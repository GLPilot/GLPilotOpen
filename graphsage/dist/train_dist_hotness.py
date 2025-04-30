import argparse
import socket
import time
from contextlib import contextmanager

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

import pandas as pd
output_data_list = []

from PyNVTX import RangePush, RangePop

import matplotlib.pyplot as plt

import cupy as cp

def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (
        g.ndata["features"][input_nodes].to(device) if load_feat else None
    )
    batch_labels = g.ndata["labels"][seeds].to(device)
    return batch_inputs, batch_labels


class DistSAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        Distributed layer-wise inference.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )

def filtered_numpy(tensor):
    return tensor.numpy()[tensor.numpy() > 0]

def compute_cdf(data):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
    return sorted_data, yvals

def coefficient_of_variation(data):
    mean = np.mean(data)
    if mean == 0:
        return np.nan
    std_dev = np.std(data)
    return std_dev / mean

def coefficient_of_variation_gpu(data):
    mean = cp.mean(data)
    if mean == 0:
        return cp.nan
    std_dev = cp.std(data)
    return std_dev / mean

def cv_process(remote_samples, remote_cv, a, b):
    cv_greater_than_one_indices = np.where((remote_cv > a) & (remote_cv <= b))[0]
    cv_greater_than_one_samples = remote_samples[cv_greater_than_one_indices]
    contributing_machines = np.argmax(cv_greater_than_one_samples, axis=1)
    machine_counts = np.bincount(contributing_machines, minlength=4)
    total_counts = machine_counts.sum()
    machine_percentages = (machine_counts / total_counts) * 100
    print(f"Machine Contribution Percentages and Counts (CV {a}-{b}):")
    for i, (count, percentage) in enumerate(zip(machine_counts, machine_percentages)):
        print(f"Machine {i}: Count = {count}, Percentage = {percentage:.2f}%")          
    print("\nSample Counts for Nodes:")
    counter = 0
    for sample in cv_greater_than_one_samples:
        if np.sum(sample) >= 15 and counter <= 10:
            print(sample)
            counter += 1

def cv_process_gpu(remote_samples, remote_cv, a, b):
    # Convert numpy arrays to cupy arrays
    remote_samples = cp.asarray(remote_samples)
    remote_cv = cp.asarray(remote_cv)

    # Perform the operations using cupy
    cv_greater_than_one_indices = cp.where((remote_cv > a) & (remote_cv <= b))[0]
    cv_greater_than_one_samples = remote_samples[cv_greater_than_one_indices]
    contributing_machines = cp.argmax(cv_greater_than_one_samples, axis=1)
    machine_counts = cp.bincount(contributing_machines, minlength=4)
    total_counts = machine_counts.sum()
    machine_percentages = (machine_counts / total_counts) * 100

    # Output results
    print(f"Machine Contribution Percentages and Counts (CV {a}-{b}):")
    for i in range(len(machine_counts)):
        count = machine_counts[i]
        percentage = machine_percentages[i]
        print(f"Machine {i}: Count = {count}, Percentage = {percentage:.2f}%")

    print("\nSample Counts for Nodes:")
    counter = 0
    for sample in cv_greater_than_one_samples:
        if cp.sum(sample) >= 15 and counter <= 10:
            print(sample)
            counter += 1


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    shuffle = True
    # prefetch_node_feats/prefetch_labels are not supported for DistGraph yet.
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    # Define model and optimizer
    model = DistSAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=device
            )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # For hotness test
    num_nodes = g.num_nodes()
    part_book = g.get_partition_book()
    part_id = part_book.partid
    print("rank: ", th.distributed.get_rank(), " part id: ", part_id)
    local_start = part_book._max_node_ids[part_id - 1] if part_id > 0 else 0
    local_end = part_book._max_node_ids[part_id]

    
    
    num_iters = len(train_nid) // 1000 + 1
    print("num_iters: ", num_iters)
    sample_count = torch.zeros(num_nodes, num_iters, dtype=torch.int8) # record total count of sampling
    # local_sample_count = torch.zeros(num_nodes, num_iters, dtype=torch.int8) # record total count of sampling of local nodes
    # remote_sample_count = torch.zeros(num_nodes, num_iters, dtype=torch.int8) # record total count of sampling of remote nodes

    # init new process group
    g0 = [0,1,2,3,4,5,6,7]
    g1 = [8,9,10,11,12,13,14,15]
    g2 = [16,17,18,19,20,21,22,23]
    g3 = [24,25,26,27,28,29,30,31]
    g0_ = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    g1_ = [0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    g2_ = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31]
    g3_ = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    group_0 = th.distributed.new_group(g0)
    group_1 = th.distributed.new_group(g1)
    group_2 = th.distributed.new_group(g2)
    group_3 = th.distributed.new_group(g3)
    group_0_ = th.distributed.new_group(g0_)
    group_1_ = th.distributed.new_group(g1_)
    group_2_ = th.distributed.new_group(g2_)
    group_3_ = th.distributed.new_group(g3_)
    
    print("start getting batch data")
    # Training loop
    epoch = 0
    for epoch in range(args.num_epochs):
        if args.breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
        tic = time.time()
        toc = time.time()

        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        if args.breakdown:
            torch.cuda.synchronize()
            torch.distributed.barrier()
        RangePush("sampling")
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph
        # as a list of blocks.
        step_time = []

        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                
                if args.hotness_test:
                    # print(input_nodes.shape[0])
                    sample_count[input_nodes, step] += 1
                    local_mask = (input_nodes >= local_start) & (input_nodes < local_end)
                    remote_mask = (input_nodes < local_start) | (input_nodes >= local_end)
                    # local_sample_count[input_nodes[local_mask], step] += 1
                    # remote_sample_count[input_nodes[remote_mask], step] += 1
                    if th.distributed.get_rank() == 0:
                        pass
                        # print(step)
                    continue
            print("getting batch data finished")
        # hotness communication and process
        rank = th.distributed.get_rank()
        if rank in g0:
            th.distributed.reduce(sample_count, 0, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_0)
        th.distributed.barrier()
        if rank in g1:
            th.distributed.reduce(sample_count, 8, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_1)
        th.distributed.barrier()
        if rank in g2:
            th.distributed.reduce(sample_count, 16, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_2)
        th.distributed.barrier()
        if rank in g3:
            th.distributed.reduce(sample_count, 24, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_3)
        th.distributed.barrier()
        if rank == 0:
            sample_count_1 = torch.zeros(num_nodes, num_iters, dtype=torch.int8)
            sample_count_2 = torch.zeros(num_nodes, num_iters, dtype=torch.int8)
            sample_count_3 = torch.zeros(num_nodes, num_iters, dtype=torch.int8)
            th.distributed.recv(sample_count_1, 8)
            th.distributed.recv(sample_count_2, 16)
            th.distributed.recv(sample_count_3, 24)
            th.distributed.barrier()
        elif rank in [8, 16, 24]:
            th.distributed.send(sample_count, 0)
            th.distributed.barrier()
        else:
            th.distributed.barrier()
        print("reduce batch data finished")

        if rank == 0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            sample_count = sample_count
            sample_count_1 = sample_count_1
            sample_count_2 = sample_count_2
            sample_count_3 = sample_count_3

            total_new_tensor = torch.stack((sample_count, sample_count_1, sample_count_2, sample_count_3), dim=1)

            # use batches
            data_length = total_new_tensor.shape[0]
            num_sub_batches = 32
            sub_batch_size = data_length // num_sub_batches
            total_sum_list = []
            cv_list = []

            for i in range(num_sub_batches):
                start_idx = i * sub_batch_size
                end_idx = data_length if i == num_sub_batches - 1 else (i + 1) * sub_batch_size

                new_tensor = total_new_tensor[start_idx:end_idx].to(device)

                sum_tensor = new_tensor.sum(dim=2)
                total_sum_tensor_batch = sum_tensor.sum(dim=1)

                # get mean and std to get cv   
                mean_tensor = sum_tensor.float().mean(dim=1)
                std_dev_tensor = sum_tensor.float().std(dim=1)
                cv_tensor_batch = std_dev_tensor / mean_tensor
                total_sum_list.append(total_sum_tensor_batch.cpu())
                cv_list.append(cv_tensor_batch.cpu())
                print(i)
            print("cv computing finished")
            # 合并所有批次的结果
            total_sum_tensor = torch.cat(total_sum_list)
            cv_tensor = torch.cat(cv_list)

            maxnum = total_sum_tensor.max()
            nan_mask = torch.isnan(cv_tensor)
            cv_tensor[nan_mask] = -1
            # # 将PyTorch张量转换为NumPy数组
            # numpy_data = total_sum_tensor.float().cpu().numpy()
            # # 使用NumPy计算分位数
            # hot_threshold = np.quantile(numpy_data, 0.9)
            hot_threshold = maxnum * 0.5
            print("hot threshold:", hot_threshold)

            condition_hot = total_sum_tensor >= hot_threshold
            condition0 = cv_tensor == -1
            condition1 = (total_sum_tensor < hot_threshold) & (total_sum_tensor > 0.25 * maxnum) & (cv_tensor >= 0) & (cv_tensor <= 0.6)
            condition2 = (total_sum_tensor < hot_threshold) & (total_sum_tensor <= 0.25 * maxnum) & (cv_tensor >= 0) & (cv_tensor <= 0.6)
            condition3 = (total_sum_tensor < hot_threshold) & (cv_tensor >= 1.4)
            condition4 = (total_sum_tensor < hot_threshold) & (cv_tensor > 0.6) & (cv_tensor < 1.4)
            
            # 获取符合各个条件的节点索引
            class_hot_nodes = torch.where(condition_hot)[0]
            class_0_nodes = torch.where(condition0)[0]
            class_1_nodes = torch.where(condition1)[0]
            class_2_nodes = torch.where(condition2)[0]
            class_3_nodes = torch.where(condition3)[0]
            class_4_nodes = torch.where(condition4)[0]
            
            # 计算总节点数
            total_nodes = total_sum_tensor.shape[0]
            
            # 计算每个类别的占比
            class_hot_percentage = len(class_hot_nodes) / total_nodes
            class_0_percentage = len(class_0_nodes) / total_nodes
            class_1_percentage = len(class_1_nodes) / total_nodes
            class_2_percentage = len(class_2_nodes) / total_nodes
            class_3_percentage = len(class_3_nodes) / total_nodes
            class_4_percentage = len(class_4_nodes) / total_nodes
            
            # 打印结果
            print("Class hot nodes percentage: {:.2%}".format(class_hot_percentage))
            print("Class 0 nodes percentage: {:.2%}".format(class_0_percentage))
            print("Class 1 nodes percentage: {:.2%}".format(class_1_percentage))
            print("Class 2 nodes percentage: {:.2%}".format(class_2_percentage))
            print("Class 3 nodes percentage: {:.2%}".format(class_3_percentage))
            print("Class 4 nodes percentage: {:.2%}".format(class_4_percentage))          

            # 分类4的节点的sum_tensor
            sum_tensor_new = total_new_tensor.sum(dim=2).to(device)
            class_4_sum_tensor = sum_tensor_new[class_4_nodes]

            # 分类为a类或b类
            class_4a_nodes = []
            class_4b_nodes = []

            # # 遍历类型四的节点
            # for idx, node_sum in enumerate(class_4_sum_tensor):
            #     total = node_sum.sum()
            #     # 检查任意两个值的组合是否满足条件
            #     for i in range(4):
            #         for j in range(i + 1, 4):
            #             if node_sum[i] + node_sum[j] > 0.9 * total and node_sum[i] > 0.25 * total and node_sum[j] > 0.25 * total:
            #                 class_4a_nodes.append(class_4_nodes[idx])
            #                 break
            #         else:
            #             continue
            #         break
            #     else:
            #         class_4b_nodes.append(class_4_nodes[idx])
            # class_4a_percentage = len(class_4a_nodes) / total_nodes
            # class_4b_percentage = len(class_4b_nodes) / total_nodes
            # # 输出结果
            # print("Class 4a nodes percentage: {:.2%}".format(class_4a_percentage))
            # print("Class 4b nodes percentage: {:.2%}".format(class_4b_percentage))

        th.distributed.barrier()



# -------------------------------------------------------------------------------------------------------------------------------------------


        # if rank == 0:
        #     print("reduce and process")
# 
        # th.distributed.reduce(sample_count, 0, op=torch.distributed.ReduceOp.SUM)
        # if rank in g0:
        #     th.distributed.reduce(local_sample_count, 0, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_0)
        # th.distributed.barrier()
        # if rank in g1:
        #     th.distributed.reduce(local_sample_count, 8, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_1)
        # th.distributed.barrier()
        # if rank in g2:
        #     th.distributed.reduce(local_sample_count, 16, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_2)
        # th.distributed.barrier()
        # if rank in g3:
        #     th.distributed.reduce(local_sample_count, 24, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_3)
        # th.distributed.barrier()
        # if rank in g0:
        #     th.distributed.reduce(remote_sample_count, 0, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_0)
        # th.distributed.barrier()
        # if rank in g1:
        #     th.distributed.reduce(remote_sample_count, 8, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_1)
        # th.distributed.barrier()
        # if rank in g2:
        #     th.distributed.reduce(remote_sample_count, 16, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_2)
        # th.distributed.barrier()
        # if rank in g3:
        #     th.distributed.reduce(remote_sample_count, 24, 
        #                           op=torch.distributed.ReduceOp.SUM, group=group_3)
        # th.distributed.barrier()
# 
        # if rank == 0:
        #     local_sample_count_1 = torch.zeros(num_nodes)
        #     local_sample_count_2 = torch.zeros(num_nodes)
        #     local_sample_count_3 = torch.zeros(num_nodes)
        #     remote_sample_count_1 = torch.zeros(num_nodes)
        #     remote_sample_count_2 = torch.zeros(num_nodes)
        #     remote_sample_count_3 = torch.zeros(num_nodes)
        #     th.distributed.recv(local_sample_count_1, src=8, tag=0)
        #     th.distributed.recv(local_sample_count_2, src=16, tag=0)
        #     th.distributed.recv(local_sample_count_3, src=24, tag=0)
        #     th.distributed.recv(remote_sample_count_1, src=8, tag=1)
        #     th.distributed.recv(remote_sample_count_2, src=16, tag=1)
        #     th.distributed.recv(remote_sample_count_3, src=24, tag=1)
        #     th.distributed.barrier()
        # elif rank in [8,16,24]:
        #     th.distributed.send(local_sample_count, dst=0, tag=0)
        #     th.distributed.send(remote_sample_count, dst=0, tag=1)
        #     th.distributed.barrier()
        # else:
        #     th.distributed.barrier()
# 
        # if rank == 0:
        #     total_local_sample_count = (
        #         local_sample_count + 
        #         local_sample_count_1 + 
        #         local_sample_count_2 + 
        #         local_sample_count_3
        #     )
# 
        #     total_remote_sample_count = (
        #         remote_sample_count + 
        #         remote_sample_count_1 + 
        #         remote_sample_count_2 + 
        #         remote_sample_count_3
        #     )
# 
        #     total_sample_count = total_local_sample_count + total_remote_sample_count
# 
        #     assert torch.equal(total_sample_count, sample_count), "Total sample count does not match the global sample count."
        #     print("equal")


            # filtered_sample_count_np = filtered_numpy(sample_count)
            # plt.figure(figsize=(12, 6))
            # plt.hist(filtered_sample_count_np, bins=range(0, int(max(filtered_sample_count_np))+1), 
            #          edgecolor='black')
            # plt.xlabel('Sampling Count')
            # plt.ylabel('Number of Nodes')
            # plt.title('Distribution of Node Global Sampling Counts')
            # plt.savefig('/home/ubuntu/workspace/figures/global_sampling_distribution.png')
            # print("1")
            # filtered_total_local_sample_count_np = filtered_numpy(total_local_sample_count)
            # plt.figure(figsize=(12, 6))
            # plt.hist(filtered_total_local_sample_count_np, bins=range(0, int(max(filtered_total_local_sample_count_np))+1), 
            #          edgecolor='black')
            # plt.xlabel('Sampling Count')
            # plt.ylabel('Number of Nodes')
            # plt.title('Distribution of Node Local Sampling Counts')
            # plt.savefig('/home/ubuntu/workspace/figures/local_sampling_distribution.png')
            # print("2")
            # filtered_total_remote_sample_count_np = filtered_numpy(total_remote_sample_count)
            # plt.figure(figsize=(12, 6))
            # plt.hist(filtered_total_remote_sample_count_np, bins=range(0, int(max(filtered_total_remote_sample_count_np))+1), 
            #          edgecolor='black')
            # plt.xlabel('Sampling Count')
            # plt.ylabel('Number of Nodes')
            # plt.title('Distribution of Node Remote Sampling Counts')
            # plt.savefig('/home/ubuntu/workspace/figures/remote_sampling_distribution.png')
            # print("3")
            # global_x, global_cdf = compute_cdf(filtered_sample_count_np)
            # local_x, local_cdf = compute_cdf(filtered_total_local_sample_count_np)
            # remote_x, remote_cdf = compute_cdf(filtered_total_remote_sample_count_np)
            # plt.figure(figsize=(12, 6))
            # plt.plot(global_x, global_cdf, label='Global Sampling Count CDF', marker='o')
            # plt.plot(local_x, local_cdf, label='Local Sampling Count CDF', marker='o')
            # plt.plot(remote_x, remote_cdf, label='Remote Sampling Count CDF', marker='o')
            # plt.xlabel('Sampling Count')
            # plt.ylabel('CDF')
            # plt.title('CDF of Node Sampling Counts')
            # plt.legend()
            # plt.savefig('/home/ubuntu/workspace/figures/cdf_sampling_distribution.png')
            # print("4")
            # # std_dev global
            # all_samples = torch.stack([
            #     local_sample_count,
            #     local_sample_count_1,
            #     local_sample_count_2,
            #     local_sample_count_3,
            #     remote_sample_count,
            #     remote_sample_count_1,
            #     remote_sample_count_2,
            #     remote_sample_count_3
            # ], dim=1).numpy()
            # std_dev = np.std(all_samples, axis=1)
            # plt.figure(figsize=(12, 6))
            # plt.scatter(total_sample_count.numpy(), std_dev, label='Standard Deviation', alpha=0.7, s=10)
            # plt.xlabel('Total Sampling Count')
            # plt.ylabel('Standard Deviation of Sampling Count Across Machines')
            # plt.title('Scatter Plot of Node Sampling Counts vs Standard Deviation')
            # plt.legend()
            # plt.savefig('/home/ubuntu/workspace/figures/scatter_std_dev.png')
# 
            # # std_dev remote
            # remote_samples = torch.stack([
            #     remote_sample_count,
            #     remote_sample_count_1,
            #     remote_sample_count_2,
            #     remote_sample_count_3
            # ], dim=1).numpy()
            # remote_std_dev = np.std(remote_samples, axis=1)
            # plt.figure(figsize=(12, 6))
            # plt.scatter(total_remote_sample_count.numpy(), remote_std_dev, label='Standard Deviation', alpha=0.7, s=10)
            # plt.xlabel('Total Sampling Count')
            # plt.ylabel('Standard Deviation of Sampling Count Across Machines')
            # plt.title('Scatter Plot of Node Sampling Counts vs Standard Deviation')
            # plt.legend()
            # plt.savefig('/home/ubuntu/workspace/figures/scatter_remote_std_dev.png')

            # all_samples = torch.stack([
            #     local_sample_count,
            #     local_sample_count_1,
            #     local_sample_count_2,
            #     local_sample_count_3,
            #     remote_sample_count,
            #     remote_sample_count_1,
            #     remote_sample_count_2,
            #     remote_sample_count_3
            # ], dim=1).numpy()
            # all_cv = np.apply_along_axis(coefficient_of_variation, 1, all_samples)
            # plt.figure(figsize=(12, 6))
            # plt.scatter(total_sample_count.numpy(), all_cv, label='Coefficient of Variation', alpha=0.7, s=5)
            # plt.xlabel('Total Sampling Count')
            # plt.ylabel('Coefficient of Variation Across Machines')
            # plt.title('Scatter Plot of Node Sampling Counts vs Coefficient of Variation')
            # plt.legend()
            # plt.savefig('/home/ubuntu/workspace/figures/scatter_cv_all.png')

            # remote_samples = torch.stack([
            #     remote_sample_count,
            #     remote_sample_count_1,
            #     remote_sample_count_2,
            #     remote_sample_count_3
            # ], dim=1).numpy()
            # remote_cv = np.apply_along_axis(coefficient_of_variation, 1, remote_samples)

            # cp.cuda.Device(0)
            # remote_samples = torch.stack([
            #     remote_sample_count,
            #     remote_sample_count_1,
            #     remote_sample_count_2,
            #     remote_sample_count_3
            # ], dim=1)
            # # 将 PyTorch Tensor 转换为 CuPy 数组
            # remote_samples = cp.array(remote_samples.numpy())
            # # 使用 CuPy 的 apply_along_axis 函数
            # remote_cv = cp.apply_along_axis(coefficient_of_variation_gpu, 1, remote_samples)

            # plt.figure(figsize=(12, 6))
            # plt.scatter(total_remote_sample_count.numpy(), remote_cv, label='Coefficient of Variation', alpha=0.7, s=5)
            # plt.xlabel('Total Remote Sampling Count')
            # plt.ylabel('Coefficient of Variation Across Machines')
            # plt.title('Scatter Plot of Remote Node Sampling Counts vs Coefficient of Variation')
            # plt.legend()
            # plt.savefig('/home/ubuntu/workspace/figures/scatter_remote_cv.png')

            # # 确定每个点的颜色（基于贡献最大的机器）
            # contributing_machine = np.argmax(remote_samples, axis=1)
            # colors = ['red', 'blue', 'green', 'purple']
            # point_colors = [colors[machine] for machine in contributing_machine]
            # # 绘制散点图
            # plt.figure(figsize=(12, 6))
            # plt.scatter(total_remote_sample_count.numpy(), remote_cv, label='Coefficient of Variation', c=point_colors, alpha=0.7, s=5)
            # plt.xlabel('Total Remote Sampling Count')
            # plt.ylabel('Coefficient of Variation Across Machines')
            # plt.title('Scatter Plot of Remote Node Sampling Counts vs Coefficient of Variation')
            # plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Machine {i+1}') for i, color in enumerate(colors)])
            # plt.savefig('/home/ubuntu/workspace/figures/scatter_remote_cv_colored.png')
            
            # print("5")
            # cv_process_gpu(remote_samples, remote_cv, 0, 0.6)
            # cv_process_gpu(remote_samples, remote_cv, 0.6, 0.8)
            # cv_process_gpu(remote_samples, remote_cv, 0.8, 1)
            # cv_process_gpu(remote_samples, remote_cv, 1, 1.2)
            # cv_process_gpu(remote_samples, remote_cv, 1.2, 1.4)
            # cv_process_gpu(remote_samples, remote_cv, 1.4, 1.6)
            # cv_process_gpu(remote_samples, remote_cv, 1.6, 2)

            # cv_greater_than_one_indices = np.where((remote_cv > 1) & (remote_cv < 1.2))[0]
            # cv_greater_than_one_samples = remote_samples[cv_greater_than_one_indices]
            # contributing_machines = np.argmax(cv_greater_than_one_samples, axis=1)
            # machine_counts = np.bincount(contributing_machines, minlength=4)
            # total_counts = machine_counts.sum()
            # machine_percentages = (machine_counts / total_counts) * 100
            # print("Machine Contribution Percentages (1-1.2):")
            # for i, percentage in enumerate(machine_percentages):
            #     print(f"Machine {i}: {percentage:.2f}%")
# 
            # cv_greater_than_one_indices = np.where((remote_cv > 1.2) & (remote_cv <= 1.4))[0]
            # cv_greater_than_one_samples = remote_samples[cv_greater_than_one_indices]
            # contributing_machines = np.argmax(cv_greater_than_one_samples, axis=1)
            # machine_counts = np.bincount(contributing_machines, minlength=4)
            # total_counts = machine_counts.sum()
            # machine_percentages = (machine_counts / total_counts) * 100
            # print("Machine Contribution Percentages and Counts (1.2-1.4):")
            # for i, (count, percentage) in enumerate(zip(machine_counts, machine_percentages)):
            #     print(f"Machine {i}: Count = {count}, Percentage = {percentage:.2f}%")          
            # # print("\nSample Counts for Nodes with CV > 1.2:")
            # # for sample in cv_greater_than_one_samples:
            # #     print(sample)
# 
            # cv_greater_than_one_indices = np.where((remote_cv > 1.4) & (remote_cv <= 1.6))[0]
            # cv_greater_than_one_samples = remote_samples[cv_greater_than_one_indices]
            # contributing_machines = np.argmax(cv_greater_than_one_samples, axis=1)
            # machine_counts = np.bincount(contributing_machines, minlength=4)
            # total_counts = machine_counts.sum()
            # machine_percentages = (machine_counts / total_counts) * 100
            # print("Machine Contribution Percentages and Counts (1.4-1.6):")
            # for i, (count, percentage) in enumerate(zip(machine_counts, machine_percentages)):
            #     print(f"Machine {i}: Count = {count}, Percentage = {percentage:.2f}%")          
            # # print("\nSample Counts for Nodes with CV > 1.2:")
            # # for sample in cv_greater_than_one_samples:
            # #     print(sample)
# 
            # cv_greater_than_one_indices = np.where(remote_cv > 1.6)[0]
            # cv_greater_than_one_samples = remote_samples[cv_greater_than_one_indices]
            # contributing_machines = np.argmax(cv_greater_than_one_samples, axis=1)
            # machine_counts = np.bincount(contributing_machines, minlength=4)
            # total_counts = machine_counts.sum()
            # machine_percentages = (machine_counts / total_counts) * 100
            # print("Machine Contribution Percentages and Counts (Over CV 1.6):")
            # for i, (count, percentage) in enumerate(zip(machine_counts, machine_percentages)):
            #     print(f"Machine {i}: Count = {count}, Percentage = {percentage:.2f}%")          
            # # print("\nSample Counts for Nodes with CV > 1.6:")
            # # for sample in cv_greater_than_one_samples:
            # #     if np.sum(sample) >= 15:
            # #         print(sample)

            # local_sample_count, _ = th.sort(local_sample_count, descending=True)
            # local_sample_count_1, _ = th.sort(local_sample_count, descending=True)
            # local_sample_count_2, _ = th.sort(local_sample_count, descending=True)
            # local_sample_count_3, _ = th.sort(local_sample_count, descending=True)
            # remote_sample_count, _ = th.sort(remote_sample_count, descending=True)
            # remote_sample_count_1, _ = th.sort(remote_sample_count_1, descending=True)
            # remote_sample_count_2, _ = th.sort(remote_sample_count_2, descending=True)
            # remote_sample_count_3, _ = th.sort(remote_sample_count_3, descending=True)
            # print(local_sample_count)
            # print(local_sample_count_1)
            # print(local_sample_count_2)
            # print(local_sample_count_3)
            # print(remote_sample_count)
            # print(remote_sample_count_1)
            # print(remote_sample_count_2)
            # print(remote_sample_count_3)
        #     th.distributed.barrier()
        # else:
        #     th.distributed.barrier()

        #
        epoch_data = {
            "Part": g.rank(),
            "Epoch Time(s)": toc - tic,
            "sample": sample_time,
            "load": load_time,
            "forward": forward_time,
            "backward": backward_time,
            "update": update_time,
            "#seeds": num_seeds,
            "#inputs": num_inputs
        }

        output_data_list.append(epoch_data)

        epoch += 1

        # if epoch % args.eval_every == 0 and epoch != 0:
        #     start = time.time()
        #     val_acc, test_acc = evaluate(
        #         model if args.standalone else model.module,
        #         g,
        #         g.ndata["features"],
        #         g.ndata["labels"],
        #         val_nid,
        #         test_nid,
        #         args.batch_size_eval,
        #         device,
        #     )
        #     print(
        #         "Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format(
        #             g.rank(), val_acc, test_acc, time.time() - start
        #         )
        #     )

    # if g.rank() == 0 and args.outlog is not None:
    #     df = pd.DataFrame(output_data_list)
    #     df.to_excel("/home/ubuntu/workspace/logs/"+args.graph_name+"_"+args.outlog+".xlsx", index=False)


def main(args):
    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(socket.gethostname(), "rank:", g.rank())

    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        "part {}, train: {} (local: {}), val: {} (local: {}), test: {} "
        "(local: {})".format(
            g.rank(),
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    del local_nid
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    n_classes = args.n_classes
    if n_classes == 0:
        labels = g.ndata["labels"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["features"].shape[1]
    if g.rank() == 0:
        print("in_feats: ", in_feats)
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    parser.add_argument("--outlog", type=str, default=None)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--hotness_test", action="store_true")
    args = parser.parse_args()

    print(args)
    main(args)
