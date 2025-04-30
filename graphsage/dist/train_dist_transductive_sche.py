import argparse
import time

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from train_dist import DistSAGE, compute_acc

import pandas as pd
output_data_list = []

from PyNVTX import RangePush, RangePop

from update_scheduler import SparseAdam
from sparse_emb import DistEmbedding
from shmtensor.shm_tensor import ShmTensor

def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    arr.uniform_(-1, 1)
    return arr

class EmbBuffer():
    def __init__(
        self, num_nodes, emb_size, emb_name=None, local_group=None
    ):
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.local_group = local_group
        assert emb_name is not None
        assert local_group is not None
        self.emb_name = emb_name
        self.local_group = local_group
        self.embedding_buffer = ShmTensor(self.emb_name + "-embbuff-", (self.num_nodes, self.emb_size), torch.distributed.get_rank(local_group),
                                          torch.distributed.get_world_size(local_group), None, torch.float32)
        self.embedding_valid = ShmTensor(self.emb_name + "-embvalid-", (self.num_nodes, ), torch.distributed.get_rank(local_group),
                                        torch.distributed.get_world_size(local_group), None, torch.bool)
        self.gradient_buffer = ShmTensor(self.emb_name + "-gradbuff-", (self.num_nodes, self.emb_size), torch.distributed.get_rank(local_group),
                                         torch.distributed.get_world_size(local_group), None, torch.float32)
        self.gradient_valid = ShmTensor(self.emb_name + "-gradvalid-", (self.num_nodes, ), torch.distributed.get_rank(local_group),
                                        torch.distributed.get_world_size(local_group), None, torch.int32)
        self.embedding_buffer.tensor_[:] = 0
        self.embedding_valid.tensor_[:] = 0
        self.gradient_buffer.tensor_[:] = 0
        self.gradient_valid.tensor_[:] = 0

    def get_emb(self, idx):
        valid_mask = self.embedding_valid.tensor_[idx]
        return_emb = torch.empty((idx.shape[0], self.emb_size), dtype=torch.float32)
        return_emb[:] = self.embedding_buffer.tensor_[idx].detach()
        return return_emb, valid_mask

    def set_emb(self, idx, emb):
        self.embedding_buffer.tensor_[idx] = emb
        self.embedding_valid.tensor_[idx] = 1

    def get_grad(self, idx):
        valid_mask = self.gradient_valid.tensor_[idx]
        return_grad = torch.empty((idx.shape[0], self.emb_size), dtype=torch.float32)
        return_grad[:] = self.gradient_buffer.tensor_[idx]
        return_grad[valid_mask==0] = 0
        self.gradient_valid.tensor_[idx] = 0
        return return_grad

    def add_grad(self, idx, grad):
        valid_mask = self.gradient_valid.tensor_[idx]
        self.gradient_buffer.tensor_[idx[valid_mask==0]] = grad[valid_mask==0]
        self.gradient_buffer.tensor_[idx[valid_mask>0]] += grad[valid_mask>0]
        self.gradient_valid.tensor_[idx] += 1

class DistEmb(nn.Module):
    def __init__(
            self, num_nodes, emb_size, dgl_sparse_emb=False, dev_id="cpu",
            local_embedding_buffer=None, local_start=None, local_end=None
    ):
        super().__init__()
        self.dev_id = dev_id
        self.emb_size = emb_size
        self.dgl_sparse_emb = dgl_sparse_emb
        assert local_embedding_buffer is not None
        self.local_embedding_buffer = local_embedding_buffer 
        if dgl_sparse_emb:
            self.sparse_emb = DistEmbedding(
                num_nodes, emb_size, name="sage", init_func=initializer, 
                local_embedding_buffer=local_embedding_buffer,
                local_start=local_start, local_end=local_end
            )
        else:
            self.sparse_emb = th.nn.Embedding(num_nodes, emb_size, sparse=True)
            nn.init.uniform_(self.sparse_emb.weight, -1.0, 1.0)

    def forward(self, idx, iter_num=None, local_policy_fetch=None, eval_=False, emb_buffer=None):
        idx = idx.cpu()
        if self.dgl_sparse_emb:
            return self.sparse_emb(idx, device=self.dev_id, 
                                   iter_num=iter_num, local_policy_fetch=local_policy_fetch,
                                   eval_=eval_, emb_buffer=emb_buffer)
        else:
            return self.sparse_emb(idx).to(self.dev_id)


def load_embs(standalone, emb_layer, g):
    nodes = dgl.distributed.node_split(
        np.arange(g.num_nodes()), g.get_partition_book(), force_even=True
    )
    x = dgl.distributed.DistTensor(
        (
            g.num_nodes(),
            emb_layer.module.emb_size
            if isinstance(emb_layer, th.nn.parallel.DistributedDataParallel)
            else emb_layer.emb_size,
        ),
        th.float32,
        "eval_embs",
        persistent=True,
    )
    num_nodes = nodes.shape[0]
    for i in range((num_nodes + 1023) // 1024):
        idx = nodes[
            i * 1024: (i + 1) * 1024
            if (i + 1) * 1024 < num_nodes
            else num_nodes
        ]
        embeds = emb_layer(idx, eval_=True).cpu()
        x[idx] = embeds

    if not standalone:
        g.barrier()

    return x


def evaluate(
    standalone,
    model,
    emb_layer,
    g,
    labels,
    val_nid,
    test_nid,
    batch_size,
    device,
):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    if not standalone:
        model = model.module
    model.eval()
    emb_layer.eval()
    with th.no_grad():
        inputs = load_embs(standalone, emb_layer, g)
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    emb_layer.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, n_classes, g, in_feats = data
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    # Define model and optimizer
    if args.num_embdim == 0:
        embdim = in_feats
    else:
        embdim = args.num_embdim


    # init communication groups
    rank = torch.distributed.get_rank()
    g0 = [i for i in range(args.num_gpus)]
    g1 = [i + args.num_gpus for i in range(args.num_gpus)]
    g2 = [i + args.num_gpus * 2 for i in range(args.num_gpus)]
    g3 = [i + args.num_gpus * 3 for i in range(args.num_gpus)]
    gmain = [i * args.num_gpus for i in range(args.num_machines)]

    if rank == 0:
        print("g0: ", g0)
        print("g1: ", g1)
        print("g2: ", g2)
        print("g3: ", g3)
        print("gmain: ", gmain)
        

    group_0 = th.distributed.new_group(g0)
    group_1 = th.distributed.new_group(g1)
    group_2 = th.distributed.new_group(g2)
    group_3 = th.distributed.new_group(g3)
    group_main = th.distributed.new_group(gmain)

    # init buffer
    # prepare tensor for getting sampled batches data
    num_nodes = g.num_nodes()
    num_iters = len(train_nid) // 1000 + 1

    if rank in g0:
        local_group = group_0
    if rank in g1:
        local_group = group_1
    if rank in g2:
        local_group = group_2
    if rank in g3:
        local_group = group_3

    emb_buffer = EmbBuffer(g.num_nodes(), embdim, emb_name="sage", local_group=local_group)

    local_embedding_buffer = ShmTensor("sage-embbuff", (g.num_nodes(), embdim), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.float32)
    local_gradient_buffer = ShmTensor("sage-gradbuff", (g.num_nodes(), embdim), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.float32)
    local_policy_fetch = ShmTensor("policy-fetch", (g.num_nodes(), num_iters), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.bool)
    local_policy_push = ShmTensor("policy-push", (g.num_nodes(), num_iters), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.bool)


    # split local id and remote id
    part_book = g.get_partition_book()
    num_parts = part_book.num_partitions()  # 获取 partition 的总数
    partition_ranges = {}


    part_tensor = torch.zeros((g.num_nodes()), dtype = torch.int)
    print(part_tensor.shape)
    for part_id in range(num_parts):
        local_start = part_book._max_node_ids[part_id - 1] if part_id > 0 else 0
        local_end = part_book._max_node_ids[part_id]
        partition_ranges[part_id] = (local_start, local_end)
        part_tensor[local_start:local_end] = part_id
        if part_book.partid == part_id:
            print(f"Partition {part_id}: Start = {local_start}, End = {local_end}")
    local_start = partition_ranges[part_book.partid][0]
    local_end = partition_ranges[part_book.partid][1]
    
    print("emb dim is: ", embdim)
    emb_layer = DistEmb(
        g.num_nodes(),
        embdim,
        dgl_sparse_emb=args.dgl_sparse,
        dev_id=device,
        local_embedding_buffer = local_embedding_buffer,
        local_start=local_start,
        local_end=local_end
    )

    model = DistSAGE(
        embdim,
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
            dev_id = th.distributed.get_rank() % args.num_gpus

            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[dev_id], output_device=dev_id
            )

            if not args.dgl_sparse:
                emb_layer = th.nn.parallel.DistributedDataParallel(emb_layer)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.dgl_sparse:
        # emb_optimizer = dgl.distributed.optim.SparseAdam(
        #     [emb_layer.sparse_emb], lr=args.sparse_lr
        # )
        emb_optimizer = SparseAdam(
            [emb_layer.sparse_emb], lr=args.sparse_lr, 
            local_gradient_buffer=local_gradient_buffer,
            local_start=local_start,
            local_end=local_end
        )
        print("optimize DGL sparse embedding:", emb_layer.sparse_emb)
    elif args.standalone:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print("optimize Pytorch sparse embedding:", emb_layer.sparse_emb)
    else:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.module.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print(
            "optimize Pytorch sparse embedding:",
            emb_layer.module.sparse_emb
        )


    # prepare local tensor as buffer for embedding and gradientg
    # warining not shared tensor only for 1GPU training
    # assert args.num_gpus == 1, "we only support one GPU per machine for now, if need multi-gpus we need use shared memory."

    # Training loop
    iter_tput = []
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()
        sample_time = 0
        load_block_label = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        RangePush("sampling")
        start = time.time()


        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            step_time = []
            dataloader_list = []
            pre_sample_time = [0, 0, 0, 0]
            sample_count = torch.zeros(num_nodes, num_iters, dtype=torch.int8, device=device)
            sample_count_per_machine_sum = torch.zeros(num_nodes, dtype=torch.int32, device=device)
            if rank == 0:
                print("start pre-sampling")
            
            presample_start = time.time()

            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # save sampled batches and count sampling result
                dataloader_list.append((input_nodes, seeds, blocks))
                sample_count[input_nodes, step] += 1
                sample_count_per_machine_sum[input_nodes] += 1

            pre_sample_time[0] += time.time() - presample_start

            if rank == 0:
                print("get sampled batches data finished")
                print("start all reduce sum")

            reducesum_start = time.time()
            
            # reduce sample count per machine to get all nodes global result
            th.distributed.all_reduce(sample_count_per_machine_sum, op=torch.distributed.ReduceOp.SUM)
            th.distributed.barrier()

            pre_sample_time[1] += time.time() - reducesum_start

            if rank == 0:
                print("all reduce sum finished")
                print("start sync sample data")

            syncsample_start = time.time()

            not_zero_mask = sample_count_per_machine_sum > 5
            masked_sample_count = sample_count[not_zero_mask]
            masked_part_tensor = part_tensor[not_zero_mask.to("cpu")].to(device)
            
            if rank == 0:
                print(masked_sample_count.shape[0], " ", sample_count.shape[0])

            # gather sample count - 1
            if args.num_gpus != 1:
                if rank in g0:
                    th.distributed.reduce(masked_sample_count, 0, 
                                          op=torch.distributed.ReduceOp.SUM, group=group_0)
                if rank in g1:
                    th.distributed.reduce(masked_sample_count, args.num_gpus*1, 
                                          op=torch.distributed.ReduceOp.SUM, group=group_1)
                if rank in g2:
                    th.distributed.reduce(masked_sample_count, args.num_gpus*2, 
                                          op=torch.distributed.ReduceOp.SUM, group=group_2)
                if rank in g3:
                    th.distributed.reduce(masked_sample_count, args.num_gpus*3, 
                                          op=torch.distributed.ReduceOp.SUM, group=group_3)

            th.distributed.barrier()

            # gather sample count - 2
            if rank in [0, args.num_gpus*1, args.num_gpus*2, args.num_gpus*3]:
                
                input_tensor_list = [masked_sample_count for i in range(4)]
                sample_count_per_machine_list = [torch.zeros(masked_sample_count.shape, dtype=torch.int8, device=device) for _ in range(args.num_machines)]
                th.distributed.all_to_all(sample_count_per_machine_list, input_tensor_list, group=group_main)
                # 在发送和接收之前同步所有进程

                # send_reqs = []
                # recv_reqs = []
# 
                # for i in range(args.num_machines):
                #     if i == rank // args.num_gpus:
                #         print("rank{} self to {}".format(rank, i))
                #         sample_count_per_machine_list[i][:] = masked_sample_count[:]
                #         # print("rank{} self to {}".format(rank, i))
                #     else:
                #         print("rank{} send to {} with tag{}".format(rank, i*args.num_gpus, rank))
                #         send_req = th.distributed.isend(tensor=masked_sample_count, dst=i*args.num_gpus, tag=rank)
                #         # print("rank{} send to {} with tag{}".format(rank, i*args.num_gpus, rank))
                #         send_reqs.append(send_req)
                #         print("rank{} recv from {} with tag{}".format(rank, i*args.num_gpus, i*args.num_gpus))
                #         recv_req = th.distributed.irecv(tensor=sample_count_per_machine_list[i], src=i*args.num_gpus, tag=i*args.num_gpus)
                #         # print("rank{} recv from {} with tag{}".format(rank, i*args.num_gpus, i*args.num_gpus))
                #         recv_reqs.append(recv_req)
# 
                # for req in send_reqs:
                #     req.wait()
                # for req in recv_reqs:
                #     req.wait()
            
            th.distributed.barrier()

            pre_sample_time[2] += time.time() - syncsample_start

            if rank == 0:
                print("sync sample data finished")
                print("start cv computing")

            computecod_start = time.time()

            # CV computing
            if rank in [0, args.num_gpus*1, args.num_gpus*2, args.num_gpus*3]:

                total_new_tensor = torch.stack(sample_count_per_machine_list, dim=1)
                # total_new_tensor = total_new_tensor.to(device)

                # 计算总和
                batch_size_ = 1024  # 根据你的内存容量选择合适的batch大小
                total_sum_list = []

                # 批量处理
                for start_idx in range(0, total_new_tensor.shape[0], batch_size_):
                    end_idx = min(start_idx + batch_size_, total_new_tensor.shape[0])
                    batch_tensor = total_new_tensor[start_idx:end_idx]

                    sum_tensor_batch = batch_tensor.sum(dim=2)
                    total_sum_list.append(sum_tensor_batch)

                # 合并所有批次的结果
                sum_tensor = torch.cat(total_sum_list, dim=0)

                # sum_tensor = total_new_tensor.sum(dim=2)
                total_sum_tensor = sum_tensor.sum(dim=1)

                # 获取均值和标准差来计算cv
                mean_tensor = sum_tensor.float().mean(dim=1)
                std_dev_tensor = sum_tensor.float().std(dim=1)
                cv_tensor = std_dev_tensor / mean_tensor

                # 处理cv_tensor中的NaN值
                nan_mask = torch.isnan(cv_tensor)
                cv_tensor[nan_mask] = -1

                # 条件判断
                condition0 = cv_tensor == -1
                condition1 = (cv_tensor >= 0) & (cv_tensor < 0.5) # average
                condition2 = (cv_tensor >= 0.5) & (cv_tensor < 0.9)
                condition3 = (cv_tensor >= 0.9) & (cv_tensor < 1.3)
                condition4 = (cv_tensor >= 1.3) & (cv_tensor < 1.7)
                condition5 = (cv_tensor >= 1.7) # local only
                
                # condition4
                sum_tensor_condition4 = sum_tensor[condition4]
                compared_columns = masked_part_tensor[condition4]
                rows_sums= total_sum_tensor[condition4]
                selected_values = sum_tensor_condition4[torch.arange(sum_tensor_condition4.size(0)), compared_columns]
                not_condition4_mask = selected_values < 0.7 * rows_sums
                if rank == 0:
                    print("not condition4 normal: ", len(torch.where(not_condition4_mask)[0]))
                    print("total condition4: ", len(torch.where(condition4)[0]) )

                # condition2 and condition3
                condition23 = (cv_tensor >= 0.5) & (cv_tensor < 1.3)
                sum_tensor_condition23 = sum_tensor[condition23]
                compared_columns = masked_part_tensor[condition23]
                row_indices = torch.arange(sum_tensor_condition23.shape[0])
                mask = torch.ones_like(sum_tensor_condition23, dtype=torch.bool)
                mask[row_indices, compared_columns] = False
                remaining_values = sum_tensor_condition23[mask].view(sum_tensor_condition23.shape[0], sum_tensor_condition23.shape[1]-1)
                remaining_sums = remaining_values.sum(dim=1)
                not_condition23 = remaining_values >= 0.6 * remaining_sums.unsqueeze(1)
                not_condition23_mask = ~(not_condition23.any(dim=1))
                if rank == 0:
                    print("not condition23 normal: ", len(torch.where(not_condition23_mask)[0]))
                    print("total condition23: ", len(torch.where(condition23)[0]))

                indices_mask23 = torch.nonzero(condition23).squeeze()
                indices_n_mask23 = torch.nonzero(not_condition23_mask).squeeze()
                final_indices_23 = indices_mask23[indices_n_mask23]
                cv_tensor[final_indices_23] = -1
                indices_mask4 = torch.nonzero(condition4).squeeze()
                indices_n_mask4 = torch.nonzero(not_condition4_mask).squeeze()
                final_indices_4 = indices_mask4[indices_n_mask4]
                cv_tensor[final_indices_4] = -1

                cv_tensor_ = torch.zeros_like(not_zero_mask, dtype=cv_tensor.dtype)
                cv_tensor_[:] = -1
                cv_tensor_[not_zero_mask] = cv_tensor[:]

                condition0 = cv_tensor_ == -1 # normal
                condition1 = (cv_tensor_ >= 0) & (cv_tensor_ < 0.5) # average
                condition2 = (cv_tensor_ >= 0.5) & (cv_tensor_ < 0.9) # local remote
                condition3 = (cv_tensor_ >= 0.9) & (cv_tensor_ < 1.3) # local remote
                condition4 = (cv_tensor_ >= 1.3) & (cv_tensor_ < 1.7) # local
                condition5 = (cv_tensor_ >= 1.7) # local only

                # 获取符合各个条件的节点索引
                class_0_nodes = torch.where(condition0)[0]
                class_1_nodes = torch.where(condition1)[0]
                class_2_nodes = torch.where(condition2)[0]
                class_3_nodes = torch.where(condition3)[0]
                class_4_nodes = torch.where(condition4)[0]
                class_5_nodes = torch.where(condition5)[0]

                # 计算总节点数
                total_nodes = cv_tensor_.shape[0]

                # 计算每个类别的占比
                class_0_percentage = len(class_0_nodes) / total_nodes
                class_1_percentage = len(class_1_nodes) / (total_nodes)
                class_2_percentage = len(class_2_nodes) / (total_nodes)
                class_3_percentage = len(class_3_nodes) / (total_nodes)
                class_4_percentage = len(class_4_nodes) / (total_nodes)
                class_5_percentage = len(class_5_nodes) / (total_nodes)

                # 打印结果
                if rank == 0:
                    print("Class 0 nodes percentage: {:.2%}".format(class_0_percentage))
                    print("Class 1 nodes percentage: {:.2%}".format(class_1_percentage))
                    print("Class 2 nodes percentage: {:.2%}".format(class_2_percentage))
                    print("Class 3 nodes percentage: {:.2%}".format(class_3_percentage))
                    print("Class 4 nodes percentage: {:.2%}".format(class_4_percentage))
                    print("Class 5 nodes percentage: {:.2%}".format(class_5_percentage))

            pre_sample_time[3] += time.time() - computecod_start

            th.distributed.barrier()

        # generate policy
            if rank in [0, args.num_gpus*1, args.num_gpus*2, args.num_gpus*3]:
                local_policy_fetch.tensor_[:] = th.zeros((g.num_nodes(), num_iters), dtype=th.bool)
                local_policy_push.tensor_[:] = th.zeros((g.num_nodes(), num_iters), dtype=th.bool)

                # local_policy_fetch.tensor_[condition4] = 1
                # local_policy_push.tensor_[condition4] = 1
                # local_policy_fetch.tensor_[condition3] = 1
                # local_policy_push.tensor_[condition3] = 1
                # local_policy_fetch.tensor_[condition2] = 1
                # local_policy_push.tensor_[condition2] = 1


                
            th.distributed.barrier()


            for step, (input_nodes, seeds, blocks) in enumerate(dataloader_list):
                # print(input_nodes.shape[0])
                RangePop()
                tic_step = time.time()
                sample_time += tic_step - start
                load_bl_start = time.time()
                RangePush("loading")
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                blocks = [block.to(device) for block in blocks]
                batch_labels = g.ndata["labels"][seeds].long().to(device)
                load_block_label += time.time() - load_bl_start
                # Embedding layer counts for loading time
                load_start = time.time()
                batch_inputs = emb_layer(input_nodes, step, local_policy_fetch, False, emb_buffer)
                RangePop()
                load_end = time.time()
                load_time += load_end - load_bl_start
                start = time.time()
                RangePush("forward") 
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                RangePop() 
                forward_end = time.time()
                RangePush("backward")    
                emb_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                RangePop()
                compute_end = time.time()
                forward_time += forward_end - start
                backward_time += compute_end - forward_end
                RangePush("update") 
                emb_optimizer.step_sche(step, local_policy_push, args.num_gpus, local_group, emb_buffer)
                # emb_optimizer.step()
                optimizer.step()
                RangePop()
                update_time += time.time() - compute_end
            
                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    print(
                        "Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                        "Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU "
                        "{:.1f} MB | time {:.3f} s".format(
                            th.distributed.get_rank() % args.num_gpus,
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            np.mean(iter_tput[3:]),
                            gpu_mem_alloc,
                            np.sum(step_time[-args.log_every:]),
                        )
                    )
                RangePush("sampling")
                start = time.time()
        
        RangePop()
        toc = time.time()

        print(
            "Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, loadlb: {:.4f}, load: {:.4f}, forward"
            ": {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs"
            ": {}".format(
                th.distributed.get_rank() % args.num_gpus,
                toc - tic,
                sample_time,
                load_block_label,
                load_time,
                forward_time,
                backward_time,
                update_time,
                num_seeds,
                num_inputs,
            )
        )
        
        # print breakdown
        # th.distributed.barrier()
        # emb_layer.sparse_emb.print_time()
        # th.distributed.barrier()
        # emb_optimizer.print_time()
        # th.distributed.barrier()
        # emb_optimizer.print_count()

        epoch_data = {
            "Part": th.distributed.get_rank(),
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

        if epoch % args.eval_every == 0 and epoch != 0:
            start = time.time()
            val_acc, test_acc = evaluate(
                args.standalone,
                model,
                emb_layer,
                g,
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )

            # Use all_reduce to get the sum across all processes
            torch.distributed.all_reduce(val_acc, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(test_acc, op=torch.distributed.ReduceOp.SUM)

            # Get the average by dividing by the world size (total number of processes)
            world_size = torch.distributed.get_world_size()
            val_acc_avg = val_acc / world_size
            test_acc_avg = test_acc / world_size

            # Only rank 0 prints the results
            if th.distributed.get_rank() % args.num_gpus == 0:
                print(
                    "Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format(
                        th.distributed.get_rank() % args.num_gpus, val_acc_avg.item(), test_acc_avg.item(), time.time() - start
                    )
                )
            
    if th.distributed.get_rank() % args.num_gpus == 0 and args.outlog is not None:
        df = pd.DataFrame(output_data_list)
        df.to_excel("/home/ubuntu/workspace/logs/"+args.graph_name+"_"+args.outlog+".xlsx", index=False)


def main(args):
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(
            args.graph_name,
            part_config=args.part_config
        )

    local_rank = th.distributed.get_rank() % args.num_gpus
    print("rank:", th.distributed.get_rank() % args.num_gpus)

    pb = g.get_partition_book()
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
            th.distributed.get_rank() % args.num_gpus,
            len(train_nid),
            len(np.intersect1d(train_nid.numpy(), local_nid)),
            len(val_nid),
            len(np.intersect1d(val_nid.numpy(), local_nid)),
            len(test_nid),
            len(np.intersect1d(test_nid.numpy(), local_nid)),
        )
    )
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = th.distributed.get_rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["features"].shape[1]
    if th.distributed.get_rank() % args.num_gpus == 0:
        print("in_feats: ", in_feats)
    data = train_nid, val_nid, test_nid, n_classes, g, in_feats
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
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--n_classes", type=int, help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_embdim", type=int, default=0)
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
        "--dgl_sparse",
        action="store_true",
        help="Whether to use DGL sparse embedding",
    )
    parser.add_argument(
        "--sparse_lr", type=float, default=1e-2, help="sparse lr rate"
    )
    parser.add_argument("--outlog", type=str, default=None)
    parser.add_argument("--num_machines", type=int, default=4)
    args = parser.parse_args()

    print(args)
    main(args)
