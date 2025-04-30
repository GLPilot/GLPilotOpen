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


def run(args, device, data):
    # unpack data
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
    # set embedding size
    if args.num_embdim == 0:
        embdim = in_feats
    else:
        embdim = args.num_embdim

    # init communication groups
    # assumue we have for machines
    # TODO: support all machines
    rank = th.distributed.get_rank()
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

    if rank in g0:
        local_group = group_0
    if rank in g1:
        local_group = group_1
    if rank in g2:
        local_group = group_2
    if rank in g3:
        local_group = group_3

    num_nodes = g.num_nodes()
    num_iters = len(train_nid) // 1000 + 1

    # init buffer
    # TODO: finish emb_buffer
    emb_buffer = EmbBuffer(
        g.num_nodes(), 
        embdim, 
        emb_name="sage", 
        local_group=local_group
    )
    # TODO: need to remove
    local_embedding_buffer = ShmTensor("sage-embbuff", (g.num_nodes(), embdim), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.float32)
    local_gradient_buffer = ShmTensor("sage-gradbuff", (g.num_nodes(), embdim), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.float32)
    # policy tensor
    local_policy_fetch = ShmTensor("policy-fetch", (g.num_nodes(), num_iters), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.bool)
    local_policy_push = ShmTensor("policy-push", (g.num_nodes(), num_iters), torch.distributed.get_rank(local_group),
                                 torch.distributed.get_world_size(local_group), None, torch.bool)

    # get partition ranges
    part_book = g.get_partition_book()
    num_parts = part_book.num_partitions()
    partition_ranges = {}

    part_tensor = torch.zeros((g.num_nodes()), dtype = torch.int)
    for part_id in range(num_parts):
        local_start = part_book._max_node_ids[part_id - 1] if part_id > 0 else 0
        local_end = part_book._max_node_ids[part_id]
        partition_ranges[part_id] = (local_start, local_end)
        part_tensor[local_start:local_end] = part_id
        if part_book.partid == part_id:
            print(f"Partition {part_id}: Start = {local_start}, End = {local_end}")
    local_start = partition_ranges[part_book.partid][0]
    local_end = partition_ranges[part_book.partid][1]

    # init emb_layer
    # TODO: finish emb_layer
    emb_layer = DistEmb(
        g.num_nodes(),
        embdim,
        dgl_sparse_emb=args.dgl_sparse,
        dev_id=device,
        local_embedding_buffer = local_embedding_buffer, ## no need?
        local_start=local_start,
        local_end=local_end
    )

    # init model
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
    
    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # init embedding optimizer
    if args.dgl_sparse:
        # TODO: finish emb_optimizer
        emb_optimizer = SparseAdam(
            [emb_layer.sparse_emb], lr=args.sparse_lr, 
            local_gradient_buffer=local_gradient_buffer, ## no need?
            local_start=local_start,
            local_end=local_end
        )
        print("optimize DGL sparse embedding:", emb_layer.sparse_emb)
    else:
        assert args.dgl_sparse is True

    # Training loop
    iter_output = []
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()
        sample_time = 0
        load_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        RangePush("sampling")
        epoch_start = time.time()

        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            step_time = []
            
            # prepare pre-sampling data
            dataloader_list = []
            sample_count = torch.zeros(
                num_nodes, 
                num_iters, 
                dtype=torch.int8, 
                device=device
            )
            sample_count_per_machine_sum = torch.zeros(
                num_nodes, 
                dtype=torch.int32, 
                device=device
            )
            if rank == 0:
                print("start pre-sampling")
            
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # save sampled batches and count sampling result
                dataloader_list.append((input_nodes, seeds, blocks))
                sample_count[input_nodes, step] += 1
                sample_count_per_machine_sum[input_nodes] += 1
                
            if rank == 0:
                print("get sampled batches data finished")

            # TODO: finish reduce_per_sample_data
            group_list = g0, g1, g2, g3, gmain, group_0, group_1, group_2, group_3, group_main
            sample_count_per_machine_list, valid_total_count_mask = reduce_per_sample_data(
                sample_count, 
                sample_count_per_machine_sum,
                part_tensor,
                device,
                args,
                group_list
            )

            if rank == 0:
                print("sync sample data finished")

            # CV computing
            cv_tensor = cv_computing(sample_count_per_machine_list, valid_total_count_mask, args)

            if rank in [0, args.num_gpus*1, args.num_gpus*2, args.num_gpus*3]:
                condition0 = cv_tensor == -1 # normal
                condition1 = (cv_tensor >= 0) & (cv_tensor < 0.5) # average
                condition2 = (cv_tensor >= 0.5) & (cv_tensor < 0.9) # local remote
                condition3 = (cv_tensor >= 0.9) & (cv_tensor < 1.3) # local remote
                condition4 = (cv_tensor >= 1.3) & (cv_tensor < 1.7) # local
                condition5 = (cv_tensor >= 1.7) # local only

            if rank == 0:
                class_0_nodes = torch.where(condition0)[0]
                class_1_nodes = torch.where(condition1)[0]
                class_2_nodes = torch.where(condition2)[0]
                class_3_nodes = torch.where(condition3)[0]
                class_4_nodes = torch.where(condition4)[0]
                class_5_nodes = torch.where(condition5)[0]
                class_0_percentage = len(class_0_nodes) / num_nodes
                class_1_percentage = len(class_1_nodes) / num_nodes
                class_2_percentage = len(class_2_nodes) / num_nodes
                class_3_percentage = len(class_3_nodes) / num_nodes
                class_4_percentage = len(class_4_nodes) / num_nodes
                class_5_percentage = len(class_5_nodes) / num_nodes
                print("Class 0 nodes percentage: {:.2%}".format(class_0_percentage))
                print("Class 1 nodes percentage: {:.2%}".format(class_1_percentage))
                print("Class 2 nodes percentage: {:.2%}".format(class_2_percentage))
                print("Class 3 nodes percentage: {:.2%}".format(class_3_percentage))
                print("Class 4 nodes percentage: {:.2%}".format(class_4_percentage))
                print("Class 5 nodes percentage: {:.2%}".format(class_5_percentage))

            if rank == 0:
                print("cv computing finished")

            th.distributed.barrier()

            if rank in [0, args.num_gpus*1, args.num_gpus*2, args.num_gpus*3]:
                local_policy_fetch.tensor_[:] = th.zeros((g.num_nodes(), num_iters), dtype=th.bool)
                local_policy_push.tensor_[:] = th.zeros((g.num_nodes(), num_iters), dtype=th.bool)
            
            th.distributed.barrier()

            sample_time = time.time() - epoch_start

            for step, (input_nodes, seeds, blocks) in enumerate(dataloader_list):
                # load
                load_start = time.time()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                blocks = [block.to(device) for block in blocks]
                batch_labels = g.ndata["labels"][seeds].long().to(device)
                batch_inputs = emb_layer(
                    input_nodes, 
                    step, 
                    local_policy_fetch, 
                    False, 
                    emb_buffer
                )
                load_time += time.time() - load_start
                # forward
                forward_start = time.time()
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                forward_time += time.time() - forward_start
                # backward
                backward_start = time.time()
                emb_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                backward_time += time.time()
                # update
                update_start = time.time()
                emb_optimizer.step_sche(
                    step, 
                    local_policy_push, 
                    args.num_gpus, 
                    local_group, 
                    emb_buffer
                )
                optimizer.step()
                update_time += time.time() - update_start
                step_t = time.time() - load_start
                step_time.append(step_t)
                iter_output.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
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
                            np.mean(iter_output[3:]),
                            gpu_mem_alloc,
                            np.sum(step_time[-args.log_every:]),
                        )
                    )
            epoch_start = time.time()

        toc = time.time()
        print(
            "Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, load: {:.4f}, forward"
            ": {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs"
            ": {}".format(
                th.distributed.get_rank() % args.num_gpus,
                toc - tic,
                sample_time,
                load_time,
                forward_time,
                backward_time,
                update_time,
                num_seeds,
                num_inputs,
            )
        )
        epoch += 1


def cv_computing(sample_count_per_machine_list, valid_total_count_mask, args):
    rank = th.distributed.get_rank()
    if rank in [0, args.num_gpus*1, args.num_gpus*2, args.num_gpus*3]:
        total_new_tensor = torch.stack(sample_count_per_machine_list, dim=1)
        process_batch_size = 1024
        total_sum_list = []

        for start_idx in range(0, total_new_tensor.shape[0], process_batch_size):
            end_idx = min(start_idx + process_batch_size, total_new_tensor.shape[0])
            batch_tensor = total_new_tensor[start_idx:end_idx]
            sum_tensor_batch = batch_tensor.sum(dim=2)
            total_sum_list.append(sum_tensor_batch)
        sum_tensor = torch.cat(total_sum_list, dim=0)
        total_sum_tensor = sum_tensor.sum(dim=1)
        # computing cv
        mean_tensor = sum_tensor.float().mean(dim=1)
        std_dev_tensor = sum_tensor.float().std(dim=1)
        cv_tensor = std_dev_tensor / mean_tensor
        # set nan to -1
        nan_mask = torch.isnan(cv_tensor)
        cv_tensor[nan_mask] = -1
        cv_tensor_ = torch.zeros_like(valid_total_count_mask, dtype=cv_tensor.dtype)
        cv_tensor_[:] = -1
        cv_tensor_[valid_total_count_mask] = cv_tensor[:]
    else:
        cv_tensor_ = None
    th.distributed.barrier()
    return cv_tensor_


def reduce_per_sample_data(
    sample_count, 
    sample_count_per_machine_sum, 
    part_tensor,
    device,
    args,
    group_list
):
    g0, g1, g2, g3, gmain, group_0, group_1, group_2, group_3, group_main = group_list 
    rank = th.distributed.get_rank()

    th.distributed.all_reduce(
        sample_count_per_machine_sum, 
        op=torch.distributed.ReduceOp.SUM
    )

    # only total sample count is over 5, we set it as valid
    valid_total_count_mask = sample_count_per_machine_sum > 5
    valid_sample_count = sample_count[valid_total_count_mask]
    valid_part_tensor = part_tensor[valid_total_count_mask.to("cpu")].to(device)
    
    if rank == 0:
        print("valid count:{} total count:{}".format(
            valid_sample_count.shape[0],
            sample_count.shape[0]
        ))

    # reduce valid_sample_count to local rank 0
    if args.num_gpus != 1:
        if rank in g0:
            th.distributed.reduce(valid_sample_count, 0, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_0)
        if rank in g1:
            th.distributed.reduce(valid_sample_count, args.num_gpus*1, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_1)
        if rank in g2:
            th.distributed.reduce(valid_sample_count, args.num_gpus*2, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_2)
        if rank in g3:
            th.distributed.reduce(valid_sample_count, args.num_gpus*3, 
                                  op=torch.distributed.ReduceOp.SUM, group=group_3)
    th.distributed.barrier()

    # send valid_sample_count to all local rank 0
    # after this all local rank 0 get all machine sample data
    if rank in [0, args.num_gpus*1, args.num_gpus*2, args.num_gpus*3]:
        input_tensor_list = [valid_sample_count for i in range(4)]
        sample_count_per_machine_list = [torch.zeros(valid_sample_count.shape, dtype=torch.int8, device=device) for _ in range(args.num_machines)]
        th.distributed.all_to_all(
            sample_count_per_machine_list, 
            input_tensor_list, 
            group=group_main)
    else:
        sample_count_per_machine_list = []
    th.distributed.barrier()

    return sample_count_per_machine_list, valid_total_count_mask


def main(args):
    # init dgl
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(
            args.graph_name,
            part_config=args.part_config
        )
    # set local rank
    local_rank = th.distributed.get_rank() & args.num_gpus
    # partition book
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
    # use cpu or gpu
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = th.distributed.get_rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    # labels and class
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print("#labels:", n_classes)
    # pack data
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
        "--part_config", type=str, 
        help="The path to the partition config file"
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
    parser.add_argument("--num_machines", type=int, default=4)
    args = parser.parse_args()

    print(args)
    main(args)