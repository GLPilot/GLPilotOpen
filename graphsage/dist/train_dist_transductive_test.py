import argparse
import time

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.distributed import DistEmbedding
from train_dist import DistSAGE, compute_acc

import pandas as pd
output_data_list = []

from PyNVTX import RangePush, RangePop

def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    arr.uniform_(-1, 1)
    return arr


class DistEmb(nn.Module):
    def __init__(
            self, num_nodes, emb_size, dgl_sparse_emb=False, dev_id="cpu"
    ):
        super().__init__()
        self.dev_id = dev_id
        self.emb_size = emb_size
        self.dgl_sparse_emb = dgl_sparse_emb
        if dgl_sparse_emb:
            self.sparse_emb = DistEmbedding(
                num_nodes, emb_size, name="sage", init_func=initializer
            )
        else:
            self.sparse_emb = th.nn.Embedding(num_nodes, emb_size, sparse=True)
            nn.init.uniform_(self.sparse_emb.weight, -1.0, 1.0)

    def forward(self, idx):
        # embeddings are stored in cpu
        idx = idx.cpu()
        if self.dgl_sparse_emb:
            return self.sparse_emb(idx, device=self.dev_id)
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
        embeds = emb_layer(idx).cpu()
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
    print("emb dim is: ", embdim)
    emb_layer = DistEmb(
        g.num_nodes(),
        embdim,
        dgl_sparse_emb=args.dgl_sparse,
        dev_id=device,
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
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[dev_id], output_device=dev_id
            )
            if not args.dgl_sparse:
                emb_layer = th.nn.parallel.DistributedDataParallel(emb_layer)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.dgl_sparse:
        emb_optimizer = dgl.distributed.optim.SparseAdam(
            [emb_layer.sparse_emb], lr=args.sparse_lr
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

    # Training loop
    iter_tput = []
    epoch = 0

    # for getting hotness of nodes
    num_nodes = g.num_nodes()
    hotness_test = 1
    sample_count = torch.zeros(num_nodes)

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
        start = time.time()
        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            step_time = []
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                if hotness_test == 1:
                    sample_count[input_nodes] += 1
                    continue

                # print(input_nodes.shape[0])
                RangePop()
                tic_step = time.time()
                sample_time += tic_step - start
                load_start = time.time()
                RangePush("loading")
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                blocks = [block.to(device) for block in blocks]
                batch_labels = g.ndata["labels"][seeds].long().to(device)
                # Embedding layer counts for loading time
                batch_inputs = emb_layer(input_nodes)
                RangePop()
                load_end = time.time()
                load_time += load_end - load_start
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
                if args.delay == 1:
                    emb_optimizer.step()
                else:
                    emb_optimizer.step_delay(mask_hot, step, args.delay)
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
                            g.rank(),
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

            # now we need to allreduce these hotness data and reset hotness_test for next training
            if hotness_test == 1:
                if not args.standalone:
                    # standalone do not need allreduce
                    th.distributed.all_reduce(sample_count, op=th.distributed.ReduceOp.SUM)
                if args.block_mode == "hot":
                    hot_threshold = th.quantile(sample_count.float(), 1-args.block_rate)
                    mask_hot = sample_count >= hot_threshold
                    print("mask shape: ", mask_hot.shape)
                    true_count = torch.sum(mask_hot)
                    false_count = mask_hot.numel() - true_count.item()
                    print("hot: ", true_count/mask_hot.numel(), false_count/mask_hot.numel())
                elif args.block_mode == "cold":
                    hot_threshold = th.quantile(sample_count.float(), args.block_rate)
                    mask_hot = sample_count <= hot_threshold
                    print("mask shape: ", mask_hot.shape)
                    true_count = torch.sum(mask_hot)
                    false_count = mask_hot.numel() - true_count.item()
                    print("cold: ", true_count.item()/mask_hot.numel(), false_count/mask_hot.numel())

                hotness_test = 0
                print("get hotness data, start training now!")

        RangePop()
        toc = time.time()
        print(
            "Part {}, Epoch Time(s): {:.4f}, sample: {:.4f}, load: {:.4f}, forward"
            ": {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs"
            ": {}".format(
                g.rank(),
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
            print(
                "Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}".format
                (
                    g.rank(), val_acc, test_acc, time.time() - start
                )
            )

            epoch_data = {
                "val acc": val_acc.item(),
                "test acc": test_acc.item()
            }
    
            output_data_list.append(epoch_data)
            
    if g.rank() == 0 and args.outlog is not None:
        df = pd.DataFrame(output_data_list)
        df.to_excel("/home/ubuntu/workspace/logs/"+args.graph_name+"_"+args.outlog+".xlsx", index=False)


def main(args):
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        th.distributed.init_process_group(backend="gloo")
    g = dgl.distributed.DistGraph(
            args.graph_name,
            part_config=args.part_config
        )
    print("rank:", g.rank())

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
            g.rank(),
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
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    if args.standalone:
        device = th.device("cuda:" + str(args.device_num))
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["features"].shape[1]
    if g.rank() == 0:
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
    parser.add_argument("--lr", type=float, default=0.0003)
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
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--device_num", type=int, default=0,
                        help="device number used in standalone mode")
    parser.add_argument("--block_rate", type=float, default=0.1)
    parser.add_argument("--block_mode", type=str, default="hot")
    args = parser.parse_args()

    print(args)
    main(args)
