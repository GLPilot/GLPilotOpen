import argparse
import os

import dgl
import dgl.nn as dglnn

import torch
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from shmtensor.shm_tensor import ShmTensor
from sparse_optim import SparseAdam
import dgl.backend as dF



def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    arr.uniform_(-1, 1)
    return arr


class DistEmb(nn.Module):
    def __init__(
            self, num_nodes, emb_size, dgl_sparse_emb=False, dev_id="cpu", num_iters=None, part_config=None
    ):
        super().__init__()
        self.dev_id = dev_id
        self.emb_size = emb_size
        self.dgl_sparse_emb = dgl_sparse_emb
        if dgl_sparse_emb:
            self.sparse_emb = DistEmbedding(
                num_nodes, emb_size, name="sage", init_func=initializer, num_iters=num_iters, part_config=part_config
            )
        else:
            self.sparse_emb = th.nn.Embedding(num_nodes, emb_size, sparse=True)
            nn.init.uniform_(self.sparse_emb.weight, -1.0, 1.0)


    def forward(self, idx, step=None):
        # embeddings are stored in cpu
        idx = idx.cpu()
        if self.dgl_sparse_emb:
            return self.sparse_emb(idx, device=self.dev_id, step=step)
        else:
            return self.sparse_emb(idx).to(self.dev_id)


class DistEmbedding:
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        name=None,
        init_func=None,
        part_policy=None,
        num_iters=None,
        part_config=None
    ):
        self._tensor = ShmTensor("sage-emb", (num_embeddings, embedding_dim), dist.get_rank(),
                                 dist.get_world_size(), None, torch.float32)
        self._trace = []
        self._name = name
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        if th.distributed.is_initialized():
            self._rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
        self._optm_state = None
        self._part_policy = part_policy
        self._num_iters = num_iters
        self._part_config = part_config
        assert self._part_config is not None
        
        # set access info data
        self.access_tensor_list = []
        self.sum_tensor_list = []
        self.total_access_tensor = ShmTensor("sage-acctensor",
                                             (num_embeddings, num_iters),
                                             dist.get_rank(),
                                             dist.get_world_size(),
                                             None,
                                             torch.int16)
        self.total_sum_tensor = ShmTensor("sage-sumtensor",
                                          (num_embeddings,),
                                          dist.get_rank(),
                                          dist.get_world_size(),
                                          None,
                                          torch.int16)
        self.total_access_tensor.tensor_[:] = 0
        self.total_sum_tensor.tensor_[:] = 0
        # init access info data
        for i in range(self._world_size):
            self.access_tensor_list.append(ShmTensor("sage-acctensor" + str(i),
                                                     (num_embeddings, num_iters),
                                                     dist.get_rank(),
                                                     dist.get_world_size(),
                                                     None,
                                                     torch.int16))
            self.sum_tensor_list.append(ShmTensor("sage-sumtensor" + str(i),
                                                  (num_embeddings,),
                                                  dist.get_rank(),
                                                  dist.get_world_size(),
                                                  None,
                                                  torch.int16))
        for i in range(self._world_size):
            self.access_tensor_list[i].tensor_[:] = 0
            self.sum_tensor_list[i].tensor_[:] = 0

        # simulate all machine only get the access info to themselves
        # local_access_tensor_list have num_machine buffers
        # each buffer contain num_machine tensor
        # tensor_i_j stands for access data from machine j to machine i
        self.local_access_tensor_list = []
        for i in range(self._world_size):
            self.local_access_tensor_list.append([])
            for j in range(self._world_size):
                self.local_access_tensor_list[i].append(ShmTensor("sage-acctensor" + str(i) + str(j),
                                                        (num_embeddings, num_iters),
                                                        dist.get_rank(),
                                                        dist.get_world_size(),
                                                        None,
                                                        torch.int16))
                self.local_access_tensor_list[i][j].tensor_[:] = 0

        # Initialize embedding buffer and lifetime tracker
        self.embedding_buffer = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32, device='cpu')
        self.gradient_buffer = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32, device='cpu')
        self.lifetime_tracker = torch.zeros(num_embeddings, dtype=torch.int64, device='cpu')  # Initialize as 0


    def generate_lifetime(self, idx):
        return torch.randint(1, 4, (len(idx),))

    def edit_lifetime(self):
        self.lifetime_tracker[self.lifetime_tracker > 0] -= 1

    def accumulate_gradient(self, idx, grad):
        self.gradient_buffer[idx.cpu()] += grad.cpu()

    def apply_gradient(self, idx):
        grad_to_apply = self.gradient_buffer[idx]
        self.gradient_buffer[idx] = 0
        return grad_to_apply

    def process_gradients(self):
        # get last trace entry
        idx = self._trace[-1][0]
        grad = self._trace[-1][1].grad.data

        device = grad.device
        lifetime_tracker = self.lifetime_tracker[idx.cpu()].to(device)

        mask_gt1 = lifetime_tracker > 1
        mask_eq1 = lifetime_tracker == 1

        if mask_gt1.any():
            idx_gt1 = idx[mask_gt1]
            grad_gt1 = grad[mask_gt1]
            self.accumulate_gradient(idx_gt1, grad_gt1)

        if mask_eq1.any():
            idx_eq1 = idx[mask_eq1]
            grad_eq1 = grad[mask_eq1]

            accumulated_grad = self.gradient_buffer[idx_eq1.cpu()].to(grad_eq1.device)
            grad[mask_eq1] += accumulated_grad
            self.gradient_buffer[idx_eq1] = 0 
        
        if mask_gt1.any():
            updated_tensor = self._trace[-1][1][~mask_gt1].clone()
            updated_tensor.grad = grad[~mask_gt1]
            self._trace[-1] = (idx[~mask_gt1], updated_tensor)
     

    def update_access_tensor(self, input_nodes, step):
        if step is None:
            return
        self.access_tensor_list[self._rank].tensor_[input_nodes, step] += 1
        # simulate all machine only get the access info to themselves
        for i in range(self._world_size):
            input_nodes_temp = input_nodes[(input_nodes >= self._part_config[i]) & (input_nodes < self._part_config[i+1])]
            self.local_access_tensor_list[i][self._rank].tensor_[input_nodes_temp, step] += 1

        self.sliding_win_avg_batch(input_nodes, step)
        

    def get_total_access_tensor(self):
        if self._rank == 0:
            for i in range(self._world_size):
                self.total_access_tensor.tensor_[:] += self.access_tensor_list[i].tensor_[:]
            print(self.total_access_tensor.tensor_)


    def get_access_tensor_sum(self):
        if self._rank == 0:
            for i in range(self._world_size):
                sum_tensor = torch.sum(self.access_tensor_list[i].tensor_, dim=1)
                self.sum_tensor_list[i].tensor_[:] = sum_tensor
                print(self.sum_tensor_list[i].tensor_)
            self.total_sum_tensor.tensor_[:] = torch.sum(self.total_access_tensor.tensor_, dim=1)
            print(self.total_sum_tensor.tensor_)
            print(torch.sort(self.total_sum_tensor.tensor_))


    def sliding_win_avg_total(self, step_num):
        if self._rank == 0:
            num_nodes, num_iters = self.total_access_tensor.tensor_.shape
            avg_access_tensor = torch.zeros((num_nodes, num_iters - step_num + 1), dtype=torch.float32)

            for i in range(num_iters - step_num + 1):
                window_sum = torch.sum(self.total_access_tensor.tensor_[:, i:i + step_num], dim=1)
                avg_access_tensor[:, i] = window_sum.float() / step_num
            print(avg_access_tensor)
            print(num_iters - step_num + 1)


    def sliding_win_avg_batch(self, idx, step, window_length=3):
        if step is None or step < window_length:
            return None
        
        temp_list_i = []
        for i in range(self._world_size):
            idx_temp = idx[(idx >= self._part_config[i]) & (idx < self._part_config[i+1])]
            temp_list_j = []
            for j in range(self._world_size):
                window_sum = torch.sum(self.local_access_tensor_list[i][j].tensor_[idx_temp, step - window_length : step], dim=1)
                avg = window_sum.float() / window_length
                temp_list_j.append(avg.unsqueeze(1))
            temp_list_i.append(torch.cat(temp_list_j, dim=1))
        
        result = torch.zeros((idx.shape[0], self._world_size), dtype=torch.float32)
        for i in range(self._world_size):
            result[(idx >= self._part_config[i]) & (idx < self._part_config[i+1])] = temp_list_i[i]
        
        # print(result.shape)
            
    def print_local_access_tensor_list(self):
        for i in range(self._world_size):
            print(self.local_access_tensor_list[self._rank][i].tensor_[self._part_config[self._rank]:self._part_config[self._rank+1]])


    
    def __call__(self, idx, device=torch.device("cpu"), step=None):
        self.update_access_tensor(idx, step)
        emb = self._tensor.tensor_[idx].to(device, non_blocking=True)
        if dF.is_recording():
            emb = dF.attach_grad(emb)
            self._trace.append((idx.to(device, non_blocking=True), emb))
        return emb

    def reset_trace(self):
        """Reset the traced data."""
        self._trace = []

    @property
    def part_policy(self):
        return self._part_policy

    @property
    def num_embeddings(self):
        return self._num_embeddings

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def optm_state(self):
        return self._optm_state

    @property
    def weight(self):
        return self._tensor

    @property
    def name(self):
        return self._name
    

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference_with_embeddings(self, g, emb_layer, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])

        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )

            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size if l != len(self.layers) - 1 else self.out_size,
                )
            )

            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                if l == 0:
                    x = emb_layer(input_nodes)
                else:
                    x = blocks[0].srcdata["h"]

                h = layer(blocks[0], x)

                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)

                y[output_nodes] = h.to(y.device, non_blocking=True)

            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y

    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size
                    if l != len(self.layers) - 1
                    else self.out_size,
                )
            )
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y


def evaluate(model, g, num_classes, dataloader, emb_layer):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            # x = blocks[0].srcdata["feat"]
            x = emb_layer(input_nodes)
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(
    proc_id, device, g, num_classes, nid, model, use_uva, batch_size=2**16
):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        print("Test Accuracy {:.4f}".format(acc.item()))


def layerwise_infer_embedding(
    proc_id, device, g, num_classes, nid, model, emb_layer, use_uva, batch_size=2**16
):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference_with_embeddings(g, emb_layer, device, batch_size, use_uva)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        print("Test Accuracy {:.4f}".format(acc.item()))


def train(
    proc_id, nprocs, device, g, num_classes, train_idx, val_idx, model, use_uva, emb_layer, emb_optimizer, args
):
    num_iters = len(train_idx) // args.batch_size + 1
    num_pseduo_iters = num_iters // args.fake_gpus + 1
    sampler = NeighborSampler(
        [10, 10, 10], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=torch.device("cpu"),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=False,
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-3)
    for epoch in range(args.num_epoch):
        model.train()
        total_loss = 0
        #------------------------------------------------------
        dataloader_list = []
        sample_count = torch.zeros(g.num_nodes(), num_pseduo_iters, dtype=torch.int8)
        #------------------------------------------------------
        if dist.get_rank() == 0:
            print("pre-sampling epoch: ", epoch, " begin")
        
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            pseduo_step = it // args.fake_gpus
            dataloader_list.append(((input_nodes, output_nodes, blocks)))
            sample_count[input_nodes.cpu(), pseduo_step] += 1

        #-- here to process pre-sampling data - begin -----------------
        if dist.get_rank() == 0:
            sample_count_per_machine_list = [torch.zeros(g.num_nodes(), num_pseduo_iters, dtype=torch.int8) for i in range(nprocs)]
            for i in range(nprocs):
                if i == 0:
                    sample_count_per_machine_list[i][:] = sample_count[:]
                else:
                    sample_count_per_machine_list[i] = sample_count_per_machine_list[i].to(device)
                    dist.recv(sample_count_per_machine_list[i], i)
                    sample_count_per_machine_list[i] = sample_count_per_machine_list[i].to("cpu")
        else:
            dist.send(sample_count.to(device), 0)
        dist.barrier()
        if dist.get_rank() == 0:
            for i in range(nprocs-1):
                sample_count[:] +=  sample_count_per_machine_list[i+1][:]

        # if dist.get_rank() == 0:
        #     print(sample_count)
        #     for i in range(nprocs):
        #         print(sample_count_per_machine_list[i])

        if dist.get_rank() == 0:
            print("pre-sampling epoch: ", epoch, " finish")


        if args.pre_sampling_only:
            continue

        #-- here to process pre-sampling data - finish -----------------
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            dataloader_list
        ):
            input_nodes = input_nodes.to(device)
            blocks = [_.to(device) for _ in blocks]
            if it % args.fake_gpus == 0 and it !=0:
                emb_layer.sparse_emb.edit_lifetime()
            # x = blocks[0].srcdata["feat"]
            pseduo_step = it // args.fake_gpus

            x = emb_layer(input_nodes, step=pseduo_step)
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            if it % args.fake_gpus == 0 and it !=0:
                opt.zero_grad()
                emb_optimizer.zero_grad()
            loss.backward()
            # emb_layer.sparse_emb.process_gradients()
            if it % args.fake_gpus == 0 and it !=0:
                emb_optimizer.step()
                opt.step()
                total_loss += loss
                acc = (
                    evaluate(model, g, num_classes, val_dataloader, emb_layer).to(device) / nprocs
                )
                dist.reduce(acc, 0)
                if proc_id == 0:
                    print(
                        "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                            epoch, total_loss / (it + 1), acc.item()
                        )
                    )
        opt.zero_grad()
        emb_optimizer.zero_grad()
        emb_optimizer.step()
        opt.step()
        total_loss += loss
        acc = (
            evaluate(model, g, num_classes, val_dataloader, emb_layer).to(device) / nprocs
        )
        dist.reduce(acc, 0)
        if proc_id == 0:
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, total_loss / (it + 1), acc.item()
                )
            )


def run(proc_id, nprocs, devices, g, data, args):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dgl.distributed.initialize(args.ip_config)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )
    # generate part config
    part_config = [0, 630098, 1207232, 1837796, 2449029]

    num_classes, train_idx, val_idx, test_idx = data
    num_train = len(train_idx)
    # if proc_id == nprocs-1:
    #     train_idx = train_idx[proc_id*(num_train//nprocs+1):]
    # else:
    #     train_idx = train_idx[proc_id*(num_train//nprocs+1):(proc_id+1)*(num_train//nprocs+1)]
    train_idx_list = []
    train_idx_len_list = []
    for i in range(nprocs):
        train_idx_list.append(train_idx[(train_idx >= part_config[i]) & (train_idx < part_config[i+1])])
        train_idx_len_list.append(len(train_idx_list[-1]))
    max_train_idx_len = max(train_idx_len_list)
    train_idx = train_idx_list[proc_id]
    if len(train_idx) < max_train_idx_len:
        train_idx = torch.cat((train_idx,train_idx[0:max_train_idx_len-len(train_idx)]), dim=0)
    print("rank", torch.distributed.get_rank(), ": train ", len(train_idx),
                                                "/ val ", len(val_idx),
                                                "/ test", len(test_idx))

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    g = g.to(device if args.mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    # in_size = g.ndata["feat"].shape[1]
    in_size = args.emb_size
    model = SAGE(in_size, args.num_hidden, num_classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # Define emblayer and emboptimizer
    num_iters = len(train_idx) // args.batch_size + 1
    num_pseduo_iters = num_iters // args.fake_gpus + 1
    if torch.distributed.get_rank() == 0:
        print("num_iters: ", num_iters)
        print("num_pseduo_iters", num_pseduo_iters)
    emb_layer = DistEmb(
        g.num_nodes(),
        args.emb_size,
        dgl_sparse_emb=True,
        dev_id = torch.device("cuda:" + str(proc_id)),
        num_iters = num_pseduo_iters,
        part_config = part_config
    )

    emb_optimizer = SparseAdam(
        [emb_layer.sparse_emb], lr=args.sparse_lr, part_config=part_config
    )
    print("optimize DGL sparse embedding:", emb_layer.sparse_emb)
    # return 0
    # training + testing
    use_uva = args.mode == "mixed"
    train(
        proc_id,
        nprocs,
        device,
        g,
        num_classes,
        train_idx,
        val_idx,
        model,
        use_uva,
        emb_layer,
        emb_optimizer,
        args
    )
    # layerwise_infer(proc_id, device, g, num_classes, test_idx, model, use_uva)
    layerwise_infer_embedding(proc_id, device, g, num_classes, test_idx, model, emb_layer, use_uva)
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default="16",
        help="embedding dimension"
    )
    parser.add_argument(
        "--sparse_lr",
        type=float,
        default=1e-2,
        help="sparse lr rate"
    )
    parser.add_argument(
        "--ip_config",
        type=str,
        help="The file for IP configuration"
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=16
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10
    )
    parser.add_argument(
        "--fake_gpus",
        type=int,
        default=8
    )
    parser.add_argument(
        "--pre_sampling_only",
        action="store_true"
    )

    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # load and preprocess dataset
    print("Loading data")
    # dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    dataset, _ = dgl.load_graphs("data/combined_graph.dgl")
    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    num_classes = g.ndata['labels'].max().item() + 1
    train_idx = g.ndata['train_mask'].nonzero(as_tuple=True)[0]
    val_idx = g.ndata['val_mask'].nonzero(as_tuple=True)[0]
    test_idx = g.ndata['test_mask'].nonzero(as_tuple=True)[0]
    g.ndata['feat'] = g.ndata['features']
    g.ndata['label'] = g.ndata['labels']
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        num_classes,
        train_idx,
        val_idx,
        test_idx,
    )

    mp.spawn(run, args=(nprocs, devices, g, data, args), nprocs=nprocs)

