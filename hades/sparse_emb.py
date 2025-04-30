"""Define sparse embedding and optimizer."""

import torch as th
import torch
import time

# from .... import backend as F
import dgl.backend as F
# from ...dist_tensor import DistTensor
from dgl.distributed.dist_tensor import DistTensor

class DistEmbedding:
    """Distributed node embeddings.

    DGL provides a distributed embedding to support models that require learnable embeddings.
    DGL's distributed embeddings are mainly used for learning node embeddings of graph models.
    Because distributed embeddings are part of a model, they are updated by mini-batches.
    The distributed embeddings have to be updated by DGL's optimizers instead of
    the optimizers provided by the deep learning frameworks (e.g., Pytorch and MXNet).

    To support efficient training on a graph with many nodes, the embeddings support sparse
    updates. That is, only the embeddings involved in a mini-batch computation are updated.
    Please refer to `Distributed Optimizers <https://docs.dgl.ai/api/python/dgl.distributed.html#
    distributed-embedding-optimizer>`__ for available optimizers in DGL.

    Distributed embeddings are sharded and stored in a cluster of machines in the same way as
    :class:`dgl.distributed.DistTensor`, except that distributed embeddings are trainable.
    Because distributed embeddings are sharded
    in the same way as nodes and edges of a distributed graph, it is usually much more
    efficient to access than the sparse embeddings provided by the deep learning frameworks.

    Parameters
    ----------
    num_embeddings : int
        The number of embeddings. Currently, the number of embeddings has to be the same as
        the number of nodes or the number of edges.
    embedding_dim : int
        The dimension size of embeddings.
    name : str, optional
        The name of the embeddings. The name can uniquely identify embeddings in a system
        so that another DistEmbedding object can referent to the same embeddings.
    init_func : callable, optional
        The function to create the initial data. If the init function is not provided,
        the values of the embeddings are initialized to zero.
    part_policy : PartitionPolicy, optional
        The partition policy that assigns embeddings to different machines in the cluster.
        Currently, it only supports node partition policy or edge partition policy.
        The system determines the right partition policy automatically.

    Examples
    --------
    >>> def initializer(shape, dtype):
            arr = th.zeros(shape, dtype=dtype)
            arr.uniform_(-1, 1)
            return arr
    >>> emb = dgl.distributed.DistEmbedding(g.num_nodes(), 10, init_func=initializer)
    >>> optimizer = dgl.distributed.optim.SparseAdagrad([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     feats = emb(nids)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()

    Note
    ----
    When a ``DistEmbedding``  object is used in the forward computation, users
    have to invoke
    :py:meth:`~dgl.distributed.optim.SparseAdagrad.step` afterwards. Otherwise,
    there will be some memory leak.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        name=None,
        init_func=None,
        part_policy=None,
        local_embedding_buffer=None,
        local_start = None,
        local_end = None
    ):
        self._tensor = DistTensor(
            (num_embeddings, embedding_dim),
            F.float32,
            name,
            init_func=init_func,
            part_policy=part_policy,
        )
        self._trace = []
        self._name = name
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        # Check whether it is multi-gpu/distributed training or not
        if th.distributed.is_initialized():
            self._rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
        # [TODO] The following code is clearly wrong but changing it to "raise DGLError"
        # actually fails unit test.  ???
        # else:
        #     assert 'th.distributed should be initialized'
        self._optm_state = None  # track optimizer state
        self._part_policy = part_policy
        assert local_embedding_buffer is not None
        self.local_embedding_buffer = local_embedding_buffer
        assert local_start is not None
        assert local_end is not None
        self.local_start = local_start
        self.local_end = local_end
        self.time_list = [0,0,0,0,0]

    def print_time(self):
        time.sleep(self._rank * 0.01)
        print(self._rank, " emb_layer:", self.time_list)
        self.time_list = [0,0,0,0,0]

    def __call__(self, idx, device=th.device("cpu"), 
                 iter_num=None, local_policy_fetch=None,
                 eval_=False, emb_buffer=None):
        """
        idx : th.tensor
            Index of the embeddings to collect.
        device : th.device
            Target device to put the collected embeddings.

        Returns
        -------
        Tensor
            The requested node embeddings
        """
        if eval_ or (local_policy_fetch is None):
            emb = self._tensor[idx].to(device, non_blocking=True)
            if F.is_recording():
                emb = F.attach_grad(emb)
                self._trace.append((idx.to(device, non_blocking=True), emb))
            return emb
#---------------------------------------------------------------------------------------
        time0 = time.time()
        current_fetch_policy = local_policy_fetch.tensor_[idx][:, iter_num]
        fetch_sync_mask = current_fetch_policy == 0
        fetch_buff_mask = current_fetch_policy == 1
        idx_from_buffer = idx[fetch_buff_mask]
        time1 = time.time()
        # get embedding from buffer
        # get embedding from kv
        # push embedding into buffer
        buffered_emb, valid_mask = emb_buffer.get_emb(idx_from_buffer)
        buffered_emb[~valid_mask] = self._tensor[idx_from_buffer[~valid_mask]].to(th.device("cpu"), non_blocking=True)
        emb_buffer.set_emb(idx_from_buffer[~valid_mask], buffered_emb[~valid_mask])
        time2 = time.time()
        emb = torch.zeros(idx.shape[0], self._embedding_dim, dtype=torch.float32, device=th.device(device))
        emb[fetch_sync_mask] = self._tensor[idx[fetch_sync_mask]].to(th.device(device), non_blocking=True)
        time3 = time.time()
        # emb_buffer.set_emb(idx[fetch_sync_mask], emb[fetch_sync_mask])
        emb[fetch_buff_mask] = buffered_emb.to(device)
        emb = emb.to(device)
        time4 = time.time()
#---------------------------------------------------------------------------------------        
        # emb = self._tensor[idx].to(device, non_blocking=True)

        if F.is_recording():
            emb = F.attach_grad(emb)
            self._trace.append((idx.to(device, non_blocking=True), emb))
        time5 = time.time()
        self.time_list[0] += time1-time0
        self.time_list[1] += time2-time1
        self.time_list[2] += time3-time2
        self.time_list[3] += time4-time3
        self.time_list[4] += time5-time4
        return emb

    def reset_trace(self):
        """Reset the traced data."""
        self._trace = []

    @property
    def part_policy(self):
        """Return the partition policy

        Returns
        -------
        PartitionPolicy
            partition policy
        """
        return self._part_policy

    @property
    def name(self):
        """Return the name of the embeddings

        Returns
        -------
        str
            The name of the embeddings
        """
        return self._tensor.tensor_name

    @property
    def data_name(self):
        """Return the data name of the embeddings

        Returns
        -------
        str
            The data name of the embeddings
        """
        return self._tensor._name

    @property
    def kvstore(self):
        """Return the kvstore client

        Returns
        -------
        KVClient
            The kvstore client
        """
        return self._tensor.kvstore

    @property
    def num_embeddings(self):
        """Return the number of embeddings

        Returns
        -------
        int
            The number of embeddings
        """
        return self._num_embeddings

    @property
    def embedding_dim(self):
        """Return the dimension of embeddings

        Returns
        -------
        int
            The dimension of embeddings
        """
        return self._embedding_dim

    @property
    def optm_state(self):
        """Return the optimizer related state tensor.

        Returns
        -------
        tuple of torch.Tensor
            The optimizer related state.
        """
        return self._optm_state

    @property
    def weight(self):
        """Return the tensor storing the node embeddings

        Returns
        -------
        torch.Tensor
            The tensor storing the node embeddings
        """
        return self._tensor
