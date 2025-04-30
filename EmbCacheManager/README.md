# How to run

Install:

```shell
cd python
python setup.py install
```

Run:

1. Partition graph:

   See [example/utils/partition_graph.py](./example/utils/partition_graph.py).

2. Configure IP address:

   Edit [example/ip_config.txt](example/ip_config.txt), add the IPs of all the machines that will participate in the training (they can access each other by SSH without password). For example:

   ```
   192.168.1.51
   192.168.1.52
   ```

3. Launch EmbCacheServer:

   ```shell
   python3 ${PATH_TO_EmbCacheManager}/example/launch_server.py --work_name DistEmbSAGE --num_clients 2
   ```

4. Launch trainers:

   ```shell
   OMP_NUM_THREADS=${#omp_threads} python3 ${PATH_TO_EmbCacheManager}/example/launch_train.py --workspace ${PATH_TO_EmbCacheManager}/example/ \
     --num_trainers 2 \
     --num_samplers 0 \
     --num_servers 1 \
     --part_config ${PATH_TO_"part_config.json"} \
     --ip_config ip_config.txt \
     "python3 train_dist_transductive.py --work_name DistEmbSAGE --graph_name ogbn-products --ip_config ip_config.txt --num_gpus 2"
   ```

   Note: `--work_name` should be the same with the one used in launching server.