#bash scripts/local_run_hades_sage_link.sh | tee logs/2024-03-28-hades-sage-linkpred-products.log
#bash scripts/local_run_hades_gat_link.sh | tee logs/2024-03-28-hades-gat-linkpred-products.log
#bash scripts/local_run_dgl_sage_link.sh | tee logs/2024-03-28-dgl-sage-linkpred-products.log
#bash scripts/local_run_dgl_gat_link.sh | tee logs/2024-03-28-dgl-gat-linkpred-products.log

# python3 example/launch_train.py --workspace ~/workspace/EmbCacheManager/example/ \
#    --num_trainers 8 \
#    --num_samplers 1 \
#    --num_servers 16 \
#    --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
#    --ip_config ip_config4.txt \
#    "~/workspace/embcache_venv/bin/python3 hades_sage_nodeclass.py \
#    --hot-threshold 0.0125 --num_hidden 256 --model_num_hidden 256 \
#    --cache_server_addr 127.0.0.1 --cache_server_port 32451 --graph_name ogbn-products \
#    --ip_config ip_config4.txt --num_gpus 8 --num_epochs 10 --presampling --hot-low-threshold 0.0 \
#    --eval_every 20 --optim adam --model sage --batch_size 8192 --batch_size_eval 100000 --infer-method layerwise --state_cache"

# python3 example/launch_train.py --workspace ~/workspace/EmbCacheManager/example/ \
#    --num_trainers 8 \
#    --num_samplers 1 \
#    --num_servers 1 \
#    --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
#    --ip_config ip_config4.txt \
#    "~/workspace/embcache_venv/bin/python3 dgl_sage_nodeclass.py --num_hidden 256 --model_num_hidden 256 --graph_name ogbn-products --ip_config ip_config4.txt --num_gpus 8 --num_epochs 10 --eval_every 20 --dgl_sparse --optim adam --model sage --batch_size_eval 100000 --batch_size 8192 --infer-method layerwise"


#python3 example/launch_train.py --workspace ~/workspace/EmbCacheManager/example/ \
#   --num_trainers 8 \
#   --num_samplers 1 \
#   --num_servers 16 \
#   --part_config /home/ubuntu/workspace/datasets/reddit_ud_4p/reddit.json \
#   --ip_config ip_config4.txt \
#   "~/workspace/embcache_venv/bin/python3 hades_sage_nodeclass.py --num_hidden 256 --model_num_hidden 256 --cache_server_addr 127.0.0.1 --cache_server_port 32451 --graph_name reddit --ip_config ip_config4.txt --num_gpus 8 --num_epochs 20 --presampling --hot-low-threshold 0.0 --eval_every 20 --optim adam --model sage --batch_size_eval 100000 --infer-method layerwise"

# python3 example/launch_train.py --workspace ~/workspace/EmbCacheManager/example/ \
#    --num_trainers 8 \
#    --num_samplers 1 \
#    --num_servers 1 \
#    --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
#    --ip_config ip_config4.txt \
#    "~/workspace/embcache_venv/bin/python3 dgl_sage_unsupervised.py --graph_name ogbn-products \
#     --ip_config ip_config4.txt --num_gpus 8 --num_epochs 5 --eval_every 1 --num_hidden 128 --model_num_hidden 128 \
#     --optim adam --dgl_sparse --infer-method nodewise --batch_size 1565 --batch_size_eval 100000 --fan_out 5,5"

# python3 example/launch_train.py --workspace ~/workspace/EmbCacheManager/example/ \
#    --num_trainers 8 \
#    --num_samplers 1 \
#    --num_servers 1 \
#    --part_config /home/ubuntu/workspace/datasets/products_4part/ogbn-products.json \
#    --ip_config ip_config4.txt \
#    "~/workspace/embcache_venv/bin/python3 hades_sage_unsupervised.py --graph_name ogbn-products \
#     --ip_config ip_config4.txt --num_gpus 8 --num_epochs 5--eval_every 1 --num_hidden 128 --model_num_hidden 128 \
#     --optim adam --infer-method nodewise --batch_size 1565 --batch_size_eval 100000 --fan_out 5,5"

bash scripts/hades.sh
bash scripts/dgl.sh