python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 3 --fan_out '5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5_1' --delay 1"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 3 --fan_out '5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5_2' --delay 2"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 3 --fan_out '5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5_3' --delay 3"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 3 --fan_out '5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5_4' --delay 4"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 3 --fan_out '5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5_5' --delay 5"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 2 --fan_out '5,5' --eval_every 2 --dgl_sparse --outlog '5-5_1' --delay 1"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 2 --fan_out '5,5' --eval_every 2 --dgl_sparse --outlog '5-5_2' --delay 2"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 2 --fan_out '5,5' --eval_every 2 --dgl_sparse --outlog '5-5_3' --delay 3"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 2 --fan_out '5,5' --eval_every 2 --dgl_sparse --outlog '5-5_4' --delay 4"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 2 --fan_out '5,5' --eval_every 2 --dgl_sparse --outlog '5-5_5' --delay 5"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 4 --fan_out '5,5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5-5_1' --delay 1"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 4 --fan_out '5,5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5-5_2' --delay 2"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 4 --fan_out '5,5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5-5_3' --delay 3"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 4 --fan_out '5,5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5-5_4' --delay 4"

python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-products-1p/ogb-product.json \
--ip_config ip_config_local.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive_test.py --graph_name ogb-product --ip_config ip_config_local.txt --num_epochs 20 --batch_size 100 --num_gpus 8 --num_hidden 256 --num_layers 4 --fan_out '5,5,5,5' --eval_every 2 --dgl_sparse --outlog '5-5-5-5_5' --delay 5"

