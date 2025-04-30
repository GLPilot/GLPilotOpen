python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 8 \
--num_samplers 0 \
--num_servers 1 \
--part_config /home/ubuntu/workspace/data/ogbn-papers100M-4p/ogb-paper100M.json \
--ip_config ip_config.txt \
"/home/ubuntu/myenv/bin/python3 train_dist_transductive.py --graph_name ogb-paper100M --ip_config ip_config.txt --num_epochs 10 --batch_size 1000 --num_gpus 8 --num_hidden 256 --num_layers 3 --fan_out '5,10,15' --batch_size 1000 --eval_every 100 --outlog 'learnable' --dgl_sparse --breakdown"
