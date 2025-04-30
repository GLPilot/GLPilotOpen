#!/bin/bash

FILE="ip_config.txt"
FILE2="train_dist_transductive.py"
FILE3="train_dist.py"
FILE4="train_dist_hotness.py"
FILE5="train_dist_transductive_test.py"
FILE6="ip_config_local.txt"
FILE7="train_dist_transductive_sche.py"
FILE8="update_scheduler.py"
FILE9="sparse_emb.py"
FILE10="local_utils.py"
FILE11="train_dist_transductive_dgl.py"

TARGET_PATH=$(pwd)

while IFS= read -r IP_ADDRESS; do
    echo "Sending $FILE to $IP_ADDRESS..."
    scp -r $FILE $FILE2 $FILE3 $FILE4 $FILE5 $FILE6 $FILE7 $FILE8 $FILE9 $FILE10 $FILE11 $IP_ADDRESS:$TARGET_PATH
    echo "File sent to $IP_ADDRESS"
done < $FILE

scp -r /home/ubuntu/myenv/lib/python3.10/site-packages/dgl/distributed/optim/pytorch 172.31.20.36:/home/ubuntu/myenv/lib/python3.10/site-packages/dgl/distributed/optim/
scp -r /home/ubuntu/myenv/lib/python3.10/site-packages/dgl/distributed/optim/pytorch 172.31.29.103:/home/ubuntu/myenv/lib/python3.10/site-packages/dgl/distributed/optim/
scp -r /home/ubuntu/myenv/lib/python3.10/site-packages/dgl/distributed/optim/pytorch 172.31.21.118:/home/ubuntu/myenv/lib/python3.10/site-packages/dgl/distributed/optim/