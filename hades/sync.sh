#!/bin/bash

FILE="ip_config.txt"
FILE2="train_dist_embedding_hades.py"
FILE3="update_scheduler.py"
FILE4="sparse_emb.py"
FILE5="local_utils.py"
FILE6="train_dist.py"

TARGET_PATH=$(pwd)

while IFS= read -r IP_ADDRESS; do
    echo "Sending $FILE to $IP_ADDRESS..."
    scp -r $FILE $FILE2 $FILE3 $FILE4 $FILE5 $FILE6 $IP_ADDRESS:$TARGET_PATH
    echo "File sent to $IP_ADDRESS"
done < $FILE
