#!/bin/bash

log_name="megnet_dim64_e1_f0.01.log"

# 启动后台进程
nohup python megnet_train.py \
    --dim_node_embed 64 \
    --a 1 \
    --b 0.01 > "$log_name" 2>&1 &