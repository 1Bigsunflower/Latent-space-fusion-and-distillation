#!/bin/bash

dim_node_embed_values=(8 16 32 64)
a_list=(1 1 1 0.01 0.0001)
b_list=(0.0001 0.01 1 1 1)

max_jobs=8  # 最多同时运行的进程数

for dim in "${dim_node_embed_values[@]}"
do
  for i in "${!a_list[@]}"
  do
    a=${a_list[$i]}
    b=${b_list[$i]}
    echo "Running with dim_node_embed = $dim, a=$a, b=$b"

    log_name="megnet_dim${dim}_a${a}_b${b}.log"

    # 启动后台进程
    nohup python megnet_train.py \
        --dim_node_embed "$dim" \
        --a "$a" \
        --b "$b" > "$log_name" 2>&1 &

    # 检查当前后台任务数，如果 >= max_jobs 就等待
    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
      sleep 2  # 等待 2 秒再检查
    done
  done
done

# 等待剩余所有后台任务结束
wait
echo "All jobs finished."
