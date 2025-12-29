#!/bin/bash

# 创建log文件夹
log_dir="jdft2d_and_dielectric"
mkdir -p "$log_dir"

dim_node_embed_values=(8)
a_list=(1 1 1 0.01 0.0001)
b_list=(0.0001 0.01 1 1 1)

max_jobs=1  # 最多同时运行的进程数

for dim in "${dim_node_embed_values[@]}"
do
  for i in "${!a_list[@]}"
  do
    a=${a_list[$i]}
    b=${b_list[$i]}
    echo "Running with dim_node_embed = $dim, a=$a, b=$b"

    # 修改log文件路径，存放到matbench文件夹中
    log_name="megnet_dim${dim}_e${a}_f${b}.log"
    log_path="$log_dir/$log_name"

    # 启动后台进程，输出到指定文件夹中的log文件
    nohup python megnet_train.py \
        --dim_node_embed "$dim" \
        --a "$a" \
        --b "$b" > "$log_path" 2>&1 &

    # 检查当前后台任务数，如果 >= max_jobs 就等待
    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
      sleep 2  # 等待 2 秒再检查
    done
  done
done

# 等待剩余所有后台任务结束
wait
echo "All jobs finished. Logs are saved in '$log_dir' directory."