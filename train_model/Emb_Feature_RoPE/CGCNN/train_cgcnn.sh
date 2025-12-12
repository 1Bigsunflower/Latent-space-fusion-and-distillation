#!/bin/bash

atom_fea_len_value=(8 16 32 64)

# a:b combinations
a_list=(1 1 1 0.01 0.0001)
b_list=(0.0001 0.01 1 1 1)

MAX_JOBS=8   # 最大并行数

for dim in "${atom_fea_len_value[@]}"
do
    for i in "${!a_list[@]}"
    do
        a=${a_list[$i]}
        b=${b_list[$i]}

        echo "Running atom_fea_len=$dim with a=$a, b=$b"

        log_name="cgcnn_dim${dim}_a${a}_b${b}.log"

        # 启动任务
        nohup python cgcnn_lightning.py \
            --atom_fea_len "$dim" \
            --a "$a" \
            --b "$b" > "$log_name" 2>&1 &

        # 如果正在运行的任务数达到上限，则等待任意一个结束
        while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
            wait -n
        done
    done
done

# 等待所有子任务完成
wait

echo "Finish."
