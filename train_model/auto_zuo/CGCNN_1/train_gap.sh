#!/bin/bash

# 创建log文件夹
log_dir="gap"
mkdir -p "$log_dir"

subset="matbench_mp_gap"

declare -a combinations=(
    "32 1 0.0001"
    "32 1 0.01"
    "32 1 1"
    "32 0.01 1"
    "32 0.0001 1"
    "64 1 0.0001"
    "64 1 0.01"
    "64 1 1"
    "64 0.01 1"
    "64 0.0001 1"
)

# 最大同时运行进程数
max_jobs=2
current_jobs=0

for combo in "${combinations[@]}"
do
    # 解析参数组合
    read -r dim a b <<< "$combo"

    echo "Running subset=$subset, atom_fea_len=$dim, a=$a, b=$b"

    # 设置log文件名和路径
    log_name="cgcnn_${subset}_dim${dim}_e${a}_f${b}.log"
    log_path="$log_dir/$log_name"

    # 运行任务
    nohup python cgcnn_lightning.py \
        --subset "$subset" \
        --atom_fea_len "$dim" \
        --a "$a" \
        --b "$b" > "$log_path" 2>&1 &

    # 更新当前任务计数
    ((current_jobs++))

    echo "Started job $current_jobs with PID $!"

    # 如果达到最大任务数，等待其中一个完成
    if [[ $current_jobs -ge $max_jobs ]]; then
        echo "Maximum jobs reached ($max_jobs), waiting for one to complete..."
        # 等待任意一个后台任务完成
        wait -n
        # 更新当前任务计数
        ((current_jobs--))
        echo "One job completed. Current running jobs: $current_jobs"
    fi
done

# 等待所有剩余的后台任务完成
echo "Waiting for all remaining jobs to complete..."
wait

echo "All tasks finished! Logs are saved in directory: $log_dir"