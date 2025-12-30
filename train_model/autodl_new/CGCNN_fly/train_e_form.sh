#!/bin/bash

log_dir="eform"
mkdir -p "$log_dir"

# 可用的GPU设备
declare -a gpus=("0" "1" "2")  #
# 每个GPU上最大并行任务数
max_per_gpu=7  #

# 数据集和fold配置
subset="matbench_mp_e_form"
declare -a folds=("0" "1" "2" "3" "4")

# 参数组合
declare -a combinations=(
    "8 1 0.0001"
    "8 1 0.01"
    "8 1 1"
    "8 0.01 1"
    "8 0.0001 1"
    "16 1 0.0001"
    "16 1 0.01"
    "16 1 1"
    "16 0.01 1"
    "16 0.0001 1"
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

# 函数：获取可用的GPU和任务槽位
get_available_slot() {
    for gpu in "${gpus[@]}"; do
        # 统计当前GPU上运行的任务数
        running_on_gpu=$(jobs -p | xargs -I {} ps -o args= {} 2>/dev/null | grep -c "cuda_devices $gpu" || true)

        if [ "$running_on_gpu" -lt "$max_per_gpu" ]; then
            echo "$gpu"
            return 0
        fi
    done
    echo ""
}

# 函数：等待任意一个任务完成
wait_for_slot() {
    while true; do
        for gpu in "${gpus[@]}"; do
            running_on_gpu=$(jobs -p | xargs -I {} ps -o args= {} 2>/dev/null | grep -c "cuda_devices $gpu" || true)

            if [ "$running_on_gpu" -lt "$max_per_gpu" ]; then
                return 0
            fi
        done

        # 如果没有可用槽位，等待1秒后重试
        sleep 1

        # 清理已完成的作业
        wait -n 2>/dev/null || true
    done
}

# 按fold顺序执行
for fold in "${folds[@]}"; do
    echo "=============================================="
    echo "Starting fold $fold"
    echo "=============================================="

    dataset_path="../CGCNN_dataset/${subset}/${fold}"

    # 第一步：生成当前fold的数据集
    if [ -d "$dataset_path" ]; then
        echo "Dataset for fold $fold already exists"
    else
        echo "Generating dataset for fold $fold..."
        python fly_data_cgcnn.py --subset "$subset" --fold "$fold"
    fi

    # 统计当前fold的总任务数
    total_tasks=$(( ${#combinations[@]} ))
    completed_tasks=0

    # 存储当前fold的所有任务PID
    declare -a current_fold_pids=()

    for combo in "${combinations[@]}"; do
        # 解析参数组合
        read -r dim a b <<< "$combo"

        # 等待可用的GPU槽位
        wait_for_slot

        # 获取可用的GPU
        available_gpu=$(get_available_slot)

        if [ -z "$available_gpu" ]; then
            echo "No available GPU slot found, waiting..."
            wait_for_slot
            available_gpu=$(get_available_slot)
        fi

        # 设置log文件名和路径
        log_name="cgcnn_${subset}_fold${fold}_dim${dim}_e${a}_f${b}.log"
        log_path="$log_dir/$log_name"

        echo "Running fold=$fold, subset=$subset, atom_fea_len=$dim, a=$a, b=$b on GPU $available_gpu"
        echo "Task $((completed_tasks + 1)) of $total_tasks for fold $fold"

        # 运行任务
        CUDA_VISIBLE_DEVICES="$available_gpu" nohup python cgcnn_lightning.py \
            --subset "$subset" \
            --fold "$fold" \
            --atom_fea_len "$dim" \
            --a "$a" \
            --b "$b" \
            --data_root "../CGCNN_dataset" \
            --cuda_devices "$available_gpu" > "$log_path" 2>&1 &

        # 保存当前任务的PID
        pid=$!
        current_fold_pids+=($pid)

        # 更新已完成任务计数
        ((completed_tasks++))
        echo "Started job with PID $! on GPU $available_gpu"
        echo "Progress for fold $fold: $completed_tasks/$total_tasks"

        # 短暂延迟，避免任务启动冲突
        sleep 1
    done

    # 等待当前fold的所有训练任务完成（只等待当前fold的PID）
    echo "=============================================="
    echo "Waiting for all training tasks in fold $fold to complete..."
    echo "=============================================="

    # 方法1：使用进程组等待（更安全）
    for pid in "${current_fold_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done

    # 方法2：或者使用jobs命令（但需要确保在当前shell中）
    # wait "${current_fold_pids[@]}" 2>/dev/null || true

    # 清空PID数组，为下一个fold做准备
    unset current_fold_pids

    # 删除当前fold的数据文件夹
    echo "Cleaning up dataset for fold $fold..."

    if [ -d "$dataset_path" ]; then
        echo "Removing dataset directory: $dataset_path"
        rm -rf "$dataset_path"
    else
        echo "Dataset directory not found: $dataset_path"
    fi
    echo ""

    # 检查是否是最后一个fold，如果不是则继续
    if [ "$fold" != "${folds[-1]}" ]; then
        echo "Fold $fold processing completed. Ready for next fold."
        sleep 3
    fi
done

echo "=============================================="
echo "All tasks for all folds finished!"
echo "=============================================="