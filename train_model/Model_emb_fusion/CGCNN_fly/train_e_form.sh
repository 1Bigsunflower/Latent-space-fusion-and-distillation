#!/bin/bash

log_dir="eform"
mkdir -p "$log_dir"

# 可用GPU设备
declare -a gpus=("0" "1" "2" "3")
# 每个GPU上最大并行任务数
max_per_gpu=8
# 全局最大并行任务数
max_total_jobs=30

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

# 可用GPU和任务槽位
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

# 等待任务完成
wait_for_slot() {
    while true; do
        # 后台总任务数
        total_running=$(jobs -p | wc -l)

        # 超过全局上限
        if [ "$total_running" -ge "$max_total_jobs" ]; then
            sleep 1
            wait -n 2>/dev/null || true
            continue
        fi

        # 检查GPU槽位
        for gpu in "${gpus[@]}"; do
            running_on_gpu=$(jobs -p | xargs -I {} ps -o args= {} 2>/dev/null | grep -c "cuda_devices $gpu" || true)

            if [ "$running_on_gpu" -lt "$max_per_gpu" ]; then
                return 0
            fi
        done

        sleep 1
        wait -n 2>/dev/null || true
    done
}


for fold in "${folds[@]}"; do
    echo "=============================================="
    echo "Starting fold $fold"
    echo "=============================================="

    dataset_path="../CGCNN_dataset/${subset}/${fold}"

    # 生成当前fold的数据集
    if [ -d "$dataset_path" ]; then
        echo "Dataset for fold $fold already exists"
    else
        echo "Generating dataset for fold $fold..."
        python fly_data_cgcnn.py --subset "$subset" --fold "$fold"
    fi

    total_tasks=$(( ${#combinations[@]} ))
    completed_tasks=0

    declare -a current_fold_pids=()

    for combo in "${combinations[@]}"; do
        read -r dim a b <<< "$combo"
        wait_for_slot

        available_gpu=$(get_available_slot)

        if [ -z "$available_gpu" ]; then
            echo "No available GPU slot found, waiting..."
            wait_for_slot
            available_gpu=$(get_available_slot)
        fi

        log_name="cgcnn_${subset}_fold${fold}_dim${dim}_e${a}_f${b}.log"
        log_path="$log_dir/$log_name"

        echo "Running fold=$fold, subset=$subset, atom_fea_len=$dim, a=$a, b=$b on GPU $available_gpu"
        echo "Task $((completed_tasks + 1)) of $total_tasks for fold $fold"

        CUDA_VISIBLE_DEVICES="$available_gpu" nohup python cgcnn_lightning.py \
            --subset "$subset" \
            --fold "$fold" \
            --atom_fea_len "$dim" \
            --a "$a" \
            --b "$b" \
            --data_root "../CGCNN_dataset" \
            --cuda_devices "$available_gpu" > "$log_path" 2>&1 &

        pid=$!
        current_fold_pids+=($pid)

        ((completed_tasks++))
        echo "Started job with PID $! on GPU $available_gpu"
        echo "Progress for fold $fold: $completed_tasks/$total_tasks"

        sleep 1
    done

    unset current_fold_pids

done

echo "=============================================="
echo "All tasks for all folds finished!"
echo "=============================================="