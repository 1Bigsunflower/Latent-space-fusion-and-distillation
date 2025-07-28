#!/bin/bash

folders=("CGCNN-emb" "CGCNN-human" "MEGNet-emb" "MEGNet-human")
task_names=("matbench_jdft2d" "matbench_phonons" "matbench_dielectric" "matbench_log_gvrh" "matbench_log_kvrh" "matbench_perovskites" "matbench_mp_gap" "matbench_mp_e_form")
folds=(0 1 2 3 4)
dims=(8 16 32 64)

max_jobs=16
current_jobs=0

for folder in "${folders[@]}"; do
    cd "$folder"
    for task in "${task_names[@]}"; do
        for fold in "${folds[@]}"; do
            for dim in "${dims[@]}"; do
                echo "ARI----Running in $folder with task=$task, fold=$fold, dim=$dim"
                python mpnn_element_emb_ARI --task_name "$task" --fold "$fold" --dim "$dim" &
                current_jobs=$((current_jobs+1))
                if [ "$current_jobs" -ge "$max_jobs" ]; then
                    wait
                    current_jobs=0
                fi
            done
        done
    done
    cd ..
done

wait
