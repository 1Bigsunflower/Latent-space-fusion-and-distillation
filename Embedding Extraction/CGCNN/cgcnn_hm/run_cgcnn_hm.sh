#!/bin/bash

DIM_NODE_EMBEDS=(8 16 32 64)
FOLDS=(0 1 2 3 4)
TASK_NAMES=("matbench_jdft2d" "matbench_phonons" "matbench_dielectric" "matbench_log_gvrh" "matbench_log_kvrh" "matbench_perovskites" "matbench_mp_gap" "matbench_mp_e_form")

for dim_node_embed in "${DIM_NODE_EMBEDS[@]}"
do
    for fold in "${FOLDS[@]}"
    do
        for task_name in "${TASK_NAMES[@]}"
        do
            python cgcnn_get_hm.py \
                --atom_fea_len $dim_node_embed \
                --task_name $task_name \
                --fold $fold

            if [ $? -ne 0 ]; then
                echo "Error encountered with task: $task_name, dim_node_embed: $dim_node_embed, fold: $fold"
                exit 1
            fi
        done
    done
done

echo "All tasks completed successfully!"
