#!/bin/bash

DIM_NODE_EMBEDS=(8 16 32 64)
TASK_NAMES=("matbench_mp_gap" "matbench_mp_e_form")

for dim_node_embed in "${DIM_NODE_EMBEDS[@]}"
do
    for task_name in "${TASK_NAMES[@]}"
    do
        python megnet_orig.py --dim_node_embed $dim_node_embed --task_name $task_name

        if [ $? -ne 0 ]; then
            echo "Error encountered with task: $task_name, dim_node_embed: $dim_node_embed"
            exit 1
        fi
    done
done

