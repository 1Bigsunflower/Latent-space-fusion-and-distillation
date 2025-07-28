#!/bin/bash

DIM_NODE_EMBEDS=(8 16 32 64)

for dim_node_embed in "${DIM_NODE_EMBEDS[@]}"
do
    python cgcnn_lightning.py --atom_fea_len $dim_node_embed

    if [ $? -ne 0 ]; then
        echo "Error encountered with  dim_node_embed: $dim_node_embed"
        exit 1
    fi

done

