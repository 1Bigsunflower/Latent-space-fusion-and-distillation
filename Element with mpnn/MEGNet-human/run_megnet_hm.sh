#!/bin/bash

DIM_NODE_EMBEDS=(8 16 32 64)

for dim_node_embed in "${DIM_NODE_EMBEDS[@]}"
do
    python fix_embedding.py --dim_node_embed $dim_node_embed

    if [ $? -ne 0 ]; then
        echo "Error encountered  dim_node_embed: $dim_node_embed"
        exit 1
    fi
done
