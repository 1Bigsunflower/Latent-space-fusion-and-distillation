#! /bin/bash

dim_node_embed_values=(8 16 32 64)

for dim in "${dim_node_embed_values[@]}"
do
      echo "Running with dim_node_embed = $dim"

      nohup python megnet_hm.py --dim_node_embed "$dim" > "megnet_hm_dim${dim}.log" 2>&1
      pid=$!

      wait $pid
      echo
done