#! /bin/bash

atom_fea_len_value=(8 16 32 64)

# a:b combinations
a_list=(1 1 1 0.01 0.0001)
b_list=(0.0001 0.01 1 1 1)

for dim in "${atom_fea_len_value[@]}"
do
    for i in "${!a_list[@]}"
    do
        a=${a_list[$i]}
        b=${b_list[$i]}

        echo "Running atom_fea_len=$dim with a=$a, b=$b"

        log_name="cgcnn_dim${dim}_a${a}_b${b}.log"

        nohup python cgcnn_lightning.py \
            --atom_fea_len "$dim" \
            --a "$a" \
            --b "$b" > "$log_name" 2>&1 &
    done
done

echo "Finish."

