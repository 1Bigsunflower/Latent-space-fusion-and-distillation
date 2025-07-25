#! /bin/bash

original_dir=$(pwd)

# 最大并行数量
max_parallel=4
# 计数器
parallel_count=0

cgcnn_hm_path="CGCNN_hm"
cgcnn_emb_path="CGCNN_emb"

atom_fea_len_value=(8 16 32 64)
for dim in "${atom_fea_len_value[@]}"
do
      echo "Running with atom_fea_len = $dim"

      cd "$cgcnn_hm_path" || exit 1
      nohup python cgcnn_lightning.py --atom_fea_len "$dim" > "cgcnn_hm_dim${dim}.log" 2>&1 &
      cd "$original_dir" || exit 1

      ((parallel_count++))
      if ((parallel_count >= max_parallel)); then
        wait -n
        ((parallel_count--))
        echo "1/2"
      fi

      cd "$cgcnn_emb_path" || exit 1
      nohup python cgcnn_lightning.py --atom_fea_len "$dim" > "cgcnn_emb_dim${dim}.log" 2>&1 &
      cd "$original_dir" || exit 1

      ((parallel_count++))
      if ((parallel_count >= max_parallel)); then
        wait -n
        ((parallel_count--))
      fi

done
echo "Finish."

