#!/bin/bash

MAX_JOBS=24
job_count=0

task_names=(
  "matbench_jdft2d"
  "matbench_phonons"
  "matbench_dielectric"
  "matbench_log_gvrh"
  "matbench_log_kvrh"
  "matbench_perovskites"
  "matbench_mp_gap"
  "matbench_mp_e_form"
)

folds=(0 1 2 3 4)
dims=(8 16 32 64)

for task in "${task_names[@]}"; do
  for fold in "${folds[@]}"; do
    for dim in "${dims[@]}"; do
      echo "Running task=$task fold=$fold dim=$dim"
      python Fig7.py --task_name "$task" --fold "$fold" --dim "$dim" &

       ((job_count++))

      if (( job_count >= MAX_JOBS )); then
        wait -n
        ((job_count--))
      fi

    done
  done
done


wait