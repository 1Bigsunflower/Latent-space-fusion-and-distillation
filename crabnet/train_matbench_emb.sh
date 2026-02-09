#!/bin/bash

PYFILE="train_crabnet_matbench.py"
OUTDIR="matbench_composition_test"

mkdir -p ${OUTDIR}

emb_list=(
     "mat2vec"
    "MDS_32_cos_zscore"
    "MDS_64_cos_zscore"
    "MDS_F_32_cos_zscore"
    "MDS_E_32_cos_zscore"
    "MDS_F_64_cos_zscore"
    "MDS_E_64_cos_zscore"

)

subset_list=(
      "matbench_steels"
      "matbench_expt_gap"
      "matbench_expt_is_metal"
      "matbench_glass"
)

for subset in "${subset_list[@]}"; do
    for emb in "${emb_list[@]}"; do
        for fold in {0..4}; do

            log_file="${OUTDIR}/crabnet_${subset}_${fold}_${emb}.log"

            echo "Running: subset=${subset}, emb_method=${emb}, fold=${fold}"
            echo "Log -> ${log_file}"

            nohup python ${PYFILE} \
                --emb_method "${emb}" \
                --fold ${fold} \
                --subset "${subset}" \
                > "${log_file}" 2>&1

            echo "Finished: ${log_file}"
            echo "--------------------------------------"
        done
    done
done

echo "All jobs finished."
