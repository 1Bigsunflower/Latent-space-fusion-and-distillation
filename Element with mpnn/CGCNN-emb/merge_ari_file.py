import os
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))

# === 配置 ===
folder = os.path.join(script_dir, "cgcnn_emb_ari")
output_file = os.path.join(script_dir, "cgcnn_emb_ari.csv")

dims = [8, 16, 32, 64]
folds = [0, 1, 2, 3, 4]
tasks = ['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric',
    'matbench_log_gvrh', 'matbench_log_kvrh', 'matbench_perovskites',
    'matbench_mp_gap', 'matbench_mp_e_form'
]

output_path = os.path.join(folder, output_file)

with open(output_path, 'w', newline='') as out_csv:
    writer = None

    for dim in dims:
        for task in tasks:
            for fold in folds:
                filename = f"{task}_emb_fold{fold}_dim{dim}.csv"
                filepath = os.path.join(folder, filename)

                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    row_with_filename = {"task": f"{task}_emb_fold{fold}_dim{dim}"}
                    row_with_filename.update(row)

                    if writer is None:
                        fieldnames = ["task"] + list(row.keys())
                        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
                        writer.writeheader()

                    writer.writerow(row_with_filename)

