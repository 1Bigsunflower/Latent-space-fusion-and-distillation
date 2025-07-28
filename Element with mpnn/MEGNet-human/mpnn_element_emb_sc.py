import argparse
import csv

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict
import random
from element_category import build_category_lookup
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument('--task_name',
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    default='matbench_jdft2d')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--dim', type=int, default=64)
args = parser.parse_args()

task_name = args.task_name
fold = args.fold
dim = args.dim

prefix = f"{task_name}_hm/fold{fold}_dim{dim}"
files = [f"{prefix}_train.pt", f"{prefix}_val.pt", f"{prefix}_test.pt"]

embedding_keys = ["node_embedded", "node_linear", "node_encoded", "block_1", "block_2", "block_3"]

# 存储每个元素在每个阶段的特征列表
element_embeddings = {key: defaultdict(list) for key in embedding_keys}
element_categories = {}

atomic_number_to_category = build_category_lookup()

for file in files:
    data = torch.load(file)
    for batch in data:
        node_input = batch["node_input"]  # shape [N]
        atomic_numbers = node_input + 1  # 转换为元素编号

        mapping = {84: 89, 85: 90, 86: 91, 87: 92, 88: 93, 89: 94}
        atomic_numbers = torch.tensor([mapping.get(int(z.item()), int(z.item())) for z in atomic_numbers])

        for key in embedding_keys:
            embeddings = batch[key]  # shape [N, D]
            for i in range(embeddings.shape[0]):
                Z = int(atomic_numbers[i].item())
                category = atomic_number_to_category[Z]
                element_categories[Z] = category
                element_embeddings[key][Z].append(embeddings[i].cpu().numpy())

print(f"------------MEGNet提取嵌入-{task_name}_hm/fold{fold}_dim{dim}------------")
# === 求每个元素在每阶段的平均嵌入 ===
avg_embeddings = {key: {} for key in embedding_keys}
for key in embedding_keys:
    for Z, vecs in element_embeddings[key].items():
        avg = np.mean(vecs, axis=0)
        avg_embeddings[key][Z] = avg

Z_to_category = {Z: element_categories[Z] for Z in avg_embeddings[embedding_keys[0]].keys()}
Zs = sorted(Z_to_category.keys())

silhouette_scores = []

for idx, key in enumerate(embedding_keys):
    X = np.stack([avg_embeddings[key][Z] for Z in Zs])
    labels = [Z_to_category[Z] for Z in Zs]

    tsne = TSNE(n_components=2, random_state=42)  # , perplexity=5, init="random")
    X_tsne = tsne.fit_transform(X)

    sil_score = silhouette_score(X_tsne, labels)
    silhouette_scores.append(sil_score)

os.makedirs('megnet_human_sc', exist_ok=True)
csv_file = os.path.join('megnet_human_sc', f"{task_name}_hm_fold{fold}_dim{dim}.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=embedding_keys)
    writer.writeheader()
    row = {key: f"{score:.4f}" for key, score in zip(embedding_keys, silhouette_scores)}
    writer.writerow(row)
