import argparse
import csv
import sys

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict
import random
from element_category import build_category_lookup
from sklearn.metrics import silhouette_score
import json
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

with open('atom_init.json', 'r') as f:
    atom_init = json.load(f)
# 建立反向映射：向量 -> 原子序数（Z）
vector_to_Z = {tuple(map(int, v)): int(k) for k, v in atom_init.items()}

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
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

embedding_keys = ["embedding_0", "embedding_1", "embedding_2", "embedding_3"]

title_mapping = {
    "embedding_0": "node_embedded",
    "embedding_1": "block_1",
    "embedding_2": "block_2",
    "embedding_3": "block_3"
}

# 存储每个元素在每个阶段的特征列表
element_embeddings = {key: defaultdict(list) for key in embedding_keys}
element_categories = {}

atomic_number_to_category = build_category_lookup()

for file in files:
    data = torch.load(file)
    for batch in data:
        node_input = batch["atom_input"]  # shape [N]
        node_input_np = node_input.cpu().numpy()

        # 将每个输入向量转换为对应的原子序数 Z
        atomic_numbers = []
        for i in range(node_input_np.shape[0]):
            vec = tuple(map(int, node_input_np[i]))
            Z = vector_to_Z.get(vec, None)
            if Z is None:
                raise ValueError(f"未能在 atom_init.json 中匹配向量: {vec}")
            atomic_numbers.append(Z)
        print(atomic_numbers)
        for key in embedding_keys:
            embeddings = batch[key]  # shape [N, D]
            for i in range(embeddings.shape[0]):
                Z = atomic_numbers[i]
                category = atomic_number_to_category[Z]
                element_categories[Z] = category
                element_embeddings[key][Z].append(embeddings[i].cpu().numpy())

print(f"------------CGCNN提取嵌入-{task_name}_hm/fold{fold}_dim{dim}------------")
# === 求每个元素在每阶段的平均嵌入 ===
avg_embeddings = {key: {} for key in embedding_keys}
for key in embedding_keys:
    for Z, vecs in element_embeddings[key].items():
        avg = np.mean(vecs, axis=0)
        avg_embeddings[key][Z] = avg

Z_to_category = {Z: element_categories[Z] for Z in avg_embeddings[embedding_keys[0]].keys()}
Zs = sorted(Z_to_category.keys())

ari_scores = []

for idx, key in enumerate(embedding_keys):
    X = np.stack([avg_embeddings[key][Z] for Z in Zs])
    labels = [Z_to_category[Z] for Z in Zs]

    le = LabelEncoder()
    true_labels = le.fit_transform(labels)

    tsne = TSNE(n_components=2, random_state=42)  # , perplexity=5, init="random")
    X_tsne = tsne.fit_transform(X)

    n_clusters = len(set(true_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    pred_labels = kmeans.fit_predict(X_tsne)

    ari_score = adjusted_rand_score(true_labels, pred_labels)
    ari_scores.append(ari_score)


os.makedirs('cgcnn_human_ari', exist_ok=True)
csv_file = os.path.join('cgcnn_human_ari', f"{task_name}_hm_fold{fold}_dim{dim}.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=embedding_keys)
    writer.writeheader()
    row = {key: f"{score:.4f}" for key, score in zip(embedding_keys, ari_scores)}
    writer.writerow(row)