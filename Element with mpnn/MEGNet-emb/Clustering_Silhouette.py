import argparse
import csv
import sys

import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import defaultdict
import random
from element_category import build_category_lookup
from sklearn.metrics import silhouette_score
import periodictable

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    default='matbench_log_gvrh')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--dim', type=int, default=16)
args = parser.parse_args()

task_name = args.task_name
fold = args.fold
dim = args.dim

prefix = f"{task_name}_emb/fold{fold}_dim{dim}"
files = [f"{prefix}_train.pt", f"{prefix}_val.pt", f"{prefix}_test.pt"]

embedding_keys = ["block_3"]

# 存储每个元素在每个阶段的特征列表
element_embeddings = {key: defaultdict(list) for key in embedding_keys}

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
                element_embeddings[key][Z].append(embeddings[i].cpu().numpy())

print(f"------------MEGNet提取嵌入-{task_name}_emb/fold{fold}_dim{dim}------------")
# === 求每个元素在每阶段的平均嵌入 ===
avg_embeddings = {key: {} for key in embedding_keys}
for key in embedding_keys:
    for Z, vecs in element_embeddings[key].items():
        avg = np.mean(vecs, axis=0)
        avg_embeddings[key][Z] = avg

output_dir = "MEGNet_emb_kmeans_sil"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"{task_name}_fold{fold}_dim{dim}.csv")

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["k", "silhouette_score"])  # CSV标题

    # 对每个 embedding_key 进行聚类
    for key in embedding_keys:
        # 构建数据矩阵和元素列表
        Z_list = sorted(avg_embeddings[key].keys())
        X = np.array([avg_embeddings[key][Z] for Z in Z_list])  # shape [num_elements, D]

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        for k in range(2, 11):  # k = 2 to 10
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_tsne)

            score = silhouette_score(X_tsne, labels)
            writer.writerow([k, score])
