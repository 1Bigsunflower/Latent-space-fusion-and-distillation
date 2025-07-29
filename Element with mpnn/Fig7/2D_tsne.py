
import argparse
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from element_category import build_category_lookup

# === 类别颜色映射 ===
cmap = {
    "alkali metal": "#00BCA9",
    "alkaline earth metal": "#FFCF3C",
    "metal": "#FCC0C2",
    "halogen": "#B9C3D5",
    "metalloid": "#FE9F6B",
    "noble gas": "#D6B7D6",
    "nonmetal": "#65D7EA",
    "transition metal": "#ABD86E",
    "lanthanides": "#FE703C",
    "actinides": "#00D4FE",
}

parser = argparse.ArgumentParser()
parser.add_argument('--task_name',
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    default='matbench_mp_e_form')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--dim', type=int, default=64)
args = parser.parse_args()

task_name = args.task_name
fold = args.fold
dim = args.dim
point_size = 180

def CGCNN_emb(axes):
    prefix = f"../CGCNN-emb/{task_name}_emb/fold{fold}_dim{dim}"
    files = [f"{prefix}_train.pt", f"{prefix}_val.pt", f"{prefix}_test.pt"]

    embedding_keys = ["embedding_0", "embedding_1", "embedding_2", "embedding_3"]

    element_embeddings = {key: defaultdict(list) for key in embedding_keys}
    element_categories = {}

    atomic_number_to_category = build_category_lookup()

    for file in files:
        data = torch.load(file)
        for batch in data:
            node_input = batch["atom_input"]
            atomic_numbers = node_input

            for key in embedding_keys:
                embeddings = batch[key]
                for i in range(embeddings.shape[0]):
                    Z = int(atomic_numbers[i].item())
                    category = atomic_number_to_category[Z]
                    element_categories[Z] = category
                    element_embeddings[key][Z].append(embeddings[i].cpu().numpy())

    print(f"------------CGCNN提取嵌入-{task_name}_emb/fold{fold}_dim{dim}------------")
    avg_embeddings = {key: {} for key in embedding_keys}
    for key in embedding_keys:
        for Z, vecs in element_embeddings[key].items():
            avg_embeddings[key][Z] = np.mean(vecs, axis=0)

    Z_to_category = {Z: element_categories[Z] for Z in avg_embeddings[embedding_keys[0]].keys()}
    Zs = sorted(Z_to_category.keys())

    for idx, key in enumerate(embedding_keys):
        X = np.stack([avg_embeddings[key][Z] for Z in Zs])
        labels = [Z_to_category[Z] for Z in Zs]

        le = LabelEncoder()
        true_labels = le.fit_transform(labels)

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        n_clusters = len(set(true_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        pred_labels = kmeans.fit_predict(X_tsne)

        ari_score = adjusted_rand_score(true_labels, pred_labels)
        ari_str = f"ARI {ari_score:.4f}"

        ax = axes[idx]
        for i, Z in enumerate(Zs):
            cat = Z_to_category[Z]
            color = cmap[cat]
            ax.scatter(X_tsne[i, 0], X_tsne[i, 1], color=color, label=cat, s=point_size, alpha=0.8)

        ax.set_title(f"{ari_str}", fontsize=25, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.text(-0.01, 0.5, "CGCNN - Embedding", fontsize=30,
                    ha='right', va='center', transform=ax.transAxes, rotation=90)


def MEGNet_emb(axes):
    prefix = f"../MEGNet-emb/{task_name}_emb/fold{fold}_dim{dim}"
    files = [f"{prefix}_train.pt", f"{prefix}_val.pt", f"{prefix}_test.pt"]

    embedding_keys = ["node_embedded", "block_1", "block_2", "block_3"]

    element_embeddings = {key: defaultdict(list) for key in embedding_keys}
    element_categories = {}

    atomic_number_to_category = build_category_lookup()

    for file in files:
        data = torch.load(file)
        for batch in data:
            node_input = batch["node_input"]
            atomic_numbers = node_input + 1

            mapping = {84: 89, 85: 90, 86: 91, 87: 92, 88: 93, 89: 94}
            atomic_numbers = torch.tensor([mapping.get(int(z.item()), int(z.item())) for z in atomic_numbers])

            for key in embedding_keys:
                embeddings = batch[key]
                for i in range(embeddings.shape[0]):
                    Z = int(atomic_numbers[i].item())
                    category = atomic_number_to_category[Z]
                    element_categories[Z] = category
                    element_embeddings[key][Z].append(embeddings[i].cpu().numpy())

    print(f"------------MEGNet提取嵌入-{task_name}_emb/fold{fold}_dim{dim}------------")
    avg_embeddings = {key: {} for key in embedding_keys}
    for key in embedding_keys:
        for Z, vecs in element_embeddings[key].items():
            avg_embeddings[key][Z] = np.mean(vecs, axis=0)

    Z_to_category = {Z: element_categories[Z] for Z in avg_embeddings[embedding_keys[0]].keys()}
    Zs = sorted(Z_to_category.keys())

    for idx, key in enumerate(embedding_keys):
        X = np.stack([avg_embeddings[key][Z] for Z in Zs])
        labels = [Z_to_category[Z] for Z in Zs]

        le = LabelEncoder()
        true_labels = le.fit_transform(labels)

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        n_clusters = len(set(true_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        pred_labels = kmeans.fit_predict(X_tsne)

        ari_score = adjusted_rand_score(true_labels, pred_labels)
        ari_str = f"ARI {ari_score:.4f}"

        ax = axes[idx]
        for i, Z in enumerate(Zs):
            cat = Z_to_category[Z]
            color = cmap[cat]
            ax.scatter(X_tsne[i, 0], X_tsne[i, 1], color=color, label=cat, s=point_size, alpha=0.8)

        ax.set_title(f"{ari_str}", fontsize=25, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.text(-0.01, 0.5, "MEGNet - Embedding", fontsize=30,
                    ha='right', va='center', transform=ax.transAxes, rotation=90)


def CGCNN_hm(axes):
    with open('atom_init.json', 'r') as f:
        atom_init = json.load(f)
    vector_to_Z = {tuple(map(int, v)): int(k) for k, v in atom_init.items()}

    prefix = f"../CGCNN-human/{task_name}_hm/fold{fold}_dim{dim}"
    files = [f"{prefix}_train.pt", f"{prefix}_val.pt", f"{prefix}_test.pt"]
    embedding_keys = ["embedding_0", "embedding_1", "embedding_2", "embedding_3"]

    element_embeddings = {key: defaultdict(list) for key in embedding_keys}
    element_categories = {}
    atomic_number_to_category = build_category_lookup()

    for file in files:
        data = torch.load(file)
        for batch in data:
            node_input = batch["atom_input"]
            node_input_np = node_input.cpu().numpy()

            atomic_numbers = []
            for i in range(node_input_np.shape[0]):
                vec = tuple(map(int, node_input_np[i]))
                Z = vector_to_Z.get(vec, None)
                if Z is None:
                    raise ValueError(f"未能在 atom_init.json 中匹配向量: {vec}")
                atomic_numbers.append(Z)

            for key in embedding_keys:
                embeddings = batch[key]
                for i in range(embeddings.shape[0]):
                    Z = atomic_numbers[i]
                    category = atomic_number_to_category[Z]
                    element_categories[Z] = category
                    element_embeddings[key][Z].append(embeddings[i].cpu().numpy())

    print(f"------------CGCNN提取嵌入-{task_name}_hm/fold{fold}_dim{dim}------------")
    avg_embeddings = {key: {} for key in embedding_keys}
    for key in embedding_keys:
        for Z, vecs in element_embeddings[key].items():
            avg_embeddings[key][Z] = np.mean(vecs, axis=0)

    Z_to_category = {Z: element_categories[Z] for Z in avg_embeddings[embedding_keys[0]].keys()}
    Zs = sorted(Z_to_category.keys())

    for idx, key in enumerate(embedding_keys):
        X = np.stack([avg_embeddings[key][Z] for Z in Zs])
        labels = [Z_to_category[Z] for Z in Zs]

        le = LabelEncoder()
        true_labels = le.fit_transform(labels)

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        n_clusters = len(set(true_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        pred_labels = kmeans.fit_predict(X_tsne)

        ari_score = adjusted_rand_score(true_labels, pred_labels)
        ari_str = f"ARI {ari_score:.4f}"

        ax = axes[idx]
        for i, Z in enumerate(Zs):
            cat = Z_to_category[Z]
            color = cmap[cat]
            ax.scatter(X_tsne[i, 0], X_tsne[i, 1], color=color, label=cat, s=point_size, alpha=0.8)

        ax.set_title(f"{ari_str}", fontsize=25, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.text(-0.01, 0.5, "CGCNN - Human", fontsize=30,
                    ha='right', va='center', transform=ax.transAxes, rotation=90)


def MEGNet_hm(axes):
    prefix = f"../MEGNet-human/{task_name}_hm/fold{fold}_dim{dim}"
    files = [f"{prefix}_train.pt", f"{prefix}_val.pt", f"{prefix}_test.pt"]
    embedding_keys = ["node_embedded", "block_1", "block_2", "block_3"]
    title_mapping = {
        "node_embedded": "Element embedding",
        "block_1": "Block1",
        "block_2": "Block2",
        "block_3": "Block3"
    }

    element_embeddings = {key: defaultdict(list) for key in embedding_keys}
    element_categories = {}
    atomic_number_to_category = build_category_lookup()

    for file in files:
        data = torch.load(file)
        for batch in data:
            node_input = batch["node_input"]
            atomic_numbers = node_input + 1
            mapping = {84: 89, 85: 90, 86: 91, 87: 92, 88: 93, 89: 94}
            atomic_numbers = torch.tensor([mapping.get(int(z.item()), int(z.item())) for z in atomic_numbers])

            for key in embedding_keys:
                embeddings = batch[key]
                for i in range(embeddings.shape[0]):
                    Z = int(atomic_numbers[i].item())
                    category = atomic_number_to_category[Z]
                    element_categories[Z] = category
                    element_embeddings[key][Z].append(embeddings[i].cpu().numpy())

    print(f"------------MEGNet提取嵌入-{task_name}_hm/fold{fold}_dim{dim}------------")
    avg_embeddings = {key: {} for key in embedding_keys}
    for key in embedding_keys:
        for Z, vecs in element_embeddings[key].items():
            avg_embeddings[key][Z] = np.mean(vecs, axis=0)

    Z_to_category = {Z: element_categories[Z] for Z in avg_embeddings[embedding_keys[0]].keys()}
    Zs = sorted(Z_to_category.keys())

    for idx, key in enumerate(embedding_keys):
        X = np.stack([avg_embeddings[key][Z] for Z in Zs])
        labels = [Z_to_category[Z] for Z in Zs]

        le = LabelEncoder()
        true_labels = le.fit_transform(labels)

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        n_clusters = len(set(true_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        pred_labels = kmeans.fit_predict(X_tsne)

        ari_score = adjusted_rand_score(true_labels, pred_labels)
        ari_str = f"ARI {ari_score:.4f}"

        ax = axes[idx]
        for i, Z in enumerate(Zs):
            cat = Z_to_category[Z]
            color = cmap[cat]
            ax.scatter(X_tsne[i, 0], X_tsne[i, 1], color=color, label=cat, s=point_size, alpha=0.8)

        ax.set_title(f"{ari_str}", fontsize=25, pad=10)

        ax.text(0.5, -0.08, f"{title_mapping[key]}",
                fontsize=30, ha='center', va='top', transform=ax.transAxes)

        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.text(-0.01, 0.5, "MEGNet - Human", fontsize=30,
                    ha='right', va='center', transform=ax.transAxes, rotation=90)


def plot_combined():
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    CGCNN_emb(axes[0])
    CGCNN_hm(axes[1])
    MEGNet_emb(axes[2])
    MEGNet_hm(axes[3])

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                          markerfacecolor=color, markersize=22)
               for cat, color in cmap.items()]

    fig.legend(handles=handles,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.07),
               ncol=5,
               fontsize=28,
               labelspacing=0.0,
               )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    os.makedirs("tsne_pic", exist_ok=True)
    fig.savefig(f"tsne_pic/{task_name}_fold{fold}_dim{dim}-combined.pdf", bbox_inches="tight")


if __name__ == '__main__':
    plot_combined()
