import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from numpy.linalg import svd
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from bokeh.sampledata.periodic_table import elements as periodic_elements
from element_category import get_element_category
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ---------------------------
# 指标计算函数  有效秩 标准差
# ---------------------------
def calculate_effective_rank(embeddings):
    emb_matrix = np.vstack(list(embeddings.values()))
    S = svd(emb_matrix, full_matrices=False, compute_uv=False)
    S_normalized = S / np.sum(S)
    effective_rank = np.exp(-np.sum(S_normalized * np.log(S_normalized + 1e-12)))
    return effective_rank


def calculate_embedding_entropy(embeddings):
    emb_matrix = np.vstack(list(embeddings.values()))
    variances = np.var(emb_matrix, axis=0)
    prob_dist = variances / np.sum(variances)
    entropy_value = entropy(prob_dist + 1e-12)
    return entropy_value


# ---------------------------
# 数据加载与预处理
# ---------------------------
def load_embeddings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    df_emb = pd.read_csv(file_path)
    element_embeddings = {row['Element']: row.drop('Element').astype(float).values
                          for _, row in df_emb.iterrows()}
    return element_embeddings


def compute_distance_matrix(embeddings):  # 角度 θ = arccos(cos) 作为距离
    X = np.array(embeddings)
    # 单位化
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # 计算cos相似度矩阵
    cos_sim = np.dot(X, X.T)
    # 点积限制在[−1,1]范围内，避免arccos精度
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    # 角度 θ
    theta = np.arccos(cos_sim)
    return theta


# ---------------------------
# 邻接矩阵与谱分析
# ---------------------------
def build_adjacency_matrix(embeddings):
    """
    角度 θ作为距离构图 A_ij = exp(-theta_ij)
    """
    n = len(embeddings)
    # 角度距离矩阵
    distances = compute_distance_matrix(embeddings)

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            aij = np.exp(-distances[i, j])
            A[i, j] = aij
            A[j, i] = aij

    np.fill_diagonal(A, 0)
    return A


def spectral_analysis(A):
    eigenvalues_A, eigenvectors_A = np.linalg.eigh(A)
    eigenvalues_A = eigenvalues_A[::-1]
    lambda1 = eigenvalues_A[0]
    spectral_gap = lambda1 - eigenvalues_A[1]
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigenvalues_L, eigenvectors_L = np.linalg.eigh(L)
    sorted_indices = np.argsort(eigenvalues_L)
    eigenvalues_L = eigenvalues_L[sorted_indices]
    fiedler_value = eigenvalues_L[1] if len(eigenvalues_L) > 1 else 0
    return {
        'spectral_radius': lambda1,
        'spectral_gap': spectral_gap,
        'fiedler_value': fiedler_value
    }


# ---------------------------
# 聚类与评估
# ---------------------------
def perform_clustering(embeddings, n_clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    return labels


def clustering_metrics(embeddings, element_embeddings):
    # 所有元素列表，保证顺序一致
    elements = list(element_embeddings.keys())

    # 构建映射：元素符号 -> 原子序数
    element_to_atomic = {row['symbol']: row['atomic number'] for _, row in periodic_elements.iterrows()}
    true_labels = [get_element_category(element_to_atomic[el]) for el in elements]

    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels)

    cluster_labels = perform_clustering(embeddings, n_clusters=len(np.unique(true_labels_encoded)))
    ari = adjusted_rand_score(true_labels, cluster_labels)

    silhouette = silhouette_score(
        np.vstack(list(element_embeddings.values())),
        cluster_labels,
        metric='cosine'
    )

    return ari, silhouette


# ---------------------------
# MAE
# ---------------------------
def load_mae_data(file_path):
    """
    读取 megnet_emb_alltask.csv 文件，返回一个二维 numpy 数组。
    文件中每行包含 5 个数，对应 fold0-fold4 的 MAE，每 8 行为一个 dim（顺序：dim8, dim16, dim32, dim64）。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    data = pd.read_csv(file_path, header=None)
    return data.values  # shape (32, 5)


def form(model, h_or_e):
    task_names = [
        'matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
        'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'
    ]
    dims = [8, 16, 32, 64]
    folds = [0, 1, 2, 3, 4]

    mae_file = f'{h_or_e}_alltask.csv'
    mae_data = load_mae_data(mae_file)  # shape (32, 5)

    results = []
    total_iterations = len(dims) * len(task_names) * len(folds)
    with tqdm(total=total_iterations, desc=f'{model}-{h_or_e}') as pbar:
        for dim_idx, dim in enumerate(dims):
            for task_idx, task_name in enumerate(task_names):
                mae_row_index = dim_idx * len(task_names) + task_idx
                for fold in folds:
                    file_path = f'../../Embedding_weight_orig/{model}/{h_or_e}/{task_name}_fold{fold}_dim{dim}.csv'
                    try:
                        element_embeddings = load_embeddings(file_path)
                    except FileNotFoundError as e:
                        print(e)
                        continue

                    eff_rank = calculate_effective_rank(element_embeddings)
                    emb_entropy = calculate_embedding_entropy(element_embeddings)

                    embeddings = [vec for vec in element_embeddings.values()]
                    A = build_adjacency_matrix(embeddings)
                    spec_info = spectral_analysis(A)

                    ari, silhouette = clustering_metrics(embeddings, element_embeddings)
                    mae = mae_data[mae_row_index, fold]

                    row = {
                        'Model': model,
                        'Method': h_or_e,
                        'Task': task_name,
                        'Fold': fold,
                        'Dimension': dim,
                        'Effective_Rank': eff_rank,
                        'Embedding_Entropy': emb_entropy,
                        'Spectral_Radius': spec_info['spectral_radius'],
                        'Spectral_Gap': spec_info['spectral_gap'],
                        'Fiedler_Value': spec_info['fiedler_value'],
                        'ARI': ari,
                        'Silhouette': silhouette,
                        'MAE': mae
                    }
                    results.append(row)
                    pbar.update(1)

    df_results = pd.DataFrame(results)
    output_csv = f'{model}_{h_or_e}_spectral_analysis.csv'
    df_results.to_csv(output_csv, index=False)


if __name__ == '__main__':
    form('MEGNet', 'megnet_emb')
    form('MEGNet', 'megnet_hm')
    form('CGCNN', 'cgcnn_emb')
    form('CGCNN', 'cgcnn_hm')
