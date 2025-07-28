import os
import glob
import random
import sys
import multiprocessing
import pandas as pd
import numpy as np
from itertools import combinations
from deap import base, creator, tools, algorithms
from sklearn.metrics import adjusted_rand_score
import itertools
from sklearn.metrics import matthews_corrcoef

random.seed(42)
np.random.seed(42)

TASK_name = [
    "matbench_phonons",  # 1,265
    "matbench_log_gvrh",  # 10,987
    "matbench_log_kvrh",  # 10,987
    "matbench_mp_gap",  # 106,113
    "matbench_mp_e_form"  # 132,752
]

# === PARAMETERS === #
K_MAX = 100
POP_SIZE = 500
N_GEN = 2000
METHODS = [
    "CGCNN-emb/Clustering Result/embedding_3",
    "CGCNN-human/Clustering Result/embedding_3",
    "MEGNet-emb/Clustering Result/block_3",
    "MEGNet-human/Clustering Result/block_3"
]
TRADITIONAL_CLUSTER_PATH = "traditional_cluster.csv"

folders = [
    "CGCNN-emb/CGCNN_emb_kmeans_sil/",
    "CGCNN-human/CGCNN_hm_kmeans_sil/",
    "MEGNet-emb/MEGNet_emb_kmeans_sil/",
    "MEGNet-human/MEGNet_hm_kmeans_sil/"
]
folds = [0, 1, 2, 3, 4]
dims = [8, 16, 32, 64]

DATA_PATHS = []


def read_reference_clusters(paths):
    """提取多组 元素:簇编号 映射"""
    reference_clusterings = []
    for path in paths:
        for file in glob.glob(path):
            df = pd.read_csv(file)
            cluster_map = df.set_index("Element")["Cluster"].to_dict()
            reference_clusterings.append(cluster_map)
    return reference_clusterings


def get_all_elements(traditional_cluster_path):  # 传统10分类
    df = pd.read_csv(traditional_cluster_path)
    return df["Element"].tolist(), df.set_index("Element")["Cluster"].to_dict()


def get_co_cluster_pairs(cluster_map):  # 获取元素对
    """Return set of (element1, element2) where both in same cluster."""
    pairs = set()
    cluster_to_elements = {}
    for elem, clus in cluster_map.items():
        cluster_to_elements.setdefault(clus, []).append(elem)
    for members in cluster_to_elements.values():
        for e1, e2 in combinations(sorted(members), 2):
            pairs.add((e1, e2))
    return pairs

def score_individual(individual, elements, reference_pairs_list):
    """计算该染色体（聚类方案）与所有参考方案的平均准确率 max"""
    # individual和elements组合起来。
    cluster_map = {elem: individual[i] for i, elem in enumerate(elements)}  # individual候选解

    total_score = 0
    valid_refs = 0

    for ref_map in reference_pairs_list:
        common_elems = set(ref_map.keys()).intersection(cluster_map.keys())  # 参考方案与当前方案的共同元素

        all_pairs = set(itertools.combinations(sorted(common_elems), 2))  # 所有元素对

        ref_pairs = get_co_cluster_pairs({e: ref_map[e] for e in common_elems})
        # 优化目标
        pred_pairs = get_co_cluster_pairs({e: cluster_map[e] for e in common_elems})

        TP = len(pred_pairs & ref_pairs)
        TN = len((all_pairs - pred_pairs) & (all_pairs - ref_pairs))
        total_possible = len(all_pairs)

        total_score += (TP + TN) / total_possible
        valid_refs += 1

    avg_score = total_score / valid_refs
    return (avg_score,)


# === GENETIC ALGORITHM SETUP === #

def setup_genetic_algorithm(elements, reference_pairs_list, traditional_mapping):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 单目标最大化问题weights=(1.0,)
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("seed", lambda: random.seed(42))
    toolbox.register("attr_cluster", random.randint, 0, K_MAX - 1)  # 单个“基因”取值范围
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_cluster, n=len(elements))

    def init_population(icls, n):  # 初始化整个种群
        pop = []
        # 传统分类方法个体
        trad_ind = [traditional_mapping.get(elem, random.randint(0, K_MAX - 1)) for elem in elements]
        pop.append(icls(trad_ind))

        for _ in range(n - 1):  # 其他个体随机生成
            ind = icls([random.randint(0, K_MAX - 1) for _ in range(len(elements))])
            pop.append(ind)
        return pop

    toolbox.register("population", init_population, creator.Individual)
    toolbox.register("evaluate", score_individual, elements=elements, reference_pairs_list=reference_pairs_list)  # 评估函数
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=K_MAX - 1, indpb=0.1)  # 10% 概率发生变异
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_optimization():
    for folder, method_path in zip(folders, METHODS):
        for task in TASK_name:
            for fold in folds:
                for dim in dims:
                    sil_file = os.path.join(folder, f"{task}_fold{fold}_dim{dim}.csv")
                    # 最佳 k
                    df = pd.read_csv(sil_file)

                    max_row = df.loc[df['silhouette_score'].idxmax()]
                    best_k = int(max_row['k'])

                    final_file = os.path.join(
                        method_path, f"{task}_fold{fold}_dim{dim}_k{best_k}.csv"
                    )
                    DATA_PATHS.append(final_file)

    reference_clusterings = read_reference_clusters(DATA_PATHS)
    reference_pairs_list = reference_clusterings

    elements, traditional_mapping = get_all_elements(TRADITIONAL_CLUSTER_PATH)

    toolbox = setup_genetic_algorithm(elements, reference_pairs_list, traditional_mapping)  # 配置遗传算法工具箱
    toolbox.register("seed", lambda: random.seed(42))
    toolbox.register("map", multiprocessing.Pool().map)
    pop = toolbox.population(n=POP_SIZE)  # 种群大小

    hof = tools.HallOfFame(3)  # 保存3个最好个体
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop,  # 初始种群
                                       toolbox,  # 工具箱
                                       cxpb=0.5,  # 50%的概率进行交叉
                                       mutpb=0.2,  # 20%的概率进行变异
                                       ngen=N_GEN,  # 进化的代数
                                       stats=stats,  # 每一代记录统计数据
                                       halloffame=hof,  # 每一代更新最好的个体
                                       verbose=True
                                       )

    # Save best result
    best_ind = hof[0]
    final_result = pd.DataFrame({
        "Element": elements,
        "Cluster": best_ind
    })

    # === 重编码类别编号为连续整数 ===
    def relabel_clusters(cluster_list):
        unique_labels = sorted(set(cluster_list))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return [label_map[c] for c in cluster_list]

    final_result["Cluster"] = relabel_clusters(final_result["Cluster"])

    OUTPUT_PATH = f"GA_cluster_result/Cluster_{POP_SIZE}_{N_GEN}_{best_ind.fitness.values[0]:.4f}.csv"
    final_result.to_csv(OUTPUT_PATH, index=False)

    return best_ind.fitness.values[0]


if __name__ == '__main__':
    best_score = run_optimization()
