import sys

import numpy as np
from scipy.stats import wilcoxon
import pandas as pd

def compare_models(A, B):
    """
    Compare model A vs model B (MAE, smaller is better)
    Returns the 4 recommended statistics
    """
    # 1. Relative improvement
    rel_improve = (B - A) / B
    mean_rel = np.mean(rel_improve) * 100
    median_rel = np.median(rel_improve) * 100

    # 2. Win / Tie / Loss
    wins = np.sum(A < B)
    ties = np.sum(A == B)
    losses = np.sum(A > B)
    N = len(A)

    # 3. Probability of Superiority
    PS = (wins + 0.5 * ties) / N

    # 4. Cliff's delta
    delta = (wins - losses) / N

    # 5. Wilcoxon signed-rank test
    diff = A - B
    if np.any(diff != 0):
        _, p_value = wilcoxon(diff)
    else:
        p_value = 1.0

    return {
        "mean_rel": mean_rel,
        "median_rel": median_rel,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "PS": PS,
        "delta": delta,
        "p_value": p_value,
        "N": N
    }


def print_report(base_name, model_name, stats):
    effect_size = abs(stats["delta"])
    if effect_size < 0.147:
        effect_desc = "negligible"
    elif effect_size < 0.33:
        effect_desc = "small"
    elif effect_size < 0.474:
        effect_desc = "medium"
    else:
        effect_desc = "large"

    print(f"""
在 N = {stats['N']} 个任务的配对比较中，相较于 {base_name}，{model_name} 在任务内的平均相对 MAE 改进为 {stats['mean_rel']:.2f}%。
在 {stats['wins']}/{stats['N']} 个任务中 {model_name} 的 MAE 低于 {base_name}，随机抽取一个任务进行比较，{model_name} 优于 {base_name} 的概率为 {stats['PS']:.2f}。
Wilcoxon signed-rank test （p = {stats['p_value']:.3g}）。
Cliff’s delta 为 {stats['delta']:.2f}，表明该改进具有 {effect_desc} 的效应量。
""")


if __name__ == '__main__':
    file_path = "CGCNN_MEGNet.xlsx"
    df = pd.read_excel(file_path)

    # 第 2–105 行
    data = df.iloc[0:105]

    # base_col = "cgcnn_f"  # 55开 后两个
    # base_col = "cgcnn_e"  # 14%提升 后两个
    # compare_cols = [
    #     "e1_f0.0001",
    #     "e1_f0.01",
    #     "e1_f1",
    #     "e0.01_f1",
    #     "e0.0001_f1",
    # ]

    base_col = "megnet_f"
    compare_cols = [
        "e1_f0.0001.1",
        "e1_f0.01.1",
        "e1_f1.1",
        "e0.01_f1.1",
        "e0.0001_f1.1",
    ]

    results = {}

    for col in compare_cols:
        pair = data[[base_col, col]].dropna()

        A = pair[col].values
        B = pair[base_col].values

        stats = compare_models(A, B)
        results[col] = stats

    for col, stats in results.items():
        # if stats["PS"] > 0.5:
        print_report("megnet_f", col, stats)