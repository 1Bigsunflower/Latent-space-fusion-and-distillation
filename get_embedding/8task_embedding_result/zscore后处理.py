import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_embedding_distribution(csv_path, bins=50):
    """
    csv_path: 你的嵌入 csv 文件路径
    bins: 直方图分箱数
    """
    print(f"\n===== Analyzing {csv_path} =====")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Drop non-numeric column (element name)
    numeric_df = df.drop(columns=df.columns[0])

    # ---- 🔥 忽略掉全是 0 的行 ----
    # 如果有 NaN，可以先把 NaN 当 0 再判断
    mask_nonzero = ~((numeric_df.fillna(0) == 0).all(axis=1))
    numeric_df = numeric_df[mask_nonzero]

    print(f"有效行数（已剔除全 0 行）：{len(numeric_df)}")

    # Flatten all embedding values into 1 array
    values = numeric_df.values.flatten()

    # Remove NaN if any missing
    values = values[~np.isnan(values)]

    # Basic statistics
    mean = np.mean(values)
    std = np.std(values)
    vmin = np.min(values)
    vmax = np.max(values)

    print(f"Mean: {mean:.6f}")
    print(f"Std:  {std:.6f}")
    print(f"Min:  {vmin:.6f}")
    print(f"Max:  {vmax:.6f}")

    # Simple heuristic checks
    if vmax > 50 or vmin < -50:
        print("⚠️ 发现可能存在 0~100 或更大范围的异常 embedding")
    elif std < 0.2:
        print("ℹ️ 分布可能比较集中，不太像标准高斯 N(0,1)")
    else:
        print("👍 看起来更接近高斯分布（粗略判断）")


    # # Plot histogram
    # plt.figure(figsize=(7,5))
    # plt.hist(values, bins=bins, density=True, alpha=0.7)
    # plt.title(f"Embedding Distribution: {csv_path}")
    # plt.xlabel("Value")
    # plt.ylabel("Density")
    # plt.grid(True)
    # plt.savefig("")
    # plt.show()


def analyze_multiple(files):
    for f in files:
        analyze_embedding_distribution(f)
def zscore_and_save(csv_path):
    print(f"\n===== Z-score processing {csv_path} =====")

    df = pd.read_csv(csv_path)

    id_col = df.columns[0]          # 保留第一列（名称/ID）
    numeric_df = df.drop(columns=id_col)

    # ---- 找出非全0行（用于参与 Z-score 计算）----
    mask_nonzero = ~((numeric_df.fillna(0) == 0).all(axis=1))
    valid_df = numeric_df[mask_nonzero]

    # 计算列向均值 & std（按维度）
    mean = valid_df.mean(axis=0)
    std = valid_df.std(axis=0)

    # 防止除以 0
    std = std.replace(0, 1)

    # ---- 对非全0行进行 Z-score ----
    zscored = (valid_df - mean) / std

    # ---- 构造输出 ----
    out_numeric = numeric_df.copy()
    out_numeric.loc[mask_nonzero] = zscored  # 非零行替换，零行保持不变

    out_df = pd.concat([df[[id_col]], out_numeric], axis=1)

    # 生成新文件名
    base, ext = os.path.splitext(csv_path)
    out_path = base + "_zscore" + ext

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存：{out_path}")

if __name__ == '__main__':
    # analyze_multiple(["mat2vec.csv", "all_6_classical_mds_32d.csv", "all_6_classical_mds_64d.csv","all_6_mds_32d.csv","all_6_mds_64d.csv"])
    # analyze_multiple(["mat2vec.csv", "all_6_classical_mds_32d_zscore.csv", "all_6_classical_mds_64d_zscore.csv", "all_6_mds_32d_zscore.csv",
    #                   "all_6_mds_64d_zscore.csv"])
    # analyze_multiple(['mat2vec.csv', 'all6_MDS_64d_euc_nl2.csv','all6_MDS_32d_euc_nl2.csv','all6_CMDS_64d_euc_nl2.csv','all6_CMDS_32d_euc_nl2.csv'])

    # analyze_multiple(['mat2vec.csv', 'all6_CMDS_32d_cos_nl2.csv', 'all6_CMDS_64d_cos_nl2.csv', 'all6_MDS_32d_cos_nl2.csv','all6_MDS_64d_cos_nl2.csv'])
    # analyze_multiple(['mat2vec.csv', 'all6_Gram_64d.csv', 'all6_Gram_32d.csv'])
    # for f in ["all_6_classical_mds_32d.csv", "all_6_classical_mds_64d.csv", "all_6_mds_32d.csv", "all_6_mds_64d.csv"]:
    #     zscore_and_save(f)
    for f in ["CMDS_32_cos.csv", "CMDS_32_euc.csv", "CMDS_64_cos.csv", "CMDS_64_euc.csv", "MDS_32_cos.csv", "MDS_32_euc.csv", "MDS_64_cos.csv", "MDS_64_euc.csv"]:
        zscore_and_save(f)