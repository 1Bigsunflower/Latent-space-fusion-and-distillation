import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def zscore_and_save(csv_path):
    print(f"\n===== Z-score processing {csv_path} =====")

    df = pd.read_csv(csv_path)

    id_col = df.columns[0]          # 保留第一列（名称/ID）
    numeric_df = df.drop(columns=id_col)

    mask_nonzero = ~((numeric_df.fillna(0) == 0).all(axis=1))
    valid_df = numeric_df[mask_nonzero]

    mean = valid_df.mean(axis=0)
    std = valid_df.std(axis=0)

    std = std.replace(0, 1)

    zscored = (valid_df - mean) / std

    out_numeric = numeric_df.copy()
    out_numeric.loc[mask_nonzero] = zscored  # 非零行替换，零行保持不变

    out_df = pd.concat([df[[id_col]], out_numeric], axis=1)

    base, ext = os.path.splitext(csv_path)
    out_path = base + "_zscore" + ext

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

if __name__ == '__main__':

    for f in ["MDS_F_32_cos.csv", "MDS_E_32_cos.csv", "MDS_F_64_cos.csv", "MDS_E_64_cos.csv"]:
        zscore_and_save(f)