import os
import re
import sys

import pandas as pd
import numpy as np
import pymatviz as pmv
from pymatgen.core import Element


def p_table_draw(file_name):
    elements = [Element.from_Z(i).symbol for i in range(1, 104)]

    file_path = f'{file_name}.csv'
    df = pd.read_csv(file_path)

    df = df[df["Element"].isin(elements)]

    # 原子序数
    df["Z"] = df["Element"].apply(lambda x: Element(x).Z)
    # 按原子序数排序
    df = df.sort_values("Z").reset_index(drop=True)

    # 按顺序编码 Cluster
    unique_clusters = df["Cluster"].unique()
    cluster_map = {old: new + 1 for new, old in enumerate(unique_clusters)}
    df["Cluster_ordered"] = df["Cluster"].map(cluster_map)

    series = pd.Series(
        df["Cluster_ordered"].values,
        index=df["Element"].values
    )

    k = series.nunique()
    print(f"共有 {k} 个簇")

    colors_17 = [
        "#000004", "#210c46", "#3b0c5e", "#56106c", "#6f196d",
        "#892268", "#a22b61", "#ba3853", "#d14664", "#e25b31",
        "#f0721f", "#f88f0e", "#faac17", "#f8ca36", "#f9e56a",
        "#fcffa4", "#f7b6d2"
    ]
    if k > 17:
        raise ValueError("聚类数超过 17，请扩展 colors_17 列表")

    fig = pmv.ptable_heatmap_plotly(
        series,
        fmt=".0f",
        colorscale=colors_17[:k],
        cscale_range=(1, k),
        color_bar=dict(
            title='Element Clustering Result',
            title_font=dict(color="black", size=18),
            tickfont=dict(color="black", size=14),
            tickvals=list(range(1, k + 1)),
            ticktext=[str(v) for v in range(1, k + 1)]
        ),
        bg_color='rgba(255, 255, 255, 1)'
    )

    fig.write_image(f"ptable_{file_name}.svg")


if __name__ == '__main__':
    file_name = "Cluster_500_2000_0.6418"
    p_table_draw(file_name)
