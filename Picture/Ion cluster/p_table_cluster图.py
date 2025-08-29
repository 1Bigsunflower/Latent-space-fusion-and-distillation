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

    series = pd.Series(
        df["Cluster"].values,
        index=df["Element"].values
    )

    series = series[series.index.isin(elements)]

    k = series.nunique()
    print(f"共有 {k} 个簇")

    fig = pmv.ptable_heatmap_plotly(
        series,
        fmt=".0f",  # 显示为整数
        colorscale="Inferno",  # 可选：Viridis, Inferno, etc.
        cscale_range=(1, k),  # colorbar范围设定
        color_bar=dict(
            title=f'Ion Clustering Result',
            title_font=dict(color="black", size=18),
            tickfont=dict(color="black", size=14),
        ),
        bg_color='rgba(255, 255, 255, 1)'  # 白色背景
    )

    fig.write_image(f"ptable_{file_name}.svg")


if __name__ == '__main__':
    file_name = "new_cluster"
    p_table_draw(file_name)