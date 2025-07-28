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

    # 索引为元素符号
    series = pd.Series(
        df["Cluster"].values + 1,
        index=df["Element"].values
    )

    # 1~103的元素
    series = series[series.index.isin(elements)]

    k = series.nunique()
    print(f"共有 {k} 个簇")

    tick_vals = list(range(1, k + 1))

    fig = pmv.ptable_heatmap_plotly(
        series,
        fmt=".0f",  # 显示为整数
        colorscale="Inferno",
        cscale_range=(1, k),
        color_bar=dict(
            title=f'Element Clustering Result',
            title_font=dict(color="black", size=18),
            tickfont=dict(color="black", size=14),
            tickvals=tick_vals,
            ticktext=[str(v) for v in tick_vals]
        ),
        bg_color='rgba(255, 255, 255, 1)'
    )

    fig.write_image(f"ptable_{file_name}.svg")


if __name__ == '__main__':
    file_name = "Cluster_500_2000_0.6418"
    p_table_draw(file_name)
