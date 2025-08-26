import os
import re
import sys

import pandas as pd
import numpy as np
import pymatviz as pmv
from pymatgen.core import Element


def p_table_draw(file_name):
    elements = [Element.from_Z(i).symbol for i in range(1, 104)]

    file_path = f'{file_name}'
    df = pd.read_csv(file_path)

    # 去掉未出现元素
    df = df[df["total_counts"] > 0]

    series = pd.Series(
        df["total_counts"].values,
        index=df["Element"].values
    )

    series = series[series.index.isin(elements)]

    k = df["total_counts"].max()

    fig = pmv.ptable_heatmap_plotly(
        series,
        fmt=".0f",  # 显示为整数
        colorscale="Inferno",  # Viridis, Inferno, etc.
        cscale_range=(0, 40000),  # colorbar范围设定
        color_bar=dict(
            title=f'Element Occurrence Count',
            title_font=dict(color="black", size=18),
            tickfont=dict(color="black", size=14),
            # dtick=5000,
        ),
        bg_color='rgba(255, 255, 255, 1)'
    )

    fig.write_image(f"ptable_element_counts.svg")


if __name__ == '__main__':
    file_name = "element_counts.csv"
    p_table_draw(file_name)