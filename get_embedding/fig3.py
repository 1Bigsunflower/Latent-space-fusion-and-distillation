# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
#
# mpl.rcParams["mathtext.fontset"] = "custom"
# mpl.rcParams["mathtext.rm"] = "Arial"
# mpl.rcParams["mathtext.it"] = "Arial:italic"
# mpl.rcParams["mathtext.bf"] = "Arial:bold"
PREFIXES = (
    "jdft2d",
    "phonons",
    "dielectric",
    "log_gvrh",
    "log_kvrh",
    "perovskites",
    "mp_gap",
    "mp_e_form",
)

DIMS = [8, 16, 32, 64]
RATIO_LABELS = [
    "$r=H(-\infty)$",
    r"$r=-4$",
    r"$r=-2$",
    r"$r=0$",
    r"$r=2$",
    r"$r=4$",
    "$r=L(+\infty)$"
]

DIM_COLORS = {
    8:  "#E38120",
    16: "#34963A",
    32: "#3173A3",
    64: "#BB2F2D",
}

RADIAL_TICKS = np.linspace(0, 1, 6)
# RADIAL_LABELS = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
RADIAL_LABEL_TICKS = [0, 0.5, 1]
RADIAL_LABELS = ["0", "0.5", "1"]
from matplotlib.transforms import Bbox

from matplotlib.patches import Rectangle

from matplotlib.patches import FancyBboxPatch

def draw_group_box_by_grid(
    fig,
    axes,
    rows,
    cols,
    color="black",
    lw=4,
    pad=0.01,
    rounding_size=0.02,     # 圆角大小（关键参数）
    dash_pattern=(0, (5, 3))  # 虚线：线长18，空8
):
    """
    rows : list of row indices, e.g. [0, 1]
    cols : list of col indices, e.g. [0, 1, 2, 3]
    """

    # 收集目标 axes 的位置
    boxes = [
        axes[r, c].get_position()
        for r in rows
        for c in cols
    ]

    x0 = min(b.x0 for b in boxes) - 0.03
    y0 = min(b.y0 for b in boxes) - pad
    x1 = max(b.x1 for b in boxes) + pad
    y1 = max(b.y1 for b in boxes) + pad

    rect = FancyBboxPatch(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={rounding_size}",
        transform=fig.transFigure,
        fill=False,
        ec=color,
        lw=lw,
        linestyle=dash_pattern,
        zorder=0,
    )

    fig.add_artist(rect)




def plot_radar_by_ratio(excel_path, sheet_name=0):
    # ======================
    # 1. 读数据
    # ======================
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    TASK_COL_IDX = 0
    DIM_COL_IDX = 8
    CGCNN_COL_IDXS = [1, 7, 6, 5, 4, 3, 2]
    MEGNET_COL_IDXS = [9, 15, 14, 13, 12, 11, 10]

    task_col = df.columns[TASK_COL_IDX]
    dim_col = df.columns[DIM_COL_IDX]
    cg_cols = [df.columns[i] for i in CGCNN_COL_IDXS]
    mg_cols = [df.columns[i] for i in MEGNET_COL_IDXS]

    df[task_col] = df[task_col].astype(str)
    df["TASK_BASE"] = df[task_col].str.replace(r"_fold\d+$", "", regex=True)

    df = df[df["TASK_BASE"].str.startswith(PREFIXES)]
    df[dim_col] = pd.to_numeric(df[dim_col], errors="coerce")
    df = df.dropna(subset=[dim_col])
    df[dim_col] = df[dim_col].astype(int)

    tasks = list(PREFIXES)
    n_tasks = len(tasks)
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False)
    angles_closed = np.append(angles, angles[0])

    # ======================
    # 2. 基础雷达框架
    # ======================
    def draw_base(ax, show_labels=False):
        theta = np.linspace(0, 2 * np.pi, 360)

        for r in RADIAL_TICKS[1:]:
            ax.plot(theta, [r] * len(theta), color="gray", lw=1, alpha=0.5)

        for a in angles:
            ax.plot([a, a], [0, 1], color="gray", lw=1, alpha=0.5)

        ax.set_ylim(0, 1)
        ax.grid(False)

        # ax.set_yticks(RADIAL_TICKS,)
        ax.set_yticks(RADIAL_LABEL_TICKS)
        if show_labels:
            ax.set_yticklabels(
                RADIAL_LABELS,
                fontsize=35,
                # fontweight="bold",
            )
            ax.set_xticks(angles)
            ax.set_xticklabels(
                tasks,
                fontsize=35,
                # fontweight="bold",
            )
        else:
            ax.set_yticklabels([])
            ax.set_xticks([])

    # ======================
    # 3. 画一个子图（模型 + 比例）
    # ======================
    def draw_one(ax, model_cols, ratio_idx):
        """
        一个子图：
        固定 model + ratio
        task-wise normalization (4 dims within each task)
        """
        # raw_vals[task][dim] = value
        raw_vals = {task: {} for task in tasks}

        # 1. 收集原始 MAE
        for task in tasks:
            for dim in DIMS:
                sub = df[
                    (df["TASK_BASE"] == task)
                    & (df[dim_col] == dim)
                    ]
                if len(sub) > 0:
                    v = sub[model_cols[ratio_idx]].apply(
                        pd.to_numeric, errors="coerce"
                    ).mean()
                    raw_vals[task][dim] = v
                else:
                    raw_vals[task][dim] = np.nan

        # 2. 对每个 task，单独算 min/max（只在 4 个 dim 内）
        norm_vals = {dim: [] for dim in DIMS}

        for task in tasks:
            vals = [raw_vals[task][d] for d in DIMS]
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)

            for dim in DIMS:
                v = raw_vals[task][dim]
                if np.isnan(v) or vmax == vmin:
                    nv = 0.0
                else:
                    nv = (vmax - v) / (vmax - vmin)

                norm_vals[dim].append(nv)

        # 3. 画线（每条线 = 一个 dim）
        for dim in DIMS:
            values = norm_vals[dim]
            values_closed = values + [values[0]]

            ax.plot(
                angles_closed,
                values_closed,
                color=DIM_COLORS[dim],
                lw=2,
                alpha=0.95,
            )
            ax.fill(
                angles_closed,
                values_closed,
                color=DIM_COLORS[dim],
                alpha=0.12,
            )

    # ======================
    # 4. 画整张图（4 × 4）
    # ======================
    fig, axes = plt.subplots(
        4, 4,
        figsize=(26, 20),
        subplot_kw=dict(polar=True)
    )


    # ---------- 第 1 行：CGCNN（H -4 -2 0） ----------
    for i, ratio in enumerate(["$r=H(-\infty)$",
    r"$r=-4$",
    r"$r=-2$",
    r"$r=0$",]):
        ax = axes[0, i]
        draw_base(ax)
        draw_one(ax, cg_cols, RATIO_LABELS.index(ratio))
        ax.set_title(ratio, fontsize=35, pad=20,
                     # fontweight="bold",
                     y=-0.17)

    # ---------- 第 2 行：MEGNet（H -4 -2 0） ----------
    for i, ratio in enumerate(["$r=H(-\infty)$",
    r"$r=-4$",
    r"$r=-2$",
    r"$r=0$"]):
        ax = axes[1, i]
        draw_base(ax)
        draw_one(ax, mg_cols, RATIO_LABELS.index(ratio))

    # ---------- 第 3 行：CGCNN（2 4 E + Task） ----------
    for i, ratio in enumerate([r"$r=2$",
    r"$r=4$",
    "$r=L(+\infty)$"]):
        ax = axes[2, i]
        draw_base(ax)
        draw_one(ax, cg_cols, RATIO_LABELS.index(ratio))
        ax.set_title(ratio, fontsize=35, pad=20,
                     # fontweight="bold",
                     y=-0.17)

    # Task reference（右侧）
    ax_task = axes[2, 3]
    draw_base(ax_task, show_labels=True)

    # ---------- 第 4 行：MEGNet（2 4 E + Legend） ----------
    for i, ratio in enumerate([r"$r=2$",
    r"$r=4$",
    "$r=L(+\infty)$"]):
        ax = axes[3, i]
        draw_base(ax)
        draw_one(ax, mg_cols, RATIO_LABELS.index(ratio))

    # ---------- 右下角：DIM 图例 ----------
    ax_leg = axes[3, 3]
    ax_leg.axis("off")

    ax_leg.text(
        0.5, 0.15,
        r"$r=\log_{10}$(Learnable / Handcrafted)",
        fontsize=35,
        ha="center", va="top",
        transform=ax_leg.transAxes,
    )

    y0 = 0.75
    dy = 0.15
    for i, dim in enumerate(DIMS):
        ax_leg.plot(
            [0.1, 0.3],
            [y0 - i * dy] * 2,
            color=DIM_COLORS[dim],
            lw=6,
            transform=ax_leg.transAxes,
        )
        ax_leg.text(
            0.35,
            y0 - i * dy,
            f"DIM = {dim}",
            fontsize=35,
            va="center",
            # fontweight="bold",
            transform=ax_leg.transAxes,
        )

    # ---------- 行标签 ----------
    fig.text(
        0.01, 0.865, "CGCNN",
        fontsize=35,
        # fontweight="bold",
        rotation=90, va="center", ha="center"
    )
    fig.text(
        0.01, 0.61, "MEGNet",
        fontsize=35,
        # fontweight="bold",
        rotation=90, va="center", ha="center"
    )
    fig.text(
        0.01, 0.365, "CGCNN",
        fontsize=35,
        # fontweight="bold",
        rotation=90, va="center", ha="center"
    )
    fig.text(
        0.01, 0.113, "MEGNet",
        fontsize=35,
        # fontweight="bold",
        rotation=90, va="center", ha="center"
    )



    plt.tight_layout()
    draw_group_box_by_grid(
        fig,
        axes,
        rows=[0, 1],
        cols=[0, 1, 2, 3],
        color="black",
        lw=4,
    )

    draw_group_box_by_grid(
        fig,
        axes,
        rows=[2, 3],
        cols=[0, 1, 2],
        color="black",
        lw=4,
    )
    plt.savefig("fig3_a.pdf", dpi=300, bbox_inches="tight")

    def print_task_validation(
            model_name="CGCNN",
            ratio_label="0",
            task_to_check="jdft2d"
    ):
        print("\n" + "=" * 70)
        print(f"验证子图: model={model_name}, ratio={ratio_label}, task={task_to_check}")
        print("=" * 70)

        ratio_idx = RATIO_LABELS.index(ratio_label)
        model_cols = cg_cols if model_name == "CGCNN" else mg_cols

        vals = {}

        for dim in DIMS:
            sub = df[
                (df["TASK_BASE"] == task_to_check)
                & (df[dim_col] == dim)
                ]
            if len(sub) > 0:
                v = sub[model_cols[ratio_idx]].apply(
                    pd.to_numeric, errors="coerce"
                ).mean()
                vals[dim] = v
            else:
                vals[dim] = np.nan

        vmin = np.nanmin(list(vals.values()))
        vmax = np.nanmax(list(vals.values()))

        print(f"task 内 min = {vmin:.6f}")
        print(f"task 内 max = {vmax:.6f}")
        print("-" * 50)

        for dim in DIMS:
            v = vals[dim]
            if np.isnan(v) or vmax == vmin:
                nv = 0.0
            else:
                nv = (vmax - v) / (vmax - vmin)

            print(
                f"DIM={dim:2d} | raw={v:.6f} | norm={nv:.4f}"
            )

    print_task_validation(
        model_name="MEGNet",
        ratio_label=r"$r=0$",
        task_to_check="log_kvrh"
    )


if __name__ == "__main__":
    plot_radar_by_ratio("CGCNN_MEGNet.xlsx")
