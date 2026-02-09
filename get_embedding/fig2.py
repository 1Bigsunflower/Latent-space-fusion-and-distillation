# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
# ======================
# 配置
# ======================

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
X_LABELS = [
    "$H$\n$(-\infty)$",
    r"$-4$",
    r"$-2$",
    r"$0$",
    r"$2$",
    r"$4$",
    "$L$\n$(+\infty)$"
]

RATIO_LABELS = [
    "$H$\n$(-\infty)$",
    r"$-4$",
    r"$-2$",
    r"$0$",
    r"$2$",
    r"$4$",
    "$L$\n$(+\infty)$"
]

# X_LABELS = ["$\\boldsymbol{r=H}$\n$\\boldsymbol{(+\\boldsymbol{\\infty})}$",
#     r"$\boldsymbol{r=-4}$",
#     r"$\boldsymbol{r=-2}$",
#     r"$\boldsymbol{r=0}$",
#     r"$\boldsymbol{r=2}$",
#     r"$\boldsymbol{r=4}$",
#     "$\\boldsymbol{r=L}$\n$\\boldsymbol{(-\\boldsymbol{\\infty})}$"]
# RATIO_LABELS = ["$\\boldsymbol{r=H}$\n$\\boldsymbol{(+\\boldsymbol{\\infty})}$",
#     r"$\boldsymbol{r=-4}$",
#     r"$\boldsymbol{r=-2}$",
#     r"$\boldsymbol{r=0}$",
#     r"$\boldsymbol{r=2}$",
#     r"$\boldsymbol{r=4}$",
#     "$\\boldsymbol{r=L}$\n$\\boldsymbol{(-\\boldsymbol{\\infty})}$"]
DIMS = [8, 16, 32, 64]

MODEL_MARKERS = {
    "CGCNN": "*",
    "MEGNet": "^",
}


def _format_xtick_two_lines(s: str) -> str:
    if "_" in s:
        a, b = s.split("_", 1)
        return f"{a}\n{b}"
    return s


# ======================
# 主函数
# ======================

def plot_matbench_curves(
        excel_path: str,
        sheet_name=0,
        ncols: int = 3,
        save_path: str = "final.pdf",
):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    TASK_COL_IDX = 0
    DIM_COL_IDX = 8
    CGCNN_COL_IDXS = [1, 7, 6, 5, 4, 3, 2]
    MEGNET_COL_IDXS = [9, 15, 14, 13, 12, 11, 10]

    task_col = df.columns[TASK_COL_IDX]
    dim_col = df.columns[DIM_COL_IDX]
    cgcnn_cols = [df.columns[i] for i in CGCNN_COL_IDXS]
    megnet_cols = [df.columns[i] for i in MEGNET_COL_IDXS]

    # ---------- 数据清洗 ----------
    df = df.copy()
    df[task_col] = df[task_col].astype(str)
    df["TASK_BASE"] = df[task_col].str.replace(r"_fold\d+$", "", regex=True)
    df = df[df["TASK_BASE"].str.startswith(PREFIXES)].copy()

    df[dim_col] = pd.to_numeric(df[dim_col], errors="coerce")
    df = df.dropna(subset=[dim_col])
    df[dim_col] = df[dim_col].astype(int)

    tasks = []
    for p in PREFIXES:
        tasks.extend(sorted(t for t in df["TASK_BASE"].unique() if t.startswith(p)))

    n_tasks = len(tasks)
    nrows = math.ceil((n_tasks + 1) / ncols)

    # 稍微增加高度以适应无间距布局
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10 * ncols, 7 * nrows),
        squeeze=False,
    )

    # 核心修改：垂直间距设为0，水平间距拉开给Y轴留空间
    plt.subplots_adjust(hspace=0, wspace=0.25)

    axes_flat = axes.ravel()

    x = np.arange(len(X_LABELS))
    xticklabels = [_format_xtick_two_lines(s) for s in X_LABELS]

    model_handles = []

    for i_task, task in enumerate(tasks):
        ax = axes_flat[i_task]
        sub = df[df["TASK_BASE"] == task]

        # 判断是否为该列的最底层子图（包括饼图占位的情况）
        curr_col = i_task % ncols
        is_bottom = (i_task + ncols >= n_tasks)

        # ---------- 绘制折线 ----------
        all_y_values = []
        for model, cols, color in [
            ("CGCNN", cgcnn_cols, "#3C5488"),
            ("MEGNet", megnet_cols, "#f63c42"),
        ]:
            dim_means = []
            for dim in DIMS:
                sd = sub[sub[dim_col] == dim]
                if len(sd):
                    dim_means.append(
                        sd[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
                    )

            if not dim_means:
                continue

            dim_means = np.vstack(dim_means)
            y_low = np.nanmin(dim_means, axis=0)
            y_high = np.nanmax(dim_means, axis=0)
            y_mean = sub[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
            all_y_values.extend(y_mean.dropna().tolist())

            fill = ax.fill_between(x, y_low, y_high, color=color, alpha=0.25)
            line, = ax.plot(
                x, y_mean, color=color, linewidth=4.5,
                marker=MODEL_MARKERS[model], markersize=14, label=model,
            )

            if i_task == 0:
                import matplotlib.patches as mpatches
                # 放入折线
                model_handles.append(line)
                # 创建并放入代表阴影的长方形 (Patch)
                range_patch = mpatches.Patch(color=color, alpha=0.25, label=f"{model} Dimension-wise MAE Range")
                model_handles.append(range_patch)

            idx_min = np.nanargmin(y_mean)
            ax.scatter(
                [x[idx_min]], [y_mean[idx_min]], s=700,
                edgecolors=color, facecolors="none", linewidths=2.5, zorder=6,
            )

        # ---------- 1. 任务名放到左上角 ----------
        ax.text(
            0.03, 0.96, f"matbench_{task}",
            transform=ax.transAxes,
            fontsize=30,
            # fontweight="bold",
            va="top",
            ha="left",
            # bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        # ---------- 2. X轴标签控制 ----------
        ax.set_xticks(x)
        if is_bottom:
            ax.set_xticklabels(xticklabels, fontsize=30,
                               #fontweight="bold",
            )
        else:
            ax.set_xticklabels([])  # 隐藏文字

        # ---------- 3. Y轴小数点控制 (硬核去重版) ----------
        ax.tick_params(axis="y", labelsize=30,)
        # for label in ax.get_yticklabels():
        #     label.set_fontweight('bold')
        ax.set_ylabel("MAE", fontsize=30,
                      # fontweight="bold",
                      )

        if len(all_y_values) > 0:
            y_max_val = max(all_y_values)

            # 1. 先用 Locator 选出合理的刻度位置
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))

            # 2. 定义自定义格式化函数
            def smart_formatter(x, pos, y_max=y_max_val):
                if y_max < 1:
                    return f"{x:.2f}"
                else:
                    return f"{x:g}"

            # 3. 使用 FuncFormatter 并手动处理重复标签
            formatter = mticker.FuncFormatter(smart_formatter)
            ax.yaxis.set_major_formatter(formatter)

            # 4. 【关键步骤】强制渲染后，检测并删除重复文字标签
            plt.draw()  # 必须先 draw 才能获取 ticklabels
            labels = ax.get_yticklabels()
            seen_texts = set()
            for label in labels:
                txt = label.get_text()
                if txt in seen_texts:
                    label.set_visible(False)  # 如果这个字符串出现过，就隐身
                else:
                    seen_texts.add(txt)

    # ======================
    # 饼图与 Legend 部分修改
    # ======================
    empty_idxs = list(range(n_tasks, len(axes_flat)))
    ax_pie = axes_flat[empty_idxs[0]]
    ax_pie.axis("off")

    # 调整子图布局：给下方文字留出更多空间
    ax_cg = ax_pie.inset_axes([0.0, 0.25, 0.4, 0.50])
    ax_mg = ax_pie.inset_axes([0.5, 0.25, 0.4, 0.50])

    # 数据统计
    vals_cg = df[cgcnn_cols].apply(pd.to_numeric, errors="coerce")
    vals_mg = df[megnet_cols].apply(pd.to_numeric, errors="coerce")
    cg_counts = vals_cg.idxmin(axis=1).value_counts().reindex(cgcnn_cols, fill_value=0)
    mg_counts = vals_mg.idxmin(axis=1).value_counts().reindex(megnet_cols, fill_value=0)

    cg_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 7))
    mg_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 7))

    # 绘制饼图
    # 注意：这里去掉了 label，防止它们自动跑到图例里
    wedges_cg, _ = ax_cg.pie(cg_counts.values, colors=cg_colors, startangle=90, radius=1.5,
                             wedgeprops=dict(edgecolor="black", linewidth=1))
    wedges_mg, _ = ax_mg.pie(mg_counts.values, colors=mg_colors, startangle=90, radius=1.5,
                             wedgeprops=dict(edgecolor="black", linewidth=1))

    # --- 修改 1: 修复匹配逻辑并添加白色小圆点 ---
    # 统一使用去除换行符的集合进行比对
    CHECK_SET_CLEAN = {"H", "L"}

    def apply_custom_hatch(wedges, labels):
        for w, l in zip(wedges, labels):
            if any(key in l for key in CHECK_SET_CLEAN):
                w.set_hatch('O')
                w.set_edgecolor('white')
                w.set_linewidth(0)
                w.set_antialiased(False)

                import matplotlib.patches as mpatches
                outline = mpatches.Wedge(
                    w.center, w.r, w.theta1, w.theta2,
                    fill=False,
                    edgecolor='black',
                    linewidth=1.2,
                    zorder=1000
                )
                w.axes.add_patch(outline)

    apply_custom_hatch(wedges_cg, RATIO_LABELS)
    apply_custom_hatch(wedges_mg, RATIO_LABELS)

    ax_cg.set_title("CGCNN", fontsize=30, pad=30,
                    # fontweight="bold"
                    )
    ax_mg.set_title("MEGNet", fontsize=30, pad=30,
                    # fontweight="bold"
                    )

    # 其余空子图关掉
    for idx in empty_idxs[1:]:
        axes_flat[idx].axis("off")

    from matplotlib.patches import Patch

    # 底部总 Legend
    legend_labels = [
        "CGCNN", "CGCNN Dimension-wise MAE Range",
        "MEGNet", "MEGNet Dimension-wise MAE Range"
    ]

    fig.legend(
        model_handles,
        legend_labels,
        loc="lower center",
        ncol=4,  # 设为 4 列
        frameon=False,
        prop={'size': 30,
              # 'weight': 'bold'
              },  # 稍微调小字号以防过挤
        bbox_to_anchor=(0.5, 0.01)
    )
    # Colorbar 调整
    cmap_cg = ListedColormap(cg_colors)
    cmap_mg = ListedColormap(mg_colors)
    bounds = np.arange(len(RATIO_LABELS) + 1)
    norm = BoundaryNorm(bounds, cmap_cg.N)
    sm_cg = ScalarMappable(norm=norm, cmap=cmap_cg)
    sm_mg = ScalarMappable(norm=norm, cmap=cmap_mg)

    cax_cg = ax_pie.inset_axes([0.0, 0.11, 0.90, 0.05])
    cax_mg = ax_pie.inset_axes([0.0, 0.05, 0.90, 0.05])

    cbar_cg = plt.colorbar(sm_cg, cax=cax_cg, orientation="horizontal", ticks=np.arange(len(RATIO_LABELS)) + 0.5)
    cbar_cg.ax.tick_params(labelbottom=False)
    # cbar_cg.ax.tick_params(labelsize=18)

    cbar_mg = plt.colorbar(sm_mg, cax=cax_mg, orientation="horizontal", ticks=np.arange(len(RATIO_LABELS)) + 0.5)
    cbar_mg.set_ticklabels(RATIO_LABELS)
    cbar_mg.ax.tick_params(labelsize=30)

    # 为第一个 colorbar 添加黑色边框
    cbar_cg.outline.set_edgecolor('black')
    cbar_cg.outline.set_linewidth(2.0)

    # 为第二个 colorbar 添加黑色边框
    cbar_mg.outline.set_edgecolor('black')
    cbar_mg.outline.set_linewidth(2.0)
    # for label in cbar_mg.ax.get_xticklabels():
    #     label.set_fontweight("bold")
    cbar_mg.set_label(r"$r=\log_{10}$(Learnable / Handcrafted)", fontsize=30)

    def add_extreme_hatch(cax, n_bins):
        for idx in [0, n_bins - 1]:
            # 将 hatch 修改为 'o' (小圆圈)
            # 将 edgecolor 设为 'white'
            rect = plt.Rectangle(
                (idx / n_bins, 0), 1 / n_bins, 1,
                transform=cax.transAxes,
                facecolor="none",
                edgecolor="white",  # 这里改为白色
                hatch="O",  # 这里改为圆圈纹理
                linewidth=0,  # 建议取消边框宽度，让圆圈更纯净
                zorder=10
            )
            cax.add_patch(rect)

            # 可选：如果白色圆圈在浅色背景上不明显，可以再加一层黑色的疏淡纹理或保持原样

    add_extreme_hatch(cax_cg, len(RATIO_LABELS))
    add_extreme_hatch(cax_mg, len(RATIO_LABELS))

    # ======================
    # 给每一列添加 column-wise xlabel: "r"
    # ======================
    label_y_offset = 0.02  # 控制 r 距离刻度的垂直间距（可微调）

    for col in range(ncols):
        # 找到该列最底部的 subplot
        bottom_row = nrows - 1
        ax_bottom = axes[bottom_row, col]

        # 获取该轴在 figure 坐标中的位置
        bbox = ax_bottom.get_position()

        # 计算该列的中心 x
        x_center = (bbox.x0 + bbox.x1) / 2

        # 在该列下方放置标题
        fig.text(
            x_center,
            bbox.y0 - label_y_offset,
            r"$r$",
            ha="center",
            va="top",
            fontsize=30,
            fontname="Arial"
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    # 请确保路径下有对应的 Excel 文件
    plot_matbench_curves(
        excel_path="CGCNN_MEGNet.xlsx",
        sheet_name=0,
        ncols=3,
        save_path="fig2_a.pdf",
    )