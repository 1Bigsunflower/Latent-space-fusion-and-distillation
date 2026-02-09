import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]

# ======================
# 1. 读取数据
# ======================
# —— 用于箭头角度（aa）——
df = pd.read_excel("Models_aa.xlsx")

# —— 用于颜色和 colorbar（bb）——
df_bb = pd.read_excel("Models_bb.xlsx")

datasets = [
    "matbench_jdft2d", "matbench_phonons", "matbench_dielectric",
    "matbench_log_gvrh", "matbench_log_kvrh", "matbench_perovskites",
    "matbench_mp_gap", "matbench_mp_e_form",
]
models = ["CGCNN", "MEGNet"]
dims = [8, 16, 32, 64]
ratio_columns = ["e0.0001_f1.0", "e0.01_f1.0", "e1.0_f1.0", "e1.0_f0.01", "e1.0_f0.0001"]

x_labels = [
    r"$r=-4$",
    r"$r=-2$",
    r"$r=0$",
    r"$r=2$",
    r"$r=4$",
]

# ======================
# 2. 颜色归一化（只基于 Models_bb.xlsx）
# ======================
angle_values_bb = df_bb[ratio_columns].values.flatten()
vmin_cb = np.nanmin(angle_values_bb)
vmax_cb = np.nanmax(angle_values_bb)

norm_cb = Normalize(vmin=vmin_cb, vmax=vmax_cb)
cmap = plt.cm.viridis

# ======================
# 3. 画布与 GridSpec
# ======================
fig = plt.figure(figsize=(28, 28))

main_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)

left_gs = gridspec.GridSpecFromSubplotSpec(
    4, len(ratio_columns), subplot_spec=main_gs[0], wspace=0.05, hspace=0.1
)
right_gs = gridspec.GridSpecFromSubplotSpec(
    4, len(ratio_columns), subplot_spec=main_gs[1], wspace=0.05, hspace=0.1
)

# ======================
# 4. 核心绘图
# ======================
for i, task in enumerate(datasets):
    is_left = i < 4
    row = i if is_left else i - 4
    current_gs = left_gs if is_left else right_gs

    for j, ratio in enumerate(ratio_columns):
        ax_outer = fig.add_subplot(current_gs[row, j])
        ax_outer.axis("off")

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            4, 2, subplot_spec=current_gs[row, j], wspace=0.0, hspace=0.0
        )

        for r, dim in enumerate(dims):
            for c, model in enumerate(models):
                ax = fig.add_subplot(inner_gs[r, c])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

                # ========= 箭头角度：来自 Models_aa =========
                val_angle = df.loc[
                    (df["TASK"] == task)
                    & (df["model"] == model)
                    & (df["Dim"] == dim),
                    ratio,
                ]

                # ========= 方块颜色：来自 Models_bb =========
                val_color = df_bb.loc[
                    (df_bb["TASK"] == task)
                    & (df_bb["model"] == model)
                    & (df_bb["Dim"] == dim),
                    ratio,
                ]

                if not val_angle.empty and not val_color.empty:
                    angle = val_angle.values[0]
                    color_value = val_color.values[0]

                    theta = np.deg2rad(angle)

                    # —— 颜色块（bb）——
                    ax.add_patch(
                        Rectangle(
                            (0, 0), 1, 1,
                            facecolor=cmap(norm_cb(color_value)),
                            edgecolor="none"
                        )
                    )

                    origin = (0.2, 0.2)
                    length = 0.65
                    arrow_color = "white" if norm_cb(color_value) < 0.4 else "black"

                    # baseline arrow
                    ax.add_patch(
                        FancyArrowPatch(
                            origin, (0.8, origin[1]),
                            arrowstyle="-|>",
                            linewidth=4,
                            mutation_scale=10,
                            color=arrow_color
                        )
                    )

                    # angle arrow
                    dx, dy = length * np.cos(theta), length * np.sin(theta)
                    ax.add_patch(
                        FancyArrowPatch(
                            origin, (origin[0] + dx, origin[1] + dy),
                            arrowstyle="-|>",
                            linewidth=4,
                            mutation_scale=10,
                            color=arrow_color
                        )
                    )

        # ======================
        # 5. 标签（与你原来一致）
        # ======================
        if j == 0 and is_left:
            ax_outer.text(
                -0.15, 0.5, task.replace("matbench_", ""),
                transform=ax_outer.transAxes,
                va="center", ha="center", fontsize=35, rotation=90
            )

        if j == len(ratio_columns) - 1 and not is_left:
            ax_outer.text(
                1.15, 0.5, task.replace("matbench_", ""),
                transform=ax_outer.transAxes,
                va="center", ha="center", fontsize=35, rotation=90
            )

        if is_left and j == len(ratio_columns) - 1:
            for r, dim in enumerate(dims):
                y_center = 1 - (r + 0.5) / 4
                ax_outer.text(
                    1.38, y_center, f"Dim{dim}",
                    transform=ax_outer.transAxes,
                    va="center", ha="center", fontsize=35
                )

        if row == 0:
            if j == 2:
                ax_outer.text(
                    0.5, 1.12,
                    r"$r=\log_{10}$(Learnable / Handcrafted)",
                    transform=ax_outer.transAxes,
                    ha="center", va="bottom", fontsize=35
                )
            ax_outer.text(
                0.5, 1.0, x_labels[j],
                transform=ax_outer.transAxes,
                ha="center", va="bottom", fontsize=35
            )

        if row == 3:
            ax_outer.text(
                0.25, -0.01, "CGCNN",
                transform=ax_outer.transAxes,
                ha="center", va="top", fontsize=35, rotation=90
            )
            ax_outer.text(
                0.75, -0.01, "MEGNet",
                transform=ax_outer.transAxes,
                ha="center", va="top", fontsize=35, rotation=90
            )

# ======================
# 6. Colorbar（完全基于 Models_bb）
# ======================
cax = fig.add_axes([0.2, 0.02, 0.6, 0.015])
sm = ScalarMappable(norm=norm_cb, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

ticks = np.linspace(vmin_cb, vmax_cb, 5)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

cbar.ax.tick_params(labelsize=35)
cbar.set_label(r"Angle variance ($\mathrm{deg}^2$)", fontsize=35)


# ======================
plt.savefig("fig4aa.pdf", dpi=300, bbox_inches="tight")
