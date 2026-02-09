import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
# ====================
# 1. 读数据
# =====================
df = pd.read_excel("element_cosine_distance.xlsx", index_col=0)
# df = pd.read_excel("mat2vec_cosine_distance.xlsx", index_col=0)
vals = df.values[np.tril_indices_from(df, k=0)]
vmin = np.nanmin(vals)
vmax = np.nanmax(vals)
# vmin = 0
# vmax = 1
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# 右侧color显示
cos_angle_max = 0.7
# cos_angle_max = 0.4
# 原始 cmap
base_cmap = plt.cm.viridis

# 1 在归一化中的位置
cut = (cos_angle_max - vmin) / (vmax - vmin)
cut = np.clip(cut, 0, 1)

# 构造新 colormap
colors = base_cmap(np.linspace(0, 1, 256))
cut_idx = int(cut * 256)

colors[cut_idx:, :] = np.array([1, 1, 1, 1])  # 白色 RGBA

new_cmap = LinearSegmentedColormap.from_list(
    "viridis_white_tail",
    colors
)

# =====================
# 2. 构造 mask
# =====================
# 下三角（含对角线）可见
mask_upper = np.tril(np.ones(df.shape), k=0)

# 上三角可见
mask_lower = np.triu(np.ones(df.shape), k=1)

# =====================
# 3. 上三角：只保留 distance < 1
# =====================
df_upper = df.copy()
df_upper[df_upper >= cos_angle_max] = np.nan
np.fill_diagonal(df_upper.values, np.nan)

# =====================
# 4. 画图
# =====================
fig, ax = plt.subplots(figsize=(36, 34))

# ---- 下三角：完整范围 + 横向 colorbar ----
hm_lower = sns.heatmap(
    df,
    mask=mask_lower,
    cmap="viridis",
    square=True,
    norm=norm,
    annot=False,
    cbar=True,
    cbar_kws={
        "label": "Cosine Distance",
        "orientation": "horizontal",
        "pad": 0.05,
        "shrink": 0.6,
        "aspect": 40
    },
    ax=ax
)

# ---- 上三角：< 1 的部分 ----
hm_upper = sns.heatmap(
    df_upper,
    mask=mask_upper,
    cmap=new_cmap,
    norm=norm,
    square=True,
    annot=False,
    cbar=False,  # 移除seaborn的colorbar
    ax=ax
)

# =====================
# 5. 强制显示全部刻度
# =====================
n = len(df)

# x 轴：整体向右移动 0.5
ax.set_xticks(np.arange(n) + 0.5)

# y 轴：整体向下移动 0.5
ax.set_yticks(np.arange(n) + 0.5)

ax.set_xticklabels(df.columns, rotation=0, fontsize=32, fontname='Arial')
ax.set_yticklabels(df.index, rotation=0, fontsize=32, fontname='Arial')

# =====================
# 6. 错位刻度标签（你的原逻辑）
# =====================
for i, label in enumerate(ax.get_xticklabels()):
    if i % 2 == 0:
        label.set_y(0)
    else:
        label.set_y(-0.03)

for i, label in enumerate(ax.get_yticklabels()):
    if i % 2 == 0:
        label.set_x(0)
    else:
        label.set_x(-0.03)

# =====================
# 7. colorbar 字体和位置调整
# =====================
# 下三角 colorbar（横向）- 调整为与右侧colorbar相似大小
cbar_lower = hm_lower.collections[0].colorbar
n_ticks = 6
ticks = np.linspace(vmin, vmax, n_ticks)

cbar_lower.set_ticks(ticks)
cbar_lower.ax.tick_params(labelsize=40)
cbar_lower.set_label("Cosine Distance", fontsize=40, fontname='Arial')

formatter = FormatStrFormatter('%.2f')
cbar_lower.ax.xaxis.set_major_formatter(formatter)

cbar_lower.ax.set_position([
    0.15,  # 左边距
    0.05,  # 底部高度
    0.5,   # 宽度
    0.02   # 高度（横条要很薄）
])


# =====================
# 8. 在右侧添加自定义colorbar（大于1为白色）
# =====================
# 创建一个独立的axes用于右侧colorbar
cax = fig.add_axes([
    0.8,  # 左边界位置（heatmap右侧）左右
    0.28,  # 底部位置
    0.015,  # 宽度
    0.7    # 高度
])

# 创建自定义colorbar，使用你的new_cmap
cbar_right = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
    cax=cax,
    orientation='vertical'
)

cbar_right.set_ticks(ticks)
cbar_right.ax.yaxis.set_major_formatter(formatter)
cbar_right.set_label(f"Cosine Distance (values < {cos_angle_max:.1f})", fontsize=40, fontname='Arial')
cbar_right.ax.tick_params(labelsize=40)


# =====================
# 9. 添加整个热力图的黑色边框
# =====================
# 获取当前坐标轴的位置信息
ax_pos = ax.get_position()

# 在图形坐标中创建一个黑色边框
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

# 创建一个矩形框，覆盖整个坐标轴区域
# 使用图形坐标 (0,0) 到 (1,1) 的相对位置
border = Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                   facecolor='none', transform=ax.transAxes)
ax.add_patch(border)

# 或者更简单的方法：设置坐标轴边框颜色和宽度
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# 设置边框颜色为黑色，线宽适中
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)


# =====================
# 9. 布局 & 保存
# =====================
plt.tight_layout()
# 调整布局以避免右侧colorbar被裁剪
plt.subplots_adjust(right=0.85)

# print("Global vmax (norm):", norm.vmax)
# print("Lower colorbar max tick:", cbar_lower.get_ticks()[-1])
# print("Right colorbar max tick:", cbar_right.get_ticks()[-1])
# print("Upper triangle visible max:", cos_angle_max)
plt.savefig("Element_cosine_triangle_2.pdf", bbox_inches="tight")
# plt.savefig("mat2vec_cosine_triangle_2.pdf", bbox_inches="tight")