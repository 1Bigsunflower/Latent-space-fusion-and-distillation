import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

features = ['Dimension',
            'Effective_Rank',
            'Embedding_Entropy',
            'Spectral_Radius', 'Spectral_Gap', 'Fiedler_Value',
            'ARI', 'Silhouette',
            ]

files = ['CGCNN_scaled_129', 'MEGNet_scaled_129']
model_map = {
    'CGCNN_scaled_129': 'CGCNN',
    'MEGNet_scaled_129': 'MEGNet'
}

method_map = {
    'cgcnn_hm': 'Human',
    'cgcnn_emb': 'Embedding',
    'megnet_hm': 'Human',
    'megnet_emb': 'Embedding'
}

desired_order = [
    "CGCNN_Human_Pearson", "CGCNN_Human_Spearman",
    "CGCNN_Embedding_Pearson", "CGCNN_Embedding_Spearman",
    "MEGNet_Human_Pearson", "MEGNet_Human_Spearman",
    "MEGNet_Embedding_Pearson", "MEGNet_Embedding_Spearman"
]

task_names = [
    # 'matbench_jdft2d',
    'matbench_phonons',
    # 'matbench_dielectric',
    'matbench_log_gvrh',
    'matbench_log_kvrh',
    # 'matbench_perovskites',
    'matbench_mp_gap', 'matbench_mp_e_form'
]

result_df = pd.DataFrame(index=features)

# 存放单元格折线图数据
line_data = {}

for file in files:
    df = pd.read_csv(file).round(8)
    model_name = model_map[file]

    for method in df['Method'].unique():
        if method in method_map:
            subset = df[df['Method'] == method]
            subset = subset[~subset['Task'].isin(['matbench_perovskites', 'matbench_dielectric', 'matbench_jdft2d'])]

            method_label = method_map[method]
            pearson_name = f"{model_name}_{method_label}_Pearson"
            spearman_name = f"{model_name}_{method_label}_Spearman"

            line_data[pearson_name] = {}
            line_data[spearman_name] = {}

            mean_corr = {}
            mean_corr_s = {}

            for feature in features:
                vec_p = [
                    subset[subset['Task'] == task][[feature, 'MAE_Norm']].corr(method='pearson').iloc[0, 1]
                    if not subset[subset['Task'] == task].empty else np.nan
                    for task in task_names
                ]
                vec_s = [
                    subset[subset['Task'] == task][[feature, 'MAE_Norm']].corr(method='spearman').iloc[0, 1]
                    if not subset[subset['Task'] == task].empty else np.nan
                    for task in task_names
                ]
                line_data[pearson_name][feature] = vec_p
                line_data[spearman_name][feature] = vec_s

                mean_corr[feature] = np.nanmean(vec_p)
                mean_corr_s[feature] = np.nanmean(vec_s)

            result_df[pearson_name] = pd.Series(mean_corr)
            result_df[spearman_name] = pd.Series(mean_corr_s)

result_plot = result_df.T
result_plot = result_plot.loc[desired_order]


def split_label(label, max_parts=3):
    parts = label.split('_')
    return '\n'.join(parts[:max_parts])


orig_row_labels = result_plot.index.tolist()

result_plot.index = [split_label(idx, 3) for idx in result_plot.index]
result_plot.columns = [split_label(col, 3) for col in result_plot.columns]

# vmin = result_plot.min().min()
# vmax = result_plot.max().max()

vmin = -1.0
vmax = 1.0

plt.figure(figsize=(14, 8))
ax = sns.heatmap(result_plot,
                 vmin=vmin, vmax=vmax,
                 # annot=False,
                 center=0, fmt=".4f",
                 linewidths=0.8, cmap="RdBu_r",
                 cbar_kws={"shrink": 0.8,
                           "ticks": np.linspace(vmin, vmax, 7)}
                 )

plt.xticks(fontsize=14)
plt.yticks(fontsize=15)
plt.tight_layout()

cbar_ax = ax.collections[0].colorbar.ax
pos = cbar_ax.get_position()
cbar_ax.set_position([pos.x0, pos.y0 + 0.08, pos.width, pos.height])

ymin, ymax = -1, 1

vmin = result_plot.replace({np.nan: 0}).values.min()
vmax = result_plot.replace({np.nan: 0}).values.max()
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.RdBu_r

n_rows, n_cols = result_plot.shape

for i in range(n_rows):
    cell_key = orig_row_labels[i]  # 原始标签，例："CGCNN_Human_Pearson"
    for j in range(n_cols):
        feature = features[j]

        cell_value = result_plot.iloc[i, j]

        cell_color = cmap(norm(cell_value))
        # 颜色亮度（近似公式：0.299R+0.587G+0.114B）
        brightness = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
        # 背景色暗白色，否则黑色
        line_color = 'w' if brightness < 0.5 else 'k'

        cell_left = j / n_cols
        cell_bottom = 1 - (i + 1) / n_rows
        cell_width = 1 / n_cols
        cell_height = 1 / n_rows

        inset_margin = 0.15
        inset_ax = ax.inset_axes([cell_left + inset_margin * cell_width,
                                  cell_bottom + inset_margin * cell_height,
                                  cell_width * (1 - 2 * inset_margin),
                                  cell_height * (1 - 2 * inset_margin)])

        x_vals = range(len(task_names))
        y_vals = line_data[cell_key][feature]
        inset_ax.plot(x_vals, y_vals, color=line_color, linewidth=2)

        inset_ax.axhline(0, color=line_color, linestyle='--', linewidth=1)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        inset_ax.set_xlim(0, len(task_names) - 1)
        inset_ax.set_ylim(ymin, ymax)

        inset_ax.patch.set_alpha(0)

plt.savefig("heatmap.pdf", bbox_inches="tight")

