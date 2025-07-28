import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.distance import cdist


def plot_embeddings(task_name, file_paths, ax):
    """
    在指定 ax 上绘制 embedding 曲线。
    """
    colors = ['#b4403e', '#256ea2', '#ea841e', '#399335']
    legend_labels = ['CGCNN-Embedding', 'CGCNN-Human knowledge', 'MEGNet-Embedding', 'MEGNet-Human knowledge']
    x_labels = ['Element\nembedding', 'Block1', 'Block2', 'Block3']
    x = np.arange(len(x_labels))

    for i, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)

        if '_emb' in file_path:
            pattern = f'^{task_name}.*_emb'
        elif '_hm' in file_path:
            pattern = f'^{task_name}.*_hm'

        filtered_df = df[df['task'].str.contains(pattern, regex=True)]
        values_list = []
        for _, row in filtered_df.iterrows():
            values = [row[1], row[-3], row[-2], row[-1]]
            values_list.append(values)

        values_array = np.array(values_list)
        rep_index = 0
        if len(values_array) > 1:
            dist_matrix = cdist(values_array, values_array)
            total_distances = dist_matrix.sum(axis=1)
            rep_index = np.argmin(total_distances)

        for j, y in enumerate(values_array):
            x_smooth = np.linspace(x.min(), x.max(), 100)
            spline = make_interp_spline(x, y, k=2)
            y_smooth = spline(x_smooth)

            if j == rep_index:
                ax.plot(x_smooth, y_smooth,
                        color=colors[i],
                        linewidth=2.5,
                        alpha=1.0,
                        label=legend_labels[i])

                ax.scatter(x, y,
                           color=colors[i],
                           edgecolor='black',
                           linewidths=0.3,
                           s=15,
                           zorder=5)
            else:
                ax.plot(x_smooth, y_smooth,
                        color=colors[i],
                        linewidth=1,
                        alpha=0.2)

                ax.scatter(x, y,
                           color=colors[i],
                           alpha=0.2,
                           edgecolor='gray',
                           linewidths=0.2,
                           s=8,
                           zorder=4)

    ax.set_title(f'{task_name}', fontsize=18)
    ax.set_ylabel('Adjusted Rand Index', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=16, rotation=0)
    ax.tick_params(axis='y', labelsize=16)


if __name__ == '__main__':
    file_list = [
        'cgcnn_emb_ari.csv',
        'cgcnn_hm_ari.csv',
        'megnet_emb_ari.csv',
        'megnet_hm_ari.csv'
    ]
    tasks = [
        'matbench_phonons',
        'matbench_log_gvrh',
        'matbench_log_kvrh',
        'matbench_mp_gap',
        'matbench_mp_e_form'
    ]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()

    for i, task in enumerate(tasks):
        plot_embeddings(task, file_list, axs[i])

    fig.delaxes(axs[5])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower right',
               bbox_to_anchor=(0.98, 0.12),
               fontsize=18)

    plt.tight_layout()
    plt.savefig("picture/all_tasks_ari.pdf", bbox_inches="tight")
    # plt.show()
