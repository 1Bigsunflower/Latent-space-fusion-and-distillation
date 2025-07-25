import csv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib
matplotlib.use('TkAgg')

def draw_errorbar(emb_list, cgcnn_list, benchemark, benchemark2, task_name, samples_list, unit_list):
    num_rows = 2
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    bar_width = 0.25  # 柱状图宽度
    for i, (emb_mae_data,  cgcnn_mae_data) in enumerate(zip(emb_list,  cgcnn_list)):
        ax = axes[i // num_cols, i % num_cols]

        # 平均值和标准差
        means_emb = [np.mean(values) for key, values in emb_mae_data.items()]
        stds_emb = [np.std(values) for key, values in emb_mae_data.items()]

        means_cgcnn = [np.mean(values) for key, values in cgcnn_mae_data.items()]
        stds_cgcnn = [np.std(values) for key, values in cgcnn_mae_data.items()]

        x = np.arange(len(cgcnn_mae_data))
        emb_bars = ax.bar(x - 0.5*bar_width, means_emb, width=bar_width,  # yerr=stds_emb,  capsize=4,
               color=(157 / 255, 210 / 255, 253 / 255), label='Embedding')
        cgcnn_bars = ax.bar(x+0.5*bar_width, means_cgcnn, width=bar_width,  # yerr=stds_cgcnn,  capsize=4,
               color=(249 / 255, 91 / 255, 93 / 255), label='Cgcnn')

        # 基准线
        ax.axhline(y=benchemark[i], color='black', linestyle='--', linewidth=2, dashes=(5, 8))
        ax.axhline(y=benchemark2[i], color='gray', linestyle='--', linewidth=2, dashes=(2, 3))
        ax.set_xticks(x)
        ax.set_xticklabels(cgcnn_mae_data.keys(), fontsize=18)
        ax.set_xlim(-0.5, len(emb_mae_data) - 0.5)
        ax.set_title(task_name[i], fontsize=18, pad=20)
        ax.tick_params(axis='y', labelsize=18)

        # 设置y轴范围
        min_val = min(min(means_emb), min(means_cgcnn), benchemark[i], benchemark2[i])
        max_val = max(max(means_emb), min(means_cgcnn), benchemark[i], benchemark2[i])
        margin1 = (max_val - min_val)*0.05
        margin2 = (max_val - min_val)*0.18
        ax.set_ylim(min_val - margin1, max_val + margin2)
        # 样本和单位信息
        ax.text(0.5, 1.035, f"(samples: {samples_list[i]}, unit: {unit_list[i]})", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='black')

        min_emb_idx = np.argmin(means_emb)
        min_cgcnn_idx = np.argmin(means_cgcnn)


        if means_emb[min_emb_idx] < means_cgcnn[min_cgcnn_idx]:
            emb_bars[min_emb_idx].set_hatch('**')
            emb_bars[min_emb_idx].set_edgecolor('black')
            emb_bars[min_emb_idx].set_linewidth(1)
        else:
            cgcnn_bars[min_cgcnn_idx].set_hatch('..')
            cgcnn_bars[min_cgcnn_idx].set_edgecolor('black')
            cgcnn_bars[min_cgcnn_idx].set_linewidth(1)

    fig.text(0.5, 0.04, 'Embedding Dimension', ha='center', fontsize=25)
    fig.text(0.04, 0.5, 'Mean Absolute Error (MAE)', va='center', rotation='vertical', fontsize=25)

    # legend
    handles = [
        mlines.Line2D([], [], color=(157 / 255, 210 / 255, 253 / 255), marker='s', markersize=15, linestyle='None',
                      label='Embedding'),
        mlines.Line2D([], [], color=(249 / 255, 91 / 255, 93 / 255), marker='s',
                      markersize=15, linestyle='None', label='Human Knowledge'),
        mlines.Line2D([], [], color='black', linestyle='--', label='Top Performance on Leaderboard', linewidth=2, dashes=(5, 8)),
        mlines.Line2D([], [], color='gray', linestyle='--', label='Official Model Performance', linewidth=2, dashes=(2, 3))
    ]

    fig.legend(handles=handles, fontsize=17, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))


    plt.subplots_adjust(hspace=0.3, wspace=0.4)

    plt.savefig("CGCNN_compare_2025_7_21.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.clf()


def modified_csv(csv_path):
    keys = ['8', '16', '32', '64']
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    for i in range(len(data)):
        key_index = i // 8 % len(keys)
        data[i].insert(0, keys[key_index])
    return data


def process_cleaned_data(cleaned_data, task):
    processed_dict = {}
    for i in range(task, len(cleaned_data), 8):
        dim = cleaned_data[i][0]
        mae_values = [float(val) for val in cleaned_data[i][1:]]
        processed_dict[dim] = mae_values
    return processed_dict


if __name__ == '__main__':
    csv_file1 = 'cgcnn_emb_alltask.csv'
    mae_data1 = modified_csv(csv_file1)
    emb_list = [process_cleaned_data(mae_data1, i) for i in range(8)]

    csv_file2 = 'cgcnn_cgcnn_alltask.csv'
    mae_data2 = modified_csv(csv_file2)
    cgcnn_list = [process_cleaned_data(mae_data2, i) for i in range(8)]

    benchmark_list = [33.1918, 28.7606, 0.2711, 0.0670, 0.0491, 0.0269, 0.1559, 0.0170]
    cgcnn_benchmark_list = [49.2440, 57.7635, 0.5988, 0.0895, 0.0712, 0.0452, 0.2972, 0.0337]
    task_name = ["matbench_jdft2d", "matbench_phonons", "matbench_dielectric", "matbench_log_gvrh", "matbench_log_kvrh",
                 "matbench_perovskites", "matbench_mp_gap", "matbench_mp_e_form"]
    unit_list = ["meV/atom", "cm^-1", "unitless", "log10(GPa)", "log10(GPa)", "eV/unit cell", "eV", "eV/atom"]
    samples_list = ["636", "1,265", "4,764", "10,987", "10,987", "18,928", "106,113", "132,752"]
    draw_errorbar(emb_list, cgcnn_list, benchmark_list, cgcnn_benchmark_list, task_name, samples_list, unit_list)
