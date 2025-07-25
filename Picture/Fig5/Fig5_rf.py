import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.metrics import r2_score

matplotlib.use('TkAgg')

task_names = [
    'matbench_jdft2d',
    'matbench_phonons',
    'matbench_dielectric',
    'matbench_log_gvrh',
    'matbench_log_kvrh',
    'matbench_perovskites',
    'matbench_mp_gap',
    'matbench_mp_e_form'
]

custom_task_order = [
    'matbench_jdft2d',
    'matbench_dielectric',
    'matbench_perovskites',
    'matbench_phonons',
    'matbench_log_gvrh',
    'matbench_log_kvrh',
    'matbench_mp_gap',
    'matbench_mp_e_form'
]

samples_list = ["636", "1,265", "4,764", "10,987", "10,987", "18,928", "106,113", "132,752"]
samples_list = [int(s.replace(",", "")) for s in samples_list]
task_to_samples = dict(zip(task_names, samples_list))

# Baseline & dummy MAE
matbench_dict = {
    "matbench_jdft2d": 33.1918,
    "matbench_phonons": 28.7606,
    "matbench_dielectric": 0.2711,
    "matbench_log_gvrh": 0.0670,
    "matbench_log_kvrh": 0.0491,
    "matbench_perovskites": 0.0269,
    "matbench_mp_gap": 0.1559,
    "matbench_mp_e_form": 0.0170,
}

matbench_dummy_dict = {
    "matbench_jdft2d": 67.2851,
    "matbench_phonons": 323.9822,
    "matbench_dielectric": 0.8088,
    "matbench_log_gvrh": 0.2931,
    "matbench_log_kvrh": 0.2897,
    "matbench_perovskites": 0.5660,
    "matbench_mp_gap": 1.3272,
    "matbench_mp_e_form": 1.0059,
}


def compute_relative_mae_performance(row):
    task = row['Task']
    mae = row['MAE']
    dummy = matbench_dummy_dict.get(task)
    baseline = matbench_dict.get(task)
    if dummy is not None and baseline is not None and dummy != baseline:
        perf = (dummy - mae) / (dummy - baseline) * 100
        return max(0, min(perf, 100))
    return None


def compute_relative_r2_performance(task_name, r2, r2_ref_df):
    row = r2_ref_df[r2_ref_df['Task'] == task_name]
    if not row.empty:
        dummy_r2 = row['R2_Dummy'].values[0]
        best_r2 = row['R2_Best'].values[0]
        if best_r2 != dummy_r2:
            perf = (r2 - dummy_r2) / (best_r2 - dummy_r2) * 100
            return max(0, min(perf, 100))
    return None


def draw_plt(data_df, mae_perf_df, r2_perf_df):
    task_order = custom_task_order
    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(24, 10),
                                   gridspec_kw={'width_ratios': [1, 1]}
                                   )

    # 小提琴图
    sns.violinplot(
        data=data_df,
        x='task_name',
        y='r2_score',
        inner='box',
        ax=ax1,
        order=task_order,
        palette=['#FF6158', '#FFA78F', '#FF889F', '#FDBE6E', '#FFFDC4', '#AF0F00', '#C5F1F8', '#4376B5']
    )

    ax1.set_xlabel('')
    ax1.set_ylabel('R² score', fontsize=23)
    current_ymin, _ = ax1.get_ylim()
    ax1.set_ylim(current_ymin, 1)
    ax1.tick_params(axis='y', labelsize=23)
    ax1.set_xticklabels([label.replace("_", "\n", 1) for label in task_order],
                        fontsize=23,
                        rotation=45)

    divider_position = 3 - 0.5
    ax1.axvline(x=divider_position, color='gray', linestyle='--', linewidth=3)

    # 折线图
    x = list(range(len(task_order)))
    mae_y = [mae_perf_df.loc[mae_perf_df['Task'] == t, 'Performance (%)'].values[0] for t in task_order]
    r2_y = [r2_perf_df.loc[r2_perf_df['task_name'] == t, 'R2_Perf (%)'].values[0] for t in task_order]

    bar_width = 0.45
    x_indices = np.arange(len(task_order))

    ax2.bar(x_indices - bar_width / 2, mae_y, width=bar_width,
            label='MAE %', color='#9C88F7', edgecolor='k')

    ax2.bar(x_indices + bar_width / 2, r2_y, width=bar_width,
            label='R² %', color='#5CD3E7', edgecolor='k')

    ax2.set_xticks(x)
    ax2.set_xticklabels([label.replace("_", "\n", 1) for label in task_order],
                        fontsize=23,
                        rotation=45)
    for i, (mae, r2) in enumerate(zip(mae_y, r2_y)):
        ax2.text(i - bar_width / 2, mae + 2, f"{mae:.1f}", ha='center', fontsize=15)
        ax2.text(i + bar_width / 2, r2 + 2, f"{r2:.1f}", ha='center', fontsize=15)
    ax2.set_ylabel('Position between dummy\nand benchmark leader (%)', fontsize=23)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelsize=23)
    ax2.legend(loc='lower right', fontsize=23)

    ax2.axvline(x=divider_position, color='gray', linestyle='--', linewidth=3)

    plt.tight_layout(w_pad=2)
    plt.savefig("RF_composition_R2MAE_split_7.21.pdf", bbox_inches="tight")


if __name__ == '__main__':
    df = pd.read_csv('rf_composition_vector_results.csv')
    df = df.dropna(subset=['r2_score'])
    df['samples'] = df['task_name'].map(task_to_samples)
    r2_ref_df = pd.read_csv('task_r2_best_dummy.csv')

    # MAE per task
    mae_df = df.groupby('task_name')['mae'].mean().reset_index()
    mae_df.rename(columns={'task_name': 'Task', 'mae': 'MAE'}, inplace=True)
    mae_df['Performance (%)'] = mae_df.apply(compute_relative_mae_performance, axis=1)

    # R² per task
    r2_df = df.groupby('task_name')['r2_score'].mean().reset_index()

    r2_df['R2_Perf (%)'] = r2_df.apply(
        lambda row: compute_relative_r2_performance(row['task_name'], row['r2_score'], r2_ref_df), axis=1
    )

    # print("\n=== R² Performance Check ===")
    # for idx, row in r2_df.iterrows():
    #     task = row['task_name']
    #     current_r2 = row['r2_score']
    #     perf = row['R2_Perf (%)']
    #
    #     ref_row = r2_ref_df[r2_ref_df['Task'] == task]
    #     if not ref_row.empty:
    #         r2_dummy = ref_row['R2_Dummy'].values[0]
    #         r2_best = ref_row['R2_Best'].values[0]
    #         manual_perf = (current_r2 - r2_dummy) / (r2_best - r2_dummy) * 100 if r2_best != r2_dummy else None
    #
    #         print(f"  Task: {task}")
    #         print(f"  - R2 (Dummy):   {r2_dummy:.4f}")
    #         print(f"  - R2 (Best):    {r2_best:.4f}")
    #         print(f"  - R2 (Current): {current_r2:.4f}")


    df['task_name'] = pd.Categorical(df['task_name'], categories=custom_task_order, ordered=True)
    df = df.sort_values('task_name')

    mae_df['Task'] = pd.Categorical(mae_df['Task'], categories=custom_task_order, ordered=True)
    mae_df = mae_df.sort_values('Task')

    r2_df['task_name'] = pd.Categorical(r2_df['task_name'], categories=custom_task_order, ordered=True)
    r2_df = r2_df.sort_values('task_name')

    draw_plt(df, mae_df, r2_df)
