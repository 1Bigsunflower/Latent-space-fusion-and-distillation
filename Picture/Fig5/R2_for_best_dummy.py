import os
import sys

from matbench.bench import MatbenchBenchmark
import json
import pandas as pd
from sklearn.metrics import r2_score


def get_testdata(task_name, fold):
    # 任务数据
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = list(mb.tasks)[0]
    task.load()
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    return test_inputs, test_outputs


def extract_task_fold_data(file_path, task_name, fold_number):
    """
    从 matbench benchmark JSON 文件中提取指定 task 和 fold 的 data 部分。

    参数:
        file_path (str): JSON 文件路径
        task_name (str): 任务名称，如 'matbench_dielectric'
        fold_number (int): fold 编号，如 0 表示 'fold_0'

    返回:
        pd.DataFrame: 包含 'id' 和 'value' 两列的 DataFrame
    """
    with open(file_path, 'r') as f:
        benchmark_data = json.load(f)

    fold_key = f'fold_{fold_number}'

    data_dict = benchmark_data['tasks'][task_name]['results'][fold_key]['data']

    df = pd.DataFrame(list(data_dict.items()), columns=['id', 'value'])
    return df


if __name__ == '__main__':
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

    fold_list = [0, 1, 2, 3, 4]

    matbench_dict = {
        "matbench_jdft2d": 'MODNet',  # MODNet
        "matbench_phonons": 'MEGNet',  # MEGNet
        "matbench_dielectric": 'MODNet',  # MODNet
        "matbench_log_gvrh": 'coNGN',  # coNGN
        "matbench_log_kvrh": 'coNGN',  # coNGN
        "matbench_perovskites": 'coGN',  # coGN
        "matbench_mp_gap": 'coGN',  # coGN
        "matbench_mp_e_form": 'coGN',  # coGN
    }

    task_r2_results = []

    for task_name in task_names:
        best_method = matbench_dict[task_name]
        best_path = os.path.join('benchmark', f'{best_method}_results.json')
        dummy_path = os.path.join('benchmark', f'dummy_results.json')

        r2_best_list = []
        r2_dummy_list = []

        for fold in fold_list:
            print(f"▶ Task={task_name}, Fold={fold}")
            _, test_outputs = get_testdata(task_name=task_name, fold=fold)

            df_best = extract_task_fold_data(best_path, task_name, fold)
            df_dummy = extract_task_fold_data(dummy_path, task_name, fold)

            # 索引对齐
            test_outputs = test_outputs.sort_index()

            # df_best
            df_best = df_best.set_index('id').sort_index()
            y_pred_best = df_best.loc[test_outputs.index, 'value']

            # df_dummy
            df_dummy = df_dummy.set_index('id').sort_index()
            y_pred_dummy = df_dummy.loc[test_outputs.index, 'value']

            # R²
            r2_best = r2_score(test_outputs, y_pred_best)
            r2_dummy = r2_score(test_outputs, y_pred_dummy)

            r2_best_list.append(r2_best)
            r2_dummy_list.append(r2_dummy)

        avg_r2_best = sum(r2_best_list) / len(r2_best_list)
        avg_r2_dummy = sum(r2_dummy_list) / len(r2_dummy_list)

        task_r2_results.append({
            'Task': task_name,
            'R2_Best': avg_r2_best,
            'R2_Dummy': avg_r2_dummy
        })

    df_r2 = pd.DataFrame(task_r2_results)
    df_r2.to_csv('task_r2_best_dummy.csv', index=False)