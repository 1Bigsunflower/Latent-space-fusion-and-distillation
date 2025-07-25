import sys

from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure
from sklearn.metrics import mean_absolute_error
from pymatgen.core.periodic_table import Element

# ---------- 元素（Z=1~103） ----------
ALL_ELEMENTS = [Element.from_Z(i).symbol for i in range(1, 104)]

# ---------- 构造晶体嵌入 ----------
def get_element_fraction_vector(structure: Structure):
    comp = structure.composition.fractional_composition
    vec = np.zeros(103)
    for el, frac in comp.items():
        idx = ALL_ELEMENTS.index(el.symbol)
        vec[idx] = frac
    return vec


def run_rf_with_composition_vector(task_name, fold):
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = list(mb.tasks)[0]
    task.load()

    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    X_train = np.array([
        get_element_fraction_vector(s)
        for s in train_inputs
    ])
    y_train = np.array(train_outputs)

    X_test = np.array([
        get_element_fraction_vector(s)
        for s in test_inputs
    ])
    y_test = np.array(test_outputs)

    # 标准化 label
    y_scaler = StandardScaler()
    y_train_std = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_std = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_std)
    y_pred = model.predict(X_test)
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    r2 = r2_score(y_test_std, y_pred)
    mae = mean_absolute_error(y_test, y_pred_orig)
    print(f"[{task_name}] RF R² = {r2:.4f}, MAE = {mae:.4f}")

    return r2, mae


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

    rf_results = []

    for task_name in task_names:
        for fold in fold_list:
            print(f"[RF] Task={task_name}, Fold={fold}")
            r2, mae = run_rf_with_composition_vector(
                task_name=task_name,
                fold=fold,
            )
            rf_results.append({
                'task_name': task_name,
                'fold': fold,
                'r2_score': r2,
                'mae': mae
            })

    output_file = 'rf_composition_vector_results.csv'
    pd.DataFrame(rf_results).to_csv(output_file, index=False)
