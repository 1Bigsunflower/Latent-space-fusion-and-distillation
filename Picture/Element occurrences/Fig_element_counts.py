import sys

import os
import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element

# ---------- 元素（Z=1~103） ----------
ALL_ELEMENTS = [Element.from_Z(i).symbol for i in range(1, 104)]


def count_elements_in_task(task_name, fold=0):
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = list(mb.tasks)[0]
    task.load()

    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    all_structs = list(train_inputs) + list(test_inputs)

    element_counter = {el: 0 for el in ALL_ELEMENTS}

    for struct in all_structs:
        # 样本包含元素集合
        elems = set([el.symbol for el in struct.composition.elements])
        for el in elems:
            element_counter[el] += 1
    return pd.Series(element_counter, name=task_name)


if __name__ == '__main__':
    task_names = [
        # 'matbench_jdft2d',
        'matbench_phonons',
        # 'matbench_dielectric',
        'matbench_log_gvrh',
        'matbench_log_kvrh',
        # 'matbench_perovskites',
        'matbench_mp_gap',
        'matbench_mp_e_form'
    ]

    all_stats = []
    for task in task_names:
        counts = count_elements_in_task(task)
        all_stats.append(counts)

    df = pd.concat(all_stats, axis=1).fillna(0).astype(int)

    df.to_csv("element_counts_task.csv", index_label="Element")

    # 整体
    overall = df.sum(axis=1).to_frame(name="total_counts")
    overall.to_csv("element_counts.csv", index_label="Element")
