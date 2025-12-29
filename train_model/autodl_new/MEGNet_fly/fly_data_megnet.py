from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd
import torch
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MEGNetDataset
from pymatgen.core import Element
from pymatgen.core import Structure
from tqdm import tqdm
from pytorch_lightning import seed_everything
import numpy as np
import pytorch_lightning as pl
import torch
from dgl.data.utils import split_dataset
from dgl.dataloading import GraphDataLoader
from matbench.bench import MatbenchBenchmark
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
from matgl.models._megnet import MEGNet
from matgl.utils.training import ModelLightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
parser = argparse.ArgumentParser(description='gen and dump')
parser.add_argument('--fold', type=int, default=0, help='')
parser.add_argument('--subset', default='matbench_mp_e_form', type=str,
                    choices=['matbench_mp_gap', 'matbench_mp_e_form'],
                    help='subset dataset to use')
args = parser.parse_args()

# Dataset Preparation
def get_data(input, output):
    structures = []
    for structure_str in input:
        structures.append(structure_str)
    return structures, output.tolist()


def gen_and_dump(name, dataset, folder):
    io_buffer = []  # 用于存储生成的数据字典的缓冲区
    io_cnt = 0  # 计数生成的文件数量
    for i in range(len(dataset)):
        io_buffer.append(dataset[i])
        if len(io_buffer) > 100 or i == len(dataset) - 1:
            for data in io_buffer:
                save_path = "{}/{}/{}.pth".format(folder, name, io_cnt)
                # Check if the directory exists, and create it if not
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                # Save the data to the specified path
                torch.save(data, save_path)
                io_cnt += 1
            io_buffer.clear()
    print("文件数量：", io_cnt)

def main():
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_seed = 42
    seed_everything(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            args.subset
        ]
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    for task in mb.tasks:
        task.load()
        for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
        folder = f'../MEGNet_dataset/{args.subset}/{args.fold}'
        if not os.path.exists(folder):
            os.makedirs(folder)  # 如果文件夹不存在，则创建它
            train_inputs, train_outputs = task.get_train_and_val_data(args.fold)
            structures, train_y = get_data(train_inputs, train_outputs)

            elem_list = DEFAULT_ELEMENTS
            # 结构转化为图
            converter = Structure2Graph(element_types=elem_list, cutoff=5.0)
            # 把原数据集转化为megnet数据集
            mp_dataset = MEGNetDataset(
                structures=structures,  # 结构
                labels={"Eform": train_y},  # 标签
                converter=converter,  # 图
                initial=0.0,  # 高斯扩展的初始距离
                final=5.0,  # 高斯扩展的最终距离
                num_centers=100,  # 高斯函数的数量
                width=0.5,  # 高斯函数的宽度
            )

            # 拆分数据集为训练、验证、测试集
            train_data, val_data, _ = split_dataset(
                mp_dataset,
                frac_list=[0.75, 0.25, 0.0],  # 比例
                shuffle=True,
                random_state=42,
            )

            for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass

            test_inputs, test_outputs = task.get_test_data(args.fold, include_target=True)
            test_sturcture, test_Eform = get_data(test_inputs, test_outputs)
            # 把原数据集转化为megnet数据集
            test_data = MEGNetDataset(
                structures=test_sturcture,  # 结构
                labels={"Eform": test_Eform},  # 标签
                converter=converter,  # 图
                initial=0.0,  # 高斯扩展的初始距离
                final=5.0,  # 高斯扩展的最终距离
                num_centers=100,  # 高斯函数的数量
                width=0.5,  # 高斯函数的宽度
            )

            gen_and_dump(name='train', dataset=train_data, folder=folder)
            gen_and_dump(name='val', dataset=val_data, folder=folder)
            gen_and_dump(name='test', dataset=test_data, folder=folder)

            for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass


if __name__ == '__main__':
    main()
