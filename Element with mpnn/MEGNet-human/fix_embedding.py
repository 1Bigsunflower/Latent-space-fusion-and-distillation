from __future__ import annotations
import sys
import os
import shutil
import warnings
import zipfile
from random import seed
from typing import List
import json
import numpy as np
from matgl.config import DEFAULT_ELEMENTS
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
from pytorch_lightning import seed_everything
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
from megnet_model import MEGNet
from matgl.utils.io import RemoteFile
from training import ModelLightningModule
from pymatgen.core import Element
import argparse
from torch.optim.lr_scheduler import LambdaLR
from matbench.bench import MatbenchBenchmark
from dgl.dataloading import GraphDataLoader
import csv

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='MEGNet')
parser.add_argument('--dim_node_embed', type=int, default=64, help='number of node embedding dim')

args = parser.parse_args()

# 100元素
element_100 = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb',
    'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl',
    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
)
element_103 = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb',
    'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl',
    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md',
    'No', 'Lr')


# Dataset Preparation
def get_data(input, output):
    structures = []
    for structure_str in input:
        structures.append(structure_str)
    return structures, output.tolist()


def load_pretrain_embeddings():
    """
        读取JSON文件并将其内容转换为二维数组，同时计算值的长度。
        :return: 二维数组, 值的长度
        """
    filename = 'atom_init.json'
    with open(filename, 'r') as file:
        data = json.load(file)

    first_key = next(iter(data))
    length = len(data[first_key])

    coordinates = [[0.0] * length]

    for key, value in data.items():
        coordinates.append(value)

    return coordinates, length


if __name__ == '__main__':
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    init_seed = 42
    seed_everything(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(init_seed)

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            "matbench_jdft2d",  # 636
            "matbench_phonons",  # 1,265
            "matbench_dielectric",  # 4,764
            "matbench_log_gvrh",  # 10,987
            "matbench_log_kvrh",  # 10,987
            "matbench_perovskites",  # 1w8
            "matbench_mp_gap",  # 106,113
            "matbench_mp_e_form"  # 132,752
        ]
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            structures, eform_per_atom = get_data(train_inputs, train_outputs)

            elem_list = DEFAULT_ELEMENTS

            # 结构转化为图
            converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

            mp_dataset = MEGNetDataset(
                structures=structures,  # 结构
                labels={"Eform": eform_per_atom},  # 标签
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

            train_loader, val_loader = MGLDataLoader(
                train_data=train_data,
                val_data=val_data,
                collate_fn=collate_fn,
                batch_size=128,
                num_workers=0,
                pin_memory=False,
                generator=torch.Generator().manual_seed(42),
                # prefetch_factor=4,
                # persistent_workers=True
            )

            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_sturcture, test_Eform = get_data(test_inputs, test_outputs)

            test_dataset = MEGNetDataset(
                structures=test_sturcture,  # 结构
                labels={"Eform": test_Eform},  # 标签
                converter=converter,  # 图
                initial=0.0,  # 高斯扩展的初始距离
                final=5.0,  # 高斯扩展的最终距离
                num_centers=100,  # 高斯函数的数量
                width=0.5,  # 高斯函数的宽度
            )

            kwargs = {
                "batch_size": 128,
                "num_workers": 0,
                "pin_memory": False
            }

            test_loader = GraphDataLoader(test_dataset, collate_fn=collate_fn, **kwargs)

            # define the bond expansion
            bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=25, width=0.4)
            # a Set2Set encoder for node and edge embeddings, feed-forward blocks of units [64, 32], softplus activation and gauss distance expansion with cutoff of 5A and 25 bins with 0.4 sigma. We used a larger input embedding vector [64] of atom species and added the charge as input graph attributes. We trained with MAE loss and a linear learning rate scheduler from 5e-4 to 5e-6 over 1000 epochs using Adam. We added a standard scaler for regression.

            cgcnn_init_embedding, atom_init_dim = load_pretrain_embeddings()

            model_dir = f"../../pre-train_model/megnet/megnet_hm/{task.dataset_name}_fold{fold}_dim{args.dim_node_embed}/"
            ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]

            # 确定 map_location
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            try:
                element_types_used = element_103
                # setup the architecture of MEGNet model
                model = MEGNet(
                    dim_node_embedding=args.dim_node_embed,  # 元素嵌入维度
                    dim_edge_embedding=100,
                    dim_state_embedding=2,
                    nblocks=3,
                    hidden_layer_sizes_input=(64, 32),
                    hidden_layer_sizes_conv=(64, 32, 32),
                    nlayers_set2set=1,
                    niters_set2set=3,
                    hidden_layer_sizes_output=(32, 16),
                    is_classification=False,
                    activation_type="softplus2",
                    # element_types=element_100,  # 用于计算embedding矩阵维度
                    element_types=element_types_used,
                    bond_expansion=bond_expansion,
                    cutoff=4.0,
                    gauss_width=0.5,
                    orig_elem_list=DEFAULT_ELEMENTS,  # 用于还原为原子序数
                    pretrain_embeddings=cgcnn_init_embedding,  # pretrain的二维嵌入矩阵
                    atom_init_dim=atom_init_dim  # cgcnn atom init原始维度
                )

                # 加载模型
                checkpoint_path = os.path.join(model_dir, ckpt_files[0])
                lit_module = ModelLightningModule.load_from_checkpoint(checkpoint_path, model=model, loss="mae_loss",
                                                                       map_location=map_location)
            except Exception as e:
                element_types_used = element_100
                # setup the architecture of MEGNet model
                model = MEGNet(
                    dim_node_embedding=args.dim_node_embed,  # 元素嵌入维度
                    dim_edge_embedding=100,
                    dim_state_embedding=2,
                    nblocks=3,
                    hidden_layer_sizes_input=(64, 32),
                    hidden_layer_sizes_conv=(64, 32, 32),
                    nlayers_set2set=1,
                    niters_set2set=3,
                    hidden_layer_sizes_output=(32, 16),
                    is_classification=False,
                    activation_type="softplus2",
                    # element_types=element_100,  # 用于计算embedding矩阵维度
                    element_types=element_types_used,  # mp_e_form 和 gap貌似是103
                    bond_expansion=bond_expansion,
                    cutoff=4.0,
                    gauss_width=0.5,
                    orig_elem_list=DEFAULT_ELEMENTS,  # 用于还原为原子序数
                    pretrain_embeddings=cgcnn_init_embedding,  # pretrain的二维嵌入矩阵
                    atom_init_dim=atom_init_dim  # cgcnn atom init原始维度
                )

                # 加载模型
                checkpoint_path = os.path.join(model_dir, ckpt_files[0])
                lit_module = ModelLightningModule.load_from_checkpoint(checkpoint_path, model=model, loss="mae_loss",
                                                                       map_location=map_location)

            embedding_save_path = f"./{task.dataset_name}_hm"
            os.makedirs(embedding_save_path, exist_ok=True)

            lit_module.extract_atom_embeddings(
                dataloader=train_loader,
                save_path=embedding_save_path,
                split_name=f"fold{fold}_dim{args.dim_node_embed}_train"
            )

            lit_module.extract_atom_embeddings(
                dataloader=val_loader,
                save_path=embedding_save_path,
                split_name=f"fold{fold}_dim{args.dim_node_embed}_val"
            )

            lit_module.extract_atom_embeddings(
                dataloader=test_loader,
                save_path=embedding_save_path,
                split_name=f"fold{fold}_dim{args.dim_node_embed}_test"
            )

            # This code just performs cleanup for this notebook.
            for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass