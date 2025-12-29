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
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Element
import argparse
from torch.optim.lr_scheduler import LambdaLR
from matbench.bench import MatbenchBenchmark
from dgl.dataloading import GraphDataLoader
import csv
import glob
from torch.utils.data import Dataset

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='MEGNet')
parser.add_argument('--dim_node_embed', type=int, default=64, help='number of node embedding dim')
parser.add_argument('--fold', type=int, default=0, help='number of fold')
parser.add_argument('--a', default=1.0, type=float,
                    help='weight for element embedding in final_vec = a*embedding + b*binary')
parser.add_argument('--b', default=1.0, type=float,
                    help='weight for binary features in final_vec = a*embedding + b*binary')
parser.add_argument('--subset', default='matbench_dielectric', type=str,
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    help='subset dataset to use')
parser.add_argument('--data_root', default='../../MEGNet_dataset', type=str)
parser.add_argument('--cuda_devices', type=str, default='0',
                    help='CUDA_VISIBLE_DEVICES environment variable')
args = parser.parse_args()

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


class PthFolderDataset(Dataset):
    def __init__(self, folder: str):
        self.folder = folder
        self.files = glob.glob(os.path.join(folder, "*.pth"))
        # 按 0.pth,1.pth... 排序（你的 fly_data_megnet.py 就是 io_cnt 命名）:contentReference[oaicite:6]{index=6}
        self.files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

        if len(self.files) == 0:
            raise RuntimeError(f"No .pth files found in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx], map_location="cpu")


if __name__ == '__main__':
    # torch.set_printoptions(threshold=int(1e6))
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
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

        data_dir = os.path.join(args.data_root, task.dataset_name, str(args.fold))
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        train_data = PthFolderDataset(train_dir)
        val_data = PthFolderDataset(val_dir)

        train_loader, val_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            collate_fn=collate_fn,
            batch_size=128,
            num_workers=1,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(42),
            # prefetch_factor=2,
            # persistent_workers=True
        )

        # 数据集中所有的元素类型
        elem_list = DEFAULT_ELEMENTS
        # [30, 30, 61, 61,  4,  4, 62, 62, 62, 47, 48, 48, 48, 48, 48, 48, 48, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 66, 66, 10, 10, 10, 10, 10, 10,  0, 0]
        # 数字根据原子序数大小
        # 结构转化为图

        # define the bond expansion
        bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=25, width=0.4)
        # a Set2Set encoder for node and edge embeddings, feed-forward blocks of units [64, 32], softplus activation and gauss distance expansion with cutoff of 5A and 25 bins with 0.4 sigma. We used a larger input embedding vector [64] of atom species and added the charge as input graph attributes. We trained with MAE loss and a linear learning rate scheduler from 5e-4 to 5e-6 over 1000 epochs using Adam. We added a standard scaler for regression.

        cgcnn_init_embedding, atom_init_dim = load_pretrain_embeddings()

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
            element_types=element_100,  # 用于计算embedding矩阵维度
            bond_expansion=bond_expansion,
            cutoff=4.0,
            gauss_width=0.5,
            orig_elem_list=elem_list,  # 用于还原为原子序数
            pretrain_embeddings=cgcnn_init_embedding,  # pretrain的二维嵌入矩阵
            atom_init_dim=atom_init_dim,  # cgcnn atom init原始维度
            a=args.a,
            b=args.b,
        )

        # setup the MEGNetTrainer
        lit_module = ModelLightningModule(model=model, loss="mae_loss")
        early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=500, verbose=True,
                                            mode="min")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_MAE',
            save_top_k=1,
            mode='min',
            dirpath=f'{task.dataset_name}_fold{args.fold}_dim{args.dim_node_embed}_e{args.a}_f{args.b}',
            # Directory to save the checkpoints
            filename='{epoch:04d}-train_loss{train_MAE:.4f}-val_loss{val_MAE:.4f}',
        )
        # Training
        trainer = pl.Trainer(max_epochs=10000, enable_progress_bar=True,
                             callbacks=[early_stop_callback, checkpoint_callback],
                             strategy='ddp_find_unused_parameters_true',
                             log_every_n_steps=1000)
        trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
        # 加载验证损失最小的模型权重
        best_model_path = checkpoint_callback.best_model_path
        lit_module = ModelLightningModule.load_from_checkpoint(best_model_path, model=model, loss="mae_loss")

        # 测试部分
        lit_module.eval()
        test_dir = os.path.join(data_dir, "test")
        test_dataset = PthFolderDataset(test_dir)
        kwargs = {
            "batch_size": 128,
            "num_workers": 1,
            "pin_memory": torch.cuda.is_available()
        }

        test_loader = GraphDataLoader(test_dataset, collate_fn=collate_fn, **kwargs)
        predict = trainer.test(model=lit_module, dataloaders=test_loader)
        # This code just performs cleanup for this notebook.
        for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
