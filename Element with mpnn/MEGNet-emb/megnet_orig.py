from __future__ import annotations

import argparse
import os
import sys
import warnings
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
# from matgl.models._megnet import MEGNet
from megnet_model import MEGNet
from training import ModelLightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='MEGNet')
parser.add_argument('--dim_node_embed', type=int, default=64, help='number of node embedding dim')
parser.add_argument('--task_name',
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    default='matbench_log_gvrh',
                    help='matbench task name')
args = parser.parse_args()


# Dataset Preparation
def get_data(input, output):
    structures = []
    for structure_str in input:
        structures.append(structure_str)
    return structures, output.tolist()


if __name__ == '__main__':
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
            args.task_name,
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

            structures, train_y = get_data(train_inputs, train_outputs)

            elem_list = DEFAULT_ELEMENTS

            # 结构转化为图
            converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

            mp_dataset = MEGNetDataset(
                structures=structures,  # 结构
                labels={"Eform": train_y},  # 标签
                converter=converter,  # 图
                initial=0.0,  # 高斯扩展的初始距离
                final=5.0,  # 高斯扩展的最终距离
                num_centers=100,  # 高斯函数的数量
                width=0.5,  # 高斯函数的宽度
            )

            # 拆分数据集
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
                element_types=DEFAULT_ELEMENTS,  # 更改
                bond_expansion=bond_expansion,
                cutoff=4.0,
                gauss_width=0.5,
            )

            model_dir = f"../../pre-train_model/megnet/megnet_emb/{args.task_name}_fold{fold}_dim{args.dim_node_embed}/"
            ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
            checkpoint_path = os.path.join(model_dir, ckpt_files[0])

            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            lit_module = ModelLightningModule.load_from_checkpoint(checkpoint_path, model=model, loss="mae_loss",
                                                                   map_location=map_location)

            embedding_save_path = f"./{args.task_name}_emb"
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
