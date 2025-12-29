from __future__ import annotations

import argparse
import json
import os
import warnings
from random import seed

import numpy as np
import pandas as pd
import torch
from matbench.bench import MatbenchBenchmark
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import get_element_list
from matgl.layers import BondExpansion
from matgl.utils.training import ModelLightningModule
from pymatgen.core.periodic_table import Element
from pytorch_lightning import seed_everything

from megnet_model import MEGNet
from sklearn.decomposition import PCA

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='MEGNet')
parser.add_argument('--dim_node_embed', type=int, default=16, help='number of node embedding dim')
parser.add_argument('--task_name',
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    default='matbench_perovskites',
                    help='matbench task name')
parser.add_argument('--fold', default=4, type=int, metavar='N',
                    help='fold number')
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


def get_element_embedding_megnet(model, elem_list):
    embedding_layer = model.model.embedding.fix_embed
    linear_layer = model.model.linear
    device = embedding_layer.weight.device
    element_embeddings = {}
    for elem in elem_list:
        elem_z = Element(elem).number
        elem_z_tensor = torch.tensor([elem_z], device=device)
        with torch.no_grad():
            fix_emb = embedding_layer(elem_z_tensor)
            final_emb = linear_layer(fix_emb)
        element_embeddings[elem] = final_emb.cpu().numpy().squeeze(0)
    return element_embeddings

def analyze_isotropy_pca(embeddings: np.ndarray):
    """
    embeddings: [N, D]
    """
    X = embeddings - embeddings.mean(axis=0, keepdims=True)

    pca = PCA()
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\n========== PCA 各向同性分析 ==========")
    for k in [1, 2, 4, 8, 16]:
        if k <= len(cumulative):
            print(f"前 {k} 个主成分解释方差: {cumulative[k-1]:.4f}")

    print(f"最大单一主成分占比: {explained[0]:.4f}")
    print(f"前 10 个主成分占比: {cumulative[min(9, len(cumulative)-1)]:.4f}")

    return explained

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_seed = 42
    seed_everything(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(init_seed)

    # define the bond expansion
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=25, width=0.4)
    # a Set2Set encoder for node and edge embeddings, feed-forward blocks of units [64, 32], softplus activation and gauss distance expansion with cutoff of 5A and 25 bins with 0.4 sigma. We used a larger input embedding vector [64] of atom species and added the charge as input graph attributes. We trained with MAE loss and a linear learning rate scheduler from 5e-4 to 5e-6 over 1000 epochs using Adam. We added a standard scaler for regression.
    cgcnn_init_embedding, atom_init_dim = load_pretrain_embeddings()

    model_dir = f"{args.task_name}_fold{args.fold}_dim{args.dim_node_embed}/"
    ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]

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
            if fold != args.fold:
                continue
            train_inputs, train_outputs = task.get_train_and_val_data(fold)  # 获取训练集
            structures, train_y = get_data(train_inputs, train_outputs)

            elem_list = get_element_list(structures)

            # 元素嵌入
            element_embeddings = get_element_embedding_megnet(lit_module, elem_list)

            embeddings_df = pd.DataFrame.from_dict(element_embeddings, orient='index')
            embeddings_df.reset_index(inplace=True)
            embeddings_df.columns = ['Element'] + [f'Dim_{i}' for i in range(embeddings_df.shape[1] - 1)]

            # 取 embedding 矩阵
            E = embeddings_df.drop(columns=['Element']).values  # [N_elem, dim]

            explained = analyze_isotropy_pca(E)
            # ========== PCA 各向同性分析 ==========
            # 前 1 个主成分解释方差: 0.4557
            # 前 2 个主成分解释方差: 0.7027
            # 前 4 个主成分解释方差: 0.8631
            # 前 8 个主成分解释方差: 0.9796
            # 前 16 个主成分解释方差: 1.0000
            # 最大单一主成分占比: 0.4557
            # 前 10 个主成分占比: 0.9889
