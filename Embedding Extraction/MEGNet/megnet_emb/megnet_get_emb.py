from __future__ import annotations

import argparse
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
from matgl.models._megnet import MEGNet
from matgl.utils.training import ModelLightningModule
from pytorch_lightning import seed_everything

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='MEGNet')
parser.add_argument('--dim_node_embed', type=int, default=64, help='number of node embedding dim')
parser.add_argument('--task_name',
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    default='matbench_jdft2d',
                    help='matbench task name')
parser.add_argument('--fold', default=0, type=int, metavar='N',
                    help='fold number')
args = parser.parse_args()


def get_data(input, output):
    structures = []
    for structure_str in input:
        structures.append(structure_str)
    return structures, output.tolist()


def get_element_embedding_megnet(model, elem_list):
    embedding_layer = model.model.embedding.layer_node_embedding
    element_embeddings = {}

    for elem in elem_list:
        if elem in DEFAULT_ELEMENTS:
            elem_index = DEFAULT_ELEMENTS.index(elem)
            element_embeddings[elem] = embedding_layer.weight.data[elem_index].cpu().numpy()

    return element_embeddings


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

            # 加载模型
            model_dir = f"../../../pre-train_model/megnet/megnet_emb/{args.task_name}_fold{args.fold}_dim{args.dim_node_embed}/"
            ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
            checkpoint_path = os.path.join(model_dir, ckpt_files[0])

            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            lit_module = ModelLightningModule.load_from_checkpoint(checkpoint_path, model=model, loss="mae_loss",
                                                                   map_location=map_location)

            # 元素嵌入
            element_embeddings = get_element_embedding_megnet(lit_module, elem_list)

            embeddings_df = pd.DataFrame.from_dict(element_embeddings, orient='index')
            embeddings_df.reset_index(inplace=True)
            embeddings_df.columns = ['Element'] + [f'Dim_{i}' for i in range(embeddings_df.shape[1] - 1)]

            # 构造保存路径
            save_path = f"../../../Embedding_weight_orig/MEGNet/megnet_emb/{args.task_name}_fold{args.fold}_dim{args.dim_node_embed}.csv"
            embeddings_df.to_csv(save_path, index=False)
