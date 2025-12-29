import os
import glob
import torch
import pytorch_lightning as pl
import argparse
from tqdm import tqdm
import sys

# 添加必要的导入（根据你的训练脚本）
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MEGNetDataset, collate_fn
from matgl.utils.training import ModelLightningModule
from matgl.layers import BondExpansion
from megnet_model import MEGNet
from matgl.config import DEFAULT_ELEMENTS
from dgl.dataloading import GraphDataLoader
from matbench.bench import MatbenchBenchmark
from megnet_train import get_data


def test_checkpoint(checkpoint_path, task_name, fold, dim_node_embed, a, b, device='cuda'):
    """
    测试单个checkpoint
    """

    # 设置设备
    if device == 'cuda' and torch.cuda.is_available():
        accelerator = 'gpu'
        devices = [0]  # 使用第一个GPU
    else:
        accelerator = 'cpu'
        devices = 1

    # 加载Matbench任务
    mb = MatbenchBenchmark(autoload=False, subset=[task_name])

    for task in mb.tasks:
        task.load()
        # 获取测试数据
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        test_structures, test_labels = get_data(test_inputs, test_outputs)

        elem_list = DEFAULT_ELEMENTS
        converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

        # 创建测试数据集
        test_dataset = MEGNetDataset(
            structures=test_structures,
            labels={"Eform": test_labels},
            converter=converter,
            initial=0.0,
            final=5.0,
            num_centers=100,
            width=0.5,
        )

        # 数据加载器
        kwargs = {
            "batch_size": 128,
            "num_workers": 1,
            "pin_memory": torch.cuda.is_available(),
            "shuffle": False
        }
        test_loader = GraphDataLoader(test_dataset, collate_fn=collate_fn, **kwargs)

        # 重新构建模型（需要与训练时完全相同的配置）
        bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0,
                                       num_centers=25, width=0.4)

        # 元素列表（与训练时一致）
        element_100 = (
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
            'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
            'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
            'Es', 'Fm'
        )

        # 加载预训练嵌入（如果需要）
        from megnet_train import load_pretrain_embeddings
        cgcnn_init_embedding, atom_init_dim = load_pretrain_embeddings()

        # 构建模型
        model = MEGNet(
            dim_node_embedding=dim_node_embed,
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
            element_types=element_100,
            bond_expansion=bond_expansion,
            cutoff=4.0,
            gauss_width=0.5,
            orig_elem_list=elem_list,
            pretrain_embeddings=cgcnn_init_embedding,
            atom_init_dim=atom_init_dim,
            a=a,
            b=b,
        )

        # 创建Lightning模块并加载checkpoint
        lit_module = ModelLightningModule(model=model, loss="mae_loss")

        # 创建Trainer进行测试
        from pytorch_lightning import Trainer

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=False,
            logger=False,  # 不记录日志
            enable_checkpointing=False,  # 不保存checkpoint
        )

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 如果checkpoint包含'state_dict'，需要调整键名
        # if 'state_dict' in checkpoint:
        # 调整键名以匹配ModelLightningModule
        state_dict = checkpoint['state_dict']
        # 移除可能的'model.'前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        lit_module.model.load_state_dict(new_state_dict, strict=False)
        # else:
        #     # 直接加载
        #     lit_module.load_state_dict(checkpoint)


        # 测试模型
        lit_module.eval()
        lit_module.to(device)

        trainer.test(
            model=lit_module,
            dataloaders=test_loader,
        )


if __name__ == '__main__':
    a = 1.0
    b = 1.0
    dim_node_embed = 16
    fold = 0
    task_name = "matbench_perovskites"
    checkpoint_path = 'matbench_perovskites_e1f1/matbench_perovskites_fold0_dim16/epochepoch=0410-train_losstrain_MAE=0.0160-val_lossval_MAE=0.0432.ckpt'
    test_checkpoint(checkpoint_path, task_name, fold, dim_node_embed, a, b, device='cuda')
