import argparse
import gc
import json
import os
import sys

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from random import sample, seed
from .model import CrystalGraphConvNet
from .data import StruData, collate_pool_matbench
from pytorch_lightning import seed_everything
from matgl.ext.pymatgen import get_element_list
from pymatgen.core.periodic_table import Element

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
# emb dim
parser.add_argument('--atom_fea_len', default=8, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--task_name',
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    default='matbench_jdft2d',
                    help='matbench task name')
parser.add_argument('--fold', default=0, type=int, metavar='N',
                    help='fold number')
parser.add_argument('--a', default=0.01, type=float,
                    help='weight for element embedding in final_vec = a*embedding + b*binary')
parser.add_argument('--b', default=1.0, type=float,
                    help='weight for binary features in final_vec = a*embedding + b*binary')
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class Cgcnn_lightning(pl.LightningModule):

    def __init__(self, crystalGraphConvNet, normalizer, a=1.0, b=1.0):
        super().__init__()
        self.crystalGraphConvNet = crystalGraphConvNet
        self.normalizer = normalizer
        self.save_hyperparameters(ignore=['crystalGraphConvNet', 'normalizer'])

    def training_step(self, batch, batch_idx):
        x, y = batch

        atom_fea, nbr_fea, nbr_fea_idx, elem_ids, crystal_atom_idx = x

        a = self.hparams.a
        b = self.hparams.b

        y_hat = self.crystalGraphConvNet(atom_fea, nbr_fea, nbr_fea_idx, elem_ids, crystal_atom_idx, a, b)

        target_normed = self.normalizer.norm(y)
        loss = nn.MSELoss()(y_hat, target_normed)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=128)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        atom_fea, nbr_fea, nbr_fea_idx, elem_ids, crystal_atom_idx = x

        target_normed = self.normalizer.norm(y)
        a, b = self.hparams.a, self.hparams.b
        y_hat = self.crystalGraphConvNet(atom_fea, nbr_fea, nbr_fea_idx, elem_ids, crystal_atom_idx, a, b)

        loss_fn = nn.L1Loss()  # mae
        val_loss = loss_fn(y_hat, target_normed)

        self.log('val_MAE', val_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        atom_fea, nbr_fea, nbr_fea_idx, elem_ids, crystal_atom_idx = x
        a, b = self.hparams.a, self.hparams.b
        y_hat = self.crystalGraphConvNet(atom_fea, nbr_fea, nbr_fea_idx, elem_ids, crystal_atom_idx, a, b)

        # loss
        loss_fn = nn.L1Loss()
        test_loss = loss_fn(self.normalizer.denorm(y_hat), y)
        self.log('test_MAE', test_loss, on_epoch=True, prog_bar=True, batch_size=128)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }


def get_data(input, output):
    structures = []
    for structure_str in input:
        structures.append(structure_str)
    return structures, output.tolist()


def extract_element_atom_fea(
        model,
        elem_list,
        a=1.0,
        b=1.0,
        device="cpu"
):
    """
    提取 elem_list 中每个元素的 atom_fea = a * elem_vec + b * binary_vec

    Parameters
    ----------
    model : CrystalGraphConvNet (已 load checkpoint)
    elem_list : list[int]
        元素原子序数列表，例如 [1, 6, 8, 14]
        atom_init.json 路径
    a, b : float
        线性组合系数
    device : str

    Returns
    -------
    elem_fea_dict : dict
        {Z: tensor(atom_fea_len)}
    """

    model.eval()
    model.to(device)

    ATOM_INIT_PATH = BASE_DIR / "atom_init.json"

    with open(ATOM_INIT_PATH, "r") as f:
        atom_init = json.load(f)

    elem_fea_dict = {}

    with torch.no_grad():
        for elem in elem_list:
            Z = Element(elem).number
            Z_str = str(Z)
            if Z_str not in atom_init:
                raise KeyError(f"Element Z={Z} not found in atom_init.json")

            atom_fea = torch.tensor(
                atom_init[Z_str],
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)  # (1, orig_atom_fea_len)
            # print(atom_fea.squeeze(0).int().tolist())

            elem_ids = torch.tensor([Z], dtype=torch.long, device=device)

            binary_vec = model.binary_encoder(atom_fea)
            elem_vec = model.elem_emb(elem_ids)
            atom_fea_out = a * elem_vec + b * binary_vec

            elem_fea_dict[Z] = atom_fea_out.squeeze(0).cpu().numpy()
    return elem_fea_dict


def get_emb(args):
    init_seed = 42
    seed_everything(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(init_seed)  # Random特有

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            args.task_name,
        ]
    )
    for task in mb.tasks:
        task.load()

        train_inputs, train_outputs = task.get_train_and_val_data(args.fold)  # 获取训练集
        dataset = StruData(train_inputs, train_outputs, precompute_workers=2)
        # collate_fn = collate_pool_matbench
        structures, train_y = get_data(train_inputs, train_outputs)
        elem_list = get_element_list(structures)

        if len(dataset) < 500:
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target = collate_pool_matbench(sample_data_list)
        normalizer = Normalizer(sample_target)

        # build model
        structures, _, = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]

        # map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crystalGraphConvNet = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                                  atom_fea_len=args.atom_fea_len,
                                                  n_conv=3,
                                                  h_fea_len=128,
                                                  n_h=1,
                                                  classification=False)

        # 加载模型
        model_dir = (
                BASE_DIR.parent / "cgcnn_ef_model" /
                f"Cgcnn_{args.task_name}_fold{args.fold}_dim{args.atom_fea_len}_e{args.a}_f{args.b}"
        )

        ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
        checkpoint_path = os.path.join(model_dir, ckpt_files[0])

        model = Cgcnn_lightning.load_from_checkpoint(
            checkpoint_path,
            crystalGraphConvNet=crystalGraphConvNet,
            normalizer=normalizer,
            a=args.a,
            b=args.b,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        model.eval()
        elem_atom_fea = extract_element_atom_fea(
            model.crystalGraphConvNet,
            elem_list=elem_list,
            a=args.a,
            b=args.b,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # print(elem_atom_fea)  # e.g. torch.Size([16])

        rows = []

        for Z, fea in elem_atom_fea.items():
            elem_symbol = Element.from_Z(Z).symbol

            if torch.is_tensor(fea):
                fea = fea.cpu().numpy()

            rows.append([elem_symbol] + fea.tolist())

        dim = len(rows[0]) - 1
        columns = ['Element'] + [f'Dim_{i}' for i in range(dim)]

        embeddings_df = pd.DataFrame(rows, columns=columns)

        # {args.task_name}_fold{args.fold}_dim{args.atom_fea_len}_e{args.a}_f{args.b}
        save_dir = BASE_DIR.parent / "cgcnn_ef_embedding"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = (
                save_dir /
                f"{args.task_name}_fold{args.fold}_dim{args.atom_fea_len}_e{args.a}_f{args.b}.csv"
        )

        embeddings_df.to_csv(save_path, index=False)

        del model
        del crystalGraphConvNet
        del normalizer
        del dataset
        del sample_data_list
        del elem_atom_fea
        del embeddings_df
        del rows

        torch.cuda.empty_cache()
        gc.collect()

        return None


if __name__ == '__main__':
    args = parser.parse_args()
    get_emb(args)
