import argparse
import json
import os
import sys
from random import sample, seed

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matbench.bench import MatbenchBenchmark
from matgl.ext.pymatgen import get_element_list
from pymatgen.core.periodic_table import Element
from pytorch_lightning import seed_everything
from scipy.linalg import svd
from scipy.stats import entropy

from data import StruData, collate_pool_matbench
from model import CrystalGraphConvNet
from torch.utils.data import DataLoader

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
args = parser.parse_args(sys.argv[1:])


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

    def __init__(self, crystalGraphConvNet, normalizer):
        super().__init__()
        self.crystalGraphConvNet = crystalGraphConvNet
        self.normalizer = normalizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        input_var = (x[0], x[1], x[2], x[3])
        y_hat = self.crystalGraphConvNet(*input_var)

        target_normed = self.normalizer.norm(y)
        loss = nn.MSELoss()(y_hat, target_normed)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=128)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])

        target_normed = self.normalizer.norm(y)
        y_hat = self.crystalGraphConvNet(*input_var)

        loss_fn = nn.L1Loss()  # mae
        val_loss = loss_fn(y_hat, target_normed)

        self.log('val_MAE', val_loss, on_epoch=True, prog_bar=True, batch_size=128)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        input_var = (x[0], x[1], x[2], x[3])

        y_hat = self.crystalGraphConvNet(*input_var)
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


def get_element_embedding_cgcnn(model, elem_list):
    embedding_layer = model.crystalGraphConvNet.embedding
    element_embeddings = {}

    device = embedding_layer.weight.device

    with open("atom_init.json", "r") as f:
        atom_init = json.load(f)

    for elem in elem_list:
        elem_z = Element(elem).number
        elem_fea = atom_init[str(elem_z)]
        with torch.no_grad():
            emb = embedding_layer(torch.tensor(elem_fea, device=device, dtype=torch.float))

        element_embeddings[elem] = emb.cpu().numpy()
    return element_embeddings


if __name__ == '__main__':
    init_seed = 42
    seed_everything(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(init_seed)  # Random特有

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
            dataset = StruData(train_inputs, train_outputs)
            collate_fn = collate_pool_matbench
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
            crystalGraphConvNet = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                                      atom_fea_len=args.atom_fea_len,
                                                      n_conv=3,
                                                      h_fea_len=128,
                                                      n_h=1,
                                                      classification=False)
            # 加载模型
            model_dir = f"../../../pre-train_model/cgcnn/cgcnn_hm/Cgcnn_{args.task_name}_fold{args.fold}_dim{args.atom_fea_len}/"
            ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
            checkpoint_path = os.path.join(model_dir, ckpt_files[0])

            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = Cgcnn_lightning.load_from_checkpoint(checkpoint_path,
                                                         crystalGraphConvNet=crystalGraphConvNet,
                                                         normalizer=normalizer,
                                                         map_location=map_location)
            # 元素嵌入
            element_embeddings = get_element_embedding_cgcnn(model, elem_list)

            embeddings_df = pd.DataFrame.from_dict(element_embeddings, orient='index')
            embeddings_df.reset_index(inplace=True)
            embeddings_df.columns = ['Element'] + [f'Dim_{i}' for i in range(embeddings_df.shape[1] - 1)]

            save_path = f"../../../Embedding_weight_orig/CGCNN/cgcnn_hm/{args.task_name}_fold{args.fold}_dim{args.atom_fea_len}.csv"
            embeddings_df.to_csv(save_path, index=False)

