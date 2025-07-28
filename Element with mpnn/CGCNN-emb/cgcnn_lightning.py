import argparse
import json
import os
import sys

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from matbench.bench import MatbenchBenchmark
from torch.utils.data import DataLoader
from random import sample, seed
from model import CrystalGraphConvNet
from data import StruData, get_train_loader, collate_pool_matbench
from pytorch_lightning import seed_everything


parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('--atom_fea_len', default=8, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')

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

    def extract_atom_embeddings(self, dataloader, save_path, split_name):
        self.eval()
        device = next(self.parameters()).device  # 自动获取模型所在设备

        all_batches = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                x = [t.to(device) if isinstance(t, torch.Tensor) else t for t in x]
                y = y.to(device)

                input_var = (x[0], x[1], x[2], x[3])
                # 提取所有层的atom特征
                all_atom_features = self.crystalGraphConvNet(*input_var)

                # 构建每个原子 → 晶体的索引映射
                # x[3] 是 crystal_atom_idx: list of LongTensor
                atom_to_crystal_idx = torch.zeros(all_atom_features[0].shape[0], dtype=torch.long, device=device)
                for crys_idx, atom_ids in enumerate(x[3]):
                    atom_to_crystal_idx[atom_ids] = crys_idx  # 让每个原子知道它属于哪个晶体

                # 展开 crystal-level target 为每个原子的 target
                expanded_target = y[atom_to_crystal_idx]

                # 保存每层的特征：你可以选择只存最后一层，或者所有层
                batch_data = {
                    'atom_input': all_atom_features[0].cpu(),
                    'embedding_0': all_atom_features[1].cpu(),  # 初始嵌入
                    'embedding_1': all_atom_features[2].cpu(),  # 第一层后
                    'embedding_2': all_atom_features[3].cpu(),  # ...
                    'embedding_3': all_atom_features[4].cpu(),
                    'target': expanded_target.cpu(),  # shape: [N_atoms]
                    'atom_to_crystal': atom_to_crystal_idx.cpu()  # 可选保存，用于后续分析
                }
                all_batches.append(batch_data)

        torch.save(all_batches, os.path.join(save_path, f"{split_name}.pt"))


def main():
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

            train_inputs, train_outputs = task.get_train_and_val_data(fold)  # 获取训练集

            dataset = StruData(train_inputs, train_outputs)
            collate_fn = collate_pool_matbench

            train_loader, val_loader = get_train_loader(dataset=dataset,
                                                        collate_fn=collate_fn,
                                                        batch_size=128,
                                                        train_ratio=0.75,
                                                        val_ratio=0.25
                                                        )

            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

            dataset_test = StruData(test_inputs, test_outputs)
            test_loader = DataLoader(dataset=dataset_test,
                                     batch_size=128,
                                     collate_fn=collate_fn)

            if len(dataset) < 500:
                sample_data_list = [dataset[i] for i in range(len(dataset))]
            else:
                sample_data_list = [dataset[i] for i in
                                    sample(range(len(dataset)), 500)]
            _, sample_target = collate_pool_matbench(sample_data_list)
            normalizer = Normalizer(sample_target)

            # build model
            structures, _, = dataset[0]
            # orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]

            atom_file = 'atom.json'
            assert os.path.exists(atom_file), f'{atom_file} does not exist!'
            with open(atom_file, 'r') as file:
                data = json.load(file)
            element_num = len(data)
            crystalGraphConvNet = CrystalGraphConvNet(element_num, nbr_fea_len,
                                                      atom_fea_len=args.atom_fea_len,
                                                      n_conv=3,
                                                      h_fea_len=128,
                                                      n_h=1,
                                                      classification=False)
            # 加载模型
            model_dir = f"../../pre-train_model/cgcnn/cgcnn_emb/Cgcnn_{task.dataset_name}_emb_fold{fold}_dim{args.atom_fea_len}/"
            ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
            checkpoint_path = os.path.join(model_dir, ckpt_files[0])

            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = Cgcnn_lightning.load_from_checkpoint(checkpoint_path,
                                                         crystalGraphConvNet=crystalGraphConvNet,
                                                         normalizer=normalizer,
                                                         map_location=map_location)

            embedding_save_path = f"./{task.dataset_name}_emb"
            os.makedirs(embedding_save_path, exist_ok=True)

            model.extract_atom_embeddings(train_loader, embedding_save_path, f"fold{fold}_dim{args.atom_fea_len}_train")
            model.extract_atom_embeddings(val_loader, embedding_save_path, f"fold{fold}_dim{args.atom_fea_len}_val")
            model.extract_atom_embeddings(test_loader, embedding_save_path, f"fold{fold}_dim{args.atom_fea_len}_test")


if __name__ == '__main__':
    main()
