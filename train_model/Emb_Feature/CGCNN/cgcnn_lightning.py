import argparse
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
# emb dim
parser.add_argument('--atom_fea_len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
# 超参数 a 和 b，用于融合 embedding + binary features
parser.add_argument('--a', default=1.0, type=float,
                    help='weight for element embedding in final_vec = a*embedding + b*binary')
parser.add_argument('--b', default=1.0, type=float,
                    help='weight for binary features in final_vec = a*embedding + b*binary')
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
            # "matbench_jdft2d",  # 636
            "matbench_phonons",  # 1,265
            # "matbench_dielectric",  # 4,764
            # "matbench_log_gvrh",  # 10,987
            # "matbench_log_kvrh",  # 10,987
            # "matbench_perovskites",  # 1w8
            # "matbench_mp_gap",  # 106,113
            # "matbench_mp_e_form"  # 132,752
        ]
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            #
            train_inputs, train_outputs = task.get_train_and_val_data(fold)  # 获取训练集

            dataset = StruData(train_inputs, train_outputs)
            collate_fn = collate_pool_matbench
            # 训练 验证资料
            train_loader, val_loader = get_train_loader(dataset=dataset,
                                                        collate_fn=collate_fn,
                                                        batch_size=128,
                                                        train_ratio=0.75,
                                                        val_ratio=0.25
                                                        )

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
            model = Cgcnn_lightning(crystalGraphConvNet, normalizer, a=args.a, b=args.b)

            early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=500, verbose=True,
                                                mode="min", )
            checkpoint_callback = ModelCheckpoint(
                monitor='val_MAE',
                save_top_k=1,
                mode='min',
                dirpath=f'Cgcnn_{task.dataset_name}_fold{fold}_dim{args.atom_fea_len}',
                filename='epoch{epoch:04d}-train_loss{train_loss:.4f}-val_loss{val_MAE:.4f}',
                auto_insert_metric_name=False)
            trainer = pl.Trainer(max_epochs=10000, callbacks=[early_stop_callback, checkpoint_callback],
                                 enable_progress_bar=False,
                                 log_every_n_steps=1000,
                                 # precision='16-mixed'
                                 )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            del train_loader, val_loader
            torch.cuda.empty_cache()

            # 加载验证损失最小的模型权重
            best_model_path = checkpoint_callback.best_model_path
            model = Cgcnn_lightning.load_from_checkpoint(best_model_path,
                                                         crystalGraphConvNet=crystalGraphConvNet,
                                                         normalizer=normalizer)
            model.eval()
            # 测试
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            dataset_test = StruData(test_inputs, test_outputs)
            test_loader = DataLoader(dataset=dataset_test,
                                     batch_size=128,
                                     collate_fn=collate_fn)

            trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    main()
