import os
import shutil
import sys
import argparse
import torch
import numpy as np
from random import seed

from lightning import seed_everything
from matbench.bench import MatbenchBenchmark
from data import StruData, get_train_loader, collate_pool_matbench

parser = argparse.ArgumentParser(description='gen and dump')
parser.add_argument('--fold', type=int, default=0, help='')
parser.add_argument('--subset', default='matbench_perovskites', type=str,
                    choices=['matbench_mp_gap', 'matbench_mp_e_form', 'matbench_perovskites'],
                    help='subset dataset to use')
args = parser.parse_args()


def gen_and_dump(name, dataset, folder):
    io_buffer = []  # 用于存储生成的数据字典的缓冲区
    io_cnt = 0  # 计数生成的文件数量
    for i in range(len(dataset)):
        io_buffer.append(dataset[i])
        if len(io_buffer) > 100 or i == len(dataset) - 1:
            for data in io_buffer:
                save_path = "{}/{}/{}.pth".format(folder, name, io_cnt)
                # Check if the directory exists, and create it if not
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                # Save the data to the specified path
                torch.save(data, save_path)
                io_cnt += 1
            io_buffer.clear()
    print("文件数量：", io_cnt)


def gen_and_dump(name, dataset, folder):
    io_buffer = []  # 用于存储生成的数据字典的缓冲区
    io_cnt = 0  # 计数生成的文件数量
    for i in range(len(dataset)):
        io_buffer.append(dataset[i])
        if len(io_buffer) > 100 or i == len(dataset) - 1:
            for data in io_buffer:
                save_path = "{}/{}/{}.pth".format(folder, name, io_cnt)
                # Check if the directory exists, and create it if not
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                # Save the data to the specified path
                torch.save(data, save_path)
                io_cnt += 1
            io_buffer.clear()
    print("文件数量：", io_cnt)

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

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            args.subset
        ]
    )

    for task in mb.tasks:
        task.load()
        folder = f'../CGCNN_dataset/{args.subset}/{args.fold}'
        cache_path = f"cache/{args.subset}_fold{args.fold}.pt"

        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"已删除旧文件夹: {folder}")
            except Exception as e:
                print(f"删除文件夹时出错: {e}")

        # 删除缓存文件
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"已删除旧缓存文件: {cache_path}")
            except Exception as e:
                print(f"删除缓存文件时出错: {e}")
        if not os.path.exists(folder):
            os.makedirs(folder)  # 如果文件夹不存在，则创建它
            train_inputs, train_outputs = task.get_train_and_val_data(args.fold)  # 获取训练集

            dataset = StruData(
                train_inputs, train_outputs,
                max_num_nbr=12, radius=8, dmin=0, step=0.2,
                precompute=True,
                precompute_workers=8,
                precompute_backend="process",  # 进程并行
                cache_path=cache_path,
                rebuild_cache=False,
                chunksize=32,
            )

            collate_fn = collate_pool_matbench
            # 训练 验证资料
            train_loader, val_loader = get_train_loader(dataset=dataset,
                                                        collate_fn=collate_fn,
                                                        batch_size=128,
                                                        train_ratio=0.75,
                                                        val_ratio=0.25
                                                        )
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset

            test_inputs, test_outputs = task.get_test_data(
                args.fold,
                include_target=True
            )
            test_dataset = StruData(test_inputs, test_outputs)

            gen_and_dump(name='train', dataset=train_dataset, folder=folder)
            gen_and_dump(name='val', dataset=val_dataset, folder=folder)
            gen_and_dump(name='test', dataset=test_dataset, folder=folder)

            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    print(f"已删除缓存文件: {cache_path}")
                except Exception as e:
                    print(f"删除缓存文件时出错: {e}")
            else:
                print(f"缓存文件不存在: {cache_path}")


if __name__ == '__main__':
    main()
