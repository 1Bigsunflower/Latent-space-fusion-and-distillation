import torch


def read_hyperparams_from_checkpoint(checkpoint_path):
    """从检查点文件中读取超参数"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 检查检查点的结构
    print("检查点包含的键:", checkpoint.keys())

    # 超参数通常保存在 'hyper_parameters' 键下
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print("超参数内容:", hparams)

        # 读取 a 和 b 值
        a = hparams.get('a', None)
        b = hparams.get('b', None)

        if a is not None and b is not None:
            print(f"从检查点读取的 a={a}, b={b}")
            return a, b
        else:
            print("警告: 检查点中没有找到 a 或 b 参数")
            print(f"找到的超参数键: {list(hparams.keys())}")
    else:
        print("警告: 检查点中没有 'hyper_parameters' 键")
        print("检查点结构:", {k: type(v) for k, v in checkpoint.items()})

    return None, None

checkpoint_path = "Cgcnn_matbench_dielectric_fold4_dim32/epoch0413-train_loss0.3336-val_loss0.4650.ckpt"
a, b = read_hyperparams_from_checkpoint(checkpoint_path)
if a is not None and b is not None:
    print(f"成功读取: a={a}, b={b}")
