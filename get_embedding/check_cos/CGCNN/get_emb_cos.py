import os
import gc
import json
import sys

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from random import seed, sample

import lightning.pytorch as pl
from lightning import seed_everything
from sklearn.metrics.pairwise import cosine_similarity
from pymatgen.core import Element
from matbench.bench import MatbenchBenchmark
from matgl.ext.pymatgen import get_element_list

from data import StruData, collate_pool_matbench
from model import CrystalGraphConvNet
from sklearn.metrics.pairwise import paired_cosine_distances

BASE_DIR = Path(__file__).resolve().parent


class Normalizer(object):
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean


class Cgcnn_lightning(pl.LightningModule):
    def __init__(self, crystalGraphConvNet, normalizer, a=1.0, b=1.0):
        super().__init__()
        self.crystalGraphConvNet = crystalGraphConvNet
        self.normalizer = normalizer
        self.save_hyperparameters(ignore=["crystalGraphConvNet", "normalizer"])

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

import os
import torch

def load_trained_model(subset, fold, dim, a, b):
    """
    Load trained CGCNN model WITHOUT loading dataset.
    """

    model_dir = (
        BASE_DIR.parent.parent
        / "cgcnn_ef_model"
        / f"Cgcnn_{subset}_fold{fold}_dim{dim}_e{a}_f{b}"
    )

    ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
    if len(ckpt_files) == 0:
        raise RuntimeError(f"No checkpoint found in {model_dir}")

    ckpt_path = os.path.join(model_dir, ckpt_files[0])
    print("Loading checkpoint:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # lightning 的 state_dict 一般在这里
    state_dict = ckpt["state_dict"]

    # atom_fea_len (= dim)
    atom_fea_len = state_dict[
        "crystalGraphConvNet.binary_encoder.weight"
    ].shape[0]

    # orig_atom_fea_len
    orig_atom_fea_len = state_dict[
        "crystalGraphConvNet.binary_encoder.weight"
    ].shape[1]

    # nbr_fea_len
    fc_weight = state_dict[
        "crystalGraphConvNet.convs.0.fc_full.weight"
    ]
    nbr_fea_len = fc_weight.shape[1] - 2 * atom_fea_len

    # n_conv
    n_conv = len([
        k for k in state_dict.keys()
        if k.startswith("crystalGraphConvNet.convs.")
        and k.endswith("fc_full.weight")
    ])

    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=atom_fea_len,
        n_conv=n_conv,
        h_fea_len=128,
        n_h=1,
        classification=False,
    )


    cgcnn_state_dict = {
        k.replace("crystalGraphConvNet.", ""): v
        for k, v in state_dict.items()
        if k.startswith("crystalGraphConvNet.")
    }

    model.load_state_dict(cgcnn_state_dict)
    model.eval()

    ROOT_DIR = BASE_DIR.parent.parent.parent
    elem_csv_path = (
            ROOT_DIR / "Embedding_weight_orig" / "CGCNN" / "cgcnn_emb" /
            f"{subset}_fold{fold}_dim{dim}.csv"
    )

    df = pd.read_csv(elem_csv_path)
    elements = df.iloc[:, 0].astype(str).unique()
    elem_list = tuple(sorted(elements, key=lambda el: Element(el).Z))

    return model, elem_list

# def load_trained_model(subset, fold, dim, a, b):
#     mb = MatbenchBenchmark(autoload=False, subset=[subset])
#     for task in mb.tasks:
#         task.load()
#
#         train_inputs, train_outputs = task.get_train_and_val_data(fold)
#
#         dataset = StruData(train_inputs, train_outputs,
#                            precompute=True,
#                            precompute_workers=2,
#                            precompute_backend="process")
#         elem_list = get_element_list(train_inputs)
#
#         if len(dataset) < 500:
#             sample_data_list = [dataset[i] for i in range(len(dataset))]
#         else:
#             sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
#
#         _, sample_target = collate_pool_matbench(sample_data_list)
#         normalizer = Normalizer(sample_target)
#
#         structures, _ = dataset[0]
#         orig_atom_fea_len = structures[0].shape[-1]
#         nbr_fea_len = structures[1].shape[-1]
#
#         model = CrystalGraphConvNet(
#             orig_atom_fea_len,
#             nbr_fea_len,
#             atom_fea_len=dim,
#             n_conv=3,
#             h_fea_len=128,
#             n_h=1,
#             classification=False,
#         )
#
#         model_dir = (
#                 BASE_DIR.parent.parent
#                 / "cgcnn_ef_model"
#                 / f"Cgcnn_{subset}_fold{fold}_dim{dim}_e{a}_f{b}"
#         )
#
#         ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
#         ckpt_path = os.path.join(model_dir, ckpt_files[0])
#
#         print(ckpt_files)
#         lit_model = Cgcnn_lightning.load_from_checkpoint(
#             ckpt_path,
#             crystalGraphConvNet=model,
#             normalizer=normalizer,
#             a=a,
#             b=b,
#             map_location=torch.device("cpu"),
#         )
#
#     return lit_model.crystalGraphConvNet, elem_list


def extract_two_embeddings(cgcnn, elem_list, device="cpu"):
    cgcnn.eval()
    cgcnn.to(device)

    ATOM_INIT = BASE_DIR / "atom_init.json"
    with open(ATOM_INIT, "r") as f:
        atom_init = json.load(f)

    E_sem = []
    E_bin = []

    with torch.no_grad():
        for elem in elem_list:
            Z = Element(elem).number
            z_str = str(Z)

            atom_fea = torch.tensor(atom_init[z_str], dtype=torch.float32, device=device).unsqueeze(0)
            elem_id = torch.tensor([Z], dtype=torch.long, device=device)

            sem = cgcnn.elem_emb(elem_id).squeeze(0).cpu().numpy()
            binv = cgcnn.binary_encoder(atom_fea).squeeze(0).cpu().numpy()

            E_sem.append(sem)
            E_bin.append(binv)

    return np.array(E_sem), np.array(E_bin)


def cgcnn_cos(task, fold, dim, a, b):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cgcnn, elem_list = load_trained_model(task, fold, dim, a, b)

    E_sem, E_bin = extract_two_embeddings(
        cgcnn,
        elem_list=elem_list,
        device=device,
    )
    # print(E_sem)
    # print(E_bin)
    cos_diag = 1 - paired_cosine_distances(E_sem, E_bin)
    angles = np.degrees(np.arccos(np.clip(cos_diag, -1, 1)))

    var_angle = np.var(angles)
    print(var_angle)
    # print(round(angles.mean(), 2))
    return round(angles.mean(), 2), round(var_angle,2)
    # cos_sim = cosine_similarity(E_sem, E_bin)
    # cos_flat = cos_sim.flatten()
    #
    # angles = np.degrees(np.arccos(np.clip(cos_flat, -1, 1)))
    # return round(angles.mean(), 2)


if __name__ == "__main__":
    init_seed = 42
    seed_everything(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(init_seed)  # Random特有

    subsets = [
        "matbench_jdft2d",
        "matbench_phonons",
        "matbench_dielectric",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
        "matbench_perovskites",
        "matbench_mp_gap",
        "matbench_mp_e_form"
    ]

    dims = [8, 16, 32, 64]
    folds = range(5)

    e_list = [1.0, 1.0, 1.0, 0.01, 0.0001]
    f_list = [0.0001, 0.01, 1.0, 1.0, 1.0]

    rows = []
    vars_ = []
    for subset in subsets:
        # print(f"Processing: {subset}")

        for dim in dims:
            for fold in folds:
                row = {}
                var_ = {}
                row["TASK"] = f"{subset}_fold{fold}"
                row["Dim"] = dim

                var_["TASK"] = f"{subset}_fold{fold}"
                var_["Dim"] = dim

                for e, f in zip(e_list, f_list):
                    value, var = cgcnn_cos(subset, fold, dim, e, f)
                    key = f"e{e}_f{f}"
                    row[key] = value
                    var_[key] =var
                    # print(f"{subset} fold={fold} dim={dim} e={e} f={f} → {value}")
                vars_.append(var_)
                rows.append(row)

                import gc

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        rows.append({})  # 空行
        vars_.append({})
    # print(len(vars))
    # print(vars)
    # print(np.mean(vars))
    df = pd.DataFrame(rows)
    df.to_excel("CGCNN_aa.xlsx", index=False)
    print("\nSaved to CGCNN_aa.xlsx")

    df2 = pd.DataFrame(vars_)
    df2.to_excel("CGCNN_bb.xlsx", index=False)
    print("\nSaved to CGCNN_bb.xlsx")
