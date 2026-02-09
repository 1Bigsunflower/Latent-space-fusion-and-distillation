import sys

import pandas as pd
import torch
import numpy as np
from lightning import seed_everything
from sklearn.metrics.pairwise import cosine_similarity
from megnet_model import MEGNet
from matgl.config import DEFAULT_ELEMENTS
from megnet_train import load_pretrain_embeddings
from random import seed
from matgl.utils.training import ModelLightningModule
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import subspace_angles
from pathlib import Path
from matgl.layers import BondExpansion
from matgl.config import DEFAULT_ELEMENTS
from matgl.utils.training import ModelLightningModule
from megnet_model import MEGNet
from megnet_train import load_pretrain_embeddings
from pymatgen.core import Element
from sklearn.metrics.pairwise import paired_cosine_distances

def model_cos(subset, fold, dim, e, f):
    init_seed = 42
    seed_everything(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(init_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    elem_list = DEFAULT_ELEMENTS
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
    bond_expansion = BondExpansion(
        rbf_type="Gaussian",
        initial=0.0,
        final=5.0,
        num_centers=25,
        width=0.4,
    )
    cgcnn_init_embedding, atom_init_dim = load_pretrain_embeddings()

    current_file = Path(__file__)
    root_dir = current_file.parents[2]
    model_name = f"{subset}_fold{fold}_dim{dim}_e{e}_f{f}"
    model_path = root_dir / "megnet_ef_model" / model_name

    model = MEGNet(
        dim_node_embedding=dim,
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
        a=e,
        b=f,
    )
    # print(model_path)
    ckpt_files = list(model_path.glob("*.ckpt"))
    print(ckpt_files)
    if not ckpt_files:
        raise FileNotFoundError(f"该目录下没有ckpt文件")
    ckpt_path = ckpt_files[0]

    lit_module = ModelLightningModule.load_from_checkpoint(
        ckpt_path,
        model=model,
        loss="mae_loss",
        map_location=torch.device("cpu")
    )
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    lit_module.model.load_state_dict(new_state_dict, strict=False)

    lit_module.eval()
    lit_module.to(device)

    BASE_DIR = Path(__file__).resolve().parent
    ROOT_DIR = BASE_DIR.parent.parent.parent

    elem_csv_path = (
            ROOT_DIR / "Embedding_weight_orig" / "MEGNet" / "megnet_emb" /
            f"{subset}_fold{fold}_dim{dim}.csv"
    )

    df = pd.read_csv(elem_csv_path)
    elements = df.iloc[:, 0].astype(str).unique()
    elem_list = tuple(sorted(elements, key=lambda el: Element(el).Z))

    emb = lit_module.model.embedding
    Z_list = [Element(el).Z for el in elem_list]
    Z_tensor = torch.tensor(Z_list, device=device)

    with torch.no_grad():
        # learnable semantic embedding
        E_sem = emb.learn_embed(Z_tensor).detach().cpu().numpy()

        # 属性 embedding（先 fix_embed，再过 binary_linear）
        E_attr = emb.binary_linear(
            emb.fix_embed(Z_tensor)
        ).detach().cpu().numpy()

    cos_diag = 1 - paired_cosine_distances(E_sem, E_attr)
    angles = np.degrees(np.arccos(np.clip(cos_diag, -1, 1)))

    var_angle = np.var(angles)
    # print(round(angles.mean(), 2))
    return round(angles.mean(), 2), round(var_angle,2)
    # cosine similarity
    # cos_sim = cosine_similarity(E_sem, E_attr)
    # cos_flat = cos_sim.flatten()
    #
    # angles = np.degrees(np.arccos(np.clip(cos_flat, -1, 1)))

    # print("\n========== 正交性统计 ==========")
    # print(f"cosine 均值: {cos_flat.mean():.6f}")
    # print(f"cosine 绝对值均值: {np.abs(cos_flat).mean():.6f}")
    # print(f"cosine 标准差: {cos_flat.std():.6f}")
    #
    # print("\n========== 角度统计 ==========")
    # print(f"平均夹角: {angles.mean():.2f}°")
    # print(f"夹角标准差: {angles.std():.2f}°")
    # print(f"夹角范围: [{angles.min():.2f}°, {angles.max():.2f}°]")
    #
    # return round(angles.mean(), 2)


if __name__ == '__main__':
    # subset = "matbench_jdft2d"
    # fold = 0
    # dim = 8
    # e = 1.0
    # f = 1.0
    # emb_angles = model_cos(subset, fold, dim, e, f)


    subsets = [
        'matbench_jdft2d',
        'matbench_phonons',
        'matbench_dielectric',
        'matbench_log_gvrh',
        'matbench_log_kvrh',
        'matbench_perovskites',
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
        # print(f"Processing subset: {subset}")

        for dim in dims:
            for fold in folds:
                row = {}
                var_ = {}
                row["TASK"] = f"{subset}_fold{fold}"
                row["Dim"] = dim

                var_["TASK"] = f"{subset}_fold{fold}"
                var_["Dim"] = dim
                # 计算五组 e,f
                for i, (e, f) in enumerate(zip(e_list, f_list), start=1):
                    value, var = model_cos(subset, fold, dim, e, f)
                    row[f"e{e}_f{f}"] = value
                    print(f"{subset} fold={fold} dim={dim}  e={e} f={f}  -> {value}")
                    var_[f"e{e}_f{f}"] =var
                rows.append(row)
                vars_.append(var_)

        # subset 之间插一个空行
        rows.append({})
        vars_.append({})
    # print(len(vars))
    # print(vars)
    # print(np.mean(vars))
    df = pd.DataFrame(rows)
    save_path = "MEGNet_aa.xlsx"
    df.to_excel(save_path, index=False)

    df2 = pd.DataFrame(vars_)
    df2.to_excel("MEGNet_bb.xlsx", index=False)
    print("\nSaved to MEGNet_bb.xlsx")

