import argparse
import gc
import os
import sys

import pandas as pd
import numpy as np
from lightning import seed_everything
from sklearn.metrics.pairwise import cosine_similarity
from random import seed
from matgl.utils.training import ModelLightningModule
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import subspace_angles

from matgl.layers import BondExpansion
from matgl.config import DEFAULT_ELEMENTS
from matgl.utils.training import ModelLightningModule
from .megnet_model import MEGNet
from .megnet_train import load_pretrain_embeddings
from pymatgen.core import Element
import torch
from pymatgen.core import Element

parser = argparse.ArgumentParser(description='MEGNet')
parser.add_argument('--dim_node_embed', type=int, default=64, help='number of node embedding dim')
parser.add_argument('--fold', type=int, default=0, help='number of fold')
parser.add_argument('--a', default=1.0, type=float,
                    help='weight for element embedding in final_vec = a*embedding + b*binary')
parser.add_argument('--b', default=1.0, type=float,
                    help='weight for binary features in final_vec = a*embedding + b*binary')
parser.add_argument('--subset', default='matbench_dielectric', type=str,
                    choices=['matbench_jdft2d', 'matbench_phonons', 'matbench_dielectric', 'matbench_log_gvrh',
                             'matbench_log_kvrh', 'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_e_form'],
                    help='subset dataset to use')
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent   # get_embedding/
ROOT_DIR = BASE_DIR.parent.parent

@torch.no_grad()
def get_element_embedding_megnet(embedding_block, elem_list, device, a, b):
    embedding_block.eval()

    learn_embed = embedding_block.learn_embed
    fix_embed = embedding_block.fix_embed
    binary_linear = embedding_block.binary_linear

    element_embeddings = {}

    for elem in elem_list:
        Z = Element(elem).number

        Z_tensor = torch.tensor([Z], device=device)
        with torch.no_grad():
            node_feat1 = learn_embed(Z_tensor)
            node_feat2 = binary_linear(fix_embed(Z_tensor))

            node_feat = a * node_feat1 + b * node_feat2
        element_embeddings[elem] = node_feat.squeeze(0).cpu().numpy()

    return element_embeddings


def save_element_embedding_csv(
        element_embeddings: dict,
        save_path: str
):
    """
    element_embeddings: {Element(str): np.ndarray or torch.Tensor}
    """

    rows = []
    for elem, fea in element_embeddings.items():
        if torch.is_tensor(fea):
            fea = fea.cpu().numpy()
        rows.append([elem] + fea.tolist())

    dim = len(rows[0]) - 1
    columns = ['Element'] + [f'Dim_{i}' for i in range(dim)]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(save_path, index=False)

def get_emb(args):
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

    model = MEGNet(
        dim_node_embedding=args.dim_node_embed,
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
        orig_elem_list=DEFAULT_ELEMENTS,
        pretrain_embeddings=cgcnn_init_embedding,
        atom_init_dim=atom_init_dim,
        a=args.a,
        b=args.b,
    )

    model_dir = (
            PROJECT_DIR / "megnet_ef_model" /
            f"{args.subset}_fold{args.fold}_dim{args.dim_node_embed}_e{args.a}_f{args.b}"
    )
    ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
    checkpoint_path = os.path.join(model_dir, ckpt_files[0])

    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module = ModelLightningModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        loss="mae_loss",
        map_location=map_location
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    elem_csv_path = (
            ROOT_DIR / "Embedding_weight_orig" / "MEGNet" / "megnet_emb" /
            f"{args.subset}_fold{args.fold}_dim{args.dim_node_embed}.csv"
    )
    df = pd.read_csv(elem_csv_path)
    elements = df.iloc[:, 0].astype(str).unique()
    elem_list = tuple(sorted(elements, key=lambda el: Element(el).Z))

    emb = lit_module.model.embedding
    element_emb_dict = get_element_embedding_megnet(
        embedding_block=emb,
        elem_list=elem_list,
        device=device,
        a=args.a,
        b=args.b
    )
    # print(element_emb_dict)
    save_dir = PROJECT_DIR / "megnet_ef_embedding"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = (
            save_dir /
            f"{args.subset}_fold{args.fold}_dim{args.dim_node_embed}_e{args.a}_f{args.b}.csv"
    )

    save_element_embedding_csv(element_emb_dict, save_path)

    del model
    del lit_module
    del checkpoint
    del state_dict
    del new_state_dict
    del emb
    del element_emb_dict
    del df

    torch.cuda.empty_cache()
    gc.collect()

    return None


if __name__ == '__main__':
    args = parser.parse_args()
    get_emb(args)
