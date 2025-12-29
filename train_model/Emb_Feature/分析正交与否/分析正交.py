import sys

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

# ===== 你的工程依赖 =====
from matgl.layers import BondExpansion
from matgl.config import DEFAULT_ELEMENTS
from matgl.utils.training import ModelLightningModule
from megnet_model import MEGNet
from megnet_train import load_pretrain_embeddings

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
dim_node_embed=16
a=1
b=1
# 构造模型
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

ckpt_path = ("matbench_perovskites_e1f1/matbench_perovskites_fold0_dim16/epochepoch=0410-train_losstrain_MAE=0.0160-val_lossval_MAE=0.0432.ckpt")
#ckpt_path = ("matbench_perovskites_e1f1/matbench_perovskites_fold3_dim16/epochepoch=0388-train_losstrain_MAE=0.0189-val_lossval_MAE=0.0422.ckpt")
#ckpt_path = ("matbench_perovskites_e1f1/matbench_perovskites_fold4_dim16/epochepoch=0431-train_losstrain_MAE=0.0176-val_lossval_MAE=0.0411.ckpt")


# ckpt = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(ckpt["state_dict"], strict=False)

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

emb = lit_module.model.embedding

# learnable semantic embedding
E_sem = emb.learn_embed.weight.detach().cpu().numpy()

# --- 属性 embedding（fix_embed 固定，但 linear 会随 checkpoint 变化） ---
Z = torch.arange(E_sem.shape[0], device=device)
with torch.no_grad():
    E_attr = emb.binary_linear(
        emb.fix_embed(Z)
    ).detach().cpu().numpy()

# cosine similarity
cos_sim = cosine_similarity(E_sem, E_attr)
cos_flat = cos_sim.flatten()

angles = np.degrees(np.arccos(np.clip(cos_flat, -1, 1)))

print("\n========== 正交性统计 ==========")
print(f"cosine 均值: {cos_flat.mean():.6f}")
print(f"cosine 绝对值均值: {np.abs(cos_flat).mean():.6f}")
print(f"cosine 标准差: {cos_flat.std():.6f}")

print("\n========== 角度统计 ==========")
print(f"平均夹角: {angles.mean():.2f}°")
print(f"夹角标准差: {angles.std():.2f}°")
print(f"夹角范围: [{angles.min():.2f}°, {angles.max():.2f}°]")


# angle matrix
angle_matrix = np.degrees(
    np.arccos(np.clip(cos_sim, -1.0, 1.0))
)
print(f"\n  第一个元素的语义 vs 属性:")
print(f"    - 相似度: {cos_sim[0, 0]:.6f}")
print(f"    - 夹角: {angle_matrix[0, 0]:.2f}°")


print(f"\n  第10个元素的语义 vs 属性:")
print(f"    - 相似度: {cos_sim[9, 9]:.6f}")
print(f"    - 夹角: {angle_matrix[9, 9]:.2f}°")

# ========== 正交性统计 ==========
# cosine 均值: -0.001794
# cosine 绝对值均值: 0.197429
# cosine 标准差: 0.244275
#
# ========== 角度统计 ==========
# 平均夹角: 90.11°
# 夹角标准差: 14.42°
# 夹角范围: [41.63°, 140.62°]
#
#   第一个元素的语义 vs 属性:
#     - 相似度: -0.002777
#     - 夹角: 90.16°
#
#   第10个元素的语义 vs 属性:
#     - 相似度: 0.203020
#     - 夹角: 78.29°
