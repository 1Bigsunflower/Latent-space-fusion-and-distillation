import sys

import pandas as pd
from pathlib import Path
from CGCNN_ef_get_embedding.get_embedding_ef import get_emb as cgcnn_get_emb
from MEGNet_ef_get_embedding.get_embedding_ef import get_emb as megnet_get_emb
import re
from types import SimpleNamespace
import numpy as np
import itertools
from scipy.spatial.distance import cdist
from pymatgen.core.periodic_table import Element

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.max_colwidth", None)  # 不截断字符串
pd.set_option("display.width", None)  # 自动适配行宽

from sklearn.manifold import MDS
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform


def build_normalized_distance(
        emb_df: pd.DataFrame,
        metric: str = "euclidean",
        l2_norm: bool = True,
):
    """
    给单个 emb_df 构造：
    1) 元素顺序列表
    2) 距离矩阵 D (size = n_elem x n_elem)
    并做 median 全局尺度归一化，使 median(D_ij)=1

    emb_df: 包含列 ["Element", Dim_0, Dim_1, ...]
    metric: "euclidean" 或 "cosine"
    l2_norm: 是否先对 embedding 行做 L2 normalize
    """

    assert "Element" in emb_df.columns, "需要 Element 列"

    elements = emb_df["Element"].tolist()
    vec = emb_df.drop(columns=["Element"]).values.astype(float)

    # ---- L2 normalize (可选) ----
    if l2_norm:
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vec = vec / norm

    # ---- 构造距离矩阵 ----
    if metric == "euclidean":
        D = squareform(pdist(vec, metric="euclidean"))
    elif metric == "cosine":
        D = squareform(pdist(vec, metric="cosine"))
    else:
        raise ValueError(f"不支持的 metric: {metric}")

    # ---- 全局 median 归一化 ----
    upper = D[np.triu_indices_from(D, k=1)]
    med = np.median(upper)

    if med <= 0:
        raise ValueError("该模型距离矩阵 median<=0，无法归一化")

    D = D / med

    return elements, D

def average_distance_matrices(dist_list, element_lists):
    """
    融合多个模型的距离矩阵：
    - 只在真正有 (i,j) 的模型里做平均
    - 分母 = 覆盖次数，而不是模型总数

    dist_list: [D1, D2, ...]
    element_lists: [elem_list1, elem_list2, ...]

    返回：
        global_elements: 排好序的全局元素列表
        D_avg: 共识距离矩阵
        C: 覆盖次数矩阵（可用于诊断）
    """

    # ---- 构造全局元素全集 ----
    global_elements = sorted(set().union(*element_lists))
    n = len(global_elements)

    idx_map_list = []
    for elems in element_lists:
        mapping = {e: i for i, e in enumerate(elems)}
        idx_map_list.append(mapping)

    # ---- 累积矩阵 ----
    D_sum = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=int)

    for (elems, D), mapping in zip(zip(element_lists, dist_list), idx_map_list):

        for i_local, ei in enumerate(elems):
            gi_global = global_elements.index(ei)

            for j_local, ej in enumerate(elems):
                gj_global = global_elements.index(ej)

                dij = D[i_local, j_local]
                D_sum[gi_global, gj_global] += dij
                C[gi_global, gj_global] += 1

    # ---- 只对 C>0 的项做平均 ----
    D_avg = np.zeros_like(D_sum)
    mask = C > 0
    D_avg[mask] = D_sum[mask] / C[mask]

    return global_elements, D_avg, C


def load_embedding_csv(emb_dir: Path, csv_name: str) -> pd.DataFrame:
    csv_path = emb_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Embedding CSV 不存在: {csv_path}")
    return pd.read_csv(csv_path)


def check_model_dir(row):
    path = Path(row["model_path"])
    # 目录不存在
    if not path.exists():
        return False, f"路径不存在: {path}"
    # ckpt 文件数量检查
    ckpts = list(path.glob("*.ckpt"))

    if len(ckpts) == 0:
        return False, f"没有 ckpt 文件: {path}"
    elif len(ckpts) > 1:
        return False, f"ckpt 文件多于一个 ({len(ckpts)}): {path}"

    return True, ckpts[0]


def infer_model_path(row):
    # L-P列需要清除掉后面的.1，因为名字相同
    row_name = row["row_name"]
    col = row["column_name"]
    I = int(row["I_value"])
    rng = row["range"]

    # -------- CGCNN --------
    if rng == "B-H":
        if col == df.columns[1]:  # B
            path = Path(f"../pre-train_model/cgcnn/cgcnn_hm/Cgcnn_matbench_{row_name}_dim{I}")
        elif col == df.columns[2]:  # C
            new_row = row_name.replace("_fold", "_emb_fold")
            path = Path(f"../pre-train_model/cgcnn/cgcnn_emb/Cgcnn_matbench_{new_row}_dim{I}")
        else:  # D-H
            # B-H列的D-H列保持原样，不清除后缀
            path = Path(f"cgcnn_ef_model/Cgcnn_matbench_{row_name}_dim{I}_{col}")

    # -------- MEGNet --------
    else:  # J-P
        if col == df.columns[9]:  # J
            path = Path(f"../pre-train_model/megnet/megnet_hm/matbench_{row_name}_dim{I}")
        elif col == df.columns[10]:  # K
            path = Path(f"../pre-train_model/megnet/megnet_emb/matbench_{row_name}_dim{I}")
        else:  # L-P
            # 只在L-P列清除.1后缀
            import re
            clean_col = re.sub(r'\.\d+$', '', col)  # 移除.1后缀
            path = Path(f"megnet_ef_model/matbench_{row_name}_dim{I}_{clean_col}")

    return path, path.exists()

def smacof_mds(D, dim=64, max_iter=2000, n_init=8, random_state=42):
    mds = MDS(
        n_components=dim,
        metric=True,
        dissimilarity="precomputed",
        max_iter=max_iter,
        n_init=n_init,
        n_jobs=None,
        random_state=random_state
    )
    X = mds.fit_transform(D)
    stress = mds.stress_
    return X, stress


def save_embedding_csv(elements, X, path):
    """
    elements: list[str]
    X: (n x d) numpy array
    path: 保存路径
    """
    Z1_103 = [Element.from_Z(z).symbol for z in range(1, 104)]

    # 当前嵌入 -> DataFrame
    df_exist = pd.DataFrame(X, index=elements)

    # 创建 103 x d 的零矩阵
    df_all = pd.DataFrame(
        0.0,
        index=Z1_103,
        columns=df_exist.columns
    )

    # 覆盖已有元素（pandas 会自动按元素名匹配）
    df_all.loc[elements] = df_exist

    # 转回 csv 形式
    df_all = df_all.reset_index()
    df_all.rename(columns={"index": "element"}, inplace=True)

    df_all.to_csv(path, index=False)
    print(f"保存成功: {path}")


if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent

    cgcnn_emb_dir = BASE_DIR / "cgcnn_ef_embedding"
    megnet_emb_dir = BASE_DIR / "megnet_ef_embedding"

    cgcnn_emb_dir.mkdir(parents=True, exist_ok=True)
    megnet_emb_dir.mkdir(parents=True, exist_ok=True)

    #
    path = "CGCNN_MEGNet.xlsx"
    df = pd.read_excel(path)

    df.iloc[:, 0] = df.iloc[:, 0].astype(str)

    name_list = [
        "jdft2d",
        "phonons",
        "dielectric",
        "log_gvrh",
        "log_kvrh",
        "perovskites",
        "mp_gap",
        "mp_e_form"
    ]

    results = []

    for name in name_list:
        # 第一列以 name 开头的所有行
        sub_df = df[df.iloc[:, 0].str.startswith(name, na=False)]

        for _, row in sub_df.iterrows():
            row_name = row.iloc[0]  # 第一列：行名
            i_value = row.iloc[8]  # I 列的值

            # -------- B-H 范围：固定选 B 列 (索引 1) --------
            # 原代码逻辑是 bh = row.iloc[1:8]; bh_min = bh.min(); bh_col = bh.idxmin()
            bh_col = df.columns[2]  # 获取 B 列的列名
            bh_val = row[bh_col]

            results.append({
                "row_name": row_name,
                "range": "B-H",
                "min_value": bh_val,  # 这里保留 key 名为 min_value 避免改动后面结构，实际存的是 B 列值
                "column_name": bh_col,
                "I_value": i_value
            })

            # -------- J-P 范围：固定选 J 列 (索引 9) --------
            # 原代码逻辑是 jp = row.iloc[9:16]; jp_min = jp.min(); jp_col = jp.idxmin()
            jp_col = df.columns[10]  # 获取 J 列的列名
            jp_val = row[jp_col]

            results.append({
                "row_name": row_name,
                "range": "J-P",
                "min_value": jp_val,
                "column_name": jp_col,
                "I_value": i_value
            })

    result_df = pd.DataFrame(results)
    result_df[["model_path", "path_exists"]] = result_df.apply(
        lambda r: infer_model_path(r),
        axis=1,
        result_type="expand"
    )

    # print(result_df)
    # sys.exit()
    # 模型完整性检查
    errors = []

    for _, row in result_df.iterrows():
        ok, info = check_model_dir(row)
        if not ok:
            errors.append(
                f"{row['row_name']} | {row['range']} | {row['column_name']} -> {info}"
            )
    assert len(errors) == 0, (
            "以下模型检查失败：\n" + "\n".join(errors)
    )
    print("模型路径和ckpt文件正确")

    all_elements = []
    all_D = []
    all_G = []

    for _, row in result_df.iterrows():
        model_path = str(row["model_path"])

        # ================= CGCNN EF =================
        if model_path.startswith("cgcnn_ef_model"):
            # Cgcnn_matbench_perovskites_fold0_dim8_e0.01_f1.0
            name = Path(model_path).name

            m = re.match(
                r"Cgcnn_(?P<task>.+)_fold(?P<fold>\d+)_dim(?P<dim>\d+)_e(?P<a>[\d.]+)_f(?P<b>[\d.]+)",
                name
            )
            if not m:
                raise ValueError(f"无法解析 CGCNN 路径: {name}")

            task = m.group("task")
            fold = int(m.group("fold"))
            dim = int(m.group("dim"))
            a = float(m.group("a"))
            b = float(m.group("b"))

            csv_name = f"{task}_fold{fold}_dim{dim}_e{a}_f{b}.csv"
            csv_path = cgcnn_emb_dir / csv_name
            if not csv_path.exists():
                args = SimpleNamespace(
                    task_name=m.group("task"),
                    fold=int(m.group("fold")),
                    atom_fea_len=int(m.group("dim")),
                    a=float(m.group("a")),
                    b=float(m.group("b")),
                )

                print(f"[CGCNN] running get_emb: {args}")
                cgcnn_get_emb(args)

            emb_df = pd.read_csv(csv_path)

        # ================= MEGNet EF =================
        elif model_path.startswith("megnet_ef_model"):
            # matbench_perovskites_fold0_dim8_e1.0_f1.0
            name = Path(model_path).name

            m = re.match(
                r"(?P<subset>.+)_fold(?P<fold>\d+)_dim(?P<dim>\d+)_e(?P<a>[\d.]+)_f(?P<b>[\d.]+)",
                name
            )
            if not m:
                raise ValueError(f"无法解析 MEGNet 路径: {name}")

            task = m.group("subset")
            fold = int(m.group("fold"))
            dim = int(m.group("dim"))
            a = float(m.group("a"))
            b = float(m.group("b"))

            csv_name = f"{task}_fold{fold}_dim{dim}_e{a}_f{b}.csv"
            csv_path = megnet_emb_dir / csv_name

            if not csv_path.exists():
                args = SimpleNamespace(
                    subset=m.group("subset"),
                    fold=int(m.group("fold")),
                    dim_node_embed=int(m.group("dim")),
                    a=float(m.group("a")),
                    b=float(m.group("b")),
                )

                print(f"[MEGNet] running get_emb: {args}")
                megnet_get_emb(args)

            emb_df = pd.read_csv(csv_path)

        # ================= Pre-train model =================
        elif model_path.startswith("..") or model_path.startswith("../pre-train_model"):
            p = Path(model_path)
            name = p.name

            # ---------- CGCNN ----------
            if "cgcnn" in model_path.lower():
                if "cgcnn_hm" in model_path:
                    emb_dir = Path("../Embedding_weight_orig/CGCNN/cgcnn_hm")
                    m = re.match(
                        r"Cgcnn_(?P<task>.+)_fold(?P<fold>\d+)_dim(?P<dim>\d+)",
                        name
                    )
                    if not m:
                        print(f"[WARN] 无法解析 CGCNN pre-train 名称: {name}")
                        continue

                    task = m.group("task")
                    fold = int(m.group("fold"))
                    dim = int(m.group("dim"))
                else:
                    emb_dir = Path("../Embedding_weight_orig/CGCNN/cgcnn_emb")
                    m = re.match(
                        r"Cgcnn_(?P<task>.+)_emb_fold(?P<fold>\d+)_dim(?P<dim>\d+)",
                        name
                    )
                    if not m:
                        print(f"[WARN] 无法解析 CGCNN pre-train 名称: {name}")
                        continue

                    task = m.group("task")
                    fold = int(m.group("fold"))
                    dim = int(m.group("dim"))

                csv_name = f"{task}_fold{fold}_dim{dim}.csv"

            # ---------- MEGNet ----------
            elif "megnet" in model_path.lower():
                m = re.match(
                    r"matbench_(?P<task>.+)_fold(?P<fold>\d+)_dim(?P<dim>\d+)",
                    name
                )
                if not m:
                    print(f"[WARN] 无法解析 MEGNet pre-train 名称: {name}")
                    continue

                task = f"matbench_{m.group('task')}"
                fold = int(m.group("fold"))
                dim = int(m.group("dim"))

                if "megnet_hm" in model_path:
                    emb_dir = Path("../Embedding_weight_orig/MEGNet/megnet_hm")
                else:
                    emb_dir = Path("../Embedding_weight_orig/MEGNet/megnet_emb")

                csv_name = f"{task}_fold{fold}_dim{dim}.csv"

            else:
                continue
            print(csv_name)
            csv_path = emb_dir / csv_name
            emb_df = pd.read_csv(csv_path)

        elems, D = build_normalized_distance(
            emb_df,
            metric="cosine",
            l2_norm=True
        )

        all_elements.append(elems)
        all_D.append(D)

    print("共有模型数量：", len(all_D))
    global_elements, D_avg, C = average_distance_matrices(all_D, all_elements)
    print("全局元素数：", len(global_elements))
    print("距离矩阵形状：", D_avg.shape)

    X32_s, stress32 = smacof_mds(D_avg, dim=32)
    save_embedding_csv(global_elements, X32_s, "MDS_E_32_cos.csv")
    print("stress =", stress32)

    X64_s, stress64 = smacof_mds(D_avg, dim=64)
    save_embedding_csv(global_elements, X64_s, "MDS_E_64_cos.csv")
    print("stress =", stress64)
