import pandas as pd
from pathlib import Path
from CGCNN_ef_get_embedding.get_embedding_ef import get_emb as cgcnn_get_emb
from MEGNet_ef_get_embedding.get_embedding_ef import get_emb as megnet_get_emb
import re
from types import SimpleNamespace

pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.max_colwidth", None)  # 不截断字符串
pd.set_option("display.width", None)  # 自动适配行宽


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
            path = Path(f"../pre-train_model/cgcnn/cgcnn_/Cgcnn_matbench_{new_row}_dim{I}")
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


# 保存模型嵌入csv的文件夹
BASE_DIR = Path(__file__).resolve().parent

cgcnn_emb_dir = BASE_DIR / "cgcnn_ef_embedding"
megnet_emb_dir = BASE_DIR / "megnet_ef_embedding"

cgcnn_emb_dir.mkdir(parents=True, exist_ok=True)
megnet_emb_dir.mkdir(parents=True, exist_ok=True)

#
path = "CGCNN_MEGNet.xlsx"
df = pd.read_excel(path)

df.iloc[:, 0] = df.iloc[:, 0].astype(str)

# 输入名称列表
name_list = ["perovskites"]

results = []

for name in name_list:
    # 第一列以 name 开头的所有行
    sub_df = df[df.iloc[:, 0].str.startswith(name, na=False)]

    for _, row in sub_df.iterrows():
        row_name = row.iloc[0]  # 第一列：行名
        i_value = row.iloc[8]  # I 列的值

        # -------- B-H 列（索引 1:8）--------
        bh = row.iloc[1:8]
        bh_min = bh.min()
        bh_col = bh.idxmin()

        results.append({
            "row_name": row_name,
            "range": "B-H",
            "min_value": bh_min,
            "column_name": bh_col,
            "I_value": i_value
        })

        # -------- J-P 列（索引 9:16）--------
        jp = row.iloc[9:16]
        jp_min = jp.min()
        jp_col = jp.idxmin()

        results.append({
            "row_name": row_name,
            "range": "J-P",
            "min_value": jp_min,
            "column_name": jp_col,
            "I_value": i_value
        })

result_df = pd.DataFrame(results)
result_df[["model_path", "path_exists"]] = result_df.apply(
    lambda r: infer_model_path(r),
    axis=1,
    result_type="expand"
)

print(result_df)

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

        args = SimpleNamespace(
            task_name=m.group("task"),
            fold=int(m.group("fold")),
            atom_fea_len=int(m.group("dim")),
            a=float(m.group("a")),
            b=float(m.group("b")),
        )

        print(f"[CGCNN] running get_emb: {args}")
        cgcnn_get_emb(args)

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

        args = SimpleNamespace(
            subset=m.group("subset"),
            fold=int(m.group("fold")),
            dim_node_embed=int(m.group("dim")),
            a=float(m.group("a")),
            b=float(m.group("b")),
        )

        print(f"[MEGNet] running get_emb: {args}")
        megnet_get_emb(args)

    # ================= 其他模型 =================
    else:
        # 不做任何事
        continue
