import pandas as pd
import os

input_csv = "megnet_e.csv"
output_excel = "output.xlsx"

# 列名 = csv 文件名（无 .csv）
col_name = os.path.splitext(os.path.basename(input_csv))[0]

# 读取 csv 所有行
with open(input_csv, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

assert len(lines) == 32, "CSV 必须是 32 行"

excel_rows = []

# 8 组
for i in range(8):
    # 每组 4 行：i, i+8, i+16, i+24
    for j in range(4):
        idx = i + j * 8
        numbers = lines[idx].split(",")

        for num in numbers:
            excel_rows.append([f"{float(num):.6g}"])

    # ⭐ 每一组结束后插入一个空行（最后一组可要可不要）
    excel_rows.append([""])

# 写入 Excel
df = pd.DataFrame(excel_rows, columns=[col_name])
df.to_excel(output_excel, index=False)

print("完成：已正确插入空行")
