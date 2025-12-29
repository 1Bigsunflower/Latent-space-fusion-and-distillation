import re
import csv
import os
import pandas as pd


def process_logs(log_dir, x):
    """
    处理CGCNN日志文件，提取test_MAE数据并保存到CSV文件

    参数:
    log_dir: log文件所在目录
    x: 文件名前缀（如'cgcnn'）
    """

    atom_fea_len_value = [8, 16, 32, 64]
    a_list = [1, 1, 1, 0.01, 0.0001]
    b_list = [0.0001, 0.01, 1, 1, 1]

    output_csv = f"{x}_plus.csv"

    # 期望的MAE数量
    EXPECTED_MAE_COUNT = 5

    # ======================
    # 正则匹配 test_MAE 数值
    # ======================
    pattern = re.compile(r"test_MAE\s*│\s*([\d.]+)")

    # ======================
    # 写 CSV
    # ======================
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "filename",
            "atom_fea_len",
            "e",
            "f",
            "test_MAE_list",
            "avg_test_MAE"
        ])

        for dim in atom_fea_len_value:
            for a, b in zip(a_list, b_list):
                log_name = f"{x}_dim{dim}_e{a}_f{b}.log"
                log_path = os.path.join(log_dir, log_name)

                if not os.path.exists(log_path):
                    # print(f"[Warning] File not found: {log_name}")
                    continue

                maes = []

                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = pattern.search(line)
                        if match:
                            value = float(match.group(1))
                            maes.append(value)

                # ===== 检查MAE数量并补0 =====
                mae_count = len(maes)
                if mae_count == 0:
                    print(f"[Warning] No test_MAE found in {log_name}")
                    maes = [0.0] * EXPECTED_MAE_COUNT
                elif mae_count < EXPECTED_MAE_COUNT:
                    print(
                        f"[Warning] Only {mae_count} test_MAE values found in {log_name}, expected {EXPECTED_MAE_COUNT}")
                    # 补0到10个
                    maes.extend([0.0] * (EXPECTED_MAE_COUNT - mae_count))

                    # 检查是否存在RuntimeError
                    runtime_error_flag = False
                    with open(log_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                        if "RuntimeError: Pin memory thread exited unexpectedly" in file_content:
                            runtime_error_flag = True

                    if runtime_error_flag:
                        print(
                            f"[Error] RuntimeError 'Pin memory thread exited unexpectedly' detected in {log_name}")

                elif mae_count > EXPECTED_MAE_COUNT:
                    print(
                        f"[Warning] {mae_count} test_MAE values found in {log_name}, expected {EXPECTED_MAE_COUNT}. Using first {EXPECTED_MAE_COUNT} values.")
                    # 只取前10个
                    maes = maes[:EXPECTED_MAE_COUNT]

                # ===== 计算平均MAE =====
                # 如果有实际的MAE值，计算平均（排除补0的值）
                actual_maes = [m for m in maes if m > 0]  # 过滤掉补的0
                if actual_maes:
                    avg_mae = round(sum(actual_maes) / len(actual_maes), 6)
                else:
                    avg_mae = 0.0

                writer.writerow([
                    log_name,
                    dim,
                    a,
                    b,
                    ";".join(f"{v:.6f}" for v in maes),
                    avg_mae
                ])

    # print(f"Saved results to {output_csv}")


def reorganize_mae_list_for_excel(mae_values):
    """
    将MAE列表按照要求重新组织：
    分成前5个和后5个，然后按特定顺序排列
    """
    if not mae_values or len(mae_values) == 0:
        return []

    if len(mae_values) != 10:
        print(f"警告：MAE列表长度不是10，是{len(mae_values)}")
        return mae_values

    # 分成前5个和后5个
    first_half = mae_values[:5]  # 前5个
    second_half = mae_values[5:]  # 后5个

    # 按照要求重新组织：先所有前5个，再所有后5个
    # 但在这个函数里，我们只处理一个test_MAE列表
    # 实际的重新组织会在主函数中完成
    return first_half + second_half


def csv_to_excel_corrected(a, x=None):
    """
    将CSV文件转换为Excel，按照特定要求组织数据

    参数:
    a: 文件名前缀（如'cgcnn'）
    x: 暂时保留，与原始代码兼容
    """

    # ======================
    # 读取 CSV
    # ======================
    csv_file = f"{a}_plus.csv"
    if x is not None:
        excel_file = f"{a}_plus_x{x}.xlsx"
    else:
        excel_file = f"{a}_plus.xlsx"

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误：找不到CSV文件 {csv_file}")
        return
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # ======================
    # 定义atom_fea_len的顺序
    # ======================
    atom_fea_len_order = [8, 16, 32, 64]

    # ======================
    # 收集所有唯一的e和f组合
    # ======================
    e_f_combinations = df[['e', 'f']].drop_duplicates().values

    # ======================
    # 创建Excel数据结构
    # ======================
    excel_data = {}

    # 初始化每一列
    for e_val, f_val in e_f_combinations:
        col_name = f"e{e_val}_f{f_val}"
        excel_data[col_name] = []

    # ======================
    # 按顺序填充每一列
    # 顺序：atom_fea_len 8,16,32,64
    # ======================
    for atom_fea in atom_fea_len_order:
        # 获取当前atom_fea的所有行
        atom_rows = df[df['atom_fea_len'] == atom_fea]

        # 对当前atom_fea，为每一列添加前5个值
        for e_val, f_val in e_f_combinations:
            col_name = f"e{e_val}_f{f_val}"

            # 查找对应的行
            row = atom_rows[(atom_rows['e'] == e_val) & (atom_rows['f'] == f_val)]

            if not row.empty:
                mae_str = row.iloc[0]['test_MAE_list']
                if pd.notna(mae_str):
                    try:
                        mae_values = [float(x) for x in str(mae_str).split(';')]
                        # 取前5个值
                        if len(mae_values) >= 5:
                            excel_data[col_name].extend(mae_values[:5])
                        else:
                            excel_data[col_name].extend(mae_values + [''] * (5 - len(mae_values)))
                    except ValueError:
                        excel_data[col_name].extend([''] * 5)
                else:
                    excel_data[col_name].extend([''] * 5)
            else:
                # 如果没有对应数据，填充空值
                excel_data[col_name].extend([''] * 5)

    # ======================
    # 再次按顺序添加后5个值
    # ======================
    for atom_fea in atom_fea_len_order:
        # 获取当前atom_fea的所有行
        atom_rows = df[df['atom_fea_len'] == atom_fea]

        # 对当前atom_fea，为每一列添加后5个值
        for e_val, f_val in e_f_combinations:
            col_name = f"e{e_val}_f{f_val}"

            # 查找对应的行
            row = atom_rows[(atom_rows['e'] == e_val) & (atom_rows['f'] == f_val)]

            if not row.empty:
                mae_str = row.iloc[0]['test_MAE_list']
                if pd.notna(mae_str):
                    try:
                        mae_values = [float(x) for x in str(mae_str).split(';')]
                        # 取后5个值
                        if len(mae_values) >= 10:
                            excel_data[col_name].extend(mae_values[5:10])
                        elif len(mae_values) > 5:
                            excel_data[col_name].extend(mae_values[5:] + [''] * (10 - len(mae_values)))
                        else:
                            # 如果没有后5个值，填充空值
                            excel_data[col_name].extend([''] * 5)
                    except ValueError:
                        excel_data[col_name].extend([''] * 5)
                else:
                    excel_data[col_name].extend([''] * 5)
            else:
                # 如果没有对应数据，填充空值
                excel_data[col_name].extend([''] * 5)

    # ======================
    # 创建DataFrame并保存为Excel
    # ======================
    try:
        # 确定最大行数
        max_rows = max(len(v) for v in excel_data.values())

        # 确保所有列长度一致
        for col in excel_data:
            while len(excel_data[col]) < max_rows:
                excel_data[col].append('')

        excel_df = pd.DataFrame(excel_data)

        # 保存为Excel
        excel_df.to_excel(excel_file, index=False)
        # print(f"已保存Excel文件: {excel_file}")

        # # 打印调试信息
        # print(f"\nExcel文件结构：")
        # print(f"总行数: {len(excel_df)}")
        # print(f"总列数: {len(excel_df.columns)}")
        # print(f"\n列名: {list(excel_df.columns)}")
        #
        # # 显示前几行示例
        # print(f"\n前5行示例:")
        # print(excel_df.head())

    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    log_dir = "../Emb_Feature/CGCNN/gvrh/"
    x = 'cgcnn_matbench_log_gvrh'
    process_logs(log_dir, x)

    log_dir = "../Emb_Feature/CGCNN/kvrh/"
    x = 'cgcnn_matbench_log_kvrh'
    process_logs(log_dir, x)

    log_dir = "../Emb_Feature/CGCNN/perovskites/"
    x = 'cgcnn_matbench_perovskites'
    process_logs(log_dir, x)

    log_dir = "../Emb_Feature/MEGNet_gvrh/gvrh"
    x = 'megnet_matbench_log_gvrh'
    process_logs(log_dir, x)

    log_dir = "../Emb_Feature/MEGNet_kvrh/kvrh"
    x = 'megnet_matbench_log_kvrh'
    process_logs(log_dir, x)

    log_dir = "../Emb_Feature/MEGNet_perovskites/perovskites"
    x = 'megnet_matbench_perovskites'
    process_logs(log_dir, x)

    log_dir = "../Emb_Feature/CGCNN/gap/"
    x = 'cgcnn_matbench_mp_gap'
    process_logs(log_dir, x)

    csv_to_excel_corrected('cgcnn_matbench_log_gvrh')
    csv_to_excel_corrected('cgcnn_matbench_log_kvrh')
    csv_to_excel_corrected('cgcnn_matbench_perovskites')
    csv_to_excel_corrected('cgcnn_matbench_mp_gap')

    csv_to_excel_corrected('megnet_matbench_log_gvrh')
    csv_to_excel_corrected('megnet_matbench_log_kvrh')
    csv_to_excel_corrected('megnet_matbench_perovskites')
