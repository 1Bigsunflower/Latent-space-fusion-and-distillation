import pandas as pd
import re


def process_log_files(csv_file, excel_file):
    """
    根据文件名中的模式分组，提取最后一列的值
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 定义提取模式的正则表达式
    # 提取 _a和_b之间的数字组合
    pattern = r'_a([\d\.]+)_b([\d\.]+)\.log$'

    # 创建字典来存储结果
    results_dict = {}

    for index, row in df.iterrows():
        filename = row['filename']

        # 使用正则表达式匹配模式
        match = re.search(pattern, filename)
        if match:
            a_value = match.group(1)
            b_value = match.group(2)
            pattern_key = f"a{a_value}_b{b_value}"

            # 获取最后一列的值
            avg_value = row['avg_test_MAE']

            # 添加到字典
            if pattern_key not in results_dict:
                results_dict[pattern_key] = []
            results_dict[pattern_key].append(avg_value)

    # 转换为DataFrame
    # 找到最长的列表长度
    max_len = max(len(v) for v in results_dict.values())

    # 填充None使所有列表长度相同
    for key in results_dict:
        while len(results_dict[key]) < max_len:
            results_dict[key].append(None)

    # 创建DataFrame
    result_df = pd.DataFrame(results_dict)

    # 写入Excel
    result_df.to_excel(excel_file, index=False)

    print(f"处理完成！结果已保存到 {excel_file}")
    print(f"找到 {len(results_dict)} 种模式")

    # 显示统计信息
    print("\n模式统计：")
    for pattern_key, values in results_dict.items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            avg = sum(valid_values) / len(valid_values)
            print(f"{pattern_key}: {len(valid_values)} 个值，平均值: {avg:.6f}")

    return result_df


# 使用示例
process_log_files('Orig_megnet.csv', 'pattern_results.xlsx')