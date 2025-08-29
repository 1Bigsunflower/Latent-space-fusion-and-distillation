def split_by_gap(values, percent=30):
    """
    按相邻数的百分比差异切分子组

    values : list of float
        输入的一组数
    percent : float
        相邻数差异的百分比阈值 (30 表示 30%)
    """
    if not values:
        return []

    values_sorted = sorted(values)
    groups, start = [], 0

    for i in range(len(values_sorted) - 1):
        # 计算相邻两个数的相对差
        gap = (values_sorted[i+1] - values_sorted[i]) / values_sorted[i]
        if gap >= percent / 100.0:
            groups.append(values_sorted[start:i+1])
            start = i + 1

    groups.append(values_sorted[start:])
    return groups

radii = [106.1,103.4,101.3,99.5,97.9,111,109,93.8,92.3,90.8,89.4,88.1,94,93,84.8]

groups = split_by_gap(radii)
for i, g in enumerate(groups, 1):
    print(f"组{i}: {g}")
