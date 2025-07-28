from mendeleev import element


def build_category_lookup():
    # 类金属列表
    metalloids = {5, 14, 32, 33, 51, 52, 84}

    # 非金属元素
    nonmetals = {1, 6, 7, 8, 15, 16, 34}
    lookup = {}
    for Z in range(1, 104):
        elem = element(Z)
        group = elem.group_id
        period = elem.period
    
        if Z == 1:  # 氢
            cat = "nonmetal"
        elif group == 1 and period > 1:  # 碱金属
            cat = "alkali metal"
        elif group == 2:  # 碱土金属
            cat = "alkaline earth metal"
        elif group == 17:  # 卤素
            cat = "halogen"
        elif group == 18:  # 稀有气体
            cat = "noble gas"
        elif Z in metalloids:  # 类金属
            cat = "metalloid"
        elif Z in nonmetals:  # 确定的非金属
            cat = "nonmetal"
        elif 57 <= Z <= 71:  # 镧系
            cat = "lanthanides"
        elif 89 <= Z <= 103:  # 锕系
            cat = "actinides"
        elif group in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  # 过渡金属
            cat = "transition metal"
        else:  # 其他金属
            cat = "metal"

        lookup[Z] = cat
    return lookup


if __name__ == '__main__':
    atomic_number_to_category = build_category_lookup()
