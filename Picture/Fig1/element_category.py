from mendeleev import element


def get_element_category(atomic_number):
    # 类金属列表
    metalloids = {5, 14, 32, 33, 51, 52, 84}

    # 非金属元素
    nonmetals = {1, 6, 7, 8, 15, 16, 34}

    elem = element(atomic_number)
    group = elem.group_id
    period = elem.period

    if atomic_number == 1:  # 氢
        return "nonmetal"
    elif group == 1 and period > 1:  # 碱金属
        return "alkali metal"
    elif group == 2:  # 碱土金属
        return "alkaline earth metal"
    elif group == 17:  # 卤素
        return "halogen"
    elif group == 18:  # 稀有气体
        return "noble gas"
    elif atomic_number in metalloids:  # 类金属
        return "metalloid"
    elif atomic_number in nonmetals:  # 确定的非金属
        return "nonmetal"
    elif 57 <= atomic_number <= 71:  # 镧系
        return "lanthanides"
    elif 89 <= atomic_number <= 103:  # 锕系
        return "actinides"
    elif group in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  # 过渡金属
        return "transition metal"
    else:  # 其他金属
        return "metal"

