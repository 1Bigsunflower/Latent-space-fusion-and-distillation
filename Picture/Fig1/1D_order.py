import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from mendeleev import element
from element_category import *
from pymatgen.core.periodic_table import Element
import matplotlib

matplotlib.use('TkAgg')

cmap = {
    "alkali metal": "#00BCA9",
    "alkaline earth metal": "#FFCF3C",
    "metal": "#FCC0C2",
    "halogen": "#B9C3D5",
    "metalloid": "#FE9F6B",
    "noble gas": "#D6B7D6",
    "nonmetal": "#65D7EA",
    "transition metal": "#ABD86E",
    "lanthanides": "#FE703C",
    "actinides": "#00D4FE",
}

file_names = ['Z.csv', 'P.csv', 'GA.csv', 'Pm.csv'][::-1]
sortings = [pd.read_csv(f)["element"].tolist() for f in file_names]
labels = [f.replace(".csv", "") for f in file_names]

new_y_positions = []
for i in range(len(labels)):
    base = i * 7
    new_y_positions.append((
        base, base + 1,
        base + 2, base + 3,
        base + 4, base + 5
    ))

flat_y_positions = [y for triplet in new_y_positions for y in triplet]

max_len = 0
for s in sortings:
    third = (len(s) + 2) // 3
    r1, r2, r3 = len(s[2*third:]), len(s[third:2*third]), len(s[:third])
    max_len = max(max_len, r1, r2, r3)


fig, ax = plt.subplots(figsize=(max_len * 2 + 4, len(flat_y_positions) * 2))

for i, (label, elems) in enumerate(list(zip(labels, sortings))[::-1]):
    block_size = 1.0
    y1_num, y1_elem, y2_num, y2_elem, y3_num, y3_elem = new_y_positions[::-1][i]
    third = (len(elems) + 2) // 3
    row1 = elems[2 * third:]
    row2 = elems[third:2 * third]
    row3 = elems[:third]

    rows = [(row1, y1_elem), (row2, y2_elem), (row3, y3_elem)]
    rows = rows[::-1]

    counter = 1

    for row, y_elem in rows:
        offset = (max_len - len(row)) / 2
        for j, symbol in enumerate(row):
            Z = Element(symbol).Z
            cat = get_element_category(Z)
            color = cmap[cat]
            center = j + offset
            left = j + offset

            ax.text(
                x=left + block_size / 2,
                y=y_elem + 1,
                s=str(counter),
                ha="center",
                va="center",
                fontsize=53,
                color="black"
            )
            counter += 1

            # 元素
            ax.barh(
                y=y_elem,
                width=block_size,
                left=left,
                height=block_size,
                color=color,
                edgecolor="white"
            )
            ax.text(
                x=left + block_size / 2,
                y=y_elem,
                s=symbol,
                ha="center",
                va="center",
                fontsize=65,
                color="black"
            )

ax.set_aspect('equal', 'box')

ax.set_yticks([y2_elem for (_, _, _, y2_elem, _, _) in new_y_positions[::-1]])
ax.set_yticklabels(labels[::-1], fontsize=80)


min_y = 0
max_y = max([y_elem for group in new_y_positions for y_elem in [group[1], group[3], group[5]]]) + 1.5
ax.set_ylim(min_y, max_y)

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlim(0, max_len)

legend_handles = [Patch(facecolor=cmap[k], label=k) for k in cmap]
ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=(len(cmap) + 1) // 2,
    fontsize=80,
    frameon=False
)

plt.subplots_adjust(right=0.95, bottom=0.25)
plt.tight_layout()
plt.savefig("1D_triple_number_7.7.pdf", bbox_inches="tight")
# plt.show()
