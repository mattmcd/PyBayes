# %%
import os
import numpy as np
import pandas as pd
from abc import ABC
import re
from collections import defaultdict

DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')

# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598.."""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 4361
test_out_part2 = 467835


# %%
def match_values(pat, s):
    vals = []
    for m in re.finditer(pat, s):
        vals.append((m.start(), m.end(), int(m.group(1)) ))

    return np.array(vals)

# %%
def parse(s):
    lines = s.strip().split('\n')
    n_row, n_col = len(lines), len(lines[0]) + 1  # +1 for cols due to \n
    vals = match_values(r'(\d+)', s)
    res = []
    connectors = []
    for v in vals:
        el_i, el_j = np.unravel_index(range(v[0], v[1]), (n_row, n_col))
        neighbours = []
        found = False
        for k in range(len(el_j)):
            i, j = el_i[k], el_j[k]
            # Get index of neighbours of all chars top-left, centre-left, top-right ec
            neighbours.append(np.ravel_multi_index((i-1, j-1),(n_row, n_col),  mode='clip'))
            neighbours.append(np.ravel_multi_index((i-1, j), (n_row, n_col), mode='clip'))
            neighbours.append(np.ravel_multi_index((i-1, j+1),(n_row, n_col),  mode='clip'))
            neighbours.append(np.ravel_multi_index((i, j-1),(n_row, n_col),  mode='clip'))
            neighbours.append(np.ravel_multi_index((i, j+1),(n_row, n_col),  mode='clip'))
            neighbours.append(np.ravel_multi_index((i+1, j-1),(n_row, n_col),  mode='clip'))
            neighbours.append(np.ravel_multi_index((i+1, j),(n_row, n_col),  mode='clip'))
            neighbours.append(np.ravel_multi_index((i+1, j + 1),(n_row, n_col),  mode='clip'))
            neighbours = list(set(neighbours))
            for n in neighbours:
                if len(re.findall(r'[^.0-9\n]', s[n])) > 0:
                    res.append(v[2])
                    connectors.append((n, s[n], v[2]))
                    found = True
                    break
            if found:
                break

    return res, connectors


# %%
def part1_sol(s):
    res, _ = parse(s)
    return np.array(res).sum()


# %%
def part2_sol(s):
    _, connectors = parse(s)
    df = pd.DataFrame(
        [{'loc': c[0], 'val': c[2]} for c in connectors if c[1] == '*']
    ).groupby('loc').agg(['count', 'prod'])
    return df.loc[df[('val', 'count')] == 2, ('val', 'prod')].sum()


# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part1), test_out_part2)

# %%
in_str = read_input(3)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str)
print(part2)

# %%
