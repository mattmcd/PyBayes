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
    for v in vals:
        i, j = np.unravel_index(range(v[0], v[1]), (n_row, n_col))
        neighbours = [
            np.ravel_multi_index((i + x, j + y), (n_row, n_col), mode='clip')
            for x, y in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        ]
        for n in set(np.concatenate(neighbours)):
            if len(re.findall(r'[^.0-9\n]', s[n])) > 0:
                res.append((n, s[n], v[2]))
                break

    return pd.DataFrame(res, columns=['loc', 'symbol', 'val'])


# %%
def part1_sol(s):
    df = parse(s)
    return df.val.sum()


# %%
def part2_sol(s):
    df = parse(s)
    df_g = df.query('symbol == "*"').groupby('loc').val.agg(['count', 'prod'])
    return df_g.loc[df_g['count'] == 2, 'prod'].sum()


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
