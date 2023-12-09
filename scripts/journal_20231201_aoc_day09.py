# %%
import os
import numpy as np
import pandas as pd
from functools import reduce
DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')


# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 114
test_out_part2 = 2


# %%
def parse(s):
    lines = s.strip().split('\n')
    nums = [[int(n) for n in line.split(' ')] for line in lines]
    return nums


# %%
def part1_sol(s):
    nums = parse(s)
    res = []
    for row in nums:
        last_vals = []
        v = pd.Series(row)
        last_vals.append(v.iloc[-1])
        while not (v.nunique() == 1):
            v = v.diff()
            last_vals.append(v.iloc[-1])

        res.append(np.array(last_vals).astype(int).sum())
    return sum(res)


# %%
def part2_sol(s):
    nums = parse(s)
    res = []
    for row in nums:
        first_vals = []
        ind = 0
        v = pd.Series(row)
        first_vals.append(v.iloc[ind])
        while not (v.nunique() == 1):
            v = v.diff()
            ind += 1
            first_vals.append(v.iloc[ind].astype(int))

        res.append(reduce(lambda acc, el: el - acc, first_vals[::-1], 0))
    return sum(res)


# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part1), test_out_part2)

# %%
in_str = read_input(9)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str)
print(part2)

# %%
