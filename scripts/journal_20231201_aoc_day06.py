# %%
import os
import numpy as np
import pandas as pd
import re
from io import StringIO

DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')


# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """Time:      7  15   30
Distance:  9  40  200"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 288
test_out_part2 = 71503


# %%
def get_maps(s):
    vals = []
    pat = r'(\w+)-to-(\w+) map:\n([\d \n]+)+(?:\n\n|$)'
    for m in re.finditer(pat, s):
        vals.append((m.group(1), m.group(2), m.group(3)))
    out = {
        (src, dst): pd.read_table(
            StringIO(m), sep=' ', names=['dst_start', 'src_start', 'range']
        ).assign(
            src_end=lambda x: x.src_start + x.range - 1
        ).sort_values('src_start') for src, dst, m in vals
    }
    return out


# %%
def parse(s):
    lines = s.strip().split('\n')
    times = [int(t) for t in lines[0].split(':')[1].strip().split(' ') if t != '']
    dists = [int(d) for d in lines[1].split(':')[1].strip().split(' ') if d != '']
    return times, dists


# %%
def solve_quad(T, d):
    y = np.sqrt(T ** 2 - 4 * (d + 1))
    s1 = int(np.ceil((T - y) / 2))
    s2 = int(np.floor((T + y) / 2))
    n = s2 - s1 + 1
    return n


# %%
def part1_sol(s):
    times, dists = parse(s)
    # res = []
    res = 1
    for i in range(len(times)):
        T = times[i]
        d = dists[i]
        # res.append([T, d, s1, s2, n])
        res *= solve_quad(T, d)
    return res


# %%
def part2_sol(s):
    times, dists = parse(s)
    T = int(''.join([str(n) for n in times]))
    d = int(''.join([str(n) for n in dists]))
    return solve_quad(T, d)

# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part1), test_out_part2)

# %%
in_str = read_input(6)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str)
print(part2)

# %%
