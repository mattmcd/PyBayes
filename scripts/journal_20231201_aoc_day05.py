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
test_in_part1 = """seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 35
test_out_part2 = 46


# %%
def get_maps(s):
    vals = []
    pat = r'(\w+)-to-(\w+) map:\n([\d \n]+)+(?:\n\n|$)'
    for m in re.finditer(pat, s):
        vals.append((m.group(1), m.group(2), m.group(3) ))
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
    s = s.strip()
    maps = get_maps(s)
    seed_str = s.split('\n')[0].split(': ')[1]
    # print(seed_str)
    seeds = [int(n) for n in seed_str.split(' ')]

    return seeds, maps


# %%
def part1_sol(s):
    seeds, maps = parse(s)
    steps = maps.keys()
    res = []
    for s in seeds:
        v = s
        s_res = [v]
        for step in steps:
            m = maps[step].query('@v >= src_start and @v <= src_end')
            if len(m) > 0:
                v = (m.dst_start + v - m.src_start).values[0]
            s_res.append(v)
        res.append(s_res)
    return np.array(res)[:, -1].min()


# %%
def part2_sol(s):
    seed_intervals, maps = parse(s)
    seed_intervals = np.array(seed_intervals).reshape((-1, 2)).tolist()
    steps = maps.keys()
    res = np.inf
    count = 0
    intervals = len(seed_intervals)
    for si in seed_intervals:
        count += 1
        print(f'Interval {count} of {intervals}')
        for s in range(si[0], si[0]+si[1]):
            v = s
            for step in steps:
                m = maps[step].query('@v >= src_start and @v <= src_end')
                if len(m) > 0:
                    v = (m.dst_start + v - m.src_start).values[0]
            if v < res:
                res = v

    return res


# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part1), test_out_part2)

# %%
in_str = read_input(5)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str)
print(part2)

# %%
