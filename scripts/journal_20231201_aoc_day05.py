# %%
import os
import numpy as np
import pandas as pd
import re
from io import StringIO
import jax.numpy as jnp

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
    s = s.strip()
    maps = get_maps(s)
    seed_str = s.split('\n')[0].split(': ')[1]
    # print(seed_str)
    seeds = [int(n) for n in seed_str.split(' ')]
    return seeds, maps


# %%
def lookup(maps, v):
    steps = maps.keys()
    for step in steps:
        m = maps[step]
        d = jnp.select((v >= m[:, 0]) & (v <= m[:, 1]), m[:, 2], jnp.nan)
        s = jnp.select((v >= m[:, 0]) & (v <= m[:, 1]), m[:, 0], jnp.nan)
        v = jnp.where(~jnp.isnan(d), d + v - s, v)
    return v


# %%
def lookup_df(maps, v):
    steps = maps.keys()
    for step in steps:
        m = maps[step].query('@v >= src_start and @v <= src_end')
        if len(m) > 0:
            v = (m.dst_start + v - m.src_start).values[0]
    return v


# %%
def part1_sol(s):
    seeds, maps = parse(s)
    # map_arrays = {
    #     k: jnp.array(v.loc[:, ['src_start', 'src_end', 'dst_start']].values) for k, v in maps.items()
    # }
    res = np.inf
    # j_lookup = jit(lambda x: lookup(map_arrays, x))

    for s in seeds:
        # v = j_lookup(s)
        v = lookup_df(maps, s)
        if v < res:
            res = v
    return np.array(res).min()


# %%
def part2_sol(s):
    seed_intervals, maps = parse(s)
    seed_intervals = np.array(seed_intervals).reshape((-1, 2)).tolist()
    count = 0
    this_intervals = seed_intervals
    for s, m in maps.items():
        count += 1
        # print(f'Map {count} of {len(maps)}')
        next_intervals = []
        while len(this_intervals) > 0:
            si = this_intervals.pop()
            v = si[0]
            # ml = m.query('@v >= src_start and @v <= src_end')  # 193ms per run on full input vs 88ms for .loc
            ml = m.loc[(m.src_start <= v) & (m.src_end >= v), ['src_start', 'range', 'dst_start']].values
            if len(ml) > 0:
                src_start, r, dst_start = ml.flatten().tolist()
                new_range = r - (v - src_start)
                this_range = min(si[1], new_range)
                remaining_range = si[1] - this_range
                mapped_interval = [dst_start + v - src_start, this_range]
                next_intervals.append(mapped_interval)
                if remaining_range > 0:
                    remaining_interval = [v + this_range, remaining_range]
                    this_intervals.append(remaining_interval)
                    # print(si, mapped_interval, remaining_interval)
            else:
                next_intervals.append(si)
        # print(next_intervals)
        this_intervals = next_intervals

    return np.array(this_intervals)[:, 0].min()


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
