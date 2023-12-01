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
def parse_part1(in_str):
    lines = in_str.strip().split('\n')
    digits = [re.findall(r'(\d)', x) for x in lines]
    numbers = np.array([int(''.join([d[0], d[-1]])) for d in digits])
    return numbers


# %%
def parse_part2(in_str):
    reps = list(zip('one two three four five six seven eight nine'.split(' '), [str(i) for i in range(1, 10)]))
    pat = '|'.join(list(zip(*reps))[0]) + r'|\d'
    reps_rev = list(
        zip([s[::-1] for s in 'one two three four five six seven eight nine'.split(' ')],
            [str(i) for i in range(1, 10)]
            )
    )
    pat_rev = '|'.join(list(zip(*reps_rev))[0]) + r'|\d'
    res = []
    for line in in_str.strip().split('\n'):
        matches = re.findall(pat, line)
        first = matches[0]
        for i, o in reps:
            first = re.sub(i, o, first)
        matches_rev = re.findall(pat_rev, line[::-1])
        last = matches_rev[0]
        for i, o in reps_rev:
            last = re.sub(i, o, last)
        res.append(int(first + last))
    return np.array(res)


# %%
test_in_part1 = """1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet"""

# %%
test_in_part2 = """two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen"""

# %%
test_out_part1 = 142
test_out_part2 = 281

# %%
print(parse_part1(test_in_part1).sum(), test_out_part1)

# %%
print(parse_part2(test_in_part2).sum(), test_out_part2)

# %%
in_str = read_input(1)

# %%
part1 = parse_part1(in_str).sum()
print(part1)

# %%
part2 = parse_part2(in_str).sum()
print(part2)