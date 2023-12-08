# %%
import os
import numpy as np
import pandas as pd
import re
from io import StringIO
from collections import Counter
from functools import reduce
from math import lcm
DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')


# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 6
test_out_part2 = 6


# %%
def parse(s):
    steps, transition_str = s.strip().split('\n\n')
    lines = transition_str.split('\n')
    transitions = dict()
    for line in lines:
        k, v = line.split(' = ')
        left, right = v.split(', ')
        transitions[k] = {'L': left[1:], 'R': right[:-1]}
    return list(steps), transitions


# %%
def part1_sol(s):
    steps, transitions = parse(s)
    n_step = len(steps)
    state = 'AAA'
    count = 0
    while state != 'ZZZ':
        state = transitions[state][steps[count % n_step]]
        count += 1
    return count


# %%
def part2_sol(s):
    steps, transitions = parse(s)
    n_step = len(steps)
    start_states = [t for t in transitions.keys() if t[-1] == 'A']
    end_states = [t for t in transitions.keys() if t[-1] == 'Z']
    periods = []
    for start_state in start_states:
        state = start_state
        count = 0
        while state not in end_states:
            state = transitions[state][steps[count % n_step]]
            count += 1
        periods.append(count)
    return lcm(*periods)

# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part1), test_out_part2)

# %%
in_str = read_input(8)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str)
print(part2)

# %%
