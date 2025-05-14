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
test_in_part1 = """Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 13
test_out_part2 = 30


# %%
def match_values(pat, s):
    vals = []
    for m in re.finditer(pat, s):
        vals.append((m.start(), m.end(), int(m.group(1)) ))

    return np.array(vals)


# %%
def parse(s):
    lines = s.strip().split('\n')
    games = dict()
    for line in lines:
        game_desc, numbers = line.split(':')
        game_id = re.findall(r'(\d+)', game_desc)[0]
        winning, card = numbers.split('|')
        games[int(game_id)] = {
            'winning': [int(n) for n in re.findall(r'(\d+)', winning)],
            'card': [int(n) for n in re.findall(r'(\d+)', card)]
        }

    return games


# %%
def part1_sol(s):
    games = parse(s)
    matches = np.array([len(set(g['winning']) & set(g['card'])) for i, g in games.items()])
    return (2**(matches[matches>0]-1)).sum()


# %%
def part2_sol(s):
    games = parse(s)
    # Matches and initial multiplier
    matches = np.array([(len(set(g['winning']) & set(g['card'])), 1) for i, g in games.items()])
    for i in range(matches.shape[0]):
        m = matches[i, 0]
        multiplier = matches[i, 1]
        matches[i+1:i+m+1, 1] += multiplier
        # print(i+1, m, multiplier, matches)
    return matches[:, 1].sum()


# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part1), test_out_part2)

# %%
in_str = read_input(4)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str)
print(part2)

# %%
