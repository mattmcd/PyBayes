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
test_in_part1 = """Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 8
test_out_part2 = 2286

# %%
def match_array(pat, s):
    res = []
    for m in re.finditer(pat, s):
        res.append((m.start(), int(m.group(1))))
    return np.vstack([np.array([[0, 0]]), np.array(res)])

def match_draws(s):
    res = []
    count = 1
    for m in re.finditer(r'[;\n]', s):
        res.append((m.start(), count))
        count += 1
    return  np.vstack([np.array([[0, 0]]), np.array(res)])


def match_values(pat, s):
    vals = []
    labels = []
    for m in re.finditer(pat, s):
        vals.append((m.start(), int(m.group(1)) ))
        labels.append(m.group(2))
    return np.array(vals), labels

# %%
def parse(s):
    games = match_array(r'Game (\d+)', s)
    draw_ends = match_draws(s)
    vals, labels = match_values(r'(\d+) (red|green|blue)', s)
    game_lookup = np.zeros(len(s), dtype=int)
    draw_lookup = np.zeros(len(s), dtype=int)
    for i in range(len(s)):
        game_lookup[i] = games[np.argwhere(games[:, 0] <= i)[-1][0], 1]
        draw_lookup[i] = draw_ends[np.argwhere(draw_ends[:, 0] <= i)[-1][0], 1]

    df_l = pd.DataFrame({
        'game': game_lookup[vals[:, 0]], 'draw': draw_lookup[vals[:, 0]],
        'value': vals[:, 1], 'label': labels
    })
    df = pd.pivot_table(df_l,
        index=['game', 'draw'], columns='label', values='value', aggfunc='sum'
    ).fillna(0).astype(int)

    return df


# %%
def part1_sol(df, r, g, b):
    df.loc[:, 'ok'] = np.all((df.loc[:, ['red', 'green', 'blue']].values - np.array([r, g, b])) <= 0, axis=1)
    df_r = df.reset_index().groupby('game').ok.all()
    return np.sum(df_r.index[df_r])


# %%
def part2_sol(df):
    df_r = df.reset_index().groupby('game')[['red', 'green', 'blue']].max()
    df_r[df_r == 0] = 1
    return df_r.product(axis=1).sum()


# %%
print(part1_sol(parse(test_in_part1), 12, 13, 14), test_out_part1)

# %%
print(part2_sol(parse(test_in_part1)), test_out_part2)

# %%
in_str = read_input(2)

# %%
part1 = part1_sol(parse(in_str), 12, 13, 14)
print(part1)

# # %%
part2 = part2_sol(parse(in_str))
print(part2)

# %%
