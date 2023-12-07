# %%
import os
import numpy as np
import pandas as pd
import re
from io import StringIO
from collections import Counter
from functools import reduce
DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')


# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 6440
test_out_part2 = 5905


# %%
def parse(s):
    df = pd.read_table(StringIO(s.strip()), sep=' ', names=['hand', 'bid'])
    return df


# %%
def hand_value(hand, card_rank):
    cards = list(hand)
    card_nums = list(card_rank)
    card_vals = range(len(card_nums))
    val_map = dict(zip(card_nums, card_vals))
    hand_val = reduce(
        lambda acc, el: acc+el[0]*el[1],
        zip([val_map[c] for c in cards], [13**i for i in range(5)[::-1]]),
        0
    )
    return hand_val


# %%
def part1_sol(s):
    df = parse(s)
    df.loc[:, 'num_unique'] = df['hand'].apply(lambda x: len(np.unique(list(x))))
    df.loc[:, 'max_count'] = df['hand'].apply(lambda x: max(Counter(x).values()))
    df.loc[:, 'val'] = df['hand'].apply(lambda x: hand_value(x, '23456789TJQKA'))
    df = df.sort_values(
        ['num_unique', 'max_count', 'val'], ascending=[True, False, False]
    ).assign(
        hand_rank=range(1, len(df) + 1)[::-1]
    )
    return (df.bid * df.hand_rank).sum()


# %%
def part2_sol(s):
    df = parse(s)

    def num_unique_with_jacks(hand):
        cards = list(hand)
        unique_cards = len(np.unique(cards))
        jacks = [c for c in cards if c == 'J']
        num_jacks = len(jacks)
        if num_jacks > 0 and unique_cards > 1:
            unique_cards -= 1  # i.e. replace J with most common card
        return unique_cards

    def max_count_with_jacks(hand):
        counter = Counter(hand)
        max_count = max(counter.values())
        if counter['J'] > 0 and len(counter) > 1:
            if counter.most_common(1)[0][0] == 'J':
                max_count = reduce(lambda acc, el: acc + el[1], counter.most_common(2), 0)
            else:
                max_count += counter['J']
        return max_count

    df.loc[:, 'num_unique'] = df['hand'].apply(num_unique_with_jacks)
    df.loc[:, 'max_count'] = df['hand'].apply(max_count_with_jacks)
    df.loc[:, 'val'] = df['hand'].apply(lambda x: hand_value(x, 'J23456789TQKA'))
    df = df.sort_values(
        ['num_unique', 'max_count', 'val'], ascending=[True, False, False]
    ).assign(
        hand_rank=range(1, len(df) + 1)[::-1]
    )
    return (df.bid * df.hand_rank).sum()

# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part1), test_out_part2)

# %%
in_str = read_input(7)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str)
print(part2)

# %%
