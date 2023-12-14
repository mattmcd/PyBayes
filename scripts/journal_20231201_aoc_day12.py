# %%
import os
import numpy as np
from math import comb
from itertools import permutations
import re
DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')


# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""

# %%
test_in_part2 = test_in_part1

# %%
test_out_part1 = 21
test_out_part2 = None


# %%
def parse(s):
    rows = [{'springs': springs, 'groups': [int(n) for n in groups.split(',')]} for springs, groups in [line.split(' ') for line in s.strip().split('\n')]]
    return rows


# %%
def parse_row_comb(row):
    good = np.cumsum([c == '.' for c in row['springs']])
    bad = np.cumsum([c == '#' for c in row['springs']])
    unknown = np.cumsum([c == '?' for c in row['springs']])
    unknown_remaining = unknown[-1] - unknown
    n_unknown = unknown[-1]
    n_bad = sum(row['groups'])
    n_good = len(row['groups']) - 1
    n_known_bad = bad[-1]
    n_known_good = good[-1]
    n_bad_remaining = n_bad - n_known_bad
    n_comb_bad = comb(n_unknown, n_bad_remaining)
    n_comb_good = comb(n_unknown, n_unknown - n_bad_remaining)
    return n_comb_bad  # np.vstack([good, bad, unknown_remaining])


# %%
def parse_row(row):
    springs = np.array(list(row['springs']))
    unknown = np.array([c == '?' for c in row['springs']])
    n_unknown = unknown.sum()
    grp = row['groups']
    n_bad = sum(grp)
    n_known_bad = np.sum([c == '#' for c in row['springs']])
    n_bad_to_place = n_bad - n_known_bad
    possible_locations = list(set(permutations(
        (['.'] * (n_unknown - n_bad_to_place)) + (['#'] * n_bad_to_place)
    )))
    pat = r'\.*' + ''.join([r'#{' + str(n) + r'}\.+' for n in grp[:-1]]) + '#{' + str(grp[-1]) + '}'
    good_loc = 0
    for loc in possible_locations:
        this_loc = springs.copy()
        this_loc[unknown] = loc
        this_str = ''.join(this_loc)
        # print(pat, this_str, re.match(pat, this_str))
        if re.match(pat, this_str):
            good_loc += 1
    return good_loc


# %%
def part1_sol(s):
    rows = parse(s)
    good_loc = 0
    n_row = len(rows)
    count = 1
    for row in rows:
        print(f'Row {count} of {n_row}')
        good_loc += parse_row_comb(row)
        count += 1
    return good_loc

# %%
print(part1_sol(test_in_part1), test_out_part1)


# %%
def part2_sol(s):
    pass


# %%
print(part2_sol(test_in_part2), test_out_part2)

# %%
in_str = read_input(12)

# %%
# part1 = part1_sol(in_str)
# print(part1)

# %%
# part2 = part2_sol(in_str)
# print(part2)

# %%
