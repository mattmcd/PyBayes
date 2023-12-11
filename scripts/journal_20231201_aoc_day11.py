# %%
import os
import numpy as np
import matplotlib.pyplot as plt
DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')


# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#....."""

# %%
test_in_part2 = [test_in_part1, 10]

# %%
test_out_part1 = 374
test_out_part2 = 1030


# %%
def parse(s):
    lines = s.strip().split('\n')
    i = 0
    stars = set()
    galaxy_map_list = []
    for line in lines:
        galaxy_map_list.append([True if c == '#' else False for c in line])
        j = 0
        for symbol in line:
            if symbol == '#':
                stars.add((i, j))
            j += 1
        i += 1
    galaxy_map = np.array(galaxy_map_list)
    return galaxy_map, sorted(stars)


# %%
def part1_sol(s, scale=2):
    galaxy_map, stars = parse(s)
    empty_cols = np.cumsum(~np.any(galaxy_map, axis=0))
    empty_rows = np.cumsum(~np.any(galaxy_map, axis=1))
    n_stars = len(stars)
    total_dist = 0
    for i in range(n_stars):
        for j in range(i+1, n_stars):
            s1 = stars[i]
            s2 = stars[j]
            # print(f'{s1} - {s2}: {manhattan_distance(s1,s2,empty_rows,empty_cols)}')
            total_dist += manhattan_distance(s1,s2,empty_rows,empty_cols,scale)
    # return galaxy_map, empty_cols, empty_rows, stars
    return total_dist


# %%
def manhattan_distance(s1, s2, empty_rows, empty_cols, scale=2):
    row_dist = abs(s1[0] - s2[0])
    col_dist = abs(s1[1] - s2[1])
    row_dist += (scale-1)*abs(empty_rows[s1[0]] - empty_rows[s2[0]])
    col_dist += (scale-1)*abs(empty_cols[s1[1]] - empty_cols[s2[1]])
    dist = row_dist + col_dist
    return dist


# %%
def part2_sol(s, scale):
    return part1_sol(s, scale)


# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(*test_in_part2), test_out_part2)

# %%
in_str = read_input(11)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str, 1_000_000)
print(part2)

# %%
