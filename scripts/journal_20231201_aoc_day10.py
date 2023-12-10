# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
DATA_DIR = os.path.expanduser('~/Work/Data/AoC2023')


# %%
def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2023_day{day:02}.txt'), 'r') as f:
        return f.read()


# %%
test_in_part1 = """..F7.
.FJ|.
SJ.L7
|F--J
LJ..."""

# %%
test_in_part2 = """FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L"""

# %%
test_out_part1 = 8
test_out_part2 = 10


# %%
def step(symbol, prev_loc):
    symbol_map = {
        ('-', (0, 1)): (0, 1),
        ('-', (0, -1)): (0, -1),
        ('|', (1, 0)): (1, 0),
        ('|', (-1, 0)): (-1, 0),
        ('L', (0, -1)): (-1, 0),
        ('L', (1, 0)): (0, 1),
        ('J', (0, 1)): (-1, 0),
        ('J', (1, 0)): (0, -1),
        ('7', (0, 1)): (1, 0),
        ('7', (-1, 0)): (0, -1),
        ('F', (-1, 0)): (0, 1),
        ('F', (0, -1)): (1, 0),
    }

    def inner(i, j):
        di = i - prev_loc[0]
        dj = j - prev_loc[1]
        i_step, j_step = symbol_map.get((symbol, (di, dj)), (0, 0))
        # print(symbol, (i, j), (di, dj), (i+i_step, j+j_step))

        return i+i_step, j+j_step
    return inner


# %%
def start_loc_to_pipe(pipes, start_loc):
    i, j = start_loc
    left = pipes.get((i, j - 1), '')
    right = pipes.get((i, j + 1), '')
    up = pipes.get((i - 1, j), '')
    down = pipes.get((i + 1, j), '')
    symbol = ''
    if left in {'-', 'F', 'L'} and right in {'-', 'J', '7'}:
        symbol = '-'
    elif left in {'-', 'F', 'L'} and down in {'|', 'L', 'J'}:
        symbol = '7'
    elif left in {'-', 'F', 'L'} and up in {'|', 'F', '7'}:
        symbol = 'J'
    elif right in {'-', 'J', '7'} and down in {'|', 'L', 'J'}:
        symbol = 'F'
    elif right in {'-', 'J', '7'} and up in {'|', 'F', '7'}:
        symbol = 'L'
    elif up in {'|', 'F', '7'} and down in {'|', 'L', 'J'}:
        symbol = '|'
    return symbol


# %%
def parse(s):
    lines = s.strip().split('\n')
    i = 0
    pipes = dict()
    start_loc = None
    for line in lines:
        j = 0
        for symbol in line:
            pipes[(i, j)] = symbol
            if symbol == 'S':
                start_loc = (i, j)
            j += 1
        i += 1
    start_symbol = start_loc_to_pipe(pipes, start_loc)
    pipes[start_loc] = start_symbol
    return pipes, start_loc


# %%
def part1_sol(s):
    pipes, start_loc = parse(s)
    l, r, u, d = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    possible_first_steps = [s for s in [
        step(
            pipes[start_loc], (start_loc[0] - loc[0], start_loc[1] - loc[1])
        )(*start_loc) for loc in [l, r, u, d]] if s != start_loc]
    first_step_loc = possible_first_steps[0]
    # print(f'First step {first_step_loc}')
    count = 1
    prev_loc = start_loc
    loc = first_step_loc
    visited = {start_loc}
    while loc not in visited:
        # print(pipes[loc], loc)
        visited.add(loc)
        next_loc = step(pipes[loc], prev_loc)(*loc)
        prev_loc = loc
        loc = next_loc
        count += 1

    return int(count/2)


# %%
def part2_sol(s, do_plot=False):
    pipes, start_loc = parse(s)
    l, r, u, d = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    possible_first_steps = [s for s in [
        step(
            pipes[start_loc], (start_loc[0] - loc[0], start_loc[1] - loc[1])
        )(*start_loc) for loc in [l, r, u, d]] if s != start_loc]
    first_step_loc = possible_first_steps[0]
    # print(f'First step {first_step_loc}')
    prev_loc = start_loc
    loc = first_step_loc
    visited = [start_loc]
    while loc not in visited:
        # print(pipes[loc], loc)
        visited.append(loc)
        next_loc = step(pipes[loc], prev_loc)(*loc)
        prev_loc = loc
        loc = next_loc

    n_rows = max(i for i, _ in pipes.keys()) + 1
    n_cols = max(j for _, j in pipes.keys()) + 1
    # Topological approach: points inside the curve will have an
    # odd number of crossings before reaching boundary.
    # To deal with flows between pipes expand the grid into an image
    # and choose a corner point for test in each tile so that it won't run
    # down a line of pipes.

    img = np.zeros((n_rows*3, n_cols*3), dtype=bool)
    tiles = {k: np.array(v).astype(bool) for k, v in {
        '-': [[0,0,0], [1,1,1], [0,0,0]],
        '|': [[0,1,0], [0,1,0], [0,1,0]],
        'F': [[0,0,0], [0,1,1], [0,1,0]],
        'L': [[0,1,0], [0,1,1], [0,0,0]],
        'J': [[0,1,0], [1,1,0], [0,0,0]],
        '7': [[0,0,0], [1,1,0], [0,1,0]]
    }.items()}
    for i, j in visited:
        img[3*i:3*i+3, 3*j:3*j+3] = tiles[pipes[(i,j)]]

    # Crossing test using top right corner of each tile
    inside_test = (
            (np.cumsum(img, axis=0)[::3, 2::3] % 2 == 1)  # Crossing test
            & (~img[1::3, 1::3])  # Not on curve
    )
    inside = np.sum(inside_test)

    if do_plot:
        img[np.kron(inside_test, np.ones((3,3), dtype=bool)).astype(bool)] = 0.5
        plt.imshow(img)
        plt.show()

    return inside


# %%
print(part1_sol(test_in_part1), test_out_part1)

# %%
print(part2_sol(test_in_part2), test_out_part2)

# %%
in_str = read_input(10)

# %%
part1 = part1_sol(in_str)
print(part1)

# %%
part2 = part2_sol(in_str, True)
print(part2)

# %%
