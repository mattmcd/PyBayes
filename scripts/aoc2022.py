import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser('~/Work/Data/AoC2022')


def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2022_day{day:02}.txt'), 'r') as f:
        return f.read()


class Day01:
    def __init__(self):
        self.input = read_input(1)
        self._inventory = self.parse()

    def parse(self):
        carried = self.input.strip().split('\n\n')
        inventory = {
            i: [int(x) for x in el.split('\n')] for i, el in enumerate(carried)
        }
        return inventory

    @property
    def inventory(self):
        return self._inventory

    @property
    def sum_carried(self):
        sum_carried = {k: sum(v) for k, v in self.inventory.items()}
        return sum_carried

    def part1(self):
        return max(self.sum_carried.values())

    def part2(self):
        return sum(sorted(self.sum_carried.values())[-3:])

    def max_carry_elf(self):
        # Not part of the puzzle but a nice way of inverting the map
        return max(self.sum_carried, key=self.sum_carried.get)


class Day01Pandas(Day01):
    @property
    def inventory(self):
        inventory = Day01.parse(self)
        # Not part of puzzle but an alternative representation
        n_elves = len(inventory)
        df = pd.DataFrame(
            pd.concat([pd.Series(inventory[i], name=i) for i in range(n_elves)], keys=range(n_elves)),
        ).reset_index()
        df.columns = columns=['elf_id', 'item', 'value']
        return df

    def part1(self):
        return self.inventory.groupby('elf_id')['value'].sum().max()

    def part2(self):
        return self.inventory.groupby('elf_id')['value'].sum().sort_values(ascending=False).cumsum().iloc[2]
