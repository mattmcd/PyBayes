import os
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Table, func, MetaData, select
from sqlalchemy.exc import NoSuchTableError

DATA_DIR = os.path.expanduser('~/Work/Data/AoC2022')


def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2022_day{day:02}.txt'), 'r') as f:
        return f.read()


def day_01():
    # Short version
    vals = sorted(
        [sum([int(x) for x in line.split('\n')])
         for line in read_input(1).strip().split('\n\n')]
    )[::-1]
    return vals[0], sum(vals[:3])


class Day01:
    def __init__(self):
        self.input = read_input(1)
        self.inventory = self.parse()

    def parse(self):
        carried = self.input.strip().split('\n\n')
        inventory = {
            i: [int(x) for x in el.split('\n')] for i, el in enumerate(carried)
        }
        return inventory

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

    def __repr__(self):
        return f'Part 1: {self.part1()}\nPart 2: {self.part2()}'


class Day01Pandas(Day01):
    def __init__(self):
        super(Day01Pandas, self).__init__()
        self.inventory: pd.DataFrame = self.parse_pandas()

    def parse_pandas(self) -> pd.DataFrame:
        inventory = Day01.parse(self)
        # Not part of puzzle but an alternative representation
        n_elves = len(inventory)
        df = pd.DataFrame(
            pd.concat([pd.Series(inventory[i], name=i) for i in range(n_elves)], keys=range(n_elves)),
        ).reset_index()
        df.columns = ['elf_id', 'item', 'value']
        return df

    def part1(self):
        return self.inventory.groupby('elf_id')['value'].sum().max()

    def part2(self):
        return self.inventory.groupby('elf_id')['value'].sum().sort_values(ascending=False).cumsum().iloc[2]


class Day01Sqla(Day01Pandas):
    def __init__(self):
        super(Day01Sqla, self).__init__()
        self.engine = create_engine(f'sqlite:///{os.path.join(DATA_DIR, "aoc2022.db")}')
        self.metadata = MetaData()
        self.inventory = self.parse_sqla()

    def parse_sqla(self):
        try:
            tb_inventory = Table('Day01', self.metadata, autoload=True, autoload_with=self.engine)
        except NoSuchTableError:
            inventory = self.inventory  # pd.DataFrame from parent class
            inventory.to_sql('Day01', self.engine, if_exists='fail', index=False)
            tb_inventory = Table('Day01', self.metadata, autoload=True, autoload_with=self.engine)
        return tb_inventory

    def sum_by_elf(self):
        return select(
            func.sum(self.inventory.c.value).label('sum_value')
        ).group_by('elf_id').order_by(func.sum(self.inventory.c.value).desc())

    def part1(self):
        query = self.sum_by_elf().limit(1)
        # print(str(query))
        return pd.read_sql(query, self.engine)['sum_value'][0]

    def part2(self):
        query_1 = self.sum_by_elf().limit(3).cte('query_1')
        query = select(func.sum(query_1.c.sum_value).label('sum_value'))
        # print(str(query))
        return pd.read_sql(query, self.engine)['sum_value'][0]
