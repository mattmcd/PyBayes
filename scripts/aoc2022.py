import os
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Table, func, MetaData, select
from sqlalchemy.exc import NoSuchTableError

DATA_DIR = os.path.expanduser('~/Work/Data/AoC2022')

ENGINE = create_engine(f'sqlite:///{os.path.expanduser("~/Work/Data/AoC2022/aoc2022.db")}')

METADATA = MetaData()


def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2022_day{day:02}.txt'), 'r') as f:
        return f.read()


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


class Day01Pandas(Day01):
    def __init__(self):
        super(Day01Pandas, self).__init__()
        self.inventory = self.parse_pandas()

    def parse_pandas(self):
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


class Day01Sqla(Day01Pandas):
    def __init__(self):
        super(Day01Sqla, self).__init__()
        self.inventory = self.parse_sqla()

    def parse_sqla(self):
        try:
            tb_inventory = Table('Day01', METADATA, autoload=True, autoload_with=ENGINE)
        except NoSuchTableError:
            inventory = self.inventory
            inventory.to_sql('Day01', ENGINE, if_exists='fail', index=False)
            tb_inventory = Table('Day01', METADATA, autoload=True, autoload_with=ENGINE)
        return tb_inventory

    def part1(self):
        query = select(
            func.sum(self.inventory.c.value).label('sum_value')
        ).group_by('elf_id').order_by(func.sum(self.inventory.c.value).desc()).limit(1)
        print(str(query))
        return pd.read_sql(query, ENGINE)['sum_value'][0]

    def part2(self):
        query = select(
            func.sum(self.inventory.c.value).label('sum_value')
        ).group_by('elf_id').order_by(func.sum(self.inventory.c.value).desc()).limit(3)
        # query = select(func.sum(select(query_1.c.sum_value)).label('sum_value'))
        # print(str(query))
        return pd.read_sql(query, ENGINE)['sum_value'].sum()