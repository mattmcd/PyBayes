import os
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Table, func, MetaData, select, case, and_
from sqlalchemy.exc import NoSuchTableError
from abc import ABC

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


class AbstractDay(ABC):
    def part1(self):
        return '????'

    def part2(self):
        return '????'

    def __repr__(self):
        return f'Part 1: {self.part1()}\nPart 2: {self.part2()}'


class Day01(AbstractDay):
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


class Day02(AbstractDay):
    def __init__(self):
        self.engine = create_engine(f'sqlite:///{os.path.join(DATA_DIR, "aoc2022.db")}')
        self.metadata = MetaData()
        self.table_name = 'Day02'
        self.strategy = self.parse_sqla()

    def parse_sqla(self):
        try:
            tb_strategy = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)
        except NoSuchTableError:
            strategy = pd.DataFrame(
                [line.split() for line in read_input(2).strip().split('\n')],
                columns=['player_1', 'player_2']
            )
            strategy.to_sql(self.table_name, self.engine, if_exists='fail', index=False)
            tb_strategy = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)
        return tb_strategy

    def score_table(self, c_response):
        response_value = {'A': 1, 'B': 2, 'C': 3}
        win_value = {('A', 'B'): 6, ('A', 'C'): 0, ('B', 'C'):  6, ('B', 'A'): 0, ('C', 'A'): 6, ('C', 'B'): 0}
        tb = self.strategy
        c_response_val = case([(c_response == k, v) for k, v in response_value.items()]).label('response_value')
        c_win = case(
            [( and_(tb.c.player_1 == k[0], c_response == k[1]), v)
             for k, v in win_value.items()],
            else_= 3
        ).label('win_value')
        c_total = (c_response_val + c_win).label('total_value')
        tb_score = select([tb.c.player_1, tb.c.player_2, c_response, c_response_val, c_win, c_total]).cte('score')
        return tb_score

    def calc_result(self, c_response, do_test=False):
        tb_score = self.score_table(c_response)
        if do_test:
            return pd.read_sql(select(tb_score), self.engine)
        else:
            query = select(func.sum(tb_score.c.total_value).label('result'))
            return pd.read_sql(query, self.engine)['result'][0]

    def part1(self, do_test=False):
        response_rule = {'X': 'A', 'Y': 'B', 'Z': 'C'}
        c_response = case([(self.strategy.c.player_2 == k, v) for k, v in response_rule.items()]).label('response')
        return self.calc_result(c_response)

    def part2(self, do_test=False):
        p1 = self.strategy.c.player_1
        response_rule = {
            'X': case([(p1 == k, v) for k, v in {'A': 'C', 'B': 'A', 'C': 'B'}.items()]),
            'Y': case([(p1 == k, v) for k, v in {'A': 'A', 'B': 'B', 'C': 'C'}.items()]),
            'Z': case([(p1 == k, v) for k, v in {'A': 'B', 'B': 'C', 'C': 'A'}.items()])}
        c_response = case([(self.strategy.c.player_2 == k, v) for k, v in response_rule.items()]).label('response')
        return self.calc_result(c_response)

