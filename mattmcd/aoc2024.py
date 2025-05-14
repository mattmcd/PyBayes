import os
import pandas as pd
from io import StringIO
from sqlalchemy import MetaData
from .io import pg_engine


DATA_DIR = os.path.expanduser('~/Work/Data/AoC2024')

def read_input(day):
    with open(os.path.join(DATA_DIR, f'aoc2024_day{day:02}.txt'), 'r') as f:
        return f.read()

class Reader:
    def __init__(self, day, data, is_test=False):
        self.day = day
        self.data : pd.DataFrame = data
        self.engine = pg_engine()
        self.metadata = MetaData()
        self.is_test = is_test

    def to_db(self):
        test_str  = '_test' if self.is_test else ''
        self.data.to_sql(f'day{self.day:02}{test_str}', con=self.engine, schema='aoc_2024', if_exists='replace')

    @classmethod
    def day01(cls):
        data = [{'left_col': r.split()[0], 'right_col': r.split()[1]} for r in read_input(1).strip().split('\n')]
        df = pd.DataFrame(data).astype(int)
        return cls(1, df)

    @classmethod
    def day02(cls):
        data = []
        record = 0
        for line in read_input(2).strip().split('\n'):
            position = 0
            for level in line.split():
                data.append({'record': record, 'position': position, 'level': level})
                position += 1
            record += 1
        df = pd.DataFrame(data).astype(int)
        return cls(2, df)

    @classmethod
    def day03(cls):
        data = read_input(3).strip().split('\n')
        df = pd.DataFrame({'program': data})
        return cls(3, df)

    @classmethod
    def day04(cls, is_test=False):
        if is_test:
            lines="""MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX""".strip().split('\n')

        else:
            lines = read_input(4).strip().split('\n')
        row = 0
        data = []
        for line in lines:
            col = 0
            for c in line:
                data.append({'row': row, 'col': col, 'letter': c})
                col += 1
            row += 1
        df = pd.DataFrame(data)
        return cls(4, df, is_test)