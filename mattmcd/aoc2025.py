import os
import pandas as pd
from io import StringIO
from sqlalchemy import MetaData
from .io import pg_engine


DATA_DIR = os.path.expanduser('~/Work/Data/AoC2025')

def read_input(day, is_test=False):
    test_str  = '_test' if is_test else ''
    with open(os.path.join(DATA_DIR, f'aoc2025_day{day:02}{test_str}.txt'), 'r') as f:
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
        self.data.to_sql(
            f'day{self.day:02}{test_str}', con=self.engine, schema='aoc_2025', if_exists='replace'
        )

    @classmethod
    def day01(cls, is_test=False):
        data = [
            {
                'rotation_label': r,
                'rotation': int(r.replace('L', '-').replace('R', ''))
            } for r in read_input(1, is_test).strip().split('\n')]
        df = pd.DataFrame(data)
        return cls(1, df)
