import os
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Table, func, MetaData, select, case, and_, or_, literal
from sqlalchemy.exc import NoSuchTableError
from abc import ABC
import re
from collections import defaultdict


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
        c_total = self.calc_total(c_response_val, c_win)
        tb_score = select([tb.c.player_1, tb.c.player_2, c_response, c_response_val, c_win, c_total]).cte('score')
        return tb_score

    def calc_total(self, c_response_val, c_win):
        c_total = (c_response_val + c_win).label('total_value')
        return c_total

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


class Day02Group(Day02):
    def __init__(self):
        super(Day02Group, self).__init__()
        tb_r = self.strategy
        self.strategy = select(
            [
                tb_r.c.player_1,
                tb_r.c.player_2,
                func.count().label('multiplier')
            ]
        ).group_by(
            tb_r.c.player_1, tb_r.c.player_2
        ).cte('grouped')

    def calc_total(self, c_response_val, c_win):
        c_total = (self.strategy.c.multiplier*(c_response_val + c_win)).label('total_value')
        return c_total


def day_02():
    # Short version
    games = [line.split() for line in read_input(2).strip().split('\n')]
    p2_resp_part1 = lambda p1, x: 'ABC'['XYZ'.index(x)]
    p2_resp_part2 = lambda p1, x: 'ABC'[('ABC'.index(p1) + 'YZX'.index(x)) % 3]
    c = np.array([3, 6, 0])
    x = np.array([1, 0, 0])
    win_score = lambda p1, p2: c @ np.roll(x, ('ABC'.index(p2) - 'ABC'.index(p1)) % 3)
    total_score = lambda resp: lambda p1, x: 'ABC'.index(resp(p1, x)) + 1 + win_score(p1, resp(p1, x))
    total_score_p1 = total_score(p2_resp_part1)
    part_1 = sum(total_score_p1(p1, x) for p1, x in games)
    total_score_p2 = total_score(p2_resp_part2)
    part_2 = sum(total_score_p2(p1, x) for p1, x in games)
    return part_1, part_2


def day_03():
    bags = read_input(3).strip().split('\n')

    def bag_diff(bag):
        n = len(bag)
        p1 = set(bag[:n//2])
        p2 = set(bag[n//2:])
        return list(p1.intersection(p2))[0]

    def item_val(x):
        val = 1 + (ord(x) -ord('A'))
        if val > 26:
            val = 1 + ord(x) - ord('a')
        else:
            val += 26
        return val

    part_1 = sum(item_val(x) for x in [bag_diff(bag) for bag in bags])

    n = len(bags)
    n_chunk = n // 3
    joined = list()
    for i in range(n_chunk):
        joined.append(bags[(3*i): (3*(i+1))])

    def unique_in_bag_group(bags_joined):
        res = set(bags_joined[0])
        for bag in bags_joined[1:]:
            res = res.intersection(bag)
        return list(res)[0]

    part_2 = sum(item_val(unique_in_bag_group(x)) for x in joined)

    return part_1, part_2


class Day03(AbstractDay):
    def __init__(self):
        self.engine = create_engine(f'sqlite:///{os.path.join(DATA_DIR, "aoc2022.db")}')
        self.metadata = MetaData()
        self.table_name = 'Day03'
        self.inventory = self.parse_sqla()

    def parse_sqla(self):
        try:
            tb_inventory = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)
        except NoSuchTableError:
            inventory = read_input(3).strip().split('\n')
            n_elves = len(inventory)
            df = pd.DataFrame(
                pd.concat([pd.Series(list(inventory[i]), name=i) for i in range(n_elves)], keys=range(n_elves)),
            ).reset_index()
            df.columns = ['elf_id', 'item_id', 'item']
            df.to_sql(self.table_name, self.engine, if_exists='fail', index=False)
            tb_inventory = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)

        return tb_inventory

    @staticmethod
    def item_val(x):
        val = 1 + (ord(x) - ord('A'))
        if val > 26:
            val = 1 + ord(x) - ord('a')
        else:
            val += 26
        return val

    def part1(self):
        tb = self.inventory
        tb_c = select(
            [tb.c.elf_id, func.count(tb.c.item_id).label('count')]
        ).group_by(tb.c.elf_id).cte('item_count')
        tb_h = select(
            [tb.c.elf_id, tb.c.item_id, case((tb.c.item_id < tb_c.c.count / 2, 0), else_=1).label('bag_half')]
        ).select_from(
            tb.join(tb_c, tb.c.elf_id == tb_c.c.elf_id)
        ).cte('bag_half')
        tb_i = select(
            [tb.c.elf_id, tb.c.item]
        ).select_from(
            tb.join(
                tb_h,
                and_(tb.c.elf_id == tb_h.c.elf_id, tb.c.item_id == tb_h.c.item_id)
            )
        ).group_by(
            tb.c.elf_id, tb.c.item
        ).having(func.min(tb_h.c.bag_half) != func.max(tb_h.c.bag_half)).cte('item_both')

        return pd.read_sql(select(tb_i.c.item), self.engine)['item'].map(self.item_val).sum()

    def part2(self):
        tb = self.inventory
        tb_g = select(
            [tb.c.elf_id, (tb.c.elf_id / 3).label('elf_group'), tb.c.item]
        ).distinct().cte('elf_group')
        tb_i = select(
            [tb_g.c.elf_group, tb_g.c.item]
        ).group_by(
            tb_g.c.elf_group, tb_g.c.item
        ).having(func.count(tb_g.c.item) == 3).cte('item_group')

        return pd.read_sql(select(tb_i.c.item), self.engine)['item'].map(self.item_val).sum()


def day_04():
    ranges = [{'elf_id': i, 'group_id': i//2, 'lo': int(el[0]), 'hi': int(el[1])}
              for i, el in enumerate(re.findall(r'(\d+)-(\d+)', read_input(4)))]
    df = pd.DataFrame(ranges)
    df_p = df.join(
        df.groupby('group_id')[['lo', 'hi']].shift(1), lsuffix='l', rsuffix='r'
    ).dropna()

    def contains(a1, a2, b1, b2):
        return ((a1 <= b1) and (b2 <= a2)) or ((b1 <= a1) and (a2 <= b2))

    df_p['contains'] = df_p.apply(
        lambda x: contains(x['lol'], x['hil'], x['lor'], x['hir']), axis=1)

    df_p['overlap'] = df_p.apply(
        lambda x: x['lor'] <= x['hil'] and x['lol'] <= x['hir'], axis=1
    )

    return df_p.contains.sum(), df_p.overlap.sum()


class Day04(AbstractDay):
    def __init__(self):
        self.engine = create_engine(f'sqlite:///{os.path.join(DATA_DIR, "aoc2022.db")}')
        self.metadata = MetaData()
        self.table_name = 'Day04'
        self.ranges = self.parse_sqla()

    def parse_sqla(self):
        try:
            tb_ranges = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)
        except NoSuchTableError:
            ranges = [{'elf_id': i, 'group_id': i // 2, 'lo': int(el[0]), 'hi': int(el[1])}
                      for i, el in enumerate(re.findall(r'(\d+)-(\d+)', read_input(4)))]
            df = pd.DataFrame(ranges)
            df.to_sql(self.table_name, self.engine, if_exists='fail', index=False)
            tb_ranges = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)

        return tb_ranges

    @staticmethod
    def contains(a1, a2, b1, b2):
        return case((or_(and_((a1 <= b1), (b2 <= a2)), and_((b1 <= a1), (a2 <= b2))), 1), else_=0)

    @staticmethod
    def overlaps(a1, a2, b1, b2):
        return case((and_(b1 <= a2, a1 <= b2), 1), else_=0)

    def create_interval_table(self):
        tb = self.ranges
        p = {'partition_by': [tb.c.group_id]}
        tb_i = select(
            [
                tb.c.group_id,
                func.first_value(tb.c.lo).over(**p).label('a1'),
                func.first_value(tb.c.hi).over(**p).label('a2'),
                func.last_value(tb.c.lo).over(**p).label('b1'),
                func.last_value(tb.c.hi).over(**p).label('b2'),
            ]
        ).distinct().cte('intervals')
        return tb_i

    def part1(self):
        tb_i = self.create_interval_table()
        tb_r = select(
            [func.sum(self.contains(tb_i.c.a1, tb_i.c.a2, tb_i.c.b1, tb_i.c.b2)).label('result')]
        ).cte('result')
        return pd.read_sql(select(tb_r.c.result), self.engine)['result'][0]

    def part2(self):
        tb_i = self.create_interval_table()
        tb_r = select(
            [func.sum(self.overlaps(tb_i.c.a1, tb_i.c.a2, tb_i.c.b1, tb_i.c.b2)).label('result')]
        ).cte('result')
        return pd.read_sql(select(tb_r.c.result), self.engine)['result'][0]


def day_05_parse(do_test=False):
    if do_test:
        in_str = """    [D]    
[N] [C]    
[Z] [M] [P]
 1   2   3 

move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2
        """
    else:
        in_str = read_input(5).rstrip()

    move_lines = re.findall(r'move (\d+) from (\d+) to (\d+)', in_str)
    lines = in_str.split('\n')
    col_lines = []
    n_col = 0
    for i, line in enumerate(lines):
        col_ind_line = re.findall(r'^(\s+\d+\s+)+$', line)
        if col_ind_line == []:
            col_lines.append(line)
        else:
            n_col = int(col_ind_line[0].strip())
            break

    cols = defaultdict(str)
    for line in col_lines:
        for i in range(0, n_col):
            cols[i+1] += line[1+4*i]

    for i in range(0, n_col):
        cols[i+1] = cols[i+1].strip()

    moves = [{'src': int(s), 'dst': int(d), 'num': int(c)} for c, s, d in move_lines]

    return cols, moves


def day_05(do_test=False):
    cols, moves = day_05_parse(do_test)

    def part_1():
        for move in moves:
            cols[move['dst']] = cols[move['src']][:move['num']][::-1] + cols[move['dst']]
            cols[move['src']] = cols[move['src']][move['num']:]

        out_str = ''
        for i in range(len(cols)):
            out_str += cols[i+1][0]
        return out_str

    def part_2():
        for move in moves:
            cols[move['dst']] = cols[move['src']][:move['num']] + cols[move['dst']]
            cols[move['src']] = cols[move['src']][move['num']:]

        out_str = ''
        for i in range(len(cols)):
            out_str += cols[i + 1][0]
        return out_str

    p1 = part_1()
    cols, moves = day_05_parse(do_test)
    p2 = part_2()

    return p1, p2


def day_05_sqla(do_test=False):
    cols, moves = day_05_parse(do_test)
    s0 = ''.join([str(k) + v for k, v in cols.items()])

    engine = create_engine('sqlite:///:memory')
    metadata = MetaData()

    def update_state(state, src, dst, num, rev):
        # print(state, src, dst, num, rev)
        i_src = state.index(str(src))
        to_move = state[i_src+1:i_src+num+1]
        # print(to_move)
        state = state.replace(str(src) + to_move, str(src))
        # print(state)
        moved = to_move[::-1] if rev else to_move
        state = state.replace(str(dst), str(dst) + moved)
        return state

    with engine.connect() as conn:
        # Add UDF update state
        conn.connection.create_function('update_state', 5, update_state)
        try:
            rows_added = pd.DataFrame(moves).to_sql('moves', conn)
            # print(f'Rows added {rows_added}')
        except ValueError as ex:
            print(ex)
        tb_m = Table('moves', metadata, autoload_with=conn)

        q_base = select(
            [
                literal(0).label('step'),
                literal(s0).label('state_p1'),
                literal(s0).label('state_p2')
            ]).cte('base', recursive=True)
        q_rest = select(
            [
                q_base.c.step + 1,
                func.update_state(q_base.c.state_p1, tb_m.c.src, tb_m.c.dst, tb_m.c.num, True),
                func.update_state(q_base.c.state_p2, tb_m.c.src, tb_m.c.dst, tb_m.c.num, False),
            ]
        ).select_from(
            q_base.join(tb_m, q_base.c.step == tb_m.c.index)
        )
        q_full = q_base.union_all(q_rest)

        df = pd.read_sql(select(q_full).order_by(q_full.c.step.desc()).limit(1), conn)
        tb_m.drop(conn)

    part_1 = ''.join([x[0] for x in re.findall(r'\d+(\D*)', df.state_p1[0])])
    part_2 = ''.join([x[0] for x in re.findall(r'\d+(\D*)', df.state_p2[0])])

    return part_1, part_2


def day_06(do_test=False):
    stream = 'mjqjpqmgbljsphdztnvjfqwrcgsmlb' if do_test else read_input(6).strip()
    df = pd.DataFrame(list(stream), columns=['lag_0'])
    for i in range(1,4):
        df.loc[:, f'lag_{i}'] = df['lag_0'].shift(i)
    df.dropna(inplace=True)
    df.loc[:, 'n_diff'] = df.apply(lambda x: len(set(x.to_list())), axis=1)

    part_1 = df.index[df.n_diff ==4][0] + 1
    df.drop(columns={'n_diff'}, inplace=True)
    for i in range(4, 14):
        df.loc[:, f'lag_{i}'] = df['lag_0'].shift(i)
    df.dropna(inplace=True)
    df.loc[:, 'n_diff'] = df.apply(lambda x: len(set(x.to_list())), axis=1)
    part_2 = df.index[df.n_diff == 14][0] + 1

    return part_1, part_2


def day_06_sqla(do_test=False):
    stream = 'mjqjpqmgbljsphdztnvjfqwrcgsmlb' if do_test else read_input(6).strip()
    engine = create_engine('sqlite:///:memory')
    metadata = MetaData()
    tb_name = 'stream_test' if do_test else 'stream'
    pd.DataFrame(list(stream), columns=['lag_0']).to_sql(tb_name, engine)
    tb = Table(tb_name, metadata, autoload_with=engine)

    def lag_table(n):
        tb_l = select(
            [tb.c.index, tb.c.lag_0] + [
                func.lag(tb.c.lag_0, i).over(order_by=tb.c.index).label(f'lag_{i}') for i in range(1, n)
            ]
        ).cte(f'lagged_{n}')
        return tb_l

    n = 4
    tb_pt1 = lag_table(n)
    part_1 = pd.read_sql(select(tb_pt1).where(tb_pt1.c[f'lag_{n-1}'].isnot(None)), engine)
    p1 = part_1['index'][part_1.apply(lambda x: len(set(x.to_list())) == n+1, axis=1)].iloc[0] + 1

    n = 14
    tb_pt2 = lag_table(n)
    part_2 = pd.read_sql(select(tb_pt2).where(tb_pt2.c[f'lag_{n-1}'].isnot(None)), engine)
    p2 = part_2['index'][part_2.apply(lambda x: len(set(x.to_list())) == n+1, axis=1)].iloc[0] + 1

    tb.drop(engine)

    return p1, p2


def day_06_sqla2(do_test=False):
    stream = 'mjqjpqmgbljsphdztnvjfqwrcgsmlb' if do_test else read_input(6).strip()
    engine = create_engine('sqlite:///:memory')
    metadata = MetaData()
    tb_name = 'stream_test' if do_test else 'stream'
    pd.DataFrame(list(stream), columns=['value']).to_sql(tb_name, engine)
    tb = Table(tb_name, metadata, autoload_with=engine)

    def lag_table(n):
        q_base = select([literal(0).label('lag'), tb.c.index, tb.c.value]).cte('base', recursive=True)
        q_rest = select([q_base.c.lag + 1, q_base.c.index + 1, q_base.c.value]).where(q_base.c.lag < n-1)
        q_full = q_base.union_all(q_rest)
        tb_l = select(
            [q_full.c.index, func.count(q_full.c.value.distinct())]
        ).group_by(
            q_full.c.index
        ).having(
            func.count(q_full.c.value.distinct()) == n
        ).limit(1).cte(f'lagged_{n}')
        return tb_l

    n = 4
    p1 = pd.read_sql(select(lag_table(n).c.index + 1), engine).values[0][0]

    n = 14
    p2 = pd.read_sql(select(lag_table(n).c.index + 1), engine).values[0][0]

    tb.drop(engine)

    return p1, p2


class Day07:

    def __init__(self, do_test=False):
        self.dir_id = None
        self.tb_dir = None
        self.tb_file = None
    def get_dir_id(self, name, parent_id):
        dir_id = None
        for k, v in self.tb_dir.items():
            if v['name'] == name and v['parent_id'] == parent_id:
                dir_id = k
                break
        return dir_id
    def parse(self, line):
        if '$ cd ' in line:
            name = line.split('$ cd ')[1]
            if name == '..':
                # print(f'{line} -> go up')
                self.dir_id = self.tb_dir[self.dir_id]['parent_id']
            else:
                # print(f'{line} -> go to {name}')
                if self.dir_id == None:
                    self.dir_id = 0
                    self.tb_dir = {self.dir_id: {'name': name, 'parent_id': None}}
                else:
                    self.dir_id = self.get_dir_id(name, self.dir_id)
        elif '$ ls' in line:
            pass
            # print(f'{line} -> list files')
        elif 'dir ' in line:
            # print(f'{line} -> dir {line.split("dir ")[1]}')
            max_id = max(self.tb_dir.keys())
            self.tb_dir[max_id + 1] = {'name': line.split('dir ')[1], 'parent_id': self.dir_id}
        else:
            s_size, name = line.split(' ')
            size = int(s_size)
            # print(f'{line} -> file {name} {size}')
            if self.tb_file is None:
                self.tb_file = {0: {'name': name, 'size': size, 'parent_id': self.dir_id}}
            else:
                max_id = max(self.tb_file.keys())
                self.tb_file[max_id + 1] = {'name': name, 'size': size, 'parent_id': self.dir_id}
        # print(self.tb_dir)
        # print(self.tb_file)


def day_07(do_test=False):
    logs = """$ cd /
$ ls
dir a
14848514 b.txt
8504156 c.dat
dir d
$ cd a
$ ls
dir e
29116 f
2557 g
62596 h.lst
$ cd e
$ ls
584 i
$ cd ..
$ cd ..
$ cd d
$ ls
4060174 j
8033020 d.log
5626152 d.ext
7214296 k""" if do_test else read_input(7).strip()
    engine = create_engine('sqlite:///:memory')
    metadata = MetaData()
    tb_dir = 'dir_test' if do_test else 'dir'
    tb_file = 'file_test' if do_test else 'file'
    parser = Day07()
    [parser.parse(line) for line in logs.split('\n')]
    # print(parser.tb_dir)
    # print(parser.tb_file)
    df_dir = pd.DataFrame.from_dict(parser.tb_dir, orient='index')
    # df_dir.parent_id = df_dir.parent_id.astype(int)
    df_file = pd.DataFrame.from_dict(parser.tb_file, orient='index')
    return df_dir, df_file


class Day08(AbstractDay):
    def __init__(self, do_test=False):
        self.engine = create_engine(f'sqlite:///:memory')
        self.metadata = MetaData()
        self.table_name = 'Day08_test' if do_test else 'Day08'
        self.do_test = do_test
        self.trees = self.parse_sqla()

    def parse_sqla(self):
        try:
            tb_trees = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)
        except NoSuchTableError:
            in_str = """30373\n25512\n65332\n33549\n35390""" if self.do_test else read_input(8).strip()
            lines = in_str.split('\n')
            res = list()
            for i, line in enumerate(lines):
                for j in range(len(line)):
                    res.append({'row': i, 'col': j, 'value': int(line[j])})
            df = pd.DataFrame(res)
            df.to_sql(self.table_name, self.engine, if_exists='fail', index=False)
            tb_trees = Table(self.table_name, self.metadata, autoload=True, autoload_with=self.engine)

        return tb_trees

    def part1(self):
        tb = self.trees
        seen_left = func.coalesce(
            tb.c.value > func.max(tb.c.value).over(
                partition_by=tb.c.row, order_by=tb.c.col, rows=(None, -1)
            ), True).label('left')
        seen_right = func.coalesce(
            tb.c.value > func.max(tb.c.value).over(
                partition_by=tb.c.row, order_by=tb.c.col.desc(), rows=(None, -1)
            ), True).label('right')
        seen_up = func.coalesce(
            tb.c.value > func.max(tb.c.value).over(
                partition_by=tb.c.col, order_by=tb.c.row, rows=(None, -1)
            ), True).label('up')
        seen_down = func.coalesce(
            tb.c.value > func.max(tb.c.value).over(
                partition_by=tb.c.col, order_by=tb.c.row.desc(), rows=(None, -1)
            ), True).label('down')
        tb_s = select(
            [tb, case((or_(seen_left, seen_right, seen_up, seen_down), 1), else_=0).label('seen')]
        ).cte('seen')
        df = pd.read_sql(func.sum(tb_s.c.seen), self.engine)
        return df

    def part2(self):
        tb = self.trees
        seen_left = func.coalesce(
            tb.c.value >= func.last_value(tb.c.value).over(
                partition_by=tb.c.row, order_by=tb.c.col, rows=(None, -1)
            ), True).label('left')
        tb_s = select([tb, seen_left]).cte(
            'seen')
        df = pd.read_sql(select(tb_s), self.engine)
        return df
