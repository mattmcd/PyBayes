# %%
from startup import np, pd, plt, sns
from mattmcd.aoc2025 import Reader
from functools import reduce
# %%
test_reader = Reader.day02(True)
reader = Reader.day02()

# %%
df_test = test_reader.data
df = reader.data

# %%
repeated_digits = np.array([int(str(x) * 2 ) for x in range(0, 100_000)])

# %%
def part_01_fun(df, digit_list):
    return df.apply(
        lambda x: digit_list[(digit_list>= x['start']) & (digit_list<= x['end'])].sum(),
        axis=1
    )

# %%
print(part_01_fun(df_test, repeated_digits).sum())
print(part_01_fun(df, repeated_digits).sum())

# %%
# Part 2: any sequence of repeats 2 or more times is valid
all_repeated_digits = np.array(
    list(set([int(str(x) * k ) for x in range(0, 100_000) for k in range(2, 11)]))
)

# %%
print(part_01_fun(df_test, all_repeated_digits).sum())
print(part_01_fun(df, all_repeated_digits).sum())