# %%
from startup import np, pd, plt, sns
from mattmcd.aoc2025 import Reader
from functools import reduce
# %%
test_reader = Reader.day01(True)
reader = Reader.day01()

# %%
df_test = test_reader.data
df = reader.data

# %%
def rot_fun(acc, el):
    return (acc + el) % 100

# %%
def part_01_fun(df):
    positions = [50]
    for x in df.rotation:
        positions.append(rot_fun(positions[-1], x))
    return positions

# %%
positions_test = part_01_fun(df_test)
print(positions_test)

# %%
positions = part_01_fun(df)
part_01_sol = np.sum(np.array(positions) == 0)
print(part_01_sol)



# %%
def cross_fun(acc, el):
    return acc + el

# %%
def part_02_fun(df):
    positions = [50]
    for x in df.rotation:
        positions.append(positions[-1] +  x)
    pos = np.array(positions) // 100
    crosses = np.where(pos[1:] % 100 != 0, np.abs(pos[1:] - pos[:-1]), 0)
    return positions, crosses

# %%
def part_02_brute_force(df):
    clicks = [1] * 50
    for x in df.rotation:
        clicks += [1] * x if x > 0 else [-1] * (-x)
    return np.sum(np.cumsum(clicks) % 100 == 0)

# %%
positions, crosses = part_02_fun(df)

# %%
fig, ax = plt.subplots()
ax.plot(np.array(positions) // 100)
plt.show()
# %%
# print(positions)
# print(df_test.rotation.tolist())
# print(crosses)
# print(np.sum(crosses))
# %%
part_02_sol = np.sum(crosses)
print(part_02_sol)


# %%
print(part_02_brute_force(df_test))
print(part_02_brute_force(df))