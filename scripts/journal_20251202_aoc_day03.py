# %%
from startup import np, pd, plt, sns
from mattmcd.aoc2025 import Reader
from functools import reduce
# %%
test_reader = Reader.day03(True)
reader = Reader.day03()

# %%
df_test = test_reader.data
df = reader.data

# %%
# %%
def part_01_fun(A):
    # ind = np.argsort(A, axis=1)
    # top_ind = ind[:, -2:]
    # sorted_top_ind = np.sort(top_ind, axis=1)
    # el = np.take_along_axis(df, sorted_top_ind, axis=1)
    res = []
    res_v = []
    at_end = A.shape[1] - 1
    for row in A:
        poss = []
        for i in range(at_end+1):
            for j in range(i+1, at_end+1):
                poss.append(10*row[i] + row[j])
        res.append(max(poss))
        ind = np.argsort(row)
        if ind[-1] == at_end:
            res_v.append(10*row[ind[-2]] + row[ind[-1]])
        else:
            res_v.append(10*row[ind[-1]] + row[ind[ind > ind[-1]][-1]])
    res = np.array(res)
    res_v = np.array(res_v)
    return  np.sum(res), np.sum(res_v), np.hstack([A, res.reshape(-1, 1), res_v.reshape(-1, 1)])[res_v != res, :]

# %%
print(part_01_fun(df_test))
r, rv, d = part_01_fun(df)

