import pandas as pd

# %%
# PGM student example p53
difficulty = ('easy', 'hard')
intelligence = ('low', 'high')
grade = ('A', 'B', 'C')
sat = ('low', 'high')
letter = ('weak', 'strong')

# %%
# P(D)
p_d = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (difficulty, ), names=['difficulty']),
    data=[0.6, 0.4],
    columns=['prob']
)

# %%
# P(I)
p_i = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (intelligence, ), names=['intelligence']),
    data=[0.7, 0.3],
    columns=['prob']
)

# %%
# P(G|I,D)
p_g_id = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (grade, intelligence, difficulty, ),
        names=['grade', 'intelligence', 'difficulty']),
    data=[
        0.3, 0.05, 0.9, 0.5,
        0.4, 0.25, 0.08, 0.3,
        0.3, 0.7, 0.02, 0.2
    ],
    columns=['prob']
)

# %%
# P(L|G)
p_l_g = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (letter, grade, ),
        names=['letter', 'grade', ]),
    data=[
        0.1, 0.4, 0.99,
        0.9, 0.6, 0.01
    ],
    columns=['prob']
)
print(p_l_g)

# %%
# P(S|I)
p_s_i = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (sat, intelligence, ),
        names=['sat', 'intelligence', ]),
    data=[
        0.95, 0.2,
        0.05, 0.8
    ],
    columns=['prob']
)
print(p_s_i)


# %%
# Joint distribution of Grade, Intelligence, Difficulty
p_gid = p_g_id.join(
    p_i, lsuffix='_g', rsuffix='_i'
).join(
    p_d.rename(columns={'prob': 'prob_d'})
).assign(prob=lambda x: x.prod(axis=1))
print(p_gid)

# %%
# Joint distribution of Grade, Intelligence, Difficulty, Letter
p_gidl = p_gid.drop(columns=['prob']).join(
    p_l_g.rename(columns={'prob': 'prob_l'})
).assign(prob=lambda x: x.prod(axis=1))
print(p_gidl)
# %%
p_gid.query(
    'grade == "A" and difficulty == "easy"'
)['prob'].transform(lambda x: x/x.sum())

# %%
# Below we show reasoning patterns from section 3.2.1.2 of PGM, p54

# %%
# If we know nothing about the student what are their
# probabilities of getting strong or weak letter?
# Solution: marginalise over other vars by group_by -> 0.502 strong
print(
    p_gidl.groupby('letter')['prob'].sum()
)

# %%
# We know they aren't intelligent -> 0.389 strong
print(
    p_gidl.query(
        'intelligence == "low"'
    ).groupby(
        'letter'
    )['prob'].sum().transform(lambda x: x/x.sum())
)

# %%
# They got a C -> 0.079 high intelligence
print(
    p_gidl.query(
        'grade == "C"'
    ).groupby(
        'intelligence'
    )['prob'].sum().transform(lambda x: x/x.sum())
)

# %%
# ... and class is probably more difficult (-> 0.629 hard)
print(
    p_gidl.query(
        'grade == "C"'
    ).groupby(
        'difficulty'
    )['prob'].sum().transform(lambda x: x/x.sum())
)

# %%
# Weak letter only - 0.14 high intelligence
print(
    p_gidl.query(
        'letter == "weak"'
    ).groupby(
        'intelligence'
    )['prob'].sum().transform(lambda x: x/x.sum())
)

# %%
# Letter and Grade gives same info as Grade alone
print(
    p_gidl.query(
        'grade == "C" and letter == "weak"'
    ).groupby(
        'intelligence'
    )['prob'].sum().transform(lambda x: x/x.sum())
)

# %%
# Introduce SAT score
p_gidls = p_gidl.drop(columns=['prob']).join(
    p_s_i.rename(columns={'prob': 'prob_s'})
).assign(prob=lambda x: x.prod(axis=1))

# %%
# Good SAT score outweighs poor grade: 0.578 high intelligence
print(
    p_gidls.query(
        'grade == "C" and sat == "high"'
    ).groupby(
        'intelligence'
    )['prob'].sum().transform(lambda x: x/x.sum())
)

# %%
# ... and makes it more likely the course is difficult -> 0.76 hard
print(
    p_gidls.query(
        'grade == "C" and sat == "high"'
    ).groupby(
        'difficulty'
    )['prob'].sum().transform(lambda x: x/x.sum())
)
