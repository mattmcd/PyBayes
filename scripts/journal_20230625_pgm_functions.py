import pandas as pd

# PGM student example p53
difficulty = ('easy', 'hard')
intelligence = ('low', 'high')
grade = ('A', 'B', 'C')
sat = ('low', 'high')
letter = ('weak', 'strong')

# P(D)
p_d = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (difficulty, ), names=['difficulty']),
    data=[0.6, 0.4],
    columns=['prob']
)

# P(I)
p_i = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (intelligence, ), names=['intelligence']),
    data=[0.7, 0.3],
    columns=['prob']
)

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


def construct_joint_prob(*args):
    """Simple calculation of joint probability distribution by factor multiplication
    NB: this will be inefficient for large distributions
    NB: order of probability distributions is important, need to make sure the joins
      have overlapping columns e.g. p_i, p_d, p_g_id fails but p_g_id, p_i, p_d works
      (to fix at some point)

    :param args: conditional probability dataframes
    :return: dataframe of full joint distribution
    """
    n_prob = len(args)

    def rename_prob(i):
        return args[i].rename(columns={'prob': f'prob_{i}'})

    if n_prob > 1:
        p_j = rename_prob(0)
        for i in range(1, n_prob):
            p_j = p_j.join(rename_prob(i))
        p_j = p_j.assign(
            prob=lambda x: x.prod(axis=1)
        ).drop(columns=[f'prob_{i}' for i in range(n_prob)])
    else:
        p_j = args[0]
    return p_j


def infer_prob(p_joint, s_cond):
    """Infer marginal distribution given observations

    :param p_joint: full joint distribution of variables
    :param s_cond: string representing query e.g. 'var', 'var|obs == val
    :return: marginal distribution
    """

    if '|' in s_cond:
        var, obs = s_cond.split('|')
        grp = p_joint.query(obs).groupby(var)
    else:
        grp = p_joint.groupby(s_cond)
    df_i = grp['prob'].sum().transform(lambda x: x/x.sum())
    return df_i


def inference_query(p_joint):
    def inner(s_cond):
        return infer_prob(p_joint, s_cond)
    return inner


def query_printer(func):
    def inner(s_cond):
        print(f'P({s_cond})')
        print(func(s_cond))
        print()
    return inner
