import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from journal_20230625_pgm_functions import (
    sat, intelligence, grade, difficulty, letter,
    p_d, p_i, p_s_i, p_g_id, p_l_g,
)

# %%
tfd = tfp.distributions


# %%
def prob_to_mat(df_prob):
    n_var = len(df_prob.index.levels)
    X = df_prob.values.reshape([len(v) for v in df_prob.index.levels])

    return tf.transpose(
        tf.constant(X),
        perm=tf.concat([tf.range(1, n_var), [0]], axis=0)
    )

# def prob_to_dist(df_prob):
#     vals = df_prob.index.levels[0].to_list()

# %%
d_i = tfd.FiniteDiscrete(
    tf.range(len(intelligence), dtype=tf.int32),
    probs=prob_to_mat(p_i), name='Intelligence'
)
d_d = tfd.FiniteDiscrete(
    tf.range(len(difficulty), dtype=tf.int32),
    probs=prob_to_mat(p_d), name='Difficulty'
)

# %%
x_s_i = prob_to_mat(p_s_i)
n_sat = len(p_s_i.index.levels[0])
d_s_i = lambda i: tfd.FiniteDiscrete(
    tf.range(len(sat), dtype=tf.int32),
    probs=tf.gather_nd(x_s_i, [i]),  # FIXME
    name='SAT_given_Intelligence'
)

# %%
print(f'{d_i}\n{d_d}\n{d_s_i(0)}')

# %%
print(pd.value_counts(d_s_i(0).sample(1000).numpy(), normalize=True))
print(pd.value_counts(d_s_i(1).sample(1000).numpy(), normalize=True))

# %%
d_si = tfd.JointDistributionSequential(
    [d_i, d_s_i]
)

# %%
x_g_id = prob_to_mat(p_g_id)
n_grade = len(p_g_id.index.levels[0])
d_g_id = lambda i, d: tfd.FiniteDiscrete(
    tf.range(n_grade, dtype=tf.int32),
    probs=tf.gather_nd(x_g_id, tf.stack([i, d])),  # FIXME
    name='Grade_given_Intelligence_and_Difficulty'
)

# %%
print('intelligence == "high" and difficulty == "hard"')
print(p_g_id.query('intelligence == "high" and difficulty == "hard"'))

print('Sample ')
pd.value_counts(d_g_id(1, 1).sample(10000).numpy(), normalize=True)

# %%
d_gid = tfd.JointDistributionSequential(
    [
        d_d, d_i,
        d_g_id
    ]
)

# %%
print(d_gid)

# %%
