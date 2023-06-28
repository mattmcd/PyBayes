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
d_s_i = lambda Intelligence: tfd.FiniteDiscrete(
    tf.range(n_sat, dtype=tf.int32),
    probs=tf.gather_nd(x_s_i, tf.transpose(tf.stack([Intelligence]))),  # FIXME
    name='SAT_given_Intelligence'
)

# %%
print(f'{d_i}\n{d_d}\n{d_s_i(0)}')

# %%
print(pd.value_counts(d_s_i(0).sample(1000).numpy(), normalize=True))
print(pd.value_counts(d_s_i(1).sample(1000).numpy(), normalize=True))

# %%
d_si = tfd.JointDistributionSequential(
    [d_i, d_s_i], batch_ndims=0,  # use_vectorized_map=True
)

# %%
print(d_si)

# %%
x_g_id = prob_to_mat(p_g_id)
n_grade = len(p_g_id.index.levels[0])
d_g_id = lambda Intelligence, Difficulty: tfd.FiniteDiscrete(
    tf.range(n_grade, dtype=tf.int32),
    probs=tf.gather_nd(x_g_id, tf.transpose(tf.stack([Intelligence, Difficulty]))),
    name='Grade_given_Intelligence_and_Difficulty'
)

# %%
# Compare conditional probability table to result of sampling distribution to check indexing
print('intelligence == "high" and difficulty == "hard"')
print(p_g_id.query('intelligence == "high" and difficulty == "hard"'))

print('Sample ')
pd.value_counts(d_g_id(1, 1).sample(10000).numpy(), normalize=True)

# %%
d_gid = tfd.JointDistributionSequential(
    [d_d, d_i, d_g_id], batch_ndims=0,  # use_vectorized_map=True
)

# %%
print(d_gid)

# %%
print(d_gid.resolve_graph())

# %%
prior_predictive_samples = d_gid.sample(10)


# %%
def levels_to_dict(levels):
    return dict(zip(range(len(levels)), levels))


# %%
# Template model: n_student Intelligence and n_class Difficulty -> Grade matrix

n_student = 11
n_class = 7

student_intelligence = d_i.sample(n_student)
course_difficulty = d_d.sample(n_class)

print(f'Student intelligence: {student_intelligence.numpy()}')
print(f'Course difficulty: {course_difficulty.numpy()}')

student_class = tf.stack(
    [
        tf.repeat(student_intelligence, [n_class]),
        tf.tile(course_difficulty, [n_student])
    ]
)
cohort_grades = d_g_id(student_class[0], student_class[1]).sample()
df_cg = pd.DataFrame(
    {
        'student': tf.repeat(tf.range(n_student),  [n_class]),
        'course': tf.tile(tf.range(n_class), [n_student]),
        'grade': cohort_grades
    }
).assign(
    letter_grade=lambda x: x.grade.map(levels_to_dict(grade))
).set_index(['student', 'course']).unstack()

print(df_cg)

# %%
# Next: going from the Grade matrix back to predictions of course Difficulty and student Intelligence.
# Approach: use Beta distributions and fit?
