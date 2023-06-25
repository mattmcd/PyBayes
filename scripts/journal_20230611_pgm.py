import pandas as pd
from journal_20230625_pgm_functions import (
    sat, intelligence, grade, difficulty, letter,
    p_d, p_i, p_s_i, p_g_id, p_l_g,
    construct_joint_prob, query_printer, inference_query
)

# %% [markdown]
# We start with a set of conditional probability distributions for
# course Difficulty, student Intelligence, student SAT result,
# Grade of student, strength of Letter of recommendation.
#
# These are as Fig 3.4 p53 of 'Probabilistic Graphical Models'
# by Koller and Friedman
#
# The conditional probabilities can then be combined to form joint distributions
# and used for inference.  Here we currently just calculate the full
# joint distribution hypercube and slice / aggregate for the marginals
# so it will be quite inefficient for larger models as the graph structure
# is not being used effectively.
#
# Possible improvement: marginalise the conditional distributions then construct
# joint distribution.

# %%
# Joint distribution of Grade, Intelligence, Difficulty
p_gid = construct_joint_prob(p_g_id, p_i, p_d)
print(p_gid)

# %%
# Joint distribution of Grade, Intelligence, Difficulty, Letter
p_gidl = construct_joint_prob(p_g_id, p_i, p_d, p_l_g)
print(p_gidl)

# %%
# Below we show reasoning patterns from section 3.2.1.2 of PGM, p54

# %%
q_p_gidl = query_printer(inference_query(p_gidl))

# %%
# If we know nothing about the student what are their
# probabilities of getting strong or weak letter?
# Solution: marginalise over other vars by group_by -> 0.502 strong
q_p_gidl('letter')

# %%
# We know they aren't intelligent -> 0.389 strong
q_p_gidl('letter|intelligence == "low"')

# %%
# They got a C -> 0.079 high intelligence
q_p_gidl('intelligence|grade == "C"')

# %%
# ... and class is probably more difficult (-> 0.629 hard)
q_p_gidl('difficulty|grade == "C"')

# %%
# Weak letter only - 0.14 high intelligence
q_p_gidl('intelligence|letter == "weak"')

# %%
# Letter and Grade gives same info as Grade alone
q_p_gidl('intelligence|grade == "C" and letter == "weak"')

# %%
# Introduce SAT score
p_gidls = construct_joint_prob(p_g_id, p_s_i, p_l_g, p_i, p_d)

# %%
q_p_gidls = query_printer(inference_query(p_gidls))

# %%
# Good SAT score outweighs poor grade: 0.578 high intelligence
q_p_gidls('intelligence|grade == "C" and sat == "high"')

# %%
# ... and makes it more likely the course is difficult -> 0.76 hard
q_p_gidls('difficulty|grade == "C" and sat == "high"')
