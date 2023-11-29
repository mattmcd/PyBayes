# %%
# Learning BFGS optimization algorithm via a short script to
# run in PyCharm debug mode to step through
# %%
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize as jax_min
from scipy.optimize import minimize as sci_min

# %%
f = lambda x: (x - 2.).T @ (x - 2.)

# %%
jax_res = jax_min(f, jnp.array([0.]), method='bfgs')

# %%
sci_res = sci_min(f, np.array([0.]), method='bfgs')
