# %%
import os
import numpy as np
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Set KERAS_BACKEND="jax" or "tensorflow" or "torch" in environment
import keras_core as keras
import keras_core.ops as ops
from journal_20231025_keras_core_jax_metal import wrap_jax_metal

# %%
BACKEND = os.environ.get('KERAS_BACKEND')


# %%
def cheb(N):
    i = ops.arange(0, N+1).reshape(-1, 1)
    pi = ops.arccos(-1)
    x = ops.cos(pi*i/N)
    c = ops.vstack([ops.array([2]), ops.ones((N-1, 1)), ops.array([2])]) * (-1)**i
    X = ops.tile(x, (1, N+1))
    dX = X - X.T
    D_r = (c*(1/c).T)/(dX + ops.eye(N+1))
    D = D_r - ops.diag(ops.sum(D_r.T, axis=0))
    return D, x

# %%
cheb(3)

# %%
xx = ops.linspace(-1, 1, 100)
uu = ops.exp(xx)*ops.sin(5*xx)


# %%
def plot_diff(N):
    D, x = cheb(N)
    u = ops.exp(x)*ops.sin(5*x)
    du = ops.exp(x)*(ops.sin(5*x) + 5*ops.cos(5*x))
    error = D @ u - du
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(xx, uu, marker='none')
    ax[0].plot(x, u, marker='.', linestyle='none', color='k')

    ax[1].plot()
    ax[1].plot(x, error, marker='.')

    plt.show()

# %%
plot_diff(20)

# %%
