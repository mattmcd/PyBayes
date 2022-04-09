import jax
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np


def distance(a, b):
    return jnp.linalg.norm(a - b)


def all_pairs(f):
    # https://twitter.com/cgarciae88/status/1511763733676277766
    f = vmap(f, in_axes=(None, 0))
    f = vmap(f, in_axes=(0, None))
    return f


def volume(*vs):
    v = (jnp.array(vs[1:]) - vs[0])
    gram = v @ v.T
    return jnp.sqrt(jnp.linalg.det(gram))/np.math.factorial(len(vs) - 1)


def all_tuples(f, n):
    for i in range(n):
        in_axes = [0 if k == (n-1-i) else None for k in range(n)]
        f = vmap(f, in_axes=in_axes)
    return f


distances = all_tuples(volume, 2)
jit_distances = jit(distances)


def check2(use_jit=True):
    A = jnp.array([[0., 0.], [1., 1.], [2., 2.]])
    B = jnp.array([[-10., -10.], [-20., -20.]])
    f = jit_distances if use_jit else distances
    return f(A, B)


areas = all_tuples(volume, 3)
jit_areas = jit(areas)


def check3(use_jit=True):
    A = jnp.array([[0., 0.], [1., 1.], [2., 2.]])
    B = jnp.array([[1., 0.], [-1., -1.]])
    C = jnp.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    f = jit_areas if use_jit else areas
    return f(A, B, C)



def check():
    A = jnp.array([[0., 0.], [1., 1.], [2., 2.]])
    B = jnp.array([[-10., -10.], [-20., -20.]])
    print(distance(A[0], B[0]))
    distances = all_pairs(distance)
    print(distances(A, B))
