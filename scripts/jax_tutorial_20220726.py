import flax.linen.activation
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random, device_put, device_get
from jax.tree_util import tree_map, tree_leaves, tree_multimap, tree_structure
from flax.linen.activation import relu, sigmoid
from functools import reduce, partial


def init_mlp_params(layers):
    layer_maps = list(zip(layers[:-1], layers[1:]))
    params = [
        {
            'weights': np.random.normal(size=(m, n))*np.sqrt(n),
            'biases': np.random.normal(size=(m, 1)),
        }
        for n, m in layer_maps
    ]
    return params


def predict(params, x, activations=None):
    if activations is None:
        activations = ([relu] * (len(params) - 1)) + [sigmoid]

    for i in range(len(params)):
        x = activations[i](jnp.dot(params[i]['weights'], x) + params[i]['biases'])

    return x[0][0]
