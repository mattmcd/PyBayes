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
    return jnp.sqrt(jnp.linalg.det(gram))/jnp.prod(jnp.arange(1, len(vs)))


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


def to_tfjs():
    import tensorflow as tf
    from jax.experimental.jax2tf import convert
    from tensorflowjs.converters import convert_tf_saved_model
    import os
    # tf_areas = convert(areas, polymorphic_shapes=["(a, 2), (b, 2), (c, 2)"])
    # tf_areas = convert(areas, polymorphic_shapes=['(a, 2)', '(b, 2)', '(c, 2)'], with_gradient=False)
    # f_tf_areas = tf.function(tf_areas, autograph=False)

    tf_areas = convert(areas, with_gradient=True, enable_xla=False)
    f_tf_areas = tf.function(tf_areas, autograph=False,
                             input_signature=[tf.TensorSpec([1, 2]), tf.TensorSpec([1, 2]), tf.TensorSpec([1, 2]), ])

    # tf_areas = convert(areas, polymorphic_shapes=['(a, d)', '(b, d)', '(c, d)'], with_gradient=True, enable_xla=False)
    # f_tf_areas = tf.function(tf_areas, autograph=False,
    #                          input_signature=[tf.TensorSpec([None, 2]), tf.TensorSpec([None, 2]),
    #                                           tf.TensorSpec([None, 2]), ])

    # f_tf_areas = tf.function(tf_areas, autograph=False)
    # f_tf_areas.get_concrete_function(tf.TensorSpec([None, 2]), tf.TensorSpec([None, 2]), tf.TensorSpec([None, 2]))

    # f_tf_areas.get_concrete_function(tf.TensorSpec([None, 2]), tf.TensorSpec([None, 2]), tf.TensorSpec([None, 2]))
    model = tf.Module()
    model.f = f_tf_areas
    tf.saved_model.save(
        model, './scripts/simplex_volume_tf',
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=True)
    )
    restored_model = tf.saved_model.load('./scripts/simplex_volume_tf')
    restored_model.f(tf.convert_to_tensor([[0., 0.]]), tf.convert_to_tensor([[1., 0.]]),
                     tf.convert_to_tensor([[0., 1.]]))
    convert_tf_saved_model('./scripts/simplex_volume_tf',
                           os.path.expanduser('~/Work/Projects/JavaScript/understanding-modules/model'))
