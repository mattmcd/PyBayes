import tensorflow as tf
import keras
import jax
import mlx.core as mx

print(f'TensorFlow: {tf.__version__}')
print(f"Tensorflow GPU: {tf.config.list_physical_devices('GPU')}")
print(f'Keras: {keras.__version__}')
print(f'JAX: {jax.__version__}')
print(f'MLX: {mx.__version__}')
