import jax
import jax.numpy as jnp
from jax import grad, vmap, pmap, jit
import pandas as pd
# from tensorflow.keras.datasets import mnist
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds


def row_expect(image_scaled, x):
    row_sum = jnp.sum(image_scaled, axis=1)
    row_mean = (image_scaled @ x / row_sum) @ row_sum / row_sum.sum()
    return row_mean


def col_expect(image_scaled, x):
    col_sum = jnp.sum(image_scaled, axis=0)
    col_mean = (x @ image_scaled / col_sum) @ col_sum / col_sum.sum()
    return col_mean


def extract_features(image):
    # Process a 28x28 image into feature vector
    image_range = jnp.max(image) - jnp.min(image)
    image_scaled = (image - jnp.min(image))/image_range + 0.0001
    pixel_sum = jnp.round(jnp.sum(image_scaled))
    # Location of mean pixel in each row
    m, n = image_scaled.shape
    row_mean = row_expect(image_scaled, jnp.arange(n))
    col_mean = col_expect(image_scaled, jnp.arange(m))
    row_std = jnp.sqrt(row_expect(image_scaled, jnp.arange(n)**2) - row_mean**2)
    col_std = jnp.sqrt(col_expect(image_scaled, jnp.arange(m)**2) - col_mean**2)
    return jnp.array([pixel_sum, row_mean, col_mean, row_std, col_std])


feature_map = vmap(extract_features, in_axes=0)
jit_feature_map = jit(feature_map)


def create_dataframe(x_train, y_train, n_samples=1000):
    data = {
        'y': y_train[:n_samples],
        'count': [1] * n_samples
    }
    features = jit_feature_map(x_train[:n_samples, ...])

    for i in range(features.shape[1]):
        data[f'x{i}'] = features[:, i]
    df  = pd.DataFrame(data)
    return df


class CNN(nn.Module):
    """A simple CNN module"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


