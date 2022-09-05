# Imports for Flax Getting Started Tutorial
import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
from flax import serialization

import numpy as np
import optax
import tensorflow_datasets as tfds

# Imports for some libraries to play with e.g. probabilistic models
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels
import cvxpy as cp
import cvxpylayers.jax as cpl
import tensorflow as tf
import functools


# Define the model
class CNN(nn.Module):
    """A simple CNN model."""

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
        return x


# Define the loss function as cross entropy loss
def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


# Compute the performance metrics
def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    return metrics


# Problem specific parts - get the date, define the training setup etc

# Define the dataset
def get_datasets():
    """Load the MNIST train and test datasets into memory"""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = train_ds['image'] / 255.0
    test_ds['image'] = test_ds['image'] / 255.0
    return train_ds, test_ds


# Define the training setup
def create_train_state(rng, learning_rate, momentum):
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


# Define the training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        # logits = CNN().apply({'params': params}, batch['image'])  # From tutorial
        logits = state.apply_fn({'params': params}, batch['image'])  # From AWS CodeWhisperer
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics


# Define the evaluation step
@jax.jit
def eval_step(state, params, batch):
    logits = state.apply_fn({'params': params}, batch['image'])  # Tutorial has CNN().apply
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return metrics


# Define training for an epoch over mini-batches
def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch"""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in perms:
        # train_ds is dict('label': n labels, 'image': n 28x28x1 images
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

    # compute mean of metrics across all batches in epoch
    batch_metrics_np = jax.device_get(batch_metrics)  # Retrieve metrics from training device and convert to np array
    # metric_names = batch_metrics_np[0].keys()
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]
    }
    print(
        f'train epoch: {epoch:d},' +
        f' loss: {epoch_metrics_np["loss"]:0.4f}, ' +
        f'accuracy:  {epoch_metrics_np["accuracy"]*100:0.2f}')
    return state


# Evaluate model
def eval_model(state, params, test_ds):
    metrics = eval_step(state, params, test_ds)
    metrics = jax.device_get(metrics)
    summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


def run_training(data=None):
    ds_train, ds_test = data or get_datasets()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    learning_rate = 1e-2
    momentum = 0.9
    state = create_train_state(init_rng, learning_rate, momentum)
    num_epochs = 10
    batch_size = 32
    for epoch in range(1, num_epochs + 1):
        state = train_epoch(state, ds_train, batch_size, epoch, rng)
        test_loss, test_accuracy = eval_model(state, state.params, ds_test)
        print(f'test epoch: {epoch:d},' +
              f'loss: {test_loss:0.4f}, ' +
              f'accuracy:  {test_accuracy*100:0.2f}')

    return state


class CnnParams:
    def __init__(self, params=None):
        cnn = CNN()
        self._params = params or cnn.init(jax.random.PRNGKey(0), jnp.ones([1, 28, 28, 1]))['params']

    @property
    def params(self):
        return self._params

    def to_file(self, fname):
        bytes_output = serialization.to_bytes(self.params)
        with open(fname, 'wb') as f:
            f.write(bytes_output)

    def predict(self, x):
        cnn = CNN()
        return jnp.argmax(cnn.apply({'params': self.params}, x), axis=1)

    def to_tflite(self, fname):
        serving_fn = self.predict  # functools.partial(self.predict, self.params)
        x_input = jnp.zeros((1, 28, 28, 1))  # BHWC
        converter = tf.lite.TFLiteConverter.experimental_from_jax(
            [serving_fn], [[('input1', x_input)]]
        )
        tflite_model = converter.convert()
        with open(fname, 'wb') as f:
            f.write(tflite_model)
        return converter

    @classmethod
    def from_file(cls, fname):
        cnn = CnnParams()
        with open(fname, 'rb') as f:
            cnn._params = serialization.from_bytes(cnn.params, f.read())
        return cnn

