# %%
import os
import numpy as np
# Set KERAS_BACKEND="jax" or "tensorflow" or "torch" in environment
import keras
from journal_20231025_keras_core_jax_metal import wrap_jax_metal
import tensorflow as tf
legacy_adam = tf.keras.optimizers.legacy.Adam
# %%
BACKEND = os.environ.get('KERAS_BACKEND')
print(f'Using backend {BACKEND}')
# %%
# See https://keras.io/keras_core/guides/getting_started_with_keras_core/
# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# %%
# Model parameters
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# %%
model.summary()

# %%
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=(
        legacy_adam(learning_rate=1e-3) if BACKEND == 'tensorflow'
        else keras.optimizers.Adam(learning_rate=1e-3)
    ),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

# %%
use_jit = True

if use_jit and BACKEND == 'jax':
    model = wrap_jax_metal(model)
else:
    model.jit_compile = False


# %%
batch_size = 128
epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=BACKEND + '_' + "model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
score = model.evaluate(x_test, y_test, verbose=0)
print(score)

# %%
model.save(f'keras_core_final_model_{BACKEND}.keras')

# %%
predictions = model.predict(x_test)
