# %%
import os
import numpy as np
# Set KERAS_BACKEND="jax" or "tensorflow" or "torch" in environment
import keras
from keras import Model
from keras.layers import Input, Dense, Flatten
from kymatio.keras import Scattering2D
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# %%
# Model parameters
num_classes = 10
input_shape = (28, 28)

inputs = Input(shape=input_shape)
# J = number of length scales, L = number of angles
x = Scattering2D(J=3, L=8)(inputs)
x = Flatten()(x)
x_out = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, x_out)

# %%
model.summary()

# %%
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# %%
print(f'JIT compile: {model.jit_compile}')

# %%
# model.fit(x_train[:10000], y_train[:10000], epochs=15,
#           batch_size=64, validation_split=0.2)

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
model.save(f'keras_wavelets_final_model_{BACKEND}_20250824.keras')

# %%
predictions = model.predict(x_test)

# %%
y_pred = np.argmax(predictions, axis=1)

# %%
print(confusion_matrix(y_test, y_pred))

# %%
print(classification_report(y_test, y_pred))

