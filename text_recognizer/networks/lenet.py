"""LeNet network."""
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    MaxPooling2D,
    BatchNormalization,
    AveragePooling2D,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential, Model


def lenet(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    layer_size: int = 64,
    dropout_amount: float = 0.1,
    batch_norm: bool = True,
    pooling: str = "max",
    padding: str = "same",
    **kwargs,
) -> Model:
    """Return LeNet Keras model."""
    num_classes = output_shape[0]
    pool_fn = MaxPooling2D if pooling == "max" else AveragePooling2D

    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape, name="expand_dims"))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(layer_size, kernel_size=(3, 3), padding=padding, input_shape=input_shape))
    if batch_norm:
        model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.2))
    model.add(pool_fn(pool_size=(2, 2)))
    model.add(Conv2D(layer_size * 2, (3, 3), padding=padding))
    if batch_norm:
        model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.2))
    model.add(pool_fn(pool_size=(2, 2)))
    model.add(Conv2D(layer_size * 4, (3, 3), padding=padding))
    if batch_norm:
        model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation="softmax"))

    return model
