
"""Define conv network function."""
from typing import Tuple

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape

def conv(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    # max_layer_size: int = 128,
    dropout_amount: float = 0.2,
    # num_layers: int = 3,    
    )-> Model:

    image_height, image_width = input_shape
    num_classes = output_shape[0]
    model = Sequential()
    # add Convolutional layers
    model.add(Reshape((image_height, image_width, 1), input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout_amount))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout_amount))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # output layer
    model.add(Dense(num_classes, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    return model
