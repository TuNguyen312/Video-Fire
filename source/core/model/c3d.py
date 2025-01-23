import tensorflow as tf
import keras
from keras import layers


class C3D(keras.layers.Layer):
    def __init__(
        self,
        n_frames,
        HEIGHT,
        WIDTH,
        num_classes,
    ):
        super().__init__()
        input_shape = (n_frames, HEIGHT, WIDTH, 3)
        self.model = keras.Sequential(
            [
                layers.Conv3D(
                    64,
                    kernel_size=(3, 3, 3),
                    padding="same",
                    input_shape=input_shape,
                ),
                layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)),
                layers.Conv3D(
                    128,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
                layers.Conv3D(
                    256,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                layers.Conv3D(
                    256,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
                layers.Conv3D(
                    512,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                layers.Conv3D(
                    512,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
                layers.Conv3D(
                    512,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                layers.Conv3D(
                    512,
                    kernel_size=(3, 3, 3),
                    padding="same",
                ),
                layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),
                layers.Flatten(),
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(4096, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
