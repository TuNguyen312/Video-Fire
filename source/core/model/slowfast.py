import tensorflow as tf
import keras
from keras import layers


class ResBlock(keras.layers.Layer):
    """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, downsample=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.downsample = downsample
        if downsample:
            self.downsampleconv = Conv2Plus1D(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                stride=(2, 2, 2),
            )
            self.downsamplebn = layers.BatchNormalization()

            self.conv1 = Conv2Plus1D(
                in_channels, out_channels, kernel_size, padding="same", stride=(2, 2, 2)
            )
        else:
            self.conv1 = Conv2Plus1D(
                in_channels, out_channels, kernel_size, padding="same", stride=(1, 1, 1)
            )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = Conv2Plus1D(
            out_channels,
            out_channels,
            kernel_size,
            padding="same",
            stride=(1, 1, 1),
        )
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)

        if self.downsample:
            x = self.downsampleconv(x)
            x = self.downsamplebn(x)
        return layers.add([res, x])

class SlowFast 
    def __init__(self, net_name, n_frames, HEIGHT, WIDTH, num_classes):
        nets = {
            "slowfast-50": (3, 4, 6, 3),
            "slowfast-101": (3, 4, 23, 3),
            "slowfast-152": (3, 8, 36, 3),
            "slowfast-200": (3, 24, 36, 3)
        }
        layer_sizes = nets[net_name]
        input_shape = (None, n_frames, HEIGHT, WIDTH, 3)
        input = layers.Input(shape=(input_shape[1:]))
        x = input

        # Conv1
        x = Conv2Plus1D(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 7, 7),
            padding="same",
            stride=(1, 2, 2),
            first_conv=True,
        )(x)

        # Conv2_x
        x = ResLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            layer_size=layer_sizes[0],
        )(x)

        # Conv3_x
        x = ResLayer(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            layer_size=layer_sizes[1],
            downsample=True,
        )(x)

        # Conv4_x
        x = ResLayer(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3, 3),
            layer_size=layer_sizes[2],
            downsample=True,
        )(x)

        # Conv5_x
        x = ResLayer(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3, 3),
            layer_size=layer_sizes[3],
            downsample=True,
        )(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(num_classes, activation="softmax")(x)

        self.model = keras.Model(input, x)
