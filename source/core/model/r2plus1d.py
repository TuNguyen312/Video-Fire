import math
import tensorflow as tf
import keras
from keras import layers


class Conv2Plus1D(keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding="same",
        stride=(1, 1, 1),
        bias=True,
        first_conv=False,
        **kwargs
    ):
        """
        A sequence of convolutional layers that first apply the convolution operation over the
        spatial dimensions, and then the temporal dimension.
        """
        super().__init__(**kwargs)

        # paper section 3.5
        if first_conv:
            intermed_channels = 45
        else:
            intermed_channels = int(
                math.floor(
                    (
                        kernel_size[0]
                        * kernel_size[1]
                        * kernel_size[2]
                        * in_channels
                        * out_channels
                    )
                    / (
                        kernel_size[1] * kernel_size[2] * in_channels
                        + kernel_size[0] * out_channels
                    )
                )
            )

        self.seq = keras.Sequential(
            [
                # Spatial decomposition
                layers.Conv3D(
                    filters=intermed_channels,
                    kernel_size=(1, kernel_size[1], kernel_size[2]),
                    padding=padding,
                    strides=(1, stride[1], stride[2]),
                    use_bias=bias,
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                # Temporal decomposition
                layers.Conv3D(
                    filters=out_channels,
                    kernel_size=(kernel_size[0], 1, 1),
                    padding=padding,
                    strides=(stride[0], 1, 1),
                    use_bias=bias,
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

    def call(self, x):
        return self.seq(x)


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
                kernel_size=(1, 1, 1),
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


class ResLayer(keras.layers.Layer):
    """
    Residual layer with multiple residual block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layer_size,
        downsample=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block1 = ResBlock(in_channels, out_channels, kernel_size, downsample)

        self.blocks = []
        for _ in range(layer_size - 1):
            self.blocks.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                )
            )

    def call(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class R2Plus1D:
    def __init__(self, net_name, n_frames, HEIGHT, WIDTH, num_classes):
        nets = {
            "r2plus1d-18": (2, 2, 2, 2),
            "r2plus1d-34": (3, 4, 6, 2),
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
