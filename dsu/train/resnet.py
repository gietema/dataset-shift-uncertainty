import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Add, Lambda

INIT_FN = tf.keras.initializers.HeNormal()


def get_residual_block(x):
    filter_size = 16
    for layer_group in range(3):
        for block in range(3):
            if layer_group > 0 and block == 0:
                filter_size *= 2
                x = residual_block(x, filter_size, match_filter_size=True)
            else:
                x = residual_block(x, filter_size)
    return x


def residual_block(x, filters, match_filter_size=False):
    x_skip = x
    if match_filter_size:
        x = Conv2D(
            filters, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=INIT_FN, padding="same", use_bias=False
        )(x_skip)
    else:
        x = Conv2D(
            filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=INIT_FN, padding="same", use_bias=False
        )(x_skip)
    x = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.9)(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, kernel_size=(3, 3), kernel_initializer=INIT_FN, padding="same", use_bias=False)(x)
    x = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.9)(x)

    if match_filter_size:
        x_skip = Lambda(
            lambda x: tf.pad(
                x[:, ::2, ::2, :],
                tf.constant([[0, 0], [0, 0], [0, 0], [filters // 4, filters // 4]]),
                mode="CONSTANT",
            )
        )(x_skip)

    x = Add()([x, x_skip])
    x = Activation("relu")(x)
    return x
