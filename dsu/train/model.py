from keras import Model
from keras.engine.functional import Functional
from keras.optimizers.schedules.learning_rate_schedule import PiecewiseConstantDecay
from tensorflow import keras

from dsu.train.config import Config
from dsu.train.resnet import get_residual_block, INIT_FN


def get_model(cfg: Config) -> Functional:
    """
    Initialize a compiled ResNet model.
    """
    model = get_base_model(cfg)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.SGD(
            learning_rate=PiecewiseConstantDecay(cfg.lr_boundaries, cfg.lr_decay_values), momentum=cfg.momentum
        ),
        metrics=["accuracy"],
    )
    return model


def get_base_model(cfg: Config) -> Model:
    inputs = keras.layers.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=INIT_FN, padding="same", use_bias=False
    )(inputs)
    x = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = keras.layers.Activation("relu")(x)
    x = get_residual_block(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, kernel_initializer=INIT_FN)(x)
    return keras.models.Model(inputs, outputs, name=cfg.model_name)
