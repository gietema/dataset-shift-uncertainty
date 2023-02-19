from typing import List

from keras.callbacks import ModelCheckpoint, Callback
from wandb.integration.keras import WandbMetricsLogger


def get_callbacks(model_name) -> List[Callback]:
    checkpoint = ModelCheckpoint(
        "models/" + model_name + "/{epoch:02d}-{val_accuracy:.4f}",
        save_best_only=True,
        onitor="val_accuracy",
        save_freq="epoch",
    )
    callbacks = [checkpoint, WandbMetricsLogger(log_freq=5)]
    return callbacks
