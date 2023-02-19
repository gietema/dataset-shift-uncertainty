import dataclasses
import math
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from dsu.train.config import Config


@dataclasses.dataclass
class StepConfig:
    steps_per_epoch: int
    validation_steps: int


@dataclasses.dataclass
class Datasets:
    train: tf.data.Dataset
    test: tf.data.Dataset
    val: tf.data.Dataset


def get_data(cfg: Config) -> Tuple[Datasets, StepConfig]:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_mean = np.mean(train_images, axis=0)
    train_images = train_images - train_mean
    test_images = test_images - train_mean

    num_classes = len(np.unique(test_labels))
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    # split train dataset into train and val
    n_train = math.floor(len(train_images) * (1 - cfg.val_split))
    train_images, val_images = train_images[:n_train], train_images[n_train:]
    train_labels, val_labels = train_labels[:n_train], train_labels[n_train:]

    train_ds = get_ds(train_images, train_labels, cfg, shuffle_buffer_size=n_train)
    val_ds = get_ds(val_images, val_labels, cfg=cfg)
    test_ds = get_ds(test_images, test_labels, cfg=cfg)

    steps_cfg = StepConfig(tf.math.floor(n_train / cfg.batch_size), tf.math.floor(len(val_images) / cfg.batch_size))
    return Datasets(train_ds, test_ds, val_ds), steps_cfg


def augment_images(images, pad_size: int):
    images = tf.pad(images, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))
    images = tf.image.random_crop(images, (32, 32, 3))
    return images


def preprocess(images):
    images = tf.image.random_flip_left_right(images)
    images = tf.divide(tf.cast(images, tf.float32), tf.cast(255.0, tf.float32))
    return images


def get_ds(images, labels, cfg: Config, shuffle_buffer_size: Optional[int] = None) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(lambda x, y: (preprocess(x), y))
    if cfg.augment:
        ds = ds.map(lambda x, y: (augment_images(x, cfg.augment_pad_size), y))
    if cfg.batch_size is not None:
        ds = ds.batch(cfg.batch_size)
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True).repeat()
    return ds
