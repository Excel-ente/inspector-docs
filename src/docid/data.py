from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

from .config import config


def build_dataset(
    data_dir: str, batch_size: int, img_size: int | None = None, shuffle: bool = True
) -> tf.data.Dataset:
    img_size = img_size or config.img_size
    datagen = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=shuffle,
    )
    ds = datagen.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    return ds.prefetch(tf.data.AUTOTUNE)
