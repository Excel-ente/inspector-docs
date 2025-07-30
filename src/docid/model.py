from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from .config import config

SUPPORTED_MODELS = {
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
    "mobilenetv3small": tf.keras.applications.MobileNetV3Small,
}


def build_model(
    model_name: str = "efficientnetb0",
    num_classes: int | None = None,
    img_size: int | None = None,
) -> tf.keras.Model:
    num_classes = num_classes or len(config.classes)
    img_size = img_size or config.img_size
    base_cls = SUPPORTED_MODELS.get(model_name.lower())
    if base_cls is None:
        raise ValueError(f"Unsupported model {model_name}")
    base = base_cls(
        weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    return model
