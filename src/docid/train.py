from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import config
from .data import build_dataset
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train document classifier")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=config.img_size)
    parser.add_argument("--model_name", default="efficientnetb0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    train_ds = build_dataset(
        str(train_dir), args.batch_size, args.img_size, shuffle=True
    )
    val_ds = build_dataset(str(val_dir), args.batch_size, args.img_size, shuffle=False)

    model = build_model(args.model_name, len(config.classes), args.img_size)
    if args.freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False

    y_train = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(config.classes)), y=np.argmax(y_train, axis=1)
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            Path(config.models_dir) / "best_model.h5", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks,
    )

    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(config.models_dir) / "saved_model")

    metrics_dir = Path(config.out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    np.save(metrics_dir / "history.npy", history.history)


if __name__ == "__main__":
    main()
