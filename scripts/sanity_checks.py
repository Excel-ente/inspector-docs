from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from docid.config import config
from docid.infer_zip import load_model
from docid.schema import ZipResult


def main() -> None:
    dummy = np.zeros((config.img_size, config.img_size, 3), dtype=np.uint8)
    Image.fromarray(dummy).save("dummy.png")
    model = load_model(Path(config.models_dir) / "saved_model")
    pred = model.predict(np.expand_dims(dummy / 255.0, 0))
    print("Prediction", pred)
    jr = ZipResult(
        zip="dummy.zip",
        run_started_at="2020-01-01T00:00:00",
        items=[],
        summary={"by_class": {}, "low_confidence": 0},
    )
    json.loads(jr.json())
    print("Schema valid")


if __name__ == "__main__":
    main()
