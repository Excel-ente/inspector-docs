from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    classes: List[str] = os.getenv(
        "CLASSES", "dni_frente,dni_dorso,titulo_automotor,doc_sensible,otro"
    ).split(",")
    zips_in: str = os.getenv("RUTA_ZIPS_ENTRADA", "./zips_in")
    out_dir: str = os.getenv("RUTA_SALIDA", "./out")
    models_dir: str = os.getenv("RUTA_MODELOS", "./models")
    img_size: int = int(os.getenv("IMG_SIZE", 384))
    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", 0.75))
    ocr_backend: str = os.getenv("OCR_BACKEND", "tesseract")


config = Config()
