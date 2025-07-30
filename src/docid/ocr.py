from __future__ import annotations

from typing import Optional

import pytesseract

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PaddleOCR = None

_paddle_instance: Optional[PaddleOCR] = None


def ocr_tesseract(img) -> str:
    return pytesseract.image_to_string(img, lang="spa+eng", config="--psm 6")


def ocr_paddle(img) -> str:
    global _paddle_instance
    if _paddle_instance is None:
        if PaddleOCR is None:
            raise RuntimeError("paddleocr not installed")
        _paddle_instance = PaddleOCR(use_angle_cls=True, lang="es")
    result = _paddle_instance.ocr(img, cls=True)
    return "\n".join([line[1][0] for line in result])
