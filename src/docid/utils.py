from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from rich.progress import track


def pdf_to_images(pdf_path: Path) -> List[Path]:
    imgs = convert_from_path(str(pdf_path), dpi=300)
    paths = []
    for i, img in enumerate(imgs):
        out_path = pdf_path.with_suffix(f"_{i}.png")
        img.save(out_path)
        paths.append(out_path)
    return paths


def tempdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory()


def now_iso() -> str:
    return datetime.utcnow().isoformat()
