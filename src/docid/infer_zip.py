from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from .config import config
from .extractors import extract_from_dni, extract_from_titulo_automotor, extract_generic
from .model import build_model
from .ocr import ocr_paddle, ocr_tesseract
from .preprocess import preprocess_for_ocr
from .schema import Item, Summary, ZipResult
from .utils import now_iso, pdf_to_images, tempdir


def load_model(model_path: str | None = None) -> tf.keras.Model:
    path = model_path or Path(config.models_dir) / "saved_model"
    return tf.keras.models.load_model(path)


def prepare_image(path: Path, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    return np.array(img) / 255.0


def run_ocr(img_bgr: np.ndarray, backend: str) -> str:
    proc = preprocess_for_ocr(img_bgr)
    if backend == "paddle":
        return ocr_paddle(proc)
    return ocr_tesseract(proc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer documents in ZIPs")
    p.add_argument("--zips_dir", default=config.zips_in)
    p.add_argument("--conf", type=float, default=config.conf_threshold)
    p.add_argument("--ocr_backend", default=config.ocr_backend)
    p.add_argument("--model_path")
    return p.parse_args()


def process_zip(
    zip_path: Path, model: tf.keras.Model, args: argparse.Namespace
) -> ZipResult:
    with tempdir() as tmp:
        shutil.unpack_archive(str(zip_path), tmp)
        extracted = []
        for path in sorted(Path(tmp).rglob("*")):
            if path.is_dir():
                continue
            files = []
            if path.suffix.lower() == ".pdf":
                files.extend(pdf_to_images(path))
            else:
                files.append(path)
            for i, img_path in enumerate(files):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                input_arr = prepare_image(img_path, config.img_size)
                pred = model.predict(np.expand_dims(input_arr, 0))[0]
                cls_idx = int(np.argmax(pred))
                cls_name = config.classes[cls_idx]
                conf = float(pred[cls_idx])
                status = "ok" if conf >= args.conf else "manual_review"
                text = run_ocr(img, args.ocr_backend) if conf >= args.conf else ""
                fields = {}
                if conf >= args.conf:
                    if cls_name == "dni_frente":
                        fields = extract_from_dni(text)
                    elif cls_name == "titulo_automotor":
                        fields = extract_from_titulo_automotor(text)
                    else:
                        fields = extract_generic(text)
                extracted.append(
                    Item(
                        file=str(path.relative_to(tmp)),
                        page=i + 1,
                        class_name=cls_name,
                        confidence=conf,
                        status=status,
                        fields=fields,
                        ocr_chars=len(text),
                    )
                )
        by_class = Counter(item.class_name for item in extracted)
        low_conf = sum(1 for item in extracted if item.status == "manual_review")
        result = ZipResult(
            zip=zip_path.name,
            run_started_at=now_iso(),
            items=extracted,
            summary=Summary(by_class=by_class, low_confidence=low_conf),
        )
        return result


def main() -> None:
    args = parse_args()
    model = load_model(args.model_path)
    Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    for zip_file in Path(args.zips_dir).glob("*.zip"):
        result = process_zip(zip_file, model, args)
        out_path = Path(config.out_dir) / f"{zip_file.stem}.json"
        out_path.write_text(
            result.json(by_alias=True, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Processed {zip_file} -> {out_path}")


if __name__ == "__main__":
    main()
