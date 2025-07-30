"""Utility script to copy images from example folders into data/train structure."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from docid.config import config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Folder with class subfolders")
    args = p.parse_args()
    src = Path(args.src)
    for split in ["train", "val"]:
        for cls in config.classes:
            dest = Path("data") / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img in (src / cls).glob("*.png"):
                shutil.copy(img, dest / img.name)
    print("Dataset populated")


if __name__ == "__main__":
    main()
