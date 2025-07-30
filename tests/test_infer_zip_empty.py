from pathlib import Path

import pytest

from docid.config import config
from docid.infer_zip import argparse, load_model, process_zip
from docid.schema import ZipResult


def test_infer_empty(tmp_path):
    # create empty zip
    empty_zip = tmp_path / "test.zip"
    (tmp_path / "inner").mkdir()
    pytest.importorskip("zipfile").ZipFile(empty_zip, "w").close()
    model = load_model(Path(config.models_dir) / "saved_model")
    args = argparse.Namespace(
        conf=0.5, ocr_backend="tesseract", zips_dir=str(tmp_path), model_path=None
    )
    res = process_zip(empty_zip, model, args)
    assert isinstance(res, ZipResult)
