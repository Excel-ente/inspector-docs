.PHONY: venv train infer sanity

venv:
python3 -m venv venv && . venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
python -m src.docid.train --epochs 12 --batch_size 16

infer:
python -m src.docid.infer_zip --zips_dir ./zips_in

sanity:
python scripts/sanity_checks.py
