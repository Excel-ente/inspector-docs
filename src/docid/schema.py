from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Item(BaseModel):
    file: str
    page: int
    class_name: str = Field(alias="class")
    confidence: float
    status: str
    fields: Dict[str, Optional[str]]
    ocr_chars: int


class Summary(BaseModel):
    by_class: Dict[str, int]
    low_confidence: int


class ZipResult(BaseModel):
    zip: str
    run_started_at: datetime
    items: List[Item]
    summary: Summary

    class Config:
        allow_population_by_field_name = True
