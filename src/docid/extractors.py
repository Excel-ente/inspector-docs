from __future__ import annotations

import re
from typing import Dict

from unidecode import unidecode

PAT_DNI = re.compile(r"\b(\d{7,8})\b")
PAT_PATENTE = re.compile(r"\b([A-Z]{2}\s?\d{3}\s?[A-Z]{2}|[A-Z]{3}\s?\d{3})\b")
PAT_CUIT = re.compile(r"\b\d{2}-?\d{8}-?\d\b")
PAT_EMAIL = re.compile(r"[\w\.]+@[\w\.]+")


def extract_from_dni(text: str) -> Dict[str, str]:
    text = unidecode(text.upper())
    dni = PAT_DNI.search(text)
    return {"dni": dni.group(1) if dni else None}


def extract_from_titulo_automotor(text: str) -> Dict[str, str]:
    text = unidecode(text.upper())
    patente = PAT_PATENTE.search(text)
    return {"patente": patente.group(1).replace(" ", "") if patente else None}


def extract_generic(text: str) -> Dict[str, str]:
    return {
        "chars": len(text),
        "has_cuit": bool(PAT_CUIT.search(text)),
        "has_email": bool(PAT_EMAIL.search(text)),
    }
