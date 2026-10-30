"""PII masking for anything that leaves the process (traces). Loan applications are personal data.

Regex covers structured identifiers. Names written in free text ("I'm Priya") need an NER model
(e.g. Microsoft Presidio) and are a documented gap; the structured name field is masked.
"""
import re
from typing import Any

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)"), "[PHONE/ID]"),  # phone, account, national ID numbers
    (re.compile(r'("person_name"\s*:\s*)"[^"]*"'), r'\1"[NAME]"'),
    (re.compile(r"('person_name'\s*:\s*)'[^']*'"), r"\1'[NAME]'"),
    (re.compile(r"(- Name: ).+"), r"\1[NAME]"),
]


def mask_text(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mask_pii(*, data: Any, **_: Any) -> Any:
    """Langfuse MaskFunction: applied to every input, output and metadata value before export."""
    if isinstance(data, str):
        return mask_text(data)
    if isinstance(data, dict):
        return {k: "[NAME]" if k == "person_name" and v else mask_pii(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_pii(data=v) for v in data]
    return data
