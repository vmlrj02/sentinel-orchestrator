"""Sensitive data never reaches the LLM, traces or logs: card, bank account and ID numbers are removed
from the message before anything else sees it, and the applicant is told why."""
import re

REMOVED = "[removed]"


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PATTERNS = {
    "id": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"                       # US SSN
                     r"|\b[A-Z]{5}\d{4}[A-Z]\b"                     # Indian PAN
                     r"|(?<!\d)\d{4} \d{4} \d{4}(?!\d)", re.I),     # Aadhaar, as usually written
    "bank account": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}\b"               # IBAN
                               r"|(?:account|a/c|acct)(?: number| no\.?)?[:#\s]*\d{6,18}", re.I),
}


def redact(text: str) -> tuple[str, list[str]]:
    """Returns the cleaned text and the kinds of data removed (empty list if nothing was found)."""
    found: list[str] = []

    def card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.append("card")
            return REMOVED
        return match.group(0)

    text = _CARD.sub(card, text)
    for kind, pattern in _PATTERNS.items():
        text, n = pattern.subn(REMOVED, text)
        found += [kind] * n
    return text, sorted(set(found))


def only_removed(text: str) -> bool:
    return not text.replace(REMOVED, "").strip(" .,:;-")
