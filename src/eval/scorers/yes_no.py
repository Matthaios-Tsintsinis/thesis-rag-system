"""Yes/No answer scorer (QASPER).

Light normalisation: strip, lowercase, drop trailing punctuation, look
for an unambiguous 'yes' or 'no' token at the head of the prediction.
The reader prompt does not pin Yes/No format, so the scorer is liberal
in what it accepts while still requiring an unambiguous polarity.
"""

from __future__ import annotations

import re


_HEAD_RE = re.compile(r"^\W*(yes|no|true|false)\b", re.IGNORECASE)


def normalize_yes_no(predicted: str) -> bool | None:
    """Extract a polarity from the predicted answer's head.

    Returns True for yes/true, False for no/false, None when the head
    is neither (ambiguous prediction — scorer treats as wrong).
    """
    if not predicted:
        return None
    m = _HEAD_RE.match(predicted.strip())
    if not m:
        return None
    token = m.group(1).lower()
    if token in ("yes", "true"):
        return True
    if token in ("no", "false"):
        return False
    return None


def score_yes_no(predicted: str, gold_yes_no: bool) -> float:
    """1.0 on polarity match, else 0.0. Ambiguous prediction => 0.0."""
    pol = normalize_yes_no(predicted)
    return 1.0 if pol is gold_yes_no else 0.0


__all__ = ["normalize_yes_no", "score_yes_no"]
