"""Word-boundary substring match, the MultiHop answer-metric complement.

Recorded in AnswerScore.metadata beside token-F1; it never scores a cell.
"""

from __future__ import annotations

import re


_NORMALISE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, drop trailing dots, collapse whitespace, keep punctuation."""
    if not text:
        return ""
    s = text.lower().strip().rstrip(".")
    return _NORMALISE_RE.sub(" ", s).strip()


def _word_boundary_match(haystack: str, needle: str) -> bool:
    """True iff needle is in haystack with no word character on either side."""
    if not haystack or not needle:
        return False
    # \w covers digits, so "2023" never matches in "120234" or "2023rd".
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


# Not the official rule, which credits one shared lowercased token; this
# needs one whole normalised string inside the other, at word boundaries.
# harness choice: recorded beside token-F1, never scored
def substring_match(predicted: str, gold: str) -> float:
    """Return 1.0 if either normalised string contains the other, else 0.0."""
    p = _normalise(predicted)
    g = _normalise(gold)
    if not p or not g:
        return 0.0
    # Either direction counts: gold in prediction or prediction in gold.
    if _word_boundary_match(p, g):
        return 1.0
    if _word_boundary_match(g, p):
        return 1.0
    return 0.0


__all__ = ["substring_match"]
