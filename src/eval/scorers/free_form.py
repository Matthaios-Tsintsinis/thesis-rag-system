"""Free-form answer scorers (MultiHop-RAG Pass-1).

`substring_match` is a robust complement to token-F1 for short factual
answers like "Sam Bankman-Fried", "Apple", "2023" — answers where
a verbose-but-correct prediction would get penalised by lexical F1.

Word-boundary regex prevents false positives (e.g. "Sam Bank" should
NOT match inside "Sam Bankman-Fried"). The boundary respects digit
runs too: "2023" must NOT match inside "120234" or "2023rd".

Direction-symmetric: gold-in-predicted OR predicted-in-gold both count.

NOTE: Per MultiHop scorer design rulings, substring_match is recorded
in AnswerScore.metadata for analysis, but the PRIMARY scoring value
remains token_F1 — substring's leniency lets false positives like
"not Apple, it's Samsung" score 1.0 against gold "Apple", which would
inflate Pass-1 answer accuracy unfaithfully. Pass-2 swaps the primary
to an LLM judge.
"""

from __future__ import annotations

import re


_NORMALISE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding whitespace and
    trailing dot/period. Keeps internal punctuation (hyphens, commas)
    so word-boundary regex can detect them as boundaries."""
    if not text:
        return ""
    s = text.lower().strip().rstrip(".")
    return _NORMALISE_RE.sub(" ", s).strip()


def _word_boundary_match(haystack: str, needle: str) -> bool:
    """True iff `needle` appears inside `haystack` surrounded by non-`\\w`
    characters (or string edges). Treats letters AND digits as `\\w`,
    so "2023" does NOT match inside "120234" (digit before) or
    "2023rd" (letter after). Direction-asymmetric; caller checks both
    directions for the symmetric variant.
    """
    if not haystack or not needle:
        return False
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def substring_match(predicted: str, gold: str) -> float:
    """Word-boundary substring match, direction-symmetric. Returns 0.0 or 1.0."""
    p = _normalise(predicted)
    g = _normalise(gold)
    if not p or not g:
        return 0.0
    if _word_boundary_match(p, g):
        return 1.0
    if _word_boundary_match(g, p):
        return 1.0
    return 0.0


__all__ = ["substring_match"]
