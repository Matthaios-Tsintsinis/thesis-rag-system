"""Abstention detection for the unanswerable scorer.

Primary path: exact match against the canonical `ABSTENTION_RESPONSE`
the reader prompt instructs the model to emit ("No answer available.").
Fallback path: fuzzy phrase match against a list of common abstention
phrasings (the model may paraphrase the canonical instruction).

Used by QASPER's unanswerable answer-type scorer:
  - For an unanswerable-flagged annotation: score = is_abstention(predicted) ? 1 : 0.
  - For an answerable annotation answered with abstention: token-F1
    against the gold answer is already 0 via the extractive/abstractive
    scorer — abstention does NOT additionally penalise here.

Also used by the MultiHop-RAG `null_query` scoring path (Pass-1
skeleton: same abstention detection; full free-form scoring in Pass-2).
"""

from __future__ import annotations

import re

from ...config import ABSTENTION_RESPONSE


# Fuzzy fallback phrases. Substring match on normalised predicted text.
# Keep this list short and high-signal: a phrase here MUST be very
# unlikely to appear in a legitimate answer. "No information about X"
# would be too aggressive (could appear in real answers); "no answer
# available" / "cannot be answered" / "does not contain" are pinned to
# abstention.
ABSTENTION_PHRASES = (
    "no answer available",
    "no answer is available",
    "cannot be answered",
    "cannot answer this",
    "does not contain the answer",
    "does not contain an answer",
    "no information is provided",
    "insufficient evidence",
    "insufficient information",
    "the evidence does not",
    "not enough information",
    "unanswerable",
    "i don't know",
    "i do not know",
)


_NORMALISE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _NORMALISE_RE.sub(" ", text.strip().lower().rstrip(".")).strip()


# Pre-compute the canonical normalised form once.
_CANONICAL_NORMALISED = _normalise(ABSTENTION_RESPONSE)


def is_abstention(predicted: str) -> bool:
    """True iff the prediction reads as an abstention.

    Two layers:
      1. Exact canonical match against `ABSTENTION_RESPONSE` (deterministic
         because the reader prompt instructs the model to emit it
         verbatim).
      2. Fuzzy substring match against ABSTENTION_PHRASES (covers the
         cases where the model paraphrases the instruction).
    """
    if not predicted:
        return False
    norm = _normalise(predicted)
    if norm == _CANONICAL_NORMALISED:
        return True
    return any(phrase in norm for phrase in ABSTENTION_PHRASES)


def score_abstention(predicted: str) -> float:
    """1.0 if the prediction is an abstention; 0.0 otherwise.

    Used for QASPER unanswerable annotations and MultiHop null_query.
    """
    return 1.0 if is_abstention(predicted) else 0.0


__all__ = ["ABSTENTION_PHRASES", "is_abstention", "score_abstention"]
