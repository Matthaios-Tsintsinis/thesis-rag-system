"""Token-F1 and the answer normaliser every answer scorer shares.

Runs the official HotpotQA normaliser with an NFKC fold in front of it.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter


# official: hotpot_evaluate_v1.py::normalize_answer @ 36358534
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _strip_non_ascii_punctuation(s: str) -> str:
    """Drop non-ASCII punctuation (category P) the ASCII table cannot reach."""
    # harness extension (inert on ASCII): see METHODS §C.11
    # NFKC keeps a curly apostrophe (U+2019) and string.punctuation does not
    # list it, so "don't" only matches "dont" once it goes. Symbols stay,
    # as they do in the official pipeline.
    if s.isascii():
        return s
    return "".join(
        ch for ch in s
        if ch.isascii() or not unicodedata.category(ch).startswith("P")
    )


def normalize_qasper_answer(s: str) -> str:
    """Normalise an answer: NFKC, then the official HotpotQA composition."""
    if s is None:
        return ""
    # NFKC and the punctuation fold run first; the official chain follows.
    # official: hotpot_evaluate_v1.py::normalize_answer @ 36358534
    # harness extension (inert on ASCII): see METHODS §C.11
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _strip_non_ascii_punctuation(s)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def token_f1(predicted: str, gold: str) -> float:
    """Token-F1 between a prediction and one gold answer, both normalised."""
    pred_tokens = normalize_qasper_answer(predicted).split()
    gold_tokens = normalize_qasper_answer(gold).split()
    # Both empty scores 1, one empty scores 0.
    # SQuAD 2.0 evaluate-v2.0.py rule; unreachable, loaders refuse empty gold
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    # P = shared/pred, R = shared/gold, F1 = 2PR/(P+R) over token multisets.
    # official: hotpot_evaluate_v1.py::f1_score @ 36358534
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def extractive_max_f1(predicted: str, gold_spans: tuple[str, ...]) -> float:
    """Max token-F1 over alternative gold references."""
    # NarrativeQA paper: max over the two references
    # An empty reference tuple scores 0; the loaders never produce one.
    if not gold_spans:
        return 0.0
    return max(token_f1(predicted, g) for g in gold_spans)


def assert_gold_not_empty(query_id: str, gold: str, *, benchmark: str) -> None:
    """Raise at load time when an answerable gold normalises to empty."""
    # An empty gold would score 1.0 against an empty prediction, so refuse it.
    # Null queries never come here; they score under unanswerable_rule.
    if not normalize_qasper_answer(gold):
        raise ValueError(
            f"{benchmark}: query {query_id!r} has an answerable gold answer "
            f"that normalises to empty (raw {gold!r}). Ground truth is "
            "malformed; scoring it would credit an empty prediction with "
            "1.0 under the official SQuAD no-answer rule."
        )


__all__ = [
    "assert_gold_not_empty",
    "normalize_qasper_answer",
    "token_f1",
    "extractive_max_f1",
]
