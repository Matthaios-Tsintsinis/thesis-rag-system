"""QASPER extractive-answer scorer: SQuAD-style token-F1 with QASPER normaliser.

Verbatim port of the QASPER official evaluator's `normalize_answer` and
F1 computation (Dasigi et al., 2021; their qasper_eval.py). Lower-case,
strip articles (a/an/the), drop punctuation, collapse whitespace. Then
SQuAD-style token F1 between predicted and gold token sets.

IMPORTANT — what the official QASPER scorer actually does with multiple
extractive spans: it JOINS an annotator's spans into ONE reference
string (", ".join(spans)) and computes a SINGLE token-F1 against it,
taking the max only ACROSS annotators/references — never a per-span
max. The QASPER extractive path therefore joins-then-scores (see
qasper.py _score_one_annotator) to match the official evaluator.

`extractive_max_f1` below returns the MAX token-F1 over a tuple of
reference strings. That is the right metric only when the references
are genuine ALTERNATIVES (e.g. NarrativeQA's two independent reference
answers, narrativeqa.py), NOT when they are co-required spans of one
answer. Do NOT use it for QASPER extractive scoring.

token_f1 is also used for QASPER abstractive answers in Pass-1 (token
F1 vs `free_form`). Pass-2 adds an LLM-judge for abstractive semantic
equivalence on top.
"""

from __future__ import annotations

import re
import string
from collections import Counter


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_qasper_answer(s: str) -> str:
    """Lowercase + strip articles + drop punctuation + collapse whitespace.

    Ported from QASPER official `qasper_eval.normalize_answer`. The
    ordering matters: lowercase first, THEN article-strip (otherwise
    a capitalised "The" survives), THEN punctuation, THEN whitespace
    collapse.
    """
    if s is None:
        return ""
    s = s.lower()
    s = _ARTICLES_RE.sub(" ", s)
    s = s.translate(_PUNCT_TABLE)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def token_f1(predicted: str, gold: str) -> float:
    """SQuAD-style token F1 between two normalised answer strings.

    Returns 1.0 when both normalise to empty (vacuously equal);
    0.0 when only one is empty. Otherwise standard F1 over token
    multisets (Counter intersection).
    """
    pred_tokens = normalize_qasper_answer(predicted).split()
    gold_tokens = normalize_qasper_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def extractive_max_f1(predicted: str, gold_spans: tuple[str, ...]) -> float:
    """Max token-F1 over a list of gold extractive spans.

    QASPER official scorer's behaviour when a question has multiple
    extractive spans (e.g. enumerated mentions of the same concept).
    Empty gold list returns 0 — a degenerate input the loader should
    not produce.
    """
    if not gold_spans:
        return 0.0
    return max(token_f1(predicted, g) for g in gold_spans)


__all__ = [
    "normalize_qasper_answer",
    "token_f1",
    "extractive_max_f1",
]
