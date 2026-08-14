"""SQuAD-style token-F1 with the official answer normaliser.

NORMALISER COMPOSITION, corrected 2026-08-14 and VERIFIED AGAINST SOURCE
AFTER THE CHANGE. Both official evaluators compose identically:

    white_space_fix(remove_articles(remove_punc(lower(s))))

i.e. lower -> DROP PUNCTUATION -> STRIP ARTICLES -> collapse whitespace.
Checked verbatim against `hotpotqa/hotpot` `hotpot_evaluate_v1.py` and
the SQuAD 2.0 `evaluate-v2.0.py`; the two agree on order.

This module previously ran lower -> strip articles -> drop punctuation,
under a docstring claiming a verbatim port and asserting that "the
ordering matters". It does matter, and the order was inverted: on
"the-cat" the official pipeline removes the hyphen first, leaving
"thecat" with no word boundary for the article regex to match, while the
old order matched "the" against the hyphen boundary and returned "cat".
Any answer with an article adjacent to punctuation therefore produced a
different token stream from the published evaluator. That was a
MISLABELLED deviation, not a documented one, and correcting it is what
lets the deviations table claim HotpotQA matches its official metric.

STATED DIVERGENCE FROM THE PUBLISHED NORMALISER: this implementation
adds Unicode NFKC plus a non-ASCII punctuation fold that the official
evaluators do NOT perform, so that Unicode-variant tokens ("don't" vs
"don't") cannot score differently from their ASCII forms. It is a
deliberate, principled extension rather than the mislabelled inversion it
replaced, it is applied uniformly to all three live benchmarks, and it is
provably inert on ASCII input. The deviations table records this as
"official normaliser + documented NFKC/Unicode extension", not as an
unqualified match.

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
import unicodedata
from collections import Counter


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _strip_non_ascii_punctuation(s: str) -> str:
    """Drop Unicode punctuation the official ASCII table cannot reach.

    NFKC alone does NOT fold a curly apostrophe: U+2019 is not a
    compatibility character, so it survives normalisation, and it is not
    in `string.punctuation`, so the official table leaves it too — which
    would tokenise "don't" as "don't" against "dont". Measured, not
    assumed.

    EXACTNESS ON ASCII IS PRESERVED. This runs AFTER the official table,
    so every ASCII punctuation character is already gone and only
    non-ASCII survives to be considered. Category P only: `string.punctuation`
    also contains symbols (`$ + < = > ^ | ~`, category S) which the
    official pipeline removes and which must stay removed — they are
    handled by the table above, not here — while non-ASCII symbols (a
    currency sign, say) are LEFT, exactly as the official pipeline leaves
    them.
    """
    if s.isascii():
        return s
    return "".join(
        ch for ch in s
        if ch.isascii() or not unicodedata.category(ch).startswith("P")
    )


def normalize_qasper_answer(s: str) -> str:
    """NFKC, then the official composition.

    Order, matching `white_space_fix(remove_articles(remove_punc(lower(s))))`
    in both published evaluators:

      1. NFKC       — ours, so a curly apostrophe tokenises like a straight
                      one. No effect on ASCII input.
      2. lower
      3. remove_punc
      4. remove_articles
      5. white_space_fix

    Steps 3 and 4 are in the published order, which is the reverse of what
    this function did before 2026-08-14. See the module docstring for the
    "the-cat" divergence that made the difference observable.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _strip_non_ascii_punctuation(s)
    s = _ARTICLES_RE.sub(" ", s)
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
    # OFFICIAL SQuAD 2.0 rule, quoted from evaluate-v2.0.py:
    #   if len(gold_toks) == 0 or len(pred_toks) == 0:
    #     # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    #     return int(gold_toks == pred_toks)
    #
    # THE TWO OFFICIAL REFERENCES DISAGREE HERE, and the disagreement is
    # recorded rather than resolved by preference: HotpotQA's
    # hotpot_evaluate_v1.f1_score has no no-answer branch, so two empty
    # token lists fall through to `num_same == 0` and it returns 0.
    # SQuAD's rule is adopted. The branch is UNREACHABLE in this pipeline
    # either way: the loader assertion forbids a gold that normalises to
    # empty, and the pred-only-empty case is where every implementation
    # already agrees on 0. See tests/test_normalisation.py.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
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


def assert_gold_not_empty(query_id: str, gold: str, *, benchmark: str) -> None:
    """Abort at LOAD when an answerable query carries an empty gold.

    This is the real defect behind the both-empty question, and catching
    it here is why `token_f1` can keep the official SQuAD rule unchanged:
    a gold that normalises to empty is malformed ground truth, not a
    scoring edge case, and silently scoring it 1.0 against an empty
    prediction would credit a system for saying nothing.

    Null queries are EXEMPT by construction and must not be passed here —
    their gold is empty on purpose and they score under
    `unanswerable_rule`.
    """
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
