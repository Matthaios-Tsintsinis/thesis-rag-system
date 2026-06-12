"""Multiple-choice scorer with robust option extraction (QuALITY).

gpt-4o-mini answers an MC prompt in many shapes: "B", "(B)", "B.",
"Option B", "Answer: B", a verbatim restatement of the option text, a
paraphrase of it, or an abstention. `extract_choice` resolves all of
those to a letter (or None) through rules applied in PRIORITY ORDER:

  1. Explicit letter markers — a bare/decorated leading letter or an
     "option/answer/choice ... X" marker anywhere. An explicit letter
     is the strongest signal and beats everything else.
     LEADING-A GUARD: a bare leading "A" followed by ordinary words is
     indistinguishable from the English article ("A good story
     about..."), so the bare-letter-plus-words pattern covers B-D
     ONLY. A leading "A" still resolves via the fullmatch,
     punctuation-anchored, or marker patterns ("Answer: A ...").
     Bare-A-plus-words answers fall through BY DESIGN and surface in
     the analyser's unparseable rate — do NOT "fix" this by adding a
     bare ^A pattern; that reintroduces the article collision.
  2. Abstention — checked AFTER the letter rules (an explicit letter
     beats abstention phrasing: "I'm not sure, but the answer is B")
     and BEFORE text matching (so a distractor that happens to share
     tokens with abstention phrasing cannot be fuzzy-matched).
  3. Option-text match — normalised exact equality, then containment
     in either direction. Picks ONLY on a unique match.
  4. Token-F1 fallback — best option by token F1, gated by
     RULE3_MIN_F1 and a RULE3_MIN_MARGIN gap over the runner-up so
     near-ties fall through instead of guessing.
  5. Unparseable.

Both abstention and unparseable score 0.0 (QuALITY has no
unanswerable questions) but stay distinguishable in metadata so the
analyser can report them separately per system — one system landing
disproportionately in token_f1/unparseable means its output format
fights the extractor, which must be visible, not buried in zeros.
"""

from __future__ import annotations

import re
from typing import Sequence

from .extractive import normalize_qasper_answer, token_f1
from .unanswerable import is_abstention


# Token-F1 fallback gates (rule 4). Named constants so the derivation
# of any future change is visible in the diff, not buried in a literal.
RULE3_MIN_F1 = 0.5      # minimum best-option token F1
RULE3_MIN_MARGIN = 0.1  # required best-vs-runner-up gap

LETTERS = "ABCD"

# Entire output is one (decorated) letter: "B", "(b)", "**C**".
_FULLMATCH_RE = re.compile(r"^\W*([A-Da-d])\W*$")
# Leading letter anchored by punctuation: "B) ...", "b. ...", "(C): ...".
_LEAD_PUNCT_RE = re.compile(r"^\W{0,3}([A-Da-d])\s*[).:\-]")
# Leading bare letter + following words — case-sensitive, B-D ONLY
# (leading-A article-collision guard; see module docstring).
_LEAD_BARE_RE = re.compile(r"^([B-D])\b")
# Marker word + letter anywhere: "Option B", "Answer: D", "choice is (a)".
_MARKER_RE = re.compile(
    r"\b(?:option|answer|choice)\s*(?:is\s*)?[:\-]?\s*\(?([A-Da-d])\)?\b",
    re.IGNORECASE,
)
# Parenthesised letter anywhere: "... (B) ...".
_PAREN_RE = re.compile(r"\(([A-Da-d])\)")


def _letter_from_match(m: re.Match) -> str:
    return m.group(1).upper()


def extract_choice(
    predicted: str,
    options: Sequence[str],
) -> tuple[str | None, str]:
    """Resolve a free-text MC answer to a letter.

    Returns (letter, method). letter is "A"-"D" or None; method is one
    of: letter_leading, letter_marker, abstention, text_exact,
    text_containment, token_f1, unparseable.
    """
    text = (predicted or "").strip()
    if not text:
        return None, "unparseable"

    # Rule 1 — explicit letters.
    for pattern in (_FULLMATCH_RE, _LEAD_PUNCT_RE, _LEAD_BARE_RE):
        m = pattern.match(text)
        if m:
            return _letter_from_match(m), "letter_leading"
    m = _MARKER_RE.search(text)
    if m:
        return _letter_from_match(m), "letter_marker"
    m = _PAREN_RE.search(text)
    if m:
        return _letter_from_match(m), "letter_marker"

    # Rule 2 — abstention (after letters, before text matching).
    if is_abstention(text):
        return None, "abstention"

    # Rule 3 — normalised option-text match (unique matches only).
    norm_pred = normalize_qasper_answer(text)
    norm_opts = [normalize_qasper_answer(o) for o in options]
    if norm_pred:
        exact = [i for i, o in enumerate(norm_opts) if o and o == norm_pred]
        if len(exact) == 1:
            return LETTERS[exact[0]], "text_exact"
        contained = [
            i
            for i, o in enumerate(norm_opts)
            if o and (o in norm_pred or norm_pred in o)
        ]
        if len(contained) == 1:
            return LETTERS[contained[0]], "text_containment"

    # Rule 4 — token-F1 fallback, gated by threshold + margin.
    f1s = [token_f1(text, o) for o in options]
    if f1s:
        ranked = sorted(range(len(f1s)), key=lambda i: f1s[i], reverse=True)
        best = ranked[0]
        runner_up = f1s[ranked[1]] if len(ranked) > 1 else 0.0
        if f1s[best] >= RULE3_MIN_F1 and (f1s[best] - runner_up) >= RULE3_MIN_MARGIN:
            return LETTERS[best], "token_f1"

    return None, "unparseable"


def score_multiple_choice(
    predicted: str,
    options: Sequence[str],
    gold_label: int,
) -> tuple[float, str, dict]:
    """Accuracy vs a 1-indexed gold label (QuALITY convention).

    Returns (value, method, metadata): value is 1.0 iff the extracted
    letter equals the gold letter, else 0.0 (abstention and
    unparseable both score 0.0 — QuALITY has no unanswerable
    questions — but remain distinguishable via metadata).
    """
    if not 1 <= gold_label <= len(options):
        raise ValueError(
            f"gold_label must be 1..{len(options)} (1-indexed); got {gold_label}"
        )
    letter, method = extract_choice(predicted, options)
    gold_letter = LETTERS[gold_label - 1]
    value = 1.0 if letter == gold_letter else 0.0
    metadata = {
        "predicted_letter": letter,
        "gold_letter": gold_letter,
        "extraction": method,
        "abstained": method == "abstention",
        "unparseable": method == "unparseable",
    }
    return value, method, metadata


__all__ = [
    "LETTERS",
    "RULE3_MIN_F1",
    "RULE3_MIN_MARGIN",
    "extract_choice",
    "score_multiple_choice",
]
