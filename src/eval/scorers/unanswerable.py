"""Anchored abstention detection.

WHAT CHANGED AND WHY (P3). The detector used to substring-match a list of
phrases anywhere in the prediction, so "insufficient information" fired
inside a complete, informative answer: "The report gives insufficient
information about Q3, but revenue rose to 4.1 billion." was classified as
a refusal. Anchoring fixes that.

A prediction abstains iff its FIRST CLAUSE is a PURE HEDGE — a clause
that, once its refusal frame is removed, contains nothing but filler
words. That single rule covers both admissible shapes:

  * the hedge IS the whole utterance ("No answer available.")
  * the hedge is the leading clause ("I don't know, but ...")

and rejects the case above, where the refusal frame sits inside a clause
that also carries content ("the report gives ... about q3").

TWO FAMILIES OF FRAME, and the distinction is semantic, not cosmetic:

  * NEGATIVE-EXISTENCE frames ("the evidence does not mention X") consume
    the object to the end of the clause. Everything after the verb is the
    thing declared ABSENT, so it asserts nothing. This is what makes "The
    evidence does not cover 2023." a pure refusal rather than a claim
    about 2023 — the case that killed the entity/digit heuristic
    originally proposed, which would have scored it as a fabrication.
  * NOUN-PHRASE hedges ("insufficient information") are matched tightly,
    and the filler test on the remainder does the work. "There is
    insufficient information" leaves "there is" (filler, abstains);
    "the report gives insufficient information about q3" leaves "the
    report gives about q3" (contentful, does not abstain).

`detect_abstention` returns the matched SPAN as well as the boolean,
because the unanswerable rule (P2) strips the hedge and inspects what is
left. Detection and the null-query rule therefore share one primitive
instead of running two heuristics that can disagree.

WHERE THIS IS LOAD-BEARING. On answerable queries the result is pure
metadata (`answer.metadata.abstained`) and never touches a score — see
the scoring contract in `src/eval/multihop.py`. It is load-bearing only
for the MultiHop null-query rule.

DEVIATION FROM THE OFFICIAL QASPER METRIC (answer side, unanswerable
type), unchanged by P3 and restated here because it belongs with the
detector. The official QASPER evaluator scores
token_f1(prediction, "Unanswerable"); this harness scores an abstention
DETECTION instead, because the shared reader prompt instructs every
system to emit "No answer available." rather than the word
"Unanswerable", so the official reference string would score ~0 on every
correct abstention under our prompt. Applied identically to all systems.
Not comparable to published QASPER Unanswerable-F1. Methods-section note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...config import ABSTENTION_RESPONSE


# --- vocabulary -----------------------------------------------------------

# Words that may surround a hedge without making the clause contentful.
# Deliberately CLOSED and small: every addition widens what counts as a
# refusal, and the failure mode is silent (an informative answer scored
# as an abstention). Content words are absent on purpose — "report",
# "gives", "revenue" must fall outside it.
_FILLER = frozenset("""
there here it this that these those they
i we you one
is are was were be been being am
do does did doing done
have has had
can could may might will would shall should must
not no nor
a an the
of to in on at for from with about regarding concerning
and or so but yet
sorry unfortunately apologies please note also well actually really
currently given provided based unable able possible
""".split())

# NEGATIVE-EXISTENCE frames. The object runs to the end of the clause
# (`.*`) because it names what is ABSENT.
_SUBJECT = (
    r"(?:the\s+|this\s+|that\s+)?"
    r"(?:evidence|context|passage|document|documents|text|article|excerpt|"
    r"information|source|sources|material|corpus|provided\s+\w+)"
)
_NEG = (
    r"(?:does\s+not|doesn't|do\s+not|don't|did\s+not|didn't|"
    r"cannot|can\s?not|can't|is\s+not|isn't|are\s+not|aren't|"
    r"was\s+not|wasn't|fails?\s+to)"
)
_VERB = (
    r"(?:contain|include|provide|mention|specify|state|say|indicate|"
    r"cover|give|address|discuss|answer|report|reveal|show|tell|"
    r"appear\s+to\s+\w+)"
)
_FIRST_PERSON = r"(?:i|we)"
_FP_NEG = (
    r"(?:do\s+not|don't|cannot|can\s?not|can't|am\s+not\s+able\s+to|"
    r"are\s+not\s+able\s+to|was\s+not\s+able\s+to|could\s+not|couldn't)"
)
_FP_VERB = r"(?:know|say|tell|determine|answer|find|see|identify|establish)"

_CONSUMING_PATTERNS = tuple(
    re.compile(p)
    for p in (
        rf"\b{_SUBJECT}\s+{_NEG}\s+{_VERB}\b.*$",
        rf"\b{_FIRST_PERSON}\s+{_FP_NEG}\s+{_FP_VERB}\b.*$",
        r"\bunable\s+to\s+(?:answer|determine|find|say|tell)\b.*$",
        r"\bno\s+(?:answer|information|data|details?)\s+"
        r"(?:is\s+|are\s+)?(?:available|provided|given|found)\b.*$",
        # "cannot be answered FROM THE EVIDENCE" — the trailing phrase
        # names the source that fails to supply the answer, so it is
        # frame material, not an assertion. Classifying it here rather
        # than widening _FILLER to admit "evidence" keeps the filler set
        # closed: "the report gives insufficient information about Q3"
        # must still read as contentful.
        r"\b(?:cannot|can\s?not|can't)\s+be\s+"
        r"(?:answered|determined|established|found)\b.*$",
        r"\bnot\s+(?:answerable|determinable)\b.*$",
    )
)

# NOUN-PHRASE hedges. Matched tightly; the filler test on the remainder
# decides whether the clause is pure.
_TIGHT_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bno\s+answer\s+available\b",
        r"\bunanswerable\b",
        r"\binsufficient\s+(?:information|evidence|data|context|detail)\b",
        r"\b(?:not|isn't|aren't)\s+enough\s+"
        r"(?:information|evidence|data|context|detail)\b",
        r"\bunknown\b",
    )
)

# Legacy name, kept because src/eval/scorers/__init__.py re-exports it.
# The literal cores of the patterns above, for documentation only — the
# detector no longer substring-matches this tuple.
ABSTENTION_PHRASES = (
    "no answer available",
    "unanswerable",
    "insufficient information",
    "insufficient evidence",
    "not enough information",
    "cannot be answered",
    "i don't know",
    "does not contain the answer",
)

# Clause boundaries: punctuation, or a contrastive conjunction, or a
# sentence break. A hedge after one of these is NOT leading.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[,;:]\s*|\s+(?:but|however|although|though|yet|while|whereas)\s+|\.\s+"
)

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class AbstentionMatch:
    """Result of anchored detection.

    `span` indexes `text` (the NORMALISED utterance), not the raw
    prediction — the null-query rule works in normalised coordinates, and
    handing back raw offsets would invite a caller to slice the wrong
    string.
    """

    matched: bool
    text: str
    span: tuple[int, int] | None = None

    @property
    def remainder(self) -> str:
        """The normalised utterance with the hedge clause removed."""
        if not self.matched or self.span is None:
            return self.text
        lo, hi = self.span
        return (self.text[:lo] + " " + self.text[hi:]).strip()


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower().rstrip(".")).strip()


_CANONICAL_NORMALISED = _normalise(ABSTENTION_RESPONSE)


def is_filler_only(text: str) -> bool:
    """True when `text` carries no content word.

    Shared by the detector (is the clause a PURE hedge?) and by the
    null-query rule (is the REMAINDER a pure refusal?). One primitive,
    so the two cannot drift apart.
    """
    words = _WORD_RE.findall(text or "")
    return all(w in _FILLER for w in words)


def _first_clause(text: str) -> str:
    m = _CLAUSE_BOUNDARY_RE.search(text)
    return text[: m.start()] if m else text


def _clause_is_pure_hedge(clause: str) -> bool:
    if not clause:
        return False
    for pattern in _CONSUMING_PATTERNS:
        m = pattern.search(clause)
        if m and is_filler_only(clause[: m.start()] + " " + clause[m.end():]):
            return True
    for pattern in _TIGHT_PATTERNS:
        m = pattern.search(clause)
        if m and is_filler_only(clause[: m.start()] + " " + clause[m.end():]):
            return True
    return False


def detect_abstention(predicted: str) -> AbstentionMatch:
    """Anchored detection. Returns the verdict AND the hedge-clause span."""
    text = _normalise(predicted)
    if not text:
        return AbstentionMatch(False, text)
    # The canonical response the reader prompt asks for, matched exactly
    # before anything else: it is the one string every system is
    # instructed to emit verbatim, so it must never depend on the
    # grammar below continuing to cover it.
    if text == _CANONICAL_NORMALISED:
        return AbstentionMatch(True, text, (0, len(text)))
    clause = _first_clause(text)
    if _clause_is_pure_hedge(clause):
        return AbstentionMatch(True, text, (0, len(clause)))
    return AbstentionMatch(False, text)


def is_abstention(predicted: str) -> bool:
    """True iff the prediction reads as an abstention.

    ANCHORED: the hedge must be the whole utterance or the leading
    clause. A hedge buried inside an informative clause does not count.
    """
    return detect_abstention(predicted).matched


def score_abstention(predicted: str) -> float:
    """1.0 if the prediction is an abstention; 0.0 otherwise.

    NOT the null-query rule. P2 replaces the MultiHop null-query path
    with `score_unanswerable`, which additionally requires that the
    prediction assert nothing beyond the hedge. This remains for the
    dropped-benchmark callers (QASPER) that still score on detection
    alone.
    """
    return 1.0 if is_abstention(predicted) else 0.0


__all__ = [
    "ABSTENTION_PHRASES",
    "AbstentionMatch",
    "detect_abstention",
    "is_abstention",
    "is_filler_only",
    "score_abstention",
]
