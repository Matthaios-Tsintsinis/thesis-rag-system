"""Anchored abstention detection and the pure-refusal rule for null queries.
A prediction abstains when its leading clause is a pure hedge; the null
rule credits it only when the rest of the utterance is filler."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...config import ABSTENTION_RESPONSE


# --- vocabulary -----------------------------------------------------------
# harness addition: see METHODS §C.9

# Words that may surround a hedge without making the clause contentful.
# The set is closed and small: every addition widens what counts as a
# refusal, so content words like "report" or "revenue" stay outside it.
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

# Negative-existence frames. The object runs to the end of the clause
# because it names what is absent, so it asserts nothing.
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
        # "cannot be answered from the evidence": the trailing phrase names
        # the source that lacks the answer, so it is frame, not a claim.
        r"\b(?:cannot|can\s?not|can't)\s+be\s+"
        r"(?:answered|determined|established|found)\b.*$",
        r"\bnot\s+(?:answerable|determinable)\b.*$",
    )
)

# Noun-phrase hedges, matched tightly; the filler test on the remainder
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

# Re-exported by src/eval/scorers/__init__.py. The literal cores of the
# patterns above, for documentation only; the detector does not read it.
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

# Clause boundaries: punctuation, a contrastive conjunction, or a
# sentence break. A hedge after one of these is not leading.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[,;:]\s*|\s+(?:but|however|although|though|yet|while|whereas)\s+|\.\s+"
)

_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class AbstentionMatch:
    """Detector verdict plus the hedge span, in normalised-text offsets."""

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
    """Lowercase, collapse whitespace and drop a trailing full stop."""
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower().rstrip(".")).strip()


_CANONICAL_NORMALISED = _normalise(ABSTENTION_RESPONSE)


def is_filler_only(text: str) -> bool:
    """True when every word of the text is in the filler set."""
    words = _WORD_RE.findall(text or "")
    return all(w in _FILLER for w in words)


def _first_clause(text: str) -> str:
    """The text up to the first clause boundary."""
    m = _CLAUSE_BOUNDARY_RE.search(text)
    return text[: m.start()] if m else text


def _clause_is_pure_hedge(clause: str) -> bool:
    """True when a hedge frame matches and only filler surrounds it."""
    if not clause:
        return False
    # Either frame family fires when the text around the match is filler.
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
    """Return the verdict and the hedge-clause span for a prediction."""
    text = _normalise(predicted)
    if not text:
        return AbstentionMatch(False, text)
    # The canonical refusal string matches exactly, ahead of the grammar.
    # harness choice: the string the null rule recognises (METHODS §C.9)
    if text == _CANONICAL_NORMALISED:
        return AbstentionMatch(True, text, (0, len(text)))
    # Otherwise the leading clause must be a pure hedge.
    clause = _first_clause(text)
    if _clause_is_pure_hedge(clause):
        return AbstentionMatch(True, text, (0, len(clause)))
    return AbstentionMatch(False, text)


def is_abstention(predicted: str) -> bool:
    """True when the hedge is the whole utterance or its leading clause."""
    return detect_abstention(predicted).matched


def score_abstention(predicted: str) -> float:
    """1.0 when the prediction abstains, 0.0 otherwise; not the null rule."""
    return 1.0 if is_abstention(predicted) else 0.0


def score_unanswerable(predicted: str) -> float:
    """The null-query rule: credit 1.0 for a pure refusal, nothing else."""
    # harness addition: see METHODS §C.9
    # The detector must fire and the utterance minus the hedge clause must
    # carry no content word, so a hedge followed by a claim scores 0.0.
    match = detect_abstention(predicted)
    if not match.matched:
        return 0.0
    return 1.0 if is_filler_only(match.remainder) else 0.0


__all__ = [
    "ABSTENTION_PHRASES",
    "AbstentionMatch",
    "detect_abstention",
    "is_abstention",
    "is_filler_only",
    "score_abstention",
    "score_unanswerable",
]
