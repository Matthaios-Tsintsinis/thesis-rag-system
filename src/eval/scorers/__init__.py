"""Answer-side scorers for the benchmark eval layer.

extractive: QASPER's SQuAD-style token-F1 with the QASPER normaliser.
yes_no: exact-match on Yes/No after light normalisation.
unanswerable: canonical-string match plus a fuzzy-phrase abstention
              detector for non-canonical phrasings the reader may emit
              despite the prompt's exact-string instruction.
multiple_choice: QuALITY option extraction (letter markers with the
              leading-A article guard, abstention, text match,
              token-F1 fallback) + 1-indexed accuracy scoring.

All scorers operate on a SINGLE predicted answer string and a SINGLE
gold annotation; the benchmark-level wrapper (qasper.score_answer)
applies max-over-annotators per QASPER convention.
"""

from .extractive import (
    normalize_qasper_answer,
    token_f1,
    extractive_max_f1,
)
from .free_form import (
    substring_match,
)
from .multiple_choice import (
    RULE3_MIN_F1,
    RULE3_MIN_MARGIN,
    extract_choice,
    score_multiple_choice,
)
from .unanswerable import (
    ABSTENTION_PHRASES,
    is_abstention,
    score_abstention,
    score_unanswerable,
)
from .yes_no import (
    normalize_yes_no,
    score_yes_no,
)


__all__ = [
    "normalize_qasper_answer",
    "token_f1",
    "extractive_max_f1",
    "substring_match",
    "RULE3_MIN_F1",
    "RULE3_MIN_MARGIN",
    "extract_choice",
    "score_multiple_choice",
    "ABSTENTION_PHRASES",
    "is_abstention",
    "score_abstention",
    "score_unanswerable",
    "normalize_yes_no",
    "score_yes_no",
]
