"""Answer-side scorers for the benchmark eval layer.

extractive: QASPER's SQuAD-style token-F1 with the QASPER normaliser.
yes_no: exact-match on Yes/No after light normalisation.
unanswerable: canonical-string match plus a fuzzy-phrase abstention
              detector for non-canonical phrasings the reader may emit
              despite the prompt's exact-string instruction.

All three operate on a SINGLE predicted answer string and a SINGLE
gold annotation; the benchmark-level wrapper (qasper.score_answer)
applies max-over-annotators per QASPER convention.
"""

from .extractive import (
    normalize_qasper_answer,
    token_f1,
    extractive_max_f1,
)
from .unanswerable import (
    ABSTENTION_PHRASES,
    is_abstention,
    score_abstention,
)
from .yes_no import (
    normalize_yes_no,
    score_yes_no,
)


__all__ = [
    "normalize_qasper_answer",
    "token_f1",
    "extractive_max_f1",
    "ABSTENTION_PHRASES",
    "is_abstention",
    "score_abstention",
    "normalize_yes_no",
    "score_yes_no",
]
