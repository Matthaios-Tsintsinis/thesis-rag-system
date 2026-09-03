"""Answer-side scorers for the benchmark eval layer.

extractive: SQuAD/HotpotQA-style token-F1 under the official normaliser
            (+ NFKC), max over references — the ONE answer-scoring
            contract every live loader applies.
unanswerable: canonical-string match plus a fuzzy-phrase abstention
              detector (metadata only; never a score) and the
              pure-refusal null rule.
free_form: substring match, recorded in MultiHop answer metadata.

All scorers operate on a SINGLE predicted answer string and a SINGLE
gold annotation; the benchmark-level wrapper applies max-over-references.
"""

from .extractive import (
    assert_gold_not_empty,
    normalize_qasper_answer,
    token_f1,
    extractive_max_f1,
)
from .free_form import (
    substring_match,
)
from .unanswerable import (
    ABSTENTION_PHRASES,
    is_abstention,
    score_abstention,
    score_unanswerable,
)


__all__ = [
    "normalize_qasper_answer",
    "token_f1",
    "extractive_max_f1",
    "substring_match",
    "ABSTENTION_PHRASES",
    "is_abstention",
    "assert_gold_not_empty",
    "score_abstention",
    "score_unanswerable",
]
