"""M1: closed-book baseline. The reader answers from its own knowledge;
nothing is retrieved, so retrieval scores exist only for M2-M4.
"""

from __future__ import annotations

from pathlib import Path

from ..config import (
    CLOSED_BOOK_SYSTEM_PROMPT,
    DEFAULT_CONFIG,
    EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    HarnessConfig,
)
from ..prompt_packing import count_tokens
from .base import BaseSystem, PreparedQuery, RetrievedChunk


# The prompt asks for an answer from the reader's own knowledge, or the
# shared refusal string when it has none.
# harness choice: same refusal string the null rule recognises (METHODS §A.1)
class ClosedBookSystem(BaseSystem):
    """No-retrieval system: the reader alone, prompted with the question."""

    system_id = "M1"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)

    def index(self, corpus_path: Path) -> None:
        """Mark the system indexed; there is no corpus to index."""
        # Keep the harness contract (index before prepare) without an index.
        del corpus_path
        self._indexed = True

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return no chunks; M1 never retrieves."""
        del query, k
        return []

    def prepare(self, query: str, k: int | None = None) -> PreparedQuery:
        """Build the closed-book prompt: system prompt plus the question."""
        del k
        self._require_indexed()
        t0 = self._now()
        # Count prompt tokens so M1 reports n_input_tokens like every system.
        n_input_tokens = count_tokens(
            CLOSED_BOOK_SYSTEM_PROMPT + "\n" + query,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        # No evidence, no ranking: every benchmark skips M1's retrieval rows.
        return PreparedQuery(
            query=query,
            retrieved=[],
            packed=[],
            scoring_ranking=[],
            system_prompt=CLOSED_BOOK_SYSTEM_PROMPT,
            user_prompt=query,
            evidence_tokens=0,
            n_input_tokens=n_input_tokens,
            n_retrieval_calls=0,
            prepare_s=self._now() - t0,
        )
