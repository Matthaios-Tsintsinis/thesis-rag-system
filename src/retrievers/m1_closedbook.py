"""M1 — Closed-book LLM baseline.

No retrieval. The generator answers from parametric knowledge alone.
Establishes the floor for retrieval-based systems (per evaluation plan).
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


class ClosedBookSystem(BaseSystem):
    system_id = "M1"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)

    def index(self, corpus_path: Path) -> None:
        # No retrieval index needed. Mark indexed so the harness contract
        # holds; the corpus is intentionally unused.
        del corpus_path
        self._indexed = True

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        del query, k
        return []

    def prepare(self, query: str, k: int | None = None) -> PreparedQuery:
        """PHASE A for M1: assemble the prompt. There is nothing to retrieve.

        Overriding prepare/finish rather than answer() is what makes M1
        batchable — the two-phase runner calls prepare() for every query,
        then hands all the prompts to `generate_batch` at once. The
        BaseSystem default cannot be reused because it retrieves and
        packs evidence, and M1 has neither: its prompt is the bare
        closed-book system prompt plus the question.

        Per CK-4 this still populates n_input_tokens so the analyser's
        --check-budget-equality keeps a meaningful M1 column. M1's
        evidence-block count is 0 by definition; n_input_tokens is
        system + query only.
        """
        del k
        self._require_indexed()
        t0 = self._now()
        n_input_tokens = count_tokens(
            CLOSED_BOOK_SYSTEM_PROMPT + "\n" + query,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        return PreparedQuery(
            query=query,
            retrieved=[],
            packed=[],
            # Closed-book: nothing is retrieved, so nothing is ranked.
            # Its retrieval rows are skipped by every benchmark anyway.
            scoring_ranking=[],
            system_prompt=CLOSED_BOOK_SYSTEM_PROMPT,
            user_prompt=query,
            evidence_tokens=0,  # no retrieval, no evidence block
            n_input_tokens=n_input_tokens,
            n_retrieval_calls=0,
            prepare_s=self._now() - t0,
        )
