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
from ..models import generate
from ..prompt_packing import count_tokens
from .base import AnswerResult, BaseSystem, RetrievedChunk


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

    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        """M1 closed-book — overrides BaseSystem.answer (no retrieval).

        Per CK-4 still populates n_input_tokens so the analyser's
        --check-budget-equality has a meaningful M1 column. M1's
        evidence-block token count is 0 by definition; the
        n_input_tokens here is system + query tokens only.
        """
        del k
        self._require_indexed()
        t0 = self._now()
        n_input_tokens = count_tokens(
            CLOSED_BOOK_SYSTEM_PROMPT + "\n" + query,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        answer = generate(
            system_prompt=CLOSED_BOOK_SYSTEM_PROMPT,
            user_prompt=query,
            cfg=self.config.generation,
        )
        return AnswerResult(
            query=query,
            answer=answer,
            retrieved=[],
            packed=[],
            latency_s=self._now() - t0,
            n_retrieval_calls=0,
            n_input_tokens=n_input_tokens,
            evidence_tokens=0,  # no retrieval, no evidence block
        )
