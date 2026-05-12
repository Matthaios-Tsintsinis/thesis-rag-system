"""M1 — Closed-book LLM baseline.

No retrieval. The generator answers from parametric knowledge alone.
Establishes the floor for retrieval-based systems (per evaluation plan).
"""

from __future__ import annotations

from pathlib import Path

from ..config import CLOSED_BOOK_SYSTEM_PROMPT, DEFAULT_CONFIG, HarnessConfig
from ..models import generate
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
        del k
        self._require_indexed()
        t0 = self._now()
        answer = generate(
            system_prompt=CLOSED_BOOK_SYSTEM_PROMPT,
            user_prompt=query,
            cfg=self.config.generation,
        )
        return AnswerResult(
            query=query,
            answer=answer,
            retrieved=[],
            latency_s=self._now() - t0,
            n_retrieval_calls=0,
        )
