"""BaseSystem ABC.

Every benchmarked system (M1-M8) implements this interface. The harness
treats each system as a black box: `index(corpus_path)` once, then
`answer(query)` per question.

`retrieve` is exposed separately so the harness can score retrieval
quality (Recall@k, RAGAS context_precision) independently from answer
quality.

M1 (closed-book) returns an empty list from `retrieve` — same interface,
no chunks. This keeps the harness uniform.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..chunking import Chunk
from ..config import DEFAULT_CONFIG, HarnessConfig


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class AnswerResult:
    query: str
    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    latency_s: float = 0.0
    n_retrieval_calls: int = 0
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    extra: dict = field(default_factory=dict)


class BaseSystem(ABC):
    """Abstract benchmarked system."""

    system_id: str = "base"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._indexed: bool = False

    @abstractmethod
    def index(self, corpus_path: Path) -> None:
        """Parse, chunk, embed, and build whatever structures the system needs."""

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return up to k retrieved chunks, ordered by descending score."""

    @abstractmethod
    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        """Retrieve evidence then generate an answer. Records timing/usage."""

    def _require_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError(f"{self.system_id} not indexed — call .index() first")

    @staticmethod
    def _now() -> float:
        return time.perf_counter()
