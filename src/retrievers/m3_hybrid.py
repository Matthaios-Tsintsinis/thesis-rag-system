"""M3 — Hybrid dense + BM25 with reciprocal rank fusion.

Same chunk pool as M2 plus a BM25 sparse retriever. The two ranked lists
are fused with RRF (k=60) per Cormack et al. (2009). No hierarchy.
This isolates the value of hybrid retrieval separately from hierarchy,
so the M4/M7 wins cannot be confounded with "hybrid beat dense-only".

Built in memory; persistent on-disk caching arrives with src.cache in a
later commit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..chunking import Chunk, chunk_corpus
from ..config import BASE_ANSWER_SYSTEM_PROMPT, DEFAULT_CONFIG, HarnessConfig
from ..models import embed_texts, generate
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(
    rankings: list[list[int]],
    k: int,
) -> list[tuple[int, float]]:
    """RRF over multiple ranked id lists. Returns (id, score) sorted desc."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRRFSystem(BaseSystem):
    system_id = "M3"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self._dense_index: Any | None = None
        self._bm25: Any | None = None

    def index(self, corpus_path: Path) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        docs = list(
            walk_corpus(corpus_path, min_chars=self.config.chunking.min_chars_per_doc)
        )
        self.chunks = chunk_corpus(
            docs,
            chunk_words=self.config.chunking.chunk_words,
            overlap_words=self.config.chunking.overlap_words,
        )
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        embeddings = embed_texts([c.text for c in self.chunks])
        dense_index = faiss.IndexFlatIP(embeddings.shape[1])
        dense_index.add(embeddings)
        self._dense_index = dense_index

        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])
        self._indexed = True

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        cfg = self.config.retrieval
        k = k or cfg.top_k

        # Dense top-N
        q_vec = embed_texts([query])
        _, dense_idx = self._dense_index.search(q_vec, cfg.first_stage_top_k)
        dense_ranking = [i for i in dense_idx[0].tolist() if i >= 0]

        # Sparse top-N
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        order = bm25_scores.argsort()[::-1][: cfg.first_stage_top_k]
        sparse_ranking = [i for i in order.tolist() if bm25_scores[i] > 0]

        fused = rrf_fuse([dense_ranking, sparse_ranking], k=cfg.rrf_k)[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=rank)
            for rank, (i, s) in enumerate(fused)
        ]

    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        self._require_indexed()
        t0 = self._now()
        retrieved = self.retrieve(query, k)
        context = "\n\n".join(f"[{r.rank + 1}] {r.chunk.text}" for r in retrieved)
        user_prompt = f"Evidence:\n{context}\n\nQuestion: {query}"
        ans = generate(
            system_prompt=BASE_ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            cfg=self.config.generation,
        )
        return AnswerResult(
            query=query,
            answer=ans,
            retrieved=retrieved,
            latency_s=self._now() - t0,
            n_retrieval_calls=2,  # dense + sparse
        )
