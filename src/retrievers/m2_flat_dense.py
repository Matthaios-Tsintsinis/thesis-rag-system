"""M2 — Flat dense retrieval baseline.

Leaf-chunk-only FAISS index over bge-m3 embeddings (L2-normalised,
inner product == cosine). No hierarchy, no sparse retriever. Isolates
the value of hierarchical organisation when compared against M4/M7.

Index is built in memory at index() time. Persistent on-disk caching
arrives with src.cache in a later commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..chunking import Chunk, chunk_corpus
from ..config import BASE_ANSWER_SYSTEM_PROMPT, DEFAULT_CONFIG, HarnessConfig
from ..models import embed_texts, generate
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


class FlatDenseSystem(BaseSystem):
    system_id = "M2"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self._index: Any | None = None  # faiss.IndexFlatIP

    def index(self, corpus_path: Path) -> None:
        import faiss

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
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index
        self._indexed = True

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        k = k or self.config.retrieval.top_k
        q_vec = embed_texts([query])
        scores, idxs = self._index.search(q_vec, k)
        out: list[RetrievedChunk] = []
        for rank, (i, s) in enumerate(zip(idxs[0].tolist(), scores[0].tolist())):
            if i < 0:
                continue
            out.append(RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=rank))
        return out

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
            n_retrieval_calls=1,
        )
