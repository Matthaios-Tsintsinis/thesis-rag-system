"""M3 — Hybrid dense + BM25 with reciprocal rank fusion.

Same chunk pool as M2 plus a BM25 sparse retriever. The two ranked lists
are fused with RRF (k=60) per Cormack et al. (2009). No hierarchy.
This isolates the value of hybrid retrieval separately from hierarchy,
so the M4/M7 wins cannot be confounded with "hybrid beat dense-only".

Same content-addressed cache as M2; the cache key also folds in the
sparse-retriever name so M2 and M3 do not share the same key.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import paths
from ..cache import (
    CacheDir,
    Manifest,
    compute_cache_key,
    corpus_content_hash,
    load_chunks,
    load_pickle,
    save_chunks,
    save_embeddings,
    save_pickle,
)
from ..chunking import Chunk, chunk_corpus
from ..config import (
    BASE_ANSWER_SYSTEM_PROMPT,
    DEFAULT_CONFIG,
    EMBEDDER_MODEL,
    HarnessConfig,
)
from ..models import embed_texts, generate, load_embedder
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
REQUIRED_FILES = ("chunks.jsonl", "embeddings.npy", "faiss.index", "bm25.pkl")


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

        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        ckey = compute_cache_key(
            chunking_config=self.config.chunking,
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            extra={"sparse": "bm25okapi", "fusion": "rrf", "rrf_k": self.config.retrieval.rrf_k},
        )
        cdir = CacheDir(paths.cache_dir(), self.system_id, ckey)

        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.chunks = load_chunks(cdir.chunks_path)
            self._dense_index = faiss.read_index(str(cdir.faiss_path))
            self._bm25 = load_pickle(cdir.bm25_path)
            self._indexed = True
            return

        print(f"[{self.system_id}] cache miss → building index at {cdir.path}")
        docs = list(
            walk_corpus(corpus_path, min_chars=self.config.chunking.min_chars_per_doc)
        )
        embedder = (
            load_embedder() if self.config.chunking.strategy == "semantic" else None
        )
        self.chunks = chunk_corpus(docs, self.config.chunking, embedder=embedder)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        embeddings = embed_texts([c.text for c in self.chunks])
        dense_index = faiss.IndexFlatIP(embeddings.shape[1])
        dense_index.add(embeddings)
        self._dense_index = dense_index

        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

        save_chunks(self.chunks, cdir.chunks_path)
        save_embeddings(embeddings, cdir.embeddings_path)
        faiss.write_index(dense_index, str(cdir.faiss_path))
        save_pickle(self._bm25, cdir.bm25_path)
        Manifest(
            system_id=self.system_id,
            cache_key=ckey,
            chunking_config=asdict(self.config.chunking),
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
            extra={"sparse": "bm25okapi", "fusion": "rrf"},
        ).save(cdir.manifest_path)

        self._indexed = True

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        cfg = self.config.retrieval
        k = k or cfg.top_k

        q_vec = embed_texts([query])
        _, dense_idx = self._dense_index.search(q_vec, cfg.first_stage_top_k)
        dense_ranking = [i for i in dense_idx[0].tolist() if i >= 0]

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
            n_retrieval_calls=2,
        )
