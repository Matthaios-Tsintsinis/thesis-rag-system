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
from ..components import (
    ResolvedComponents,
    format_components_log,
    resolve_components,
)
from ..config import (
    DEFAULT_CONFIG,
    HarnessConfig,
)
from ..models import embed_texts, load_embedder
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
        # Populated at the top of index(); reused at query time.
        self._resolved: ResolvedComponents | None = None

    def index(self, corpus_path: Path) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        # M3 has no per-system Config namespace today; None-path returns
        # pure shared defaults.
        self._resolved = resolve_components(None, self.config)
        print(f"[components] {format_components_log(self.system_id, self._resolved)}")
        chunker_cfg = self._resolved.chunker_config
        embedder_id = self._resolved.embedder_id

        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        ckey = compute_cache_key(
            chunking_config=chunker_cfg,
            embedder_model=embedder_id,
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
        docs = list(walk_corpus(corpus_path, min_chars=chunker_cfg.min_chars_per_doc))
        embedder = (
            load_embedder(embedder_id) if chunker_cfg.strategy == "semantic" else None
        )
        self.chunks = chunk_corpus(docs, chunker_cfg, embedder=embedder)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        embeddings = embed_texts([c.text for c in self.chunks], model_name=embedder_id)
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
            chunking_config=asdict(chunker_cfg),
            embedder_model=embedder_id,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
            extra={"sparse": "bm25okapi", "fusion": "rrf"},
        ).save(cdir.manifest_path)

        self._indexed = True

    @property
    def resolved_components(self) -> ResolvedComponents | None:
        return self._resolved

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        assert self._resolved is not None
        cfg = self.config.retrieval
        # Natural top-K (FINAL_CONTEXT_CHUNKS=15 default). M3 baseline
        # feeds the generator at its paper-validated context size. RRF
        # over first_stage_top_k inputs is deterministic; the `[:k]`
        # tail respects the natural baseline.
        k = k or cfg.top_k

        q_vec = embed_texts([query], model_name=self._resolved.embedder_id)
        _, dense_idx = self._dense_index.search(q_vec, cfg.first_stage_top_k)
        dense_ranking = [i for i in dense_idx[0].tolist() if i >= 0]

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        order = bm25_scores.argsort()[::-1][: cfg.first_stage_top_k]
        # Deviation from literal Cormack-2009 RRF: drop zero-BM25 docs (no
        # lexical overlap) from the sparse list before fusion. Benign — such
        # a doc sits at the bottom of the top-K list and contributes only
        # ~1/(rrf_k+rank) to fusion; arguably more correct (a doc sharing no
        # query terms earns no sparse credit).
        #
        # SCOPE, corrected 2026-08-22 (docs/FINAL_FIDELITY_AUDIT.md AF-4).
        # This previously read "Applied uniformly in M3/M4/M7", which has
        # been false since the 2026-07-29 M4 paper rebuild: RAPTOR's
        # collapsed retrieval is dense-only, so M4 has no BM25 leg and no
        # sparse list for this filter to apply to (M4 deviation 7). The
        # filter lives in M3 and M7 only.
        sparse_ranking = [i for i in order.tolist() if bm25_scores[i] > 0]

        fused = rrf_fuse([dense_ranking, sparse_ranking], k=cfg.rrf_k)[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=rank)
            for rank, (i, s) in enumerate(fused)
        ]

    # answer() inherits BaseSystem default (CK-4 shared packer).
