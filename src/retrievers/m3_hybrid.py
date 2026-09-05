"""M3: hybrid dense + BM25 retrieval fused with reciprocal rank fusion.

Same chunk pool and cache layout as M2 plus a BM25 leg; the cache key
folds in the sparse retriever so M2 and M3 never share a substrate.
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
from ..models import embed_texts
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


# harness choice: simplest deterministic tokeniser (METHODS §A.3)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Files a complete M3 substrate holds.
REQUIRED_FILES = ("chunks.jsonl", "embeddings.npy", "faiss.index", "bm25.pkl")


def _tokenize(text: str) -> list[str]:
    """Lowercase \\w+ tokens, no stemming and no stopwords."""
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(
    rankings: list[list[int]],
    k: int,
) -> list[tuple[int, float]]:
    """Fuse ranked id lists with RRF; return (id, score), best first."""
    scores: dict[int, float] = {}
    # RRF (Cormack et al. 2009): score = sum 1/(k + rank), rank 1-based, k = 60
    # enumerate is 0-based, so rank + 1 is the 1-based rank.
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRRFSystem(BaseSystem):
    """Hybrid dense + BM25 retriever with RRF fusion over the M2 chunk pool."""
    system_id = "M3"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self._dense_index: Any | None = None
        self._bm25: Any | None = None
        # Resolved at the top of index() and reused at query time.
        self._resolved: ResolvedComponents | None = None

    def index(self, corpus_path: Path) -> None:
        """Build or load the chunk pool, the dense index and the BM25 model."""
        import faiss
        from rank_bm25 import BM25Okapi

        # M3 has no per-system config; None resolves to the shared defaults.
        # harness choice: shared default for M2/M3 (METHODS §A.2)
        # harness choice: per-paper-components rule (METHODS §A.2)
        self._resolved = resolve_components(None, self.config)
        print(f"[components] {format_components_log(self.system_id, self._resolved)}")
        chunker_cfg = self._resolved.chunker_config
        embedder_id = self._resolved.embedder_id

        # The sparse leg and fusion settings go into extra, so M2 and M3
        # keep distinct keys over the same chunk pool.
        # harness choice: content-addressed substrates (METHODS §D)
        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        ckey = compute_cache_key(
            chunking_config=chunker_cfg,
            embedder_model=embedder_id,
            corpus_hash=chash,
            extra={"sparse": "bm25okapi", "fusion": "rrf", "rrf_k": self.config.retrieval.rrf_k},
        )
        cdir = CacheDir(paths.cache_dir(), self.system_id, ckey)

        # Cache hit: load the chunks, the dense index and the BM25 model.
        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.chunks = load_chunks(cdir.chunks_path)
            self._dense_index = faiss.read_index(str(cdir.faiss_path))
            self._bm25 = load_pickle(cdir.bm25_path)
            self._indexed = True
            return

        # Cache miss: walk the corpus and chunk it.
        print(f"[{self.system_id}] cache miss → building index at {cdir.path}")
        docs = list(walk_corpus(corpus_path, min_chars=chunker_cfg.min_chars_per_doc))
        self.chunks = chunk_corpus(docs, chunker_cfg)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        # Dense leg: embed every chunk and index by inner product.
        # harness choice: exact search at this scale
        embeddings = embed_texts([c.text for c in self.chunks], model_name=embedder_id)
        dense_index = faiss.IndexFlatIP(embeddings.shape[1])
        dense_index.add(embeddings)
        self._dense_index = dense_index

        # Sparse leg: BM25 over the same chunks.
        # ref: rank_bm25 @ 47aa3ddf (BM25Okapi defaults)
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

        # Persist every artifact, manifest last, so the dir is complete
        # only once everything is on disk.
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
        """Components resolved by index(), or None before it runs."""
        return self._resolved

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Fuse the dense and BM25 top-50 lists and return the top-k chunks."""
        self._require_indexed()
        assert self._resolved is not None
        cfg = self.config.retrieval
        # Top-15 to the reader unless the caller asks for another depth.
        # harness choice: baselines at natural strength, no imposed budget (METHODS §A.2)
        k = k or cfg.top_k

        # Dense leg: top-50 by inner product.
        # harness choice: 50 per leg (METHODS §A.3)
        q_vec = embed_texts([query], model_name=self._resolved.embedder_id)
        _, dense_idx = self._dense_index.search(q_vec, cfg.first_stage_top_k)
        dense_ranking = [i for i in dense_idx[0].tolist() if i >= 0]

        # Sparse leg: top-50 by BM25 score, then drop chunks that share no
        # query term (score <= 0) so they earn no sparse credit.
        # deviation from RRF (no sparse credit without a shared term): see METHODS §A.3
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        order = bm25_scores.argsort()[::-1][: cfg.first_stage_top_k]
        sparse_ranking = [i for i in order.tolist() if bm25_scores[i] > 0]

        # Fuse both legs with k = 60 from config and keep the top-k.
        fused = rrf_fuse([dense_ranking, sparse_ranking], k=cfg.rrf_k)[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=rank)
            for rank, (i, s) in enumerate(fused)
        ]

    # answer() is the BaseSystem default: pack the chunks and call the reader.
