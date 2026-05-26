"""M2 — Flat dense retrieval baseline.

Leaf-chunk-only FAISS index over bge-m3 embeddings (L2-normalised,
inner product == cosine). No hierarchy, no sparse retriever. Isolates
the value of hierarchical organisation when compared against M4/M7.

Index artifacts (chunks, embeddings, FAISS index) are cached on disk
keyed by hash(chunking_config + embedder_model + corpus_content). A
hit means index() is essentially free across Colab sessions.
"""

from __future__ import annotations

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
    save_chunks,
    save_embeddings,
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


REQUIRED_FILES = ("chunks.jsonl", "embeddings.npy", "faiss.index")


class FlatDenseSystem(BaseSystem):
    system_id = "M2"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self._index: Any | None = None  # faiss.IndexFlatIP
        # Populated at the top of index(); used at query time by retrieve()
        # so embedder identity is consistent across index and query.
        self._resolved: ResolvedComponents | None = None

    def index(self, corpus_path: Path) -> None:
        import faiss

        # M2 has no per-system Config namespace today (it does not override
        # any component); the None path returns pure shared defaults.
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
        )
        cdir = CacheDir(paths.cache_dir(), self.system_id, ckey)

        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.chunks = load_chunks(cdir.chunks_path)
            self._index = faiss.read_index(str(cdir.faiss_path))
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
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

        save_chunks(self.chunks, cdir.chunks_path)
        save_embeddings(embeddings, cdir.embeddings_path)
        faiss.write_index(index, str(cdir.faiss_path))
        Manifest(
            system_id=self.system_id,
            cache_key=ckey,
            chunking_config=asdict(chunker_cfg),
            embedder_model=embedder_id,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
        ).save(cdir.manifest_path)

        self._indexed = True

    @property
    def resolved_components(self) -> ResolvedComponents | None:
        return self._resolved

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        assert self._resolved is not None
        # Natural top-K (FINAL_CONTEXT_CHUNKS=15 default). M2 baseline
        # feeds the generator what its paper would. The CK-4 packer
        # at answer() time is a no-op pass-through when the opt-in
        # --evidence-budget flag is OFF (the default). Callers who
        # want a deeper menu (for opt-in budget ablations) pass an
        # explicit k.
        k = k or self.config.retrieval.top_k
        q_vec = embed_texts([query], model_name=self._resolved.embedder_id)
        scores, idxs = self._index.search(q_vec, k)
        out: list[RetrievedChunk] = []
        for rank, (i, s) in enumerate(zip(idxs[0].tolist(), scores[0].tolist())):
            if i < 0:
                continue
            out.append(RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=rank))
        return out

    # answer() inherits BaseSystem default (CK-4 shared packer +
    # uniform [N] {text} prompt + n_input_tokens instrumentation).
