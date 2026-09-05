"""M2: flat dense retrieval over leaf chunks, exact FAISS search."""

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
from ..models import embed_texts
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


REQUIRED_FILES = ("chunks.jsonl", "embeddings.npy", "faiss.index")


class FlatDenseSystem(BaseSystem):
    """Embed word-window chunks with bge-m3 and search them by cosine."""

    system_id = "M2"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        """Start with no chunks, no index and no resolved components."""
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self._index: Any | None = None  # faiss.IndexFlatIP
        # index() fills this; retrieve() reads it so query and index share
        # one embedder.
        self._resolved: ResolvedComponents | None = None

    def index(self, corpus_path: Path) -> None:
        """Chunk and embed the corpus, or load the cached index for it."""
        import faiss

        # M2 overrides no component, so the None path gives the shared
        # defaults.
        # embedder BAAI/bge-m3, normalised, cosine via inner product
        # harness choice: per-paper-components rule (METHODS §A.2)
        # word window 200 words, overlap 50, docs under 200 chars dropped
        # harness choice: shared default for M2/M3 (METHODS §A.2)
        self._resolved = resolve_components(None, self.config)
        print(f"[components] {format_components_log(self.system_id, self._resolved)}")
        chunker_cfg = self._resolved.chunker_config
        embedder_id = self._resolved.embedder_id

        # Cache key = chunking config + embedder + parser identity + corpus
        # hash.
        # harness choice: content-addressed substrates (METHODS §D)
        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        ckey = compute_cache_key(
            chunking_config=chunker_cfg,
            embedder_model=embedder_id,
            corpus_hash=chash,
        )
        cdir = CacheDir(paths.cache_dir(), self.system_id, ckey)

        # A complete cache dir means chunks and index load straight from
        # disk.
        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.chunks = load_chunks(cdir.chunks_path)
            self._index = faiss.read_index(str(cdir.faiss_path))
            self._indexed = True
            return

        # Walk the corpus and chunk it.
        print(f"[{self.system_id}] cache miss → building index at {cdir.path}")
        docs = list(walk_corpus(corpus_path, min_chars=chunker_cfg.min_chars_per_doc))
        self.chunks = chunk_corpus(docs, chunker_cfg)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        # Embed every chunk and build the exact inner-product index.
        # exact FAISS IndexFlatIP
        # harness choice: exact search at this scale
        embeddings = embed_texts([c.text for c in self.chunks], model_name=embedder_id)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

        # Write chunks, embeddings and index, then the manifest last.
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
        """Return the components index() resolved, or None before index()."""
        return self._resolved

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Embed the query and return its top-k chunks by cosine score."""
        self._require_indexed()
        assert self._resolved is not None
        # k defaults to the natural top-15 the reader gets; callers who
        # want a deeper ranking pass an explicit k.
        # top-15 to the reader
        # harness choice: baselines at natural strength, no imposed budget (METHODS §A.2)
        k = k or self.config.retrieval.top_k
        q_vec = embed_texts([query], model_name=self._resolved.embedder_id)
        scores, idxs = self._index.search(q_vec, k)
        # FAISS pads short result lists with -1; skip those slots.
        out: list[RetrievedChunk] = []
        for rank, (i, s) in enumerate(zip(idxs[0].tolist(), scores[0].tolist())):
            if i < 0:
                continue
            out.append(RetrievedChunk(chunk=self.chunks[i], score=float(s), rank=rank))
        return out

    # answer() is inherited from BaseSystem: the shared packer and the
    # [N] text prompt.
