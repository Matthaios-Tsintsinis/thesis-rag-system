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
from ..config import (
    BASE_ANSWER_SYSTEM_PROMPT,
    DEFAULT_CONFIG,
    EMBEDDER_MODEL,
    HarnessConfig,
)
from ..models import embed_texts, generate, load_embedder
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


REQUIRED_FILES = ("chunks.jsonl", "embeddings.npy", "faiss.index")


class FlatDenseSystem(BaseSystem):
    system_id = "M2"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self._index: Any | None = None  # faiss.IndexFlatIP

    def index(self, corpus_path: Path) -> None:
        import faiss

        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        ckey = compute_cache_key(
            chunking_config=self.config.chunking,
            embedder_model=EMBEDDER_MODEL,
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
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

        save_chunks(self.chunks, cdir.chunks_path)
        save_embeddings(embeddings, cdir.embeddings_path)
        faiss.write_index(index, str(cdir.faiss_path))
        Manifest(
            system_id=self.system_id,
            cache_key=ckey,
            chunking_config=asdict(self.config.chunking),
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
        ).save(cdir.manifest_path)

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
