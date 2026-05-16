"""M4 — Official RAPTOR (collapsed retrieval).

The eval plan's primary headline baseline (Section 3, system M4). The
RAPTOR paper [R3] reports collapsed retrieval as the stronger variant
of RAPTOR over single-path tree traversal; this implementation matches
that variant and PIPELINE_DESIGN.md sections 3.4, 3.5, and 4.4 Axis-1
Part A.

NO cross-encoder rerank in M4 — the published collapsed-RAPTOR pipeline
has none, and reranking is M7's contribution to attribute, not M4's.
The flat collapsed index returns chunks (after summary-node expansion)
ranked by RRF of dense and BM25 first-stage retrieval.

Index-time substrate (chunks, embeddings, tree with LLM summaries,
flat collapsed index, BM25 over leaf chunks) is shared with M7 via
src/raptor.py. M7 layers Axes 2/3 on top of this exact substrate.

Cache key folds in:
  * shared: chunking + embedder + parsing + corpus content hash
  * M4 extras: tree build params, summary model + prompt version,
    include_root flag, sparse retriever name, fusion type + RRF k.
Swapping the summarizer (e.g. gpt-4o-mini -> gpt-4.1-mini) or bumping
SUMMARY_PROMPT_VERSION invalidates every cached tree summary cleanly.
The answer generator is NOT in the key (it runs at query time, leaves
no baked artifact).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..cache import (
    CacheDir,
    Manifest,
    compute_cache_key,
    corpus_content_hash,
    load_chunks,
    load_embeddings,
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
from ..raptor import (
    RAPTOR_SUBSTRATE_NAMESPACE,
    FlatCollapsedIndex,
    RaptorNode,
    RaptorTree,
    build_flat_collapsed_index,
    build_raptor_tree,
    expand_node,
    load_flat_index,
    load_raptor_tree,
    raptor_substrate_extra,
    save_flat_index,
    save_raptor_tree,
)
from ..summarization import (
    SUMMARY_PROMPT_VERSION,
    summarize_passages,
)
from .base import AnswerResult, BaseSystem, RetrievedChunk


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


REQUIRED_FILES = (
    "chunks.jsonl",
    "embeddings.npy",
    "bm25.pkl",
    "raptor_tree.json",
    "raptor_summary_embeddings.npy",
    "flat_collapsed.index",
    "flat_collapsed_meta.json",
)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _rrf_fuse(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """RRF (Cormack et al. 2009) over rank lists in a unified id space."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class RaptorSystem(BaseSystem):
    system_id = "M4"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self.chunk_embeddings: np.ndarray | None = None
        self._bm25: Any | None = None
        self._tree: RaptorTree | None = None
        self._flat: FlatCollapsedIndex | None = None

        # Index-time stats. Always populated (cheap); surfaced when trace=True.
        self._index_stats: dict = {}
        # Per-query trace. Populated only when self.config.m4.trace is True.
        self._last_trace: dict = {}

    # --- index ----------------------------------------------------------------

    def index(self, corpus_path: Path) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        m4 = self.config.m4
        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        # Shared RAPTOR substrate key: no system_id field, so M4 and M7
        # land on the same RAPTOR/<substrate_hash>/ directory and reuse
        # one copy of chunks/embeddings/bm25/tree/flat index.
        substrate_extra = raptor_substrate_extra(
            build=m4.build,
            summary_model=m4.summary_model,
            summary_prompt_version=SUMMARY_PROMPT_VERSION,
            include_root=m4.include_root_in_flat_index,
            rrf_k=m4.rrf_k,
        )
        ckey = compute_cache_key(
            chunking_config=self.config.chunking,
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            extra=substrate_extra,
        )
        cdir = CacheDir(paths.cache_dir(), RAPTOR_SUBSTRATE_NAMESPACE, ckey)

        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.chunks = load_chunks(cdir.chunks_path)
            self.chunk_embeddings = load_embeddings(cdir.embeddings_path)
            self._bm25 = load_pickle(cdir.bm25_path)
            self._tree = load_raptor_tree(
                cdir.path / "raptor_tree.json",
                cdir.path / "raptor_summary_embeddings.npy",
            )
            self._flat = load_flat_index(
                cdir.path / "flat_collapsed.index",
                cdir.path / "flat_collapsed_meta.json",
            )
            self._index_stats = self._collect_index_stats(summary_calls=0)
            self._indexed = True
            return

        print(f"[{self.system_id}] cache miss -> building index at {cdir.path}")
        docs = list(
            walk_corpus(corpus_path, min_chars=self.config.chunking.min_chars_per_doc)
        )
        embedder = (
            load_embedder() if self.config.chunking.strategy == "semantic" else None
        )
        self.chunks = chunk_corpus(docs, self.config.chunking, embedder=embedder)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        self.chunk_embeddings = embed_texts([c.text for c in self.chunks])

        # BM25 over leaf chunks only — summary nodes are paraphrased text where
        # BM25 is unreliable (PIPELINE_DESIGN section 4.4).
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

        # LLM-summarised cluster tree.
        summary_calls = [0]

        def _on_summary(_n: RaptorNode) -> None:
            summary_calls[0] += 1

        def _summarize(passages: list[str]) -> str:
            return summarize_passages(passages, model=m4.summary_model)

        self._tree = build_raptor_tree(
            chunk_texts=[c.text for c in self.chunks],
            chunk_embeddings=self.chunk_embeddings,
            params=m4.build,
            summarize_fn=_summarize,
            embed_fn=embed_texts,
            on_summary=_on_summary,
        )
        print(
            f"[{self.system_id}] tree built: {len(self._tree.nodes)} nodes, "
            f"{summary_calls[0]} gpt-4o-mini calls"
        )

        # Flat collapsed index: chunks + non-root summary embeddings, one matrix.
        self._flat = build_flat_collapsed_index(
            self._tree,
            self.chunk_embeddings,
            expansion=m4.expansion,
            include_root=m4.include_root_in_flat_index,
        )

        # --- persist ---
        save_chunks(self.chunks, cdir.chunks_path)
        save_embeddings(self.chunk_embeddings, cdir.embeddings_path)
        save_pickle(self._bm25, cdir.bm25_path)
        save_raptor_tree(
            self._tree,
            cdir.path / "raptor_tree.json",
            cdir.path / "raptor_summary_embeddings.npy",
        )
        save_flat_index(
            self._flat,
            cdir.path / "flat_collapsed.index",
            cdir.path / "flat_collapsed_meta.json",
        )

        self._index_stats = self._collect_index_stats(summary_calls=summary_calls[0])
        Manifest(
            system_id=RAPTOR_SUBSTRATE_NAMESPACE,
            cache_key=ckey,
            chunking_config=asdict(self.config.chunking),
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
            extra={
                "m4": asdict(m4),
                "index_stats": self._index_stats,
            },
        ).save(cdir.manifest_path)

        self._indexed = True

    def _collect_index_stats(self, *, summary_calls: int) -> dict:
        assert self._tree is not None and self._flat is not None
        depth_counts = Counter(n.depth for n in self._tree.nodes.values())
        type_counts = Counter(self._flat.node_types)
        return {
            "n_summary_calls_at_index": int(summary_calls),
            "tree_n_nodes": len(self._tree.nodes),
            "tree_depth_counts": {int(d): int(c) for d, c in depth_counts.items()},
            "flat_n_chunks": int(type_counts.get("chunk", 0)),
            "flat_n_summaries": int(sum(
                v for k, v in type_counts.items() if k != "chunk"
            )),
            "flat_node_type_counts": {k: int(v) for k, v in type_counts.items()},
        }

    @property
    def index_stats(self) -> dict:
        return dict(self._index_stats)

    @property
    def last_trace(self) -> dict:
        return dict(self._last_trace)

    # --- retrieve -------------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        m4 = self.config.m4
        k = k or m4.top_k_final
        trace_on = m4.trace

        assert self._flat is not None and self._tree is not None
        assert self.chunk_embeddings is not None

        q_vec = embed_texts([query])
        q_vec_1d = q_vec[0]

        # --- Dense top-K over flat collapsed index (chunks + non-root summaries) ---
        n_flat = len(self._flat.refs)
        ks = min(m4.first_stage_top_k, n_flat)
        _, dense_idx = self._flat.faiss_index.search(q_vec, ks)
        dense_flat_positions = [i for i in dense_idx[0].tolist() if i >= 0]

        # --- BM25 top-K over leaf chunks only ---
        # Chunk indices == flat positions for the chunk-typed prefix of the
        # flat index, so RRF can fuse the two rankings in flat-position space.
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = bm25_scores.argsort()[::-1][: m4.first_stage_top_k]
        sparse_flat_positions = [
            int(i) for i in bm25_order.tolist() if bm25_scores[i] > 0
        ]

        # --- RRF fuse the two rankings ---
        fused = _rrf_fuse(
            [dense_flat_positions, sparse_flat_positions], k=m4.rrf_k
        )[: m4.first_stage_top_k]

        # --- Per-node-type expansion: walk fused, expand summaries to chunks ---
        chunk_idx_score: dict[int, float] = {}
        type_counter: Counter[str] = Counter()
        paths_exercised: set[str] = set()

        for flat_pos, rrf_score in fused:
            ref = self._flat.refs[flat_pos]
            node_type = self._flat.node_types[flat_pos]
            type_counter[node_type] += 1

            if ref["type"] == "chunk":
                paths_exercised.add("leaf")
                ci = int(ref["chunk_idx"])
                if rrf_score > chunk_idx_score.get(ci, -1.0):
                    chunk_idx_score[ci] = rrf_score
                continue

            # summary node -> route via §4.4 expansion rules
            node_id = ref["node_id"]
            expanded_chunks, branch_trace = expand_node(
                node_id,
                q_vec_1d,
                self._tree,
                self.chunk_embeddings,
                expansion=m4.expansion,
                _path_trace=[],
            )
            # Tag bucket the summary lived in (high/mid/low).
            paths_exercised.add(node_type.replace("summary_", ""))
            if trace_on:
                # Branch traces are detailed for diagnostic runs only.
                self._last_trace.setdefault("branch_traces", []).append({
                    "from_node": node_id,
                    "node_type": node_type,
                    "branches": branch_trace,
                    "n_chunks": len(expanded_chunks),
                })

            # Assign the originating summary's RRF score to each expanded chunk.
            # If a chunk also appears directly later, the max wins.
            for ci in expanded_chunks:
                if rrf_score > chunk_idx_score.get(ci, -1.0):
                    chunk_idx_score[ci] = rrf_score

        # --- Rank chunks, dedupe by chunk_id, trim to k ---
        ranked = sorted(chunk_idx_score.items(), key=lambda kv: kv[1], reverse=True)

        seen_ids: set[str] = set()
        out: list[RetrievedChunk] = []
        for ci, score in ranked:
            chunk = self.chunks[ci]
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            out.append(RetrievedChunk(chunk=chunk, score=float(score), rank=len(out)))
            if len(out) >= k:
                break

        if trace_on:
            self._last_trace.update({
                "collapsed_top50_node_types": dict(type_counter),
                "paths_exercised": sorted(paths_exercised),
                "n_fused": len(fused),
                "n_unique_chunks_after_expansion": len(chunk_idx_score),
                "n_returned": len(out),
            })

        return out

    # --- answer ---------------------------------------------------------------

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
        extra: dict = {}
        if self.config.m4.trace:
            extra["trace"] = self.last_trace
        return AnswerResult(
            query=query,
            answer=ans,
            retrieved=retrieved,
            latency_s=self._now() - t0,
            n_retrieval_calls=2,
            extra=extra,
        )
