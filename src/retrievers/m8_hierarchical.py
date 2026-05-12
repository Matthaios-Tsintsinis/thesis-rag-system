"""M8 — Existing hierarchical RAG (port of the Colab notebook).

Distinguishing features (vs. M4/M7 to come):

  * MiniBatchKMeans tree with TF-IDF keywords per node — *no* LLM
    summaries. That's the line between M8 and M4 (M4 generates
    abstractive summaries with an LLM).
  * Linear `alpha_dense` fusion of dense + BM25 (not RRF; M3 owns RRF).
  * Per-doc grouping after candidate scoring + neighbor expansion
    within each doc (`context_neighbor_radius`).
  * Cross-encoder rerank top-N, with sigmoid(top-1 logit) gating
    abstention at threshold 0.35.

Index artifacts cached on disk so subsequent Colab sessions re-load
instead of re-clustering.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
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
    M8_LOW_CONFIDENCE_ANSWER,
)
from ..models import (
    embed_texts,
    generate,
    load_embedder,
    load_reranker,
    rerank_scores,
)
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


REQUIRED_FILES = (
    "chunks.jsonl",
    "embeddings.npy",
    "faiss.index",
    "bm25.pkl",
    "tree.pkl",
)


# --- Tree node structure (in-memory; pickled to tree.pkl) -----------------


@dataclass
class TreeNode:
    node_id: str
    parent_id: str | None
    depth: int
    size: int
    member_indices: list[int]
    centroid: np.ndarray
    keywords: list[str]
    children: list[str] = field(default_factory=list)


@dataclass
class HierarchicalTree:
    root_id: str
    nodes: dict[str, TreeNode]
    leaf_to_chunk_indices: dict[str, list[int]]
    chunk_idx_to_leaf: dict[int, list[str]]


# --- Stopwords ------------------------------------------------------------


def _load_en_el_stopwords() -> list[str] | None:
    """Combined English + Greek NLTK stopword list, with download fallback.

    Returns None if NLTK is unavailable; callers should treat None as
    "use sklearn's English default" rather than crashing.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
    except ImportError:
        return None

    sw: list[str] = []
    for lang in ("english", "greek"):
        try:
            sw.extend(stopwords.words(lang))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                sw.extend(stopwords.words(lang))
            except Exception:
                pass
    return sw or None


_STOPLIST_CACHE: list[str] | None = None


def _stoplist() -> list[str] | None:
    global _STOPLIST_CACHE
    if _STOPLIST_CACHE is None:
        _STOPLIST_CACHE = _load_en_el_stopwords()
    return _STOPLIST_CACHE


# --- TF-IDF keywords per tree node ----------------------------------------


def _safe_cluster_keywords(
    texts: list[str],
    top_n: int,
    min_df: int,
    max_df: float,
) -> list[str]:
    """Top TF-IDF terms across a node's member chunks.

    Notebook fix per brief: EN+EL stoplist, min_df=2, max_df=0.95.
    Returns [] when the vocabulary is too small to satisfy min_df/max_df
    (common for tiny clusters in unit tests / smoke).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []
    try:
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words=_stoplist(),
            min_df=min_df,
            max_df=max_df,
        )
        X = tfidf.fit_transform(texts)
    except ValueError:
        # min_df/max_df collapsed the vocabulary; retry without filters
        # so the tree node still gets *some* keywords on tiny clusters.
        try:
            tfidf = TfidfVectorizer(max_features=5000, stop_words=_stoplist())
            X = tfidf.fit_transform(texts)
        except ValueError:
            return []
    sums = np.asarray(X.sum(axis=0)).ravel()
    feats = np.array(tfidf.get_feature_names_out())
    order = np.argsort(-sums)[:top_n]
    return feats[order].tolist()


# --- Tree construction ----------------------------------------------------


def _compute_centroid(vectors: np.ndarray) -> np.ndarray:
    centroid = np.mean(vectors, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid.astype("float32", copy=False)


def _build_tree(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    *,
    max_depth: int,
    min_cluster_size: int,
    branching_factor: int,
    keywords_top_n: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
) -> HierarchicalTree:
    from sklearn.cluster import MiniBatchKMeans

    nodes: dict[str, TreeNode] = {}
    leaf_to_chunk_indices: dict[str, list[int]] = defaultdict(list)

    def recurse(
        member_indices: list[int],
        depth: int,
        parent_id: str | None,
        path_tokens: list[str],
    ) -> str:
        node_id = "root" if parent_id is None else "__".join(path_tokens)
        node_vectors = embeddings[member_indices]
        node_texts = [chunks[i].text for i in member_indices]
        centroid = _compute_centroid(node_vectors)
        keywords = _safe_cluster_keywords(
            node_texts, keywords_top_n, tfidf_min_df, tfidf_max_df
        )

        node = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            size=len(member_indices),
            member_indices=[int(i) for i in member_indices],
            centroid=centroid,
            keywords=keywords,
        )
        nodes[node_id] = node

        should_stop = (
            depth >= max_depth
            or len(member_indices) < min_cluster_size
            or len(member_indices) <= branching_factor
        )
        if should_stop:
            leaf_to_chunk_indices[node_id].extend(node.member_indices)
            return node_id

        n_clusters = min(
            branching_factor,
            max(2, len(member_indices) // min_cluster_size),
        )
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(2048, len(member_indices)),
            n_init=3,
        )
        local_labels = km.fit_predict(node_vectors)
        unique_labels = sorted(set(int(l) for l in local_labels.tolist()))
        if len(unique_labels) <= 1:
            leaf_to_chunk_indices[node_id].extend(node.member_indices)
            return node_id

        for child_rank, label in enumerate(unique_labels):
            child_indices = [
                member_indices[i]
                for i, lab in enumerate(local_labels)
                if int(lab) == label
            ]
            if not child_indices:
                continue
            child_path = path_tokens + [f"d{depth + 1}n{child_rank}"]
            child_id = recurse(child_indices, depth + 1, node_id, child_path)
            node.children.append(child_id)

        if not node.children:
            leaf_to_chunk_indices[node_id].extend(node.member_indices)
        return node_id

    root_id = recurse(list(range(len(chunks))), 0, None, [])
    chunk_idx_to_leaf: dict[int, list[str]] = defaultdict(list)
    for leaf_id, idxs in leaf_to_chunk_indices.items():
        for i in idxs:
            chunk_idx_to_leaf[int(i)].append(leaf_id)

    return HierarchicalTree(
        root_id=root_id,
        nodes=nodes,
        leaf_to_chunk_indices=dict(leaf_to_chunk_indices),
        chunk_idx_to_leaf=dict(chunk_idx_to_leaf),
    )


# --- Helpers --------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize_scores(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def _generate_query_views(query: str, max_views: int) -> list[str]:
    """Deterministic, templated query views.

    LLM-generated paraphrase views are explicitly *off* in M8 — that
    belongs to M7 (intent decomposition + HyDE). M8 uses only the four
    semantic-perspective templates from the notebook.
    """
    q = query.strip()
    views = [q]
    templates = [
        "main topic of: {q}",
        "key entities and concepts in: {q}",
        "evidence and passages relevant to: {q}",
        "sections discussing: {q}",
    ]
    for tpl in templates:
        if len(views) >= max_views:
            break
        views.append(tpl.format(q=q))

    dedup: list[str] = []
    seen: set[str] = set()
    for v in views:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(v.strip())
        if len(dedup) >= max_views:
            break
    return dedup


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b)
        / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9)
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --- System ---------------------------------------------------------------


class HierarchicalSystem(BaseSystem):
    system_id = "M8"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self.chunk_embeddings: np.ndarray | None = None
        self._faiss: Any | None = None
        self._bm25: Any | None = None
        self._tree: HierarchicalTree | None = None
        self._doc_chunks_map: dict[str, list[Chunk]] = {}

    # --- index ------------------------------------------------------------

    def index(self, corpus_path: Path) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        ckey = compute_cache_key(
            chunking_config=self.config.chunking,
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            extra={
                "system": "M8",
                "tree": asdict(self.config.m8),
            },
        )
        cdir = CacheDir(paths.cache_dir(), self.system_id, ckey)

        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.chunks = load_chunks(cdir.chunks_path)
            self.chunk_embeddings = np.load(cdir.embeddings_path)
            self._faiss = faiss.read_index(str(cdir.faiss_path))
            self._bm25 = load_pickle(cdir.bm25_path)
            self._tree = load_pickle(cdir.path / "tree.pkl")
            self._doc_chunks_map = self._build_doc_chunks_map(self.chunks)
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

        self.chunk_embeddings = embed_texts([c.text for c in self.chunks])
        faiss_index = faiss.IndexFlatIP(self.chunk_embeddings.shape[1])
        faiss_index.add(self.chunk_embeddings)
        self._faiss = faiss_index

        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

        m8 = self.config.m8
        self._tree = _build_tree(
            self.chunks,
            self.chunk_embeddings,
            max_depth=m8.tree_max_depth,
            min_cluster_size=m8.tree_min_cluster_size,
            branching_factor=m8.tree_branching_factor,
            keywords_top_n=m8.tree_keywords_top_n,
            tfidf_min_df=m8.tfidf_min_df,
            tfidf_max_df=m8.tfidf_max_df,
        )

        # Tag chunks with their primary leaf (notebook parity)
        for i, c in enumerate(self.chunks):
            leafs = self._tree.chunk_idx_to_leaf.get(i, [])
            c.metadata["tree_leaf_nodes"] = leafs
            c.metadata["primary_tree_leaf"] = leafs[0] if leafs else "root"

        self._doc_chunks_map = self._build_doc_chunks_map(self.chunks)

        save_chunks(self.chunks, cdir.chunks_path)
        save_embeddings(self.chunk_embeddings, cdir.embeddings_path)
        faiss.write_index(faiss_index, str(cdir.faiss_path))
        save_pickle(self._bm25, cdir.bm25_path)
        save_pickle(self._tree, cdir.path / "tree.pkl")
        Manifest(
            system_id=self.system_id,
            cache_key=ckey,
            chunking_config=asdict(self.config.chunking),
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
            extra={
                "tree": asdict(self.config.m8),
                "n_tree_nodes": len(self._tree.nodes),
                "n_leaves": len(self._tree.leaf_to_chunk_indices),
            },
        ).save(cdir.manifest_path)

        self._indexed = True

    @staticmethod
    def _build_doc_chunks_map(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
        m: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            m[c.doc_id].append(c)
        for doc_id in m:
            m[doc_id].sort(key=lambda c: c.position)
        return dict(m)

    # --- query-time ------------------------------------------------------

    def _embed_queries(self, queries: list[str]) -> dict[str, np.ndarray]:
        embs = embed_texts(queries)
        return {q: embs[i] for i, q in enumerate(queries)}

    def _traverse_tree(self, query_emb: np.ndarray) -> list[int]:
        """Top-down multi-branch traversal (ports `traverse_semantic_tree`)."""
        tree = self._tree
        top_k = self.config.m8.tree_top_branches_per_level
        active = [tree.root_id]
        leaf_nodes: set[str] = set()

        while active:
            next_active: list[str] = []
            for node_id in active:
                node = tree.nodes[node_id]
                if not node.children:
                    leaf_nodes.add(node_id)
                    continue
                scored = sorted(
                    (
                        (cid, _cosine_sim(query_emb, tree.nodes[cid].centroid))
                        for cid in node.children
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                next_active.extend(cid for cid, _ in scored[:top_k])
            active = next_active

        return sorted({
            i
            for leaf_id in leaf_nodes
            for i in tree.leaf_to_chunk_indices.get(leaf_id, [])
        })

    def _score_candidates(
        self,
        candidate_indices: list[int],
        query_views: list[str],
        query_embeddings: dict[str, np.ndarray],
    ) -> list[dict]:
        """Linear alpha_dense * dense_norm + (1-alpha) * bm25_norm.

        Stays linear by design — M3 covers the RRF variant.
        """
        if not candidate_indices:
            return []

        bm25_by_view = {
            qv: self._bm25.get_scores(_tokenize(qv)) for qv in query_views
        }

        rows: list[dict] = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            dense_vals = [
                float(np.dot(query_embeddings[qv], self.chunk_embeddings[idx]))
                for qv in query_views
            ]
            bm25_vals = [float(bm25_by_view[qv][idx]) for qv in query_views]
            rows.append({
                "idx": int(idx),
                "chunk": chunk,
                "dense_best": max(dense_vals) if dense_vals else 0.0,
                "bm25_best": max(bm25_vals) if bm25_vals else 0.0,
            })

        dense_norm = _normalize_scores(np.array([r["dense_best"] for r in rows]))
        bm25_norm = _normalize_scores(np.array([r["bm25_best"] for r in rows]))
        alpha = self.config.m8.alpha_dense
        for i, r in enumerate(rows):
            r["score_hybrid"] = float(
                alpha * dense_norm[i] + (1.0 - alpha) * bm25_norm[i]
            )
        return sorted(rows, key=lambda r: r["score_hybrid"], reverse=True)

    def _collect_source_documents(
        self,
        scored: list[dict],
    ) -> list[dict]:
        """Group candidates by doc_id; keep top-N docs by max chunk score."""
        top_docs = self.config.m8.top_docs_after_tree
        agg: dict[str, dict] = {}
        for row in scored:
            doc_id = row["chunk"].doc_id
            d = agg.setdefault(doc_id, {
                "doc_id": doc_id,
                "score": -1.0,
                "supporting": [],
            })
            d["score"] = max(d["score"], row["score_hybrid"])
            d["supporting"].append(row)
        ranked = sorted(agg.values(), key=lambda d: d["score"], reverse=True)
        for d in ranked:
            d["supporting"].sort(key=lambda r: r["score_hybrid"], reverse=True)
        return ranked[:top_docs]

    def _expand_context(self, selected_docs: list[dict]) -> list[dict]:
        """Neighbor expansion within each doc (radius = context_neighbor_radius)."""
        radius = self.config.m8.context_neighbor_radius
        per_doc = self.config.m8.top_chunks_per_doc_for_context
        fragments: list[dict] = []
        seen: set[str] = set()

        for doc in selected_docs:
            doc_chunks = self._doc_chunks_map.get(doc["doc_id"], [])
            for support in doc["supporting"][:per_doc]:
                anchor = support["chunk"]
                center = anchor.position
                left = max(0, center - radius)
                right = min(len(doc_chunks) - 1, center + radius)
                window = doc_chunks[left : right + 1]
                fid = f"{doc['doc_id']}::window::{left}-{right}"
                if fid in seen:
                    continue
                seen.add(fid)
                fragments.append({
                    "fragment_id": fid,
                    "doc_id": doc["doc_id"],
                    "anchor_chunk": anchor,
                    "window_chunks": window,
                    "text": "\n".join(c.text for c in window).strip(),
                    "score_hybrid": float(support["score_hybrid"]),
                })

        return sorted(fragments, key=lambda f: f["score_hybrid"], reverse=True)

    def _rerank_fragments(
        self,
        query: str,
        fragments: list[dict],
    ) -> list[dict]:
        n = self.config.m8.rerank_top_n
        if not fragments:
            return fragments
        head = fragments[:n]
        # touch the reranker so we surface load errors deterministically
        load_reranker()
        scores = rerank_scores(query, [f["text"] for f in head])
        for f, s in zip(head, scores):
            f["score_rerank"] = float(s)
            f["score_rerank_sigmoid"] = _sigmoid(float(s))
        tail = fragments[n:]
        head_sorted = sorted(
            head, key=lambda f: f.get("score_rerank", -1e9), reverse=True
        )
        return head_sorted + tail

    # --- public retrieve / answer ---------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        k = k or self.config.m8.top_k_final
        m8 = self.config.m8

        query_views = _generate_query_views(query, m8.max_query_views)
        query_embeddings = self._embed_queries(query_views)
        base = np.mean(
            np.vstack([query_embeddings[qv] for qv in query_views]), axis=0
        )
        base = base / (np.linalg.norm(base) + 1e-9)

        candidate_indices = self._traverse_tree(base)
        scored = self._score_candidates(candidate_indices, query_views, query_embeddings)
        docs = self._collect_source_documents(scored)
        fragments = self._expand_context(docs)
        fragments = self._rerank_fragments(query, fragments)

        out: list[RetrievedChunk] = []
        for rank, frag in enumerate(fragments[:k]):
            anchor = frag["anchor_chunk"]
            score = frag.get("score_rerank", frag["score_hybrid"])
            out.append(RetrievedChunk(chunk=anchor, score=float(score), rank=rank))
        return out

    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        self._require_indexed()
        t0 = self._now()
        retrieved = self.retrieve(query, k)

        if not retrieved:
            return AnswerResult(
                query=query,
                answer=M8_LOW_CONFIDENCE_ANSWER,
                retrieved=[],
                latency_s=self._now() - t0,
                n_retrieval_calls=2,
                extra={"abstained": True, "confidence": 0.0, "reason": "no_candidates"},
            )

        top_logit = retrieved[0].score
        confidence = _sigmoid(top_logit)
        threshold = self.config.m8.abstention_threshold

        if confidence < threshold:
            return AnswerResult(
                query=query,
                answer=M8_LOW_CONFIDENCE_ANSWER,
                retrieved=retrieved,
                latency_s=self._now() - t0,
                n_retrieval_calls=2,
                extra={
                    "abstained": True,
                    "confidence": confidence,
                    "threshold": threshold,
                    "reason": "low_confidence",
                },
            )

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
            extra={"abstained": False, "confidence": confidence, "threshold": threshold},
        )
