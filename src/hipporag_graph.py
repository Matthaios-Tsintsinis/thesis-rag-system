"""HippoRAG 1 graph construction.

Builds the phrase / fact / document graph described in section 3 of the
HippoRAG paper, faithfully reproducing the legacy `create_graph.py`
pipeline. Nodes are unique normalised phrases; edges are:

  * Triple-derived edges between (head, tail) of every extracted fact,
    symmetric, weight = number of co-occurrences across passages.
  * Synonymy edges between phrases whose Contriever embeddings have
    cosine similarity >= sim_threshold (paper default 0.8), capped at
    `synonym_top_k_cap` (paper default 100) per source phrase. Weight =
    cosine score.

Sparse phrase-fact-doc matrices:

  * docs_to_facts_mat  shape (n_chunks,  n_facts)
  * facts_to_phrases_mat shape (n_facts, n_phrases)

Built as scipy.sparse.csr_array exactly as in legacy create_graph.py.
Used at query time to propagate PPR phrase scores -> fact scores -> doc
scores (`doc_prob = docs_to_facts_mat @ (facts_to_phrases_mat @ ppr)`).

igraph holds the phrase-node graph for PPR. Constructed from the merged
edge dict, with `weight` attribute set on every edge. NOT persisted —
rebuilt deterministically from the persisted edge dict at load time,
which keeps the on-disk format pickle-free and cross-version safe.

THIS FILE IS A C4a SKELETON — function bodies raise NotImplementedError.
C4b lands the working build code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# --- Graph artefact bundle -------------------------------------------------


@dataclass
class HippoGraph:
    """Concrete on-disk representation of the M6 graph at query time.

    Field shapes / dtypes:
      phrase_to_id      : dict[str, int]  — unique normalised phrase -> id
      fact_to_id        : dict[tuple[str,str,str], int]
      docs_to_facts_mat : scipy.sparse.csr_array (n_chunks, n_facts), uint8
      facts_to_phrases_mat : scipy.sparse.csr_array (n_facts, n_phrases), uint8
      edges             : list[tuple[int, int, float]] — undirected edge list
      graph             : igraph.Graph — built from (edges, weights)
      phrase_embeddings : np.ndarray (n_phrases, 768) float32, L2-normalised
      phrase_to_num_doc : np.ndarray (n_phrases,) int — for node specificity
    """
    phrase_to_id: dict[str, int]
    fact_to_id: dict[tuple, int]
    docs_to_facts_mat: Any
    facts_to_phrases_mat: Any
    edges: list[tuple[int, int, float]]
    graph: Any  # igraph.Graph
    phrase_embeddings: np.ndarray
    phrase_to_num_doc: np.ndarray


# --- Build API (C4b) -------------------------------------------------------


def build_graph_structures(
    openie_results: list,  # list[OpenIEResult]
) -> tuple[dict[str, int], dict[tuple, int], Any, Any]:
    """Build phrase_to_id, fact_to_id, docs_to_facts_mat, facts_to_phrases_mat.

    Faithful port of `scratch_hipporag/src/create_graph.py:156-245`. Walks
    every passage's clean triples, normalising each element via
    `hipporag_openie.processing_phrases`, accumulating unique phrases and
    facts. Builds the two sparse CSR matrices via scipy.sparse.csr_array,
    same shape/dtype as legacy.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement phrase/fact graph build.")


def build_triple_edges(
    fact_to_id: dict[tuple, int],
    phrase_to_id: dict[str, int],
) -> dict[tuple[int, int], float]:
    """Symmetric triple-derived edges: per fact, +1 weight on (h, t) and (t, h).

    Faithful port of `scratch_hipporag/src/create_graph.py:188-233` triple
    iteration. Weight accumulates across passages if the same pair
    co-occurs in multiple facts. Output dict feeds add_synonymy_edges
    next.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement triple-derived edges.")


def embed_phrases(
    phrases_in_id_order: list[str],
    *,
    embedder_id: str,
) -> np.ndarray:
    """Contriever masked-mean phrase embeddings, L2-normalised, float32.

    HARD GATE in C4b: before relying on these embeddings for synonymy
    edges, the M6 system must verify that `SentenceTransformer(
    embedder_id).encode(...)`'s mean pooling matches Contriever's
    attention-mask-weighted mean (`scratch_hipporag/src/processing.py:
    mean_pooling`). Specifically: encode a 5-phrase sample via both
    paths and assert cosine >= 0.99 element-wise. If the cosine gate
    fails, port the legacy HuggingFaceWrapper masked-mean pooling
    directly here instead of trusting SentenceTransformer's auto-wrap.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement Contriever phrase embedding + pooling gate.")


def add_synonymy_edges(
    edges: dict[tuple[int, int], float],
    phrase_embeddings: np.ndarray,
    *,
    sim_threshold: float,
    top_k_cap: int,
) -> dict[tuple[int, int], float]:
    """For each phrase, top-K Contriever-cosine neighbours with score >= threshold.

    Faithful port of `scratch_hipporag/src/create_graph.py:251-307`
    synonymy loop. Threshold and cap from M6Config (paper defaults 0.8
    and 100). Weight on each new edge = the cosine score itself (not 1.0)
    per legacy line 294 `graph_plus[sim_edge] = similarity_max * score`
    with similarity_max=1.0.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement synonymy edge augmentation.")


def assemble_igraph(
    edges: dict[tuple[int, int], float],
    n_phrases: int,
) -> Any:
    """Wrap the edge dict into an igraph.Graph with `weight` attribute set.

    Faithful port of `scratch_hipporag/src/hipporag.py:441-467` (their
    build_graph method, which they run on load too). Edges are
    deduplicated (the dict keys already are), self-loops dropped.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement igraph assembly.")


# --- Persistence (C4b) -----------------------------------------------------

# Layout under cache/M6/<m6_hash>/:
#   chunks.jsonl              # shared with other systems via cache.save_chunks
#   openie.json               # list[OpenIEResult] (json, not pickle, cross-version safe)
#   phrase_to_id.json
#   fact_to_id.json           # serialised with tuple-keys -> string keys
#   docs_to_facts.npz         # scipy.sparse.save_npz
#   facts_to_phrases.npz
#   graph_edges.json          # list[[head_id, tail_id, weight]] -- igraph rebuilt on load
#   phrase_embeddings.npy     # float32 (n_phrases, 768)
#   phrase_to_num_doc.npy     # int  (n_phrases,)
#   manifest.json             # the harness Manifest dataclass

REQUIRED_FILES = (
    "chunks.jsonl",
    "openie.json",
    "phrase_to_id.json",
    "fact_to_id.json",
    "docs_to_facts.npz",
    "facts_to_phrases.npz",
    "graph_edges.json",
    "phrase_embeddings.npy",
    "phrase_to_num_doc.npy",
)


def save_graph(graph_bundle: HippoGraph, dir_path: Any) -> None:
    """Persist the full graph bundle into dir_path. NOT IMPLEMENTED in C4a."""
    raise NotImplementedError("C4b: implement graph persistence.")


def load_graph(dir_path: Any) -> HippoGraph:
    """Reverse of save_graph. Rebuilds igraph from graph_edges.json.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement graph load + igraph rebuild.")


__all__ = [
    "HippoGraph",
    "REQUIRED_FILES",
    "build_graph_structures",
    "build_triple_edges",
    "embed_phrases",
    "add_synonymy_edges",
    "assemble_igraph",
    "save_graph",
    "load_graph",
]
