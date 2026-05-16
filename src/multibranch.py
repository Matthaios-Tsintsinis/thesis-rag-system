"""Axis 1 Part B — multi-branch tree traversal (PIPELINE_DESIGN.md §4.4).

Collapsed retrieval (Axis 1a, src/raptor.py + M4) flattens the tree and
can rank the correct subtree below the top-50 on multi-aspect questions.
Multi-branch traversal forces breadth: keep the top-k branches alive at
each depth instead of committing to a single path, and carry explicit
subtree provenance on every collected chunk.

Node scoring uses summary-embedding cosine against the view vector —
the same signal src/raptor.py's collapsed-expansion child routing uses
(`_route_through_top_children`). BM25 over summary nodes is explicitly
weak (summaries are paraphrased text; that is why collapsed BM25 is
leaf-only, §4.4), so the tree-walk uses the dense signal and the hybrid
dense+BM25+RRF stays the collapsed path's job. The two pools are then
RRF-fused, not set-unioned, so a chunk strong in only one ranking can
still survive.

This module is pure (RaptorTree + numpy only) and deterministic:
score ties break by node_id / chunk_idx so two runs over the same tree
produce identical traversal output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MultiBranchParams
from .raptor import RaptorTree


@dataclass(frozen=True)
class BranchHit:
    """A leaf chunk reached by traversal, with its originating path."""
    chunk_idx: int
    score: float
    path: tuple[str, ...]   # node_id chain root → terminal node


def _score_nodes(
    node_ids: list[str],
    view_vec: np.ndarray,
    tree: RaptorTree,
) -> list[tuple[str, float]]:
    """Cosine(view, summary_embedding) per node, score-desc then node_id
    for deterministic tie-breaking. Nodes without an embedding sink last."""
    v = view_vec.astype(np.float32, copy=False)
    scored: list[tuple[str, float]] = []
    for nid in node_ids:
        emb = tree.nodes[nid].summary_embedding
        s = float(emb @ v) if emb is not None else float("-inf")
        scored.append((nid, s))
    return sorted(scored, key=lambda t: (-t[1], t[0]))


def _topk_leaves(
    member_indices: list[int],
    view_vec: np.ndarray,
    chunk_embeddings: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    if not member_indices or k <= 0:
        return []
    members = np.asarray(sorted(member_indices), dtype=int)
    sims = chunk_embeddings[members] @ view_vec.astype(np.float32, copy=False)
    order = np.argsort(-sims, kind="stable")[:k]
    return [(int(members[i]), float(sims[i])) for i in order.tolist()]


def multi_branch_traverse(
    tree: RaptorTree,
    view_vec: np.ndarray,
    chunk_embeddings: np.ndarray,
    params: MultiBranchParams,
) -> list[BranchHit]:
    """Keep top-k branches alive per depth; collect leaf chunks under each
    surviving terminal node, capped per path. Every chunk carries the
    node_id path that produced it (PIPELINE_DESIGN §4.4 Axis-1 Part B)."""
    root = tree.nodes[tree.root_id]

    # Depth 1: score the root's direct children, keep TOP_K_DEPTH_1.
    level1 = _score_nodes(root.children, view_vec, tree)[: params.top_k_depth_1]
    # frontier entries: (node_id, path-to-node inclusive)
    frontier: list[tuple[str, tuple[str, ...]]] = [
        (nid, (tree.root_id, nid)) for nid, _ in level1
    ]

    hits: list[BranchHit] = []
    seen_terminal: set[str] = set()

    def _collect(node_id: str, path: tuple[str, ...]) -> None:
        if node_id in seen_terminal:
            return
        seen_terminal.add(node_id)
        node = tree.nodes[node_id]
        for ci, sc in _topk_leaves(
            node.member_indices, view_vec, chunk_embeddings,
            params.leaves_per_path,
        ):
            hits.append(BranchHit(chunk_idx=ci, score=sc, path=path))

    # Depths 2..MAX_DEPTH: branch each surviving node into TOP_K_PER_LEVEL.
    depth = 1
    while frontier and depth < params.max_depth:
        depth += 1
        next_frontier: list[tuple[str, tuple[str, ...]]] = []
        for node_id, path in frontier:
            node = tree.nodes[node_id]
            if not node.children:
                _collect(node_id, path)            # terminal: no children
                continue
            kept = _score_nodes(node.children, view_vec, tree)[
                : params.top_k_per_level
            ]
            for cid, _ in kept:
                next_frontier.append((cid, path + (cid,)))
        frontier = next_frontier

    # Anything still alive at MAX_DEPTH is terminal — collect its leaves.
    for node_id, path in frontier:
        _collect(node_id, path)

    return hits


# --- RRF fusion (PIPELINE_DESIGN.md §4.4 "merge via RRF, not set-union") --


def rrf_fuse(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Generic Reciprocal Rank Fusion (Cormack et al. 2009) over rank
    lists in a shared id space. Deterministic: equal scores break by id."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def merge_collapsed_multibranch(
    collapsed_ranked_idx: list[int],
    multibranch_ranked_idx: list[int],
    k: int,
) -> list[tuple[int, float]]:
    """RRF-fuse the collapsed ranking and the multi-branch ranking (each
    a chunk-index list, best first). A chunk in both accumulates score;
    a chunk in only one survives if it ranks highly there. Dedup is
    implicit (RRF keys on chunk index)."""
    return rrf_fuse([collapsed_ranked_idx, multibranch_ranked_idx], k)


__all__ = [
    "BranchHit",
    "multi_branch_traverse",
    "rrf_fuse",
    "merge_collapsed_multibranch",
]
