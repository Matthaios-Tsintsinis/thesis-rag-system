"""Shared RAPTOR substrate: cluster tree with LLM summaries + flat collapsed index.

Used by M4 (collapsed-retrieval RAPTOR baseline) and later by M7
(three-axis hybrid; layers Axes 2/3 on top of this same substrate).
Anything spec'd in PIPELINE_DESIGN.md sections 3.4, 3.5, and the
per-node-type expansion rules of section 4.4 lives here so M7 imports
it unchanged.

Tree construction (top-down recursive MiniBatchKMeans per PIPELINE_DESIGN
section 3.4) operates on chunk embeddings. After topology is built,
summaries are generated bottom-up: leaf-internal nodes summarise from
their member chunk texts; higher nodes summarise from their direct
children's summary texts (RAPTOR paper [R3] style, keeps prompts
bounded). Every internal node — root included — gets a summary; the
root is excluded only from the flat collapsed index (section 3.5,
INCLUDE_ROOT_IN_FLAT_INDEX = False).

Serialisation is JSON for topology + .npy for the summary-embedding
matrix + FAISS binary for the flat collapsed index. No pickle — tree
artifacts may be re-inspected months from now during thesis writing,
and JSON survives Python upgrades that pickle does not.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .config import ExpansionParams, NodeType, RaptorBuildParams


SummarizeFn = Callable[[list[str]], str]
EmbedFn = Callable[[list[str]], np.ndarray]

FlatRefKind = Literal["chunk", "summary"]


# --- Shared substrate cache identity --------------------------------------

# Every RAPTOR-family system (M4, M7, M7 ablations, future variants) is a
# *consumer* of one shared substrate: chunks + embeddings + BM25 + the
# LLM-summarised cluster tree + flat collapsed index. These artifacts
# depend only on the corpus, the parser, the chunking config, the
# embedder, the tree topology params, and the summariser identity — never
# on which system reads them. They therefore live under a single shared
# cache namespace keyed by `raptor_substrate_extra()` with NO system_id
# field, so M4 and M7 share one copy instead of rebuilding per system.
RAPTOR_SUBSTRATE_NAMESPACE = "RAPTOR"


def raptor_substrate_extra(
    *,
    build: RaptorBuildParams,
    summary_model: str,
    summary_prompt_version: str,
    include_root: bool,
    rrf_k: int,
    sparse: str = "bm25okapi",
    fusion: str = "rrf",
) -> dict:
    """Substrate-only cache-key extras shared across RAPTOR-family systems.

    Folded into `compute_cache_key(... extra=...)` together with the
    shared chunking/embedder/parsing/corpus inputs. Deliberately omits
    any system identifier: the produced artifacts are identical no
    matter which system triggers the build, so M4 and M7 must land on
    the same `RAPTOR/<substrate_hash>/` directory. Tree topology params
    are included because the tree artifact itself depends on them.
    """
    return {
        "tree": asdict(build),
        "summary_model": summary_model,
        "summary_prompt_version": summary_prompt_version,
        "include_root_in_flat_index": bool(include_root),
        "sparse": sparse,
        "fusion": fusion,
        "rrf_k": int(rrf_k),
    }


# --- Tree dataclasses -----------------------------------------------------


@dataclass
class RaptorNode:
    node_id: str
    parent_id: str | None
    depth: int
    children: list[str] = field(default_factory=list)
    member_indices: list[int] = field(default_factory=list)
    summary_text: str = ""
    # Populated after topology + summarisation; shape (D,), float32, L2-normalised.
    summary_embedding: np.ndarray | None = None


@dataclass
class RaptorTree:
    root_id: str
    nodes: dict[str, RaptorNode]
    params: RaptorBuildParams

    def internal_nodes(self) -> list[RaptorNode]:
        """All nodes — root + descendants. Every node here is 'internal'
        in the RAPTOR sense (it groups chunks via member_indices); chunks
        themselves are not represented as nodes."""
        return list(self.nodes.values())

    def non_root_nodes(self) -> list[RaptorNode]:
        return [n for n in self.nodes.values() if n.node_id != self.root_id]

    def bottom_up_order(self) -> list[str]:
        """Node ids ordered deepest-first (leaves of the tree before parents)."""
        return sorted(self.nodes, key=lambda nid: -self.nodes[nid].depth)

    def descendants_of(self, node_id: str) -> list[int]:
        """All leaf chunk indices reachable from a node (its member_indices)."""
        return list(self.nodes[node_id].member_indices)

    def ancestors_of(self, chunk_idx: int) -> list[str]:
        """Path of node ids from the deepest internal node containing this
        chunk up to (and including) the root. Empty if not found."""
        # Linear lookup; called rarely. Cache later if needed.
        deepest: RaptorNode | None = None
        for n in self.nodes.values():
            if chunk_idx in n.member_indices:
                if deepest is None or n.depth > deepest.depth:
                    deepest = n
        if deepest is None:
            return []
        chain: list[str] = []
        cursor: RaptorNode | None = deepest
        while cursor is not None:
            chain.append(cursor.node_id)
            cursor = self.nodes[cursor.parent_id] if cursor.parent_id else None
        return chain


# --- Node-type bucketing --------------------------------------------------


def depth_to_summary_type(depth: int, expansion: ExpansionParams) -> NodeType:
    """Map a node depth to the flat-index node_type label.

    Root (depth 0) maps to summary_high but is filtered out at flat-index
    build time when include_root=False.
    """
    if depth <= expansion.high_level_max_depth:
        return "summary_high"
    if depth == expansion.mid_level_depth:
        return "summary_mid"
    return "summary_low"


# --- Tree construction (top-down k-means) ---------------------------------


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 0.0:
        return vec
    return (vec / n).astype(np.float32, copy=False)


def _cluster(
    member_vectors: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """MiniBatchKMeans labels for member_vectors. Returns int labels."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(2048, len(member_vectors)),
        n_init=3,
    )
    return km.fit_predict(member_vectors).astype(int)


def _build_topology(
    chunk_embeddings: np.ndarray,
    params: RaptorBuildParams,
) -> RaptorTree:
    """Top-down recursive clustering. Produces a tree where every node
    (root included) holds member_indices into the chunk array but no
    summaries yet."""
    nodes: dict[str, RaptorNode] = {}

    def recurse(
        member_indices: list[int],
        depth: int,
        parent_id: str | None,
        path_tokens: list[str],
    ) -> str:
        node_id = "root" if parent_id is None else "__".join(path_tokens)
        node = RaptorNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            member_indices=[int(i) for i in member_indices],
        )
        nodes[node_id] = node

        should_stop = (
            depth >= params.max_depth
            or len(member_indices) < params.min_cluster_size
            or len(member_indices) <= params.branching_factor
        )
        if should_stop:
            return node_id

        node_vectors = chunk_embeddings[member_indices]
        n_clusters = min(
            params.branching_factor,
            max(2, len(member_indices) // max(params.min_cluster_size, 1)),
        )
        labels = _cluster(node_vectors, n_clusters)
        unique_labels = sorted(set(int(l) for l in labels.tolist()))
        if len(unique_labels) <= 1:
            return node_id

        for child_rank, label in enumerate(unique_labels):
            child_indices = [
                member_indices[i]
                for i, lab in enumerate(labels)
                if int(lab) == label
            ]
            if not child_indices:
                continue
            child_path = path_tokens + [f"d{depth + 1}n{child_rank}"]
            child_id = recurse(child_indices, depth + 1, node_id, child_path)
            node.children.append(child_id)

        return node_id

    root_id = recurse(list(range(len(chunk_embeddings))), 0, None, [])
    return RaptorTree(root_id=root_id, nodes=nodes, params=params)


def build_raptor_tree(
    chunk_texts: list[str],
    chunk_embeddings: np.ndarray,
    *,
    params: RaptorBuildParams,
    summarize_fn: SummarizeFn,
    embed_fn: EmbedFn,
    on_summary: Callable[[RaptorNode], None] | None = None,
) -> RaptorTree:
    """Build a RAPTOR tree per PIPELINE_DESIGN section 3.4.

    Topology is built top-down by recursive MiniBatchKMeans on chunk
    embeddings. Summaries are generated bottom-up: leaf-internal nodes
    summarise from their member chunk texts; higher nodes summarise
    from their direct children's already-computed summary texts.

    `summarize_fn(passages)` is called exactly once per node (root
    included). The optional `on_summary` callback fires after each
    successful summarisation — used by callers to count calls in a
    manifest or trace.
    """
    if chunk_embeddings.ndim != 2:
        raise ValueError("chunk_embeddings must be 2D (n_chunks, dim)")
    if len(chunk_texts) != chunk_embeddings.shape[0]:
        raise ValueError(
            f"chunk_texts ({len(chunk_texts)}) and chunk_embeddings "
            f"({chunk_embeddings.shape[0]}) length mismatch"
        )

    tree = _build_topology(chunk_embeddings, params)

    # --- Bottom-up summarisation ---
    for node_id in tree.bottom_up_order():
        node = tree.nodes[node_id]
        if not node.children:
            passages = [chunk_texts[i] for i in node.member_indices]
        else:
            passages = [tree.nodes[cid].summary_text for cid in node.children]
        passages = [p for p in passages if p and p.strip()]
        if not passages:
            node.summary_text = ""
            continue
        node.summary_text = summarize_fn(passages)
        if on_summary is not None:
            on_summary(node)

    # --- Batch-embed all summary texts in one pass ---
    ordered = list(tree.nodes.values())
    texts = [n.summary_text for n in ordered]
    non_empty_mask = [bool(t and t.strip()) for t in texts]
    to_embed = [t for t, keep in zip(texts, non_empty_mask) if keep]
    if to_embed:
        emb = embed_fn(to_embed)  # already L2-normalised by embed_texts
        it = iter(emb)
        for n, keep in zip(ordered, non_empty_mask):
            if keep:
                n.summary_embedding = next(it).astype(np.float32, copy=False)
            else:
                n.summary_embedding = None

    return tree


# --- Flat collapsed index (PIPELINE_DESIGN section 3.5) -------------------


@dataclass
class FlatCollapsedIndex:
    """In-memory shape for the collapsed retrieval index.

    The FAISS index, node_types array, and refs list are all length-N
    where N = n_chunks + n_summary_nodes_included. They are positionally
    aligned: row i of the FAISS index corresponds to node_types[i] and
    refs[i].
    """
    faiss_index: Any
    node_types: list[NodeType]
    refs: list[dict]  # each: {"type": "chunk", "chunk_idx": int}
    #                          or {"type": "summary", "node_id": str}
    include_root: bool
    dim: int


def build_flat_collapsed_index(
    tree: RaptorTree,
    chunk_embeddings: np.ndarray,
    *,
    expansion: ExpansionParams,
    include_root: bool = False,
) -> FlatCollapsedIndex:
    """Combine every non-root summary embedding with every chunk
    embedding into a single FAISS IndexFlatIP (cosine on L2-normalised
    vectors). Root is excluded by default per PIPELINE_DESIGN section 3.5."""
    import faiss

    if chunk_embeddings.ndim != 2:
        raise ValueError("chunk_embeddings must be 2D")
    dim = int(chunk_embeddings.shape[1])

    rows: list[np.ndarray] = [chunk_embeddings.astype(np.float32, copy=False)]
    node_types: list[NodeType] = ["chunk"] * len(chunk_embeddings)
    refs: list[dict] = [
        {"type": "chunk", "chunk_idx": i} for i in range(len(chunk_embeddings))
    ]

    summary_rows: list[np.ndarray] = []
    for n in tree.nodes.values():
        if n.summary_embedding is None:
            continue
        if not include_root and n.node_id == tree.root_id:
            continue
        summary_rows.append(n.summary_embedding.reshape(1, -1))
        node_types.append(depth_to_summary_type(n.depth, expansion))
        refs.append({"type": "summary", "node_id": n.node_id})

    if summary_rows:
        rows.append(np.vstack(summary_rows).astype(np.float32, copy=False))

    combined = np.vstack(rows)
    if combined.shape[1] != dim:
        raise RuntimeError(
            f"Flat index dim mismatch: chunks={dim}, combined={combined.shape[1]}"
        )

    index = faiss.IndexFlatIP(dim)
    index.add(combined)
    return FlatCollapsedIndex(
        faiss_index=index,
        node_types=node_types,
        refs=refs,
        include_root=include_root,
        dim=dim,
    )


# --- Per-node-type expansion (PIPELINE_DESIGN section 4.4) ----------------


def _topk_descendants(
    node: RaptorNode,
    view_vec: np.ndarray,
    chunk_embeddings: np.ndarray,
    k: int,
) -> list[int]:
    if not node.member_indices:
        return []
    members = np.asarray(node.member_indices, dtype=int)
    sims = chunk_embeddings[members] @ view_vec.astype(np.float32, copy=False)
    order = np.argsort(-sims)[:k]
    return [int(members[i]) for i in order.tolist()]


def expand_node(
    node_id: str,
    view_vec: np.ndarray,
    tree: RaptorTree,
    chunk_embeddings: np.ndarray,
    *,
    expansion: ExpansionParams,
    _recursion_depth: int = 0,
    _path_trace: list[str] | None = None,
) -> tuple[list[int], list[str]]:
    """Per-node-type expansion of a summary node into chunk indices.

    Returns (chunk_indices, trace_labels). `trace_labels` records which
    branch of the §4.4 rules fired (e.g. ["high->children", "mid->direct"])
    for diagnostic / sanity-check purposes. Pass an empty list as
    `_path_trace` from the caller to accumulate across recursion.
    """
    node = tree.nodes[node_id]
    trace = _path_trace if _path_trace is not None else []

    # High-level summary: re-score children, route top-N, recurse.
    if node.depth <= expansion.high_level_max_depth:
        if (
            _recursion_depth >= expansion.max_expansion_recursion_depth
            or not node.children
        ):
            trace.append(f"high_d{node.depth}->direct_topk")
            return _topk_descendants(
                node, view_vec, chunk_embeddings,
                expansion.summary_expansion_top_k_chunks,
            ), trace
        trace.append(f"high_d{node.depth}->children")
        return _route_through_top_children(
            node, view_vec, tree, chunk_embeddings,
            expansion, _recursion_depth, trace,
        )

    # Mid-level summary: direct top-K if descendants <= cap, else route.
    if node.depth == expansion.mid_level_depth:
        if (
            len(node.member_indices)
            <= expansion.max_descendant_chunks_for_direct_expansion
        ):
            trace.append(f"mid_d{node.depth}->direct_topk")
            return _topk_descendants(
                node, view_vec, chunk_embeddings,
                expansion.summary_expansion_top_k_chunks,
            ), trace
        if not node.children:
            trace.append(f"mid_d{node.depth}->direct_topk_no_children")
            return _topk_descendants(
                node, view_vec, chunk_embeddings,
                expansion.summary_expansion_top_k_chunks,
            ), trace
        trace.append(f"mid_d{node.depth}->children")
        return _route_through_top_children(
            node, view_vec, tree, chunk_embeddings,
            expansion, _recursion_depth, trace,
        )

    # Low-level summary (depth >= low_level_min_depth): direct top-K.
    trace.append(f"low_d{node.depth}->direct_topk")
    return _topk_descendants(
        node, view_vec, chunk_embeddings,
        expansion.summary_expansion_top_k_chunks,
    ), trace


def _route_through_top_children(
    node: RaptorNode,
    view_vec: np.ndarray,
    tree: RaptorTree,
    chunk_embeddings: np.ndarray,
    expansion: ExpansionParams,
    recursion_depth: int,
    trace: list[str],
) -> tuple[list[int], list[str]]:
    """Score node's direct children by summary-embedding cosine, recurse
    into the top-K. Used by both high- and (overflow) mid-level rules."""
    children_with_emb = [
        (cid, tree.nodes[cid].summary_embedding)
        for cid in node.children
        if tree.nodes[cid].summary_embedding is not None
    ]
    if not children_with_emb:
        return _topk_descendants(
            node, view_vec, chunk_embeddings,
            expansion.summary_expansion_top_k_chunks,
        ), trace

    view_f32 = view_vec.astype(np.float32, copy=False)
    scored = sorted(
        ((cid, float(emb @ view_f32)) for cid, emb in children_with_emb),
        key=lambda t: t[1],
        reverse=True,
    )[: expansion.max_children_to_follow_from_broad_summary]

    out: list[int] = []
    for cid, _ in scored:
        chunks, _ = expand_node(
            cid, view_vec, tree, chunk_embeddings,
            expansion=expansion,
            _recursion_depth=recursion_depth + 1,
            _path_trace=trace,
        )
        out.extend(chunks)
    return out, trace


# --- Serialisation (JSON for tree + .npy for embeddings) ------------------


def _tree_to_json_obj(tree: RaptorTree) -> dict:
    """Tree topology + summary texts. Embeddings handled separately."""
    return {
        "root_id": tree.root_id,
        "params": asdict(tree.params),
        "nodes": [
            {
                "node_id": n.node_id,
                "parent_id": n.parent_id,
                "depth": n.depth,
                "children": list(n.children),
                "member_indices": list(n.member_indices),
                "summary_text": n.summary_text,
                "has_embedding": n.summary_embedding is not None,
            }
            for n in tree.nodes.values()
        ],
    }


def save_raptor_tree(
    tree: RaptorTree,
    tree_json_path: Path,
    summary_emb_path: Path,
) -> None:
    """Tree topology + summaries -> JSON; summary embeddings -> .npy.

    The .npy stores embeddings in the same node order as the JSON's
    `nodes` list (only rows for nodes with `has_embedding: true`).
    """
    tree_json_path.parent.mkdir(parents=True, exist_ok=True)
    obj = _tree_to_json_obj(tree)
    tree_json_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    embs = [
        n.summary_embedding
        for n in tree.nodes.values()
        if n.summary_embedding is not None
    ]
    if embs:
        mat = np.vstack([e.reshape(1, -1) for e in embs]).astype(np.float32)
    else:
        mat = np.zeros((0, 0), dtype=np.float32)
    np.save(summary_emb_path, mat)


def load_raptor_tree(
    tree_json_path: Path,
    summary_emb_path: Path,
) -> RaptorTree:
    obj = json.loads(tree_json_path.read_text())
    params = RaptorBuildParams(**obj["params"])
    embs = np.load(summary_emb_path) if summary_emb_path.exists() else None
    emb_iter = iter(embs) if embs is not None and len(embs) > 0 else None

    nodes: dict[str, RaptorNode] = {}
    for d in obj["nodes"]:
        node = RaptorNode(
            node_id=d["node_id"],
            parent_id=d["parent_id"],
            depth=int(d["depth"]),
            children=list(d["children"]),
            member_indices=[int(i) for i in d["member_indices"]],
            summary_text=d.get("summary_text", ""),
        )
        if d.get("has_embedding") and emb_iter is not None:
            node.summary_embedding = next(emb_iter).astype(np.float32, copy=False)
        nodes[node.node_id] = node

    return RaptorTree(root_id=obj["root_id"], nodes=nodes, params=params)


def save_flat_index(
    flat: FlatCollapsedIndex,
    faiss_path: Path,
    meta_path: Path,
) -> None:
    """FAISS binary + meta JSON. Meta carries the positional alignment."""
    import faiss

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(flat.faiss_index, str(faiss_path))
    meta = {
        "include_root": flat.include_root,
        "dim": flat.dim,
        "node_types": flat.node_types,
        "refs": flat.refs,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))


def load_flat_index(
    faiss_path: Path,
    meta_path: Path,
) -> FlatCollapsedIndex:
    import faiss

    meta = json.loads(meta_path.read_text())
    index = faiss.read_index(str(faiss_path))
    return FlatCollapsedIndex(
        faiss_index=index,
        node_types=list(meta["node_types"]),
        refs=list(meta["refs"]),
        include_root=bool(meta["include_root"]),
        dim=int(meta["dim"]),
    )


__all__ = [
    "RAPTOR_SUBSTRATE_NAMESPACE",
    "raptor_substrate_extra",
    "SummarizeFn",
    "EmbedFn",
    "RaptorNode",
    "RaptorTree",
    "FlatCollapsedIndex",
    "FlatRefKind",
    "depth_to_summary_type",
    "build_raptor_tree",
    "build_flat_collapsed_index",
    "expand_node",
    "save_raptor_tree",
    "load_raptor_tree",
    "save_flat_index",
    "load_flat_index",
]
