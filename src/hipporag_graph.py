"""HippoRAG 1 graph construction.

Builds the phrase / fact / document graph described in section 3 of the
HippoRAG paper, faithfully reproducing the legacy `create_graph.py`
pipeline. Nodes are unique normalised phrases; edges are:

  * Triple-derived edges between (head, tail) of every extracted fact,
    symmetric (both directions added with weight += 1 per co-occurrence).
  * Synonymy edges between phrases whose Contriever embeddings have
    cosine similarity >= sim_threshold (paper default 0.8), capped at
    `synonym_top_k_cap` neighbours per source phrase (paper default 100).
    Weight = cosine score. Direction matches legacy create_graph.py:282
    (single-direction add per source phrase; effectively symmetric
    because every phrase appears as source).

Sparse phrase-fact-doc matrices:

  * docs_to_facts_mat  shape (n_chunks,  n_facts), uint8 csr_array
  * facts_to_phrases_mat shape (n_facts, n_phrases), uint8 csr_array

Used at query time to propagate PPR phrase scores -> fact scores -> doc
scores (`doc_prob = docs_to_facts_mat @ (facts_to_phrases_mat @ ppr)`).

igraph holds the phrase-node graph for PPR. Constructed from the merged
edge dict, with `weight` attribute set on every edge. NOT persisted —
rebuilt deterministically from the persisted edge list at load time,
which keeps the on-disk format pickle-free and cross-version safe.

CONTRIEVER POOLING HARD GATE
============================
`facebook/contriever` is a base HF model — NOT a sentence-transformers
checkpoint. SentenceTransformer auto-wraps it with a Pooling layer; the
mean-pooling implementation MUST attention-mask-weight the token
embeddings (sum non-padding tokens / mask.sum), matching the legacy
`processing.mean_pooling`. SentenceTransformer's default pooling does
exactly this when `pooling_mode_mean_tokens=True` (the default for
auto-wrap), but a silent mismatch (CLS / first-token / unweighted mean)
would corrupt every entity linking step, producing plausible-but-wrong
retrieval. `verify_contriever_pooling` is therefore a HARD GATE called
before phrase embedding: it computes both the SentenceTransformer
encoding and a from-scratch masked-mean encoding (legacy code path,
ported verbatim) and asserts cosine >= 0.99 on a 5-phrase probe set.
If the gate fails, M6 raises immediately with diagnostic output and
the operator must port the legacy HuggingFaceWrapper pooling
directly. No M6 build is allowed to proceed past a failed pooling gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hipporag_openie import OpenIEResult, processing_phrases


# --- Graph artefact bundle -------------------------------------------------


@dataclass
class HippoGraph:
    """Concrete on-disk representation of the M6 graph at query time."""
    phrase_to_id: dict[str, int]
    fact_to_id: dict[tuple[str, str, str], int]
    docs_to_facts_mat: Any  # scipy.sparse.csr_array (n_chunks, n_facts)
    facts_to_phrases_mat: Any  # scipy.sparse.csr_array (n_facts, n_phrases)
    edges: list[tuple[int, int, float]]
    graph: Any  # igraph.Graph
    phrase_embeddings: np.ndarray  # (n_phrases, dim) float32, L2-normalised
    phrase_to_num_doc: np.ndarray  # (n_phrases,) int — for node specificity
    # Index-time stats for the manifest + smoke sanity checks.
    n_passages: int = 0
    n_triple_edges: int = 0
    n_synonymy_edges: int = 0
    n_openie_parse_failures: int = 0


# --- Build pipeline --------------------------------------------------------


def build_graph_structures(
    openie_results: list[OpenIEResult],
) -> tuple[dict[str, int], dict[tuple[str, str, str], int], Any, Any, np.ndarray]:
    """Build phrase_to_id, fact_to_id, sparse matrices, phrase_to_num_doc.

    Faithful port of `scratch_hipporag/src/create_graph.py:156-245`. Walks
    every passage's extracted triples, normalises each element via
    processing_phrases, accumulates unique phrases and facts. Builds the
    two sparse CSR matrices with `scipy.sparse.csr_array`, uint8 dtype
    (entries are 0/1 indicator).

    Phrases are id'd in first-seen order (matches legacy `unique_phrases =
    list(np.unique(entities))` modulo np.unique's sort, which we replicate
    by sorting at the end so phrase ids are deterministic across runs).

    Facts use the normalised tuple (head, rel, tail) as the key (matches
    legacy `lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}`).
    """
    from scipy.sparse import csr_array

    # Pass 1: collect unique phrases and facts.
    all_entities_set: set[str] = set()
    fact_tuples: list[tuple[str, str, str]] = []  # in encounter order
    per_doc_facts: list[list[tuple[str, str, str]]] = []
    per_doc_phrases: list[list[str]] = []

    for r in openie_results:
        doc_facts: list[tuple[str, str, str]] = []
        doc_phrases: list[str] = []
        for triple in r.extracted_triples:
            if len(triple) != 3:
                continue
            clean = tuple(processing_phrases(p) for p in triple)
            if not all(clean):  # drop if any element normalises to empty
                continue
            head, rel, tail = clean
            doc_facts.append((head, rel, tail))
            doc_phrases.extend([head, tail])
            all_entities_set.add(head)
            all_entities_set.add(tail)
        per_doc_facts.append(doc_facts)
        per_doc_phrases.append(doc_phrases)

    # Deterministic id assignment: sort alphabetically (matches legacy's
    # np.unique sort).
    unique_phrases = sorted(all_entities_set)
    phrase_to_id = {p: i for i, p in enumerate(unique_phrases)}

    # Facts ordered by first encounter, deduplicated.
    seen_facts: set[tuple[str, str, str]] = set()
    unique_facts: list[tuple[str, str, str]] = []
    for doc_facts in per_doc_facts:
        for f in doc_facts:
            if f not in seen_facts:
                seen_facts.add(f)
                unique_facts.append(f)
    fact_to_id = {f: i for i, f in enumerate(unique_facts)}

    n_passages = len(openie_results)
    n_facts = len(unique_facts)
    n_phrases = len(unique_phrases)

    # Pass 2: populate sparse matrices.
    # docs_to_facts_mat[doc_id, fact_id] = 1 if doc produced fact.
    # facts_to_phrases_mat[fact_id, phrase_id] = 1 if fact mentions phrase
    # (head or tail; NOT relation — matches legacy lines 205-209 which
    # only iterate the np.array(triple)[[0, 2]] indices, skipping rel).
    d2f_rows: list[int] = []
    d2f_cols: list[int] = []
    f2p_rows: list[int] = []
    f2p_cols: list[int] = []

    for doc_id, doc_facts in enumerate(per_doc_facts):
        for f in doc_facts:
            fact_id = fact_to_id[f]
            d2f_rows.append(doc_id)
            d2f_cols.append(fact_id)
            head, _rel, tail = f
            for phrase in (head, tail):
                f2p_rows.append(fact_id)
                f2p_cols.append(phrase_to_id[phrase])

    docs_to_facts_mat = csr_array(
        (
            np.ones(len(d2f_rows), dtype=np.uint8),
            (np.asarray(d2f_rows, dtype=np.int64), np.asarray(d2f_cols, dtype=np.int64)),
        ),
        shape=(n_passages, n_facts),
    )
    facts_to_phrases_mat = csr_array(
        (
            np.ones(len(f2p_rows), dtype=np.uint8),
            (np.asarray(f2p_rows, dtype=np.int64), np.asarray(f2p_cols, dtype=np.int64)),
        ),
        shape=(n_facts, n_phrases),
    )

    # Node specificity (legacy hipporag.py:420 phrase_to_num_doc):
    # number of distinct documents each phrase appears in (via facts).
    # doc_to_phrases_mat = docs_to_facts_mat @ facts_to_phrases_mat
    # then binarise (any nonzero -> 1) before column-summing.
    doc_to_phrases_mat = docs_to_facts_mat @ facts_to_phrases_mat
    binary = (doc_to_phrases_mat > 0).astype(np.uint32)
    phrase_to_num_doc = np.asarray(binary.sum(axis=0)).ravel().astype(np.int64)

    return phrase_to_id, fact_to_id, docs_to_facts_mat, facts_to_phrases_mat, phrase_to_num_doc


def build_triple_edges(
    fact_to_id: dict[tuple[str, str, str], int],
    phrase_to_id: dict[str, int],
) -> dict[tuple[int, int], float]:
    """Symmetric triple-derived edges per legacy create_graph.py:188-233.

    For every fact (head, rel, tail): add (head_id, tail_id) +1.0 and
    (tail_id, head_id) +1.0 to the edge dict. Same (h, t) pair across
    multiple facts accumulates weight.
    """
    edges: dict[tuple[int, int], float] = {}
    for head, _rel, tail in fact_to_id.keys():
        h_id = phrase_to_id[head]
        t_id = phrase_to_id[tail]
        if h_id == t_id:
            continue  # skip self-loops; PPR damping handles dwell
        edges[(h_id, t_id)] = edges.get((h_id, t_id), 0.0) + 1.0
        edges[(t_id, h_id)] = edges.get((t_id, h_id), 0.0) + 1.0
    return edges


# --- Contriever embedder + pooling HARD GATE -------------------------------


def _legacy_masked_mean_encode(
    texts: list[str],
    *,
    embedder_id: str,
) -> np.ndarray:
    """Reference encoding using the legacy masked-mean pooling code path.

    Verbatim port of `scratch_hipporag/src/processing.py:mean_pooling +
    mean_pooling_embedding_with_normalization`. Loads the raw HF model
    via transformers (NOT sentence-transformers) and computes the
    attention-mask-weighted mean of the last hidden state, then
    L2-normalises. Used as the ground truth in `verify_contriever_pooling`
    — if SentenceTransformer's auto-wrap disagrees with this, we trust
    this version.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(embedder_id)
    model = AutoModel.from_pretrained(embedder_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)
    token_embs = outputs.last_hidden_state  # (B, T, D)
    mask = enc["attention_mask"]  # (B, T)
    # Masked-mean (legacy mean_pooling lines 12-15):
    token_embs = token_embs.masked_fill(~mask[..., None].bool(), 0.0)
    summed = token_embs.sum(dim=1)
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = summed / counts
    # L2-normalise (legacy line 34):
    normed = mean / mean.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return normed.detach().cpu().numpy().astype(np.float32)


def verify_contriever_pooling(
    embedder_id: str = "facebook/contriever",
    *,
    sample_phrases: list[str] | None = None,
    cosine_floor: float = 0.99,
) -> dict[str, Any]:
    """HARD GATE: SentenceTransformer's masked-mean must match legacy's.

    Encodes a small probe set via both paths (SentenceTransformer
    auto-wrap vs from-scratch masked-mean over raw HF model) and
    asserts the minimum per-row cosine >= cosine_floor. Raises
    RuntimeError with diagnostics on failure.

    Cosine is computed on the L2-normalised vectors both paths return,
    so it's a straight dot product per row.

    Call once at the top of M6 indexing, before embed_phrases. The
    result dict is logged into the manifest so a later run can confirm
    the gate fired and what the actual min-cosine was.
    """
    from sentence_transformers import SentenceTransformer

    probe = sample_phrases or [
        "Radio City",
        "India",
        "private FM radio station",
        "PlanetRadiocity.com",
        "music portal",
    ]

    # Path A: SentenceTransformer auto-wrap (what M6 will use at scale).
    st_model = SentenceTransformer(embedder_id)
    st_vecs = st_model.encode(
        probe,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Path B: legacy masked-mean over raw HF model.
    legacy_vecs = _legacy_masked_mean_encode(probe, embedder_id=embedder_id)

    if st_vecs.shape != legacy_vecs.shape:
        raise RuntimeError(
            f"Contriever pooling gate FAILED: shape mismatch. "
            f"SentenceTransformer={st_vecs.shape} vs legacy={legacy_vecs.shape}. "
            "Port the legacy HuggingFaceWrapper masked-mean directly."
        )

    cosines = (st_vecs * legacy_vecs).sum(axis=1)
    min_cos = float(cosines.min())
    mean_cos = float(cosines.mean())

    report = {
        "embedder_id": embedder_id,
        "n_probe": len(probe),
        "min_cosine": min_cos,
        "mean_cosine": mean_cos,
        "cosine_floor": cosine_floor,
        "passed": bool(min_cos >= cosine_floor),
    }

    if not report["passed"]:
        raise RuntimeError(
            f"Contriever pooling HARD GATE FAILED. min_cosine={min_cos:.6f} "
            f"< floor={cosine_floor}. SentenceTransformer auto-wrap pooling "
            f"diverges from Contriever's documented masked-mean. Per-row "
            f"cosines: {cosines.tolist()}. Diagnose by inspecting the "
            f"SentenceTransformer({embedder_id}).modules() pooling config "
            "and either fix it or replace embed_phrases with "
            "_legacy_masked_mean_encode (which is known correct)."
        )

    return report


def embed_phrases(
    phrases_in_id_order: list[str],
    *,
    embedder_id: str,
) -> np.ndarray:
    """Contriever phrase embeddings via SentenceTransformer, L2-normalised float32.

    Callers MUST run verify_contriever_pooling first (M6's index method
    enforces this). This function trusts that the gate has already
    passed; it does not re-verify on every call. Returns shape
    (len(phrases), 768) for facebook/contriever.
    """
    if not phrases_in_id_order:
        return np.zeros((0, 768), dtype=np.float32)
    # Reuse the harness's shared embed_texts so the embedder identity is
    # the same one logged in the components line and used at query time.
    from .models import embed_texts
    return embed_texts(phrases_in_id_order, model_name=embedder_id)


# --- Synonymy edges --------------------------------------------------------


def add_synonymy_edges(
    edges: dict[tuple[int, int], float],
    phrase_embeddings: np.ndarray,
    *,
    sim_threshold: float,
    top_k_cap: int,
) -> tuple[dict[tuple[int, int], float], int]:
    """Augment edge dict with Contriever-cosine synonymy edges.

    Faithful port of `scratch_hipporag/src/create_graph.py:251-307`.

    For each phrase row: cosine vs ALL phrases (matrix multiply since
    rows are L2-normalised), sort descending, take neighbours with
    score >= sim_threshold, cap at top_k_cap, skip self. Edge weight =
    cosine score (legacy line 294 `graph_plus[sim_edge] = similarity_max
    * score` with similarity_max=1.0).

    Direction matches legacy: single-direction add per source phrase.
    Symmetric in aggregate because every phrase is iterated as source.

    Returns (augmented_edges_dict, n_synonymy_edges_added).
    """
    n = phrase_embeddings.shape[0]
    if n == 0:
        return edges, 0

    # Cosine matrix (rows are L2-normalised so dot = cosine).
    sim_matrix = phrase_embeddings @ phrase_embeddings.T  # (n, n)
    np.fill_diagonal(sim_matrix, -np.inf)  # exclude self from top-K

    n_added = 0
    for phrase_id in range(n):
        sims = sim_matrix[phrase_id]
        # argsort descending; take top_k_cap candidates, then filter by
        # threshold (matches legacy line 280-281 break-on-threshold).
        top_k_idx = np.argpartition(-sims, kth=min(top_k_cap, n - 1))[: top_k_cap]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]

        num_nns = 0
        for nn_id in top_k_idx:
            if num_nns >= top_k_cap:
                break
            score = float(sims[nn_id])
            if score < sim_threshold:
                break  # legacy line 280: also breaks at threshold
            sim_edge = (int(phrase_id), int(nn_id))
            edges[sim_edge] = score  # overwrite; matches legacy graph_plus[sim_edge] = score
            n_added += 1
            num_nns += 1

    return edges, n_added


# --- igraph assembly -------------------------------------------------------


def assemble_igraph(
    edges: dict[tuple[int, int], float],
    n_phrases: int,
) -> Any:
    """Wrap the edge dict into an igraph.Graph with `weight` attribute set.

    Faithful port of `scratch_hipporag/src/hipporag.py:441-467`. Edges
    deduplicated by dict-key uniqueness (the dict already enforces this).
    Self-loops excluded by build_triple_edges and add_synonymy_edges.
    """
    import igraph as ig

    edge_list = list(edges.keys())
    weights = [edges[e] for e in edge_list]
    g = ig.Graph(n=n_phrases, edges=edge_list, directed=False)
    g.es["weight"] = weights
    return g


# --- Persistence -----------------------------------------------------------

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


def save_graph(graph_bundle: HippoGraph, dir_path: Path) -> None:
    """Persist the full graph bundle. Pickle-free for cross-version safety.

    Tuple-keyed dicts are flattened to JSON-safe lists (fact_to_id keys
    are serialised as `["head", "rel", "tail"]`; phrase_to_id keys are
    already strings). Sparse matrices use scipy.sparse.save_npz which is
    forward/backward compatible across numpy 1.x/2.x. igraph itself is
    NOT persisted — rebuilt deterministically from graph_edges.json on
    load.
    """
    from scipy.sparse import save_npz

    dir_path.mkdir(parents=True, exist_ok=True)

    (dir_path / "phrase_to_id.json").write_text(
        json.dumps(graph_bundle.phrase_to_id, ensure_ascii=False)
    )
    (dir_path / "fact_to_id.json").write_text(
        json.dumps(
            [[list(k), v] for k, v in graph_bundle.fact_to_id.items()],
            ensure_ascii=False,
        )
    )
    save_npz(dir_path / "docs_to_facts.npz", graph_bundle.docs_to_facts_mat)
    save_npz(dir_path / "facts_to_phrases.npz", graph_bundle.facts_to_phrases_mat)
    (dir_path / "graph_edges.json").write_text(
        json.dumps(
            [[int(h), int(t), float(w)] for h, t, w in graph_bundle.edges],
            ensure_ascii=False,
        )
    )
    np.save(dir_path / "phrase_embeddings.npy", graph_bundle.phrase_embeddings)
    np.save(dir_path / "phrase_to_num_doc.npy", graph_bundle.phrase_to_num_doc)


def load_graph(dir_path: Path) -> HippoGraph:
    """Reverse of save_graph. Rebuilds igraph from graph_edges.json."""
    from scipy.sparse import load_npz

    phrase_to_id: dict[str, int] = json.loads(
        (dir_path / "phrase_to_id.json").read_text()
    )
    raw_fact = json.loads((dir_path / "fact_to_id.json").read_text())
    fact_to_id: dict[tuple[str, str, str], int] = {
        tuple(k): v for k, v in raw_fact
    }
    docs_to_facts_mat = load_npz(dir_path / "docs_to_facts.npz")
    facts_to_phrases_mat = load_npz(dir_path / "facts_to_phrases.npz")
    raw_edges = json.loads((dir_path / "graph_edges.json").read_text())
    edges: list[tuple[int, int, float]] = [(int(h), int(t), float(w)) for h, t, w in raw_edges]
    phrase_embeddings = np.load(dir_path / "phrase_embeddings.npy")
    phrase_to_num_doc = np.load(dir_path / "phrase_to_num_doc.npy")

    edge_dict = {(h, t): w for h, t, w in edges}
    graph = assemble_igraph(edge_dict, n_phrases=len(phrase_to_id))

    return HippoGraph(
        phrase_to_id=phrase_to_id,
        fact_to_id=fact_to_id,
        docs_to_facts_mat=docs_to_facts_mat,
        facts_to_phrases_mat=facts_to_phrases_mat,
        edges=edges,
        graph=graph,
        phrase_embeddings=phrase_embeddings,
        phrase_to_num_doc=phrase_to_num_doc,
    )


__all__ = [
    "HippoGraph",
    "REQUIRED_FILES",
    "build_graph_structures",
    "build_triple_edges",
    "verify_contriever_pooling",
    "embed_phrases",
    "add_synonymy_edges",
    "assemble_igraph",
    "save_graph",
    "load_graph",
]
