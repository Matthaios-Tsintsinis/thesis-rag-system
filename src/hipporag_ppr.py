"""HippoRAG 1 query-time PPR: NER + dense link -> reset vector -> PPR -> doc scores.

Faithful port of `scratch_hipporag/src/hipporag.py:173-294` rank_docs and
the helpers it calls (link_node_by_dpr, run_pagerank_igraph_chunk).

Damping note (CRITICAL — verified from legacy main-experiment scripts,
not from paper prose):

    The legacy class default is damping=0.1 (hipporag.py:33), but every
    published-experiment shell script in the legacy repo overrides this
    to --damping 0.5 via CLI:

        run_hipporag_main_exps.sh:5
        run_hipporag_ablations.sh:4
        run_hipporag_case_study.sh:4/7/10
        run_hipporag_ircot_main_exps.sh:4/5/11/12

    The constructor 0.1 is a code stub; 0.5 is what their headline
    numbers were produced with. Single-step + Contriever + sim 0.8 +
    damping 0.5 is the exact main-experiment config for HippoRAG-Single
    Contriever, which is our M6 target.

    igraph's `personalized_pagerank(damping=...)` is the continue-walk
    probability (1 - restart). damping=0.5 -> 50% continue, 50% restart
    from the personalised reset vector. Higher restart than vanilla
    PageRank (~0.85), consistent with strong query personalisation.

    M6Config.damping = 0.5 is therefore byte-for-byte faithful at the
    call site.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .hipporag_openie import processing_phrases
from .models import embed_texts


def link_query_entities_to_phrases(
    query_ner_strings: list[str],
    phrase_embeddings: np.ndarray,
    *,
    embedder_id: str,
) -> list[tuple[int, float]]:
    """For each query NER string, argmax cosine vs the phrase-node embeddings.

    Faithful port of `scratch_hipporag/src/hipporag.py:596-632`
    link_node_by_dpr. Each NER string is normalised via
    processing_phrases first (matches legacy hipporag.py:308), then
    embedded via Contriever using the same `embed_texts(model_name=
    embedder_id)` path used at index time — guarantees encoding parity.

    Returns list of (phrase_id, cosine_score) per NER string in input
    order. If query_ner_strings is empty, returns empty list (caller
    handles the empty-NER fallback at the system level).
    """
    if not query_ner_strings or phrase_embeddings.shape[0] == 0:
        return []

    normalised = [processing_phrases(s) for s in query_ner_strings]
    normalised = [s for s in normalised if s]  # drop strings that normalise to empty
    if not normalised:
        return []

    query_embs = embed_texts(normalised, model_name=embedder_id)
    # query_embs and phrase_embeddings are both L2-normalised, so
    # dot = cosine.
    sims = query_embs @ phrase_embeddings.T  # (n_query, n_phrases)
    best_ids = sims.argmax(axis=1)
    best_scores = sims[np.arange(len(normalised)), best_ids]
    return [(int(i), float(s)) for i, s in zip(best_ids, best_scores)]


def build_reset_vector(
    linked: list[tuple[int, float]],
    *,
    n_phrases: int,
    phrase_to_num_doc: np.ndarray,
    node_specificity: bool,
) -> np.ndarray:
    """Personalisation vector for PPR.

    Faithful port of `scratch_hipporag/src/hipporag.py:617-628`. For each
    linked phrase_id: weight = 1 / phrase_to_num_doc[phrase_id] if
    node_specificity, else 1.0. The legacy code special-cases phrases
    with num_doc == 0 (returns weight 1, line 622); we replicate exactly.

    If `linked` is empty, returns a zero vector — the caller (M6 system)
    detects empty-NER upstream and switches to the uniform fallback;
    this function never silently substitutes a uniform vector itself.
    """
    reset = np.zeros(n_phrases, dtype=np.float64)
    for phrase_id, _score in linked:
        if not (0 <= phrase_id < n_phrases):
            continue
        if node_specificity:
            n_doc = int(phrase_to_num_doc[phrase_id])
            if n_doc == 0:
                weight = 1.0  # legacy line 622 fallback
            else:
                weight = 1.0 / n_doc
        else:
            weight = 1.0
        reset[phrase_id] = weight
    return reset


def run_pagerank(
    igraph_graph: Any,
    reset_vector: np.ndarray,
    *,
    damping: float,
) -> np.ndarray:
    """One PPR run. Matches `scratch_hipporag/src/hipporag.py:524-538`.

    igraph call (continue-walk damping convention):
        g.personalized_pagerank(
            vertices=range(n_phrases),
            damping=damping,                 # 0.5 per main_exps
            directed=False,
            weights='weight',
            reset=reset_vector,
            implementation='prpack',
        )

    Returns (n_phrases,) float64.
    """
    n_phrases = igraph_graph.vcount()
    if n_phrases == 0:
        return np.zeros(0, dtype=np.float64)
    probs = igraph_graph.personalized_pagerank(
        vertices=range(n_phrases),
        damping=damping,
        directed=False,
        weights="weight",
        reset=reset_vector.tolist(),
        implementation="prpack",
    )
    return np.asarray(probs, dtype=np.float64)


def _min_max_normalize(v: np.ndarray) -> np.ndarray:
    """Verbatim port of legacy processing.min_max_normalize."""
    if v.size == 0:
        return v
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 0.0:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def propagate_phrase_to_doc(
    ppr_phrase_probs: np.ndarray,
    docs_to_facts_mat: Any,
    facts_to_phrases_mat: Any,
) -> np.ndarray:
    """phrase PPR -> fact prob -> doc prob, min-max normalised.

    Faithful port of `scratch_hipporag/src/hipporag.py:229-231`:
        fact_prob = facts_to_phrases_mat.dot(ppr_phrase_probs)
        ppr_doc_prob = docs_to_facts_mat.dot(fact_prob)
        ppr_doc_prob = min_max_normalize(ppr_doc_prob)
    """
    fact_prob = facts_to_phrases_mat @ ppr_phrase_probs
    doc_prob = docs_to_facts_mat @ fact_prob
    # csr_array @ vec returns ndarray-like; ensure ndarray for downstream.
    doc_prob = np.asarray(doc_prob).ravel().astype(np.float64)
    return _min_max_normalize(doc_prob)


def uniform_fallback(n_chunks: int) -> np.ndarray:
    """Empty-NER fallback per legacy `hipporag.py:233`.

    When query NER returns no entities the legacy code sets:
        ppr_doc_prob = np.ones(n_chunks) / n_chunks

    Effect: every chunk gets identical score, top-k is arbitrary order.
    Faithful behaviour preserved — M6 logs every empty-NER event
    prominently so the analysis phase can quantify the impact.
    """
    if n_chunks <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.ones(n_chunks, dtype=np.float64) / float(n_chunks)


__all__ = [
    "link_query_entities_to_phrases",
    "build_reset_vector",
    "run_pagerank",
    "propagate_phrase_to_doc",
    "uniform_fallback",
]
