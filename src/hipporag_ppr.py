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

THIS FILE IS A C4a SKELETON — function bodies raise NotImplementedError.
C4b lands the working query path.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def link_query_entities_to_phrases(
    query_ner_strings: list[str],
    phrase_embeddings: np.ndarray,
    *,
    embedder_id: str,
) -> list[tuple[int, float]]:
    """For each query NER string, argmax cosine vs the phrase-node embeddings.

    Faithful port of `scratch_hipporag/src/hipporag.py:596-632`
    link_node_by_dpr. Returns list of (phrase_id, cosine_score) per NER
    string, length == len(query_ner_strings).

    NER strings are first embedded via Contriever using the same
    `embed_texts(model_name=embedder_id)` path the rest of the harness
    uses — guarantees query-side encoding matches index-side encoding
    so cosine is meaningful.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement query-entity -> phrase-node dense link.")


def build_reset_vector(
    linked: list[tuple[int, float]],
    *,
    n_phrases: int,
    phrase_to_num_doc: np.ndarray,
    node_specificity: bool,
) -> np.ndarray:
    """Personalisation vector for PPR.

    Faithful port of `scratch_hipporag/src/hipporag.py:617-628`. For each
    linked phrase_id: weight = 1/n_docs_containing_phrase if
    node_specificity else 1.0. Phrases not seen in any doc get weight 1
    (their fallback line 622).

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement reset-vector construction.")


def run_pagerank(
    igraph_graph: Any,
    reset_vector: np.ndarray,
    *,
    damping: float,
) -> np.ndarray:
    """One PPR run. Matches `scratch_hipporag/src/hipporag.py:524-538` byte-for-byte.

    Call:
        g.personalized_pagerank(
            vertices=range(n_phrases),
            damping=damping,                 # 0.5 per main_exps; see module docstring
            directed=False,
            weights='weight',
            reset=reset_vector,
            implementation='prpack',
        )

    Returns np.ndarray (n_phrases,) of stationary probabilities.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement igraph personalized_pagerank call.")


def propagate_phrase_to_doc(
    ppr_phrase_probs: np.ndarray,
    docs_to_facts_mat: Any,
    facts_to_phrases_mat: Any,
) -> np.ndarray:
    """phrase PPR -> fact prob -> doc prob, then min-max normalised.

    Faithful port of `scratch_hipporag/src/hipporag.py:229-231` and
    processing.min_max_normalize. Returns (n_chunks,) float32.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement phrase->fact->doc propagation.")


def uniform_fallback(n_chunks: int) -> np.ndarray:
    """Empty-NER fallback per legacy `hipporag.py:233`.

    When query NER returns no entities the legacy code sets:
        ppr_doc_prob = np.ones(n_chunks) / n_chunks

    Effect: every chunk gets identical score, top-k is arbitrary order.
    Faithful behaviour, even though it scores badly on benchmarks.
    M6Config keeps this exact fallback (no silent fix); the M6 system
    logs every empty-NER event prominently so the analysis phase can
    quantify the impact.

    NOT IMPLEMENTED in C4a.
    """
    raise NotImplementedError("C4b: implement uniform fallback.")


__all__ = [
    "link_query_entities_to_phrases",
    "build_reset_vector",
    "run_pagerank",
    "propagate_phrase_to_doc",
    "uniform_fallback",
]
