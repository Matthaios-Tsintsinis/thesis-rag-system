"""Archived per-system configuration dataclasses and constants.

These previously lived in `src/config.py` but were moved here when M5
and M8 were dropped from the active evaluation roster. The archived
modules in this directory import `M5Config`, `M8Config`, and
`M8_LOW_CONFIDENCE_ANSWER` from `src.config`; those import paths are
preserved-as-was in the archived files (they will not resolve until
resurrected), but the definitions themselves are kept here so the code
still has somewhere to land if M5 or M8 is reactivated.

See `README.md` in this directory for the archival rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import EMBEDDER_MODEL, FINAL_CONTEXT_CHUNKS, JUDGE_MODEL


M8_LOW_CONFIDENCE_ANSWER = (
    "I do not have sufficient evidence in the provided documents to answer "
    "this question reliably."
)


@dataclass(frozen=True)
class M8Config:
    """M8-specific knobs (ported from existing hierarchical-RAG notebook).

    Linear `alpha_dense` fusion stays here — that's what distinguishes M8.
    M3 already covers the RRF variant. Don't switch this.
    """

    # tree
    tree_max_depth: int = 4
    tree_min_cluster_size: int = 24
    tree_branching_factor: int = 4
    tree_top_branches_per_level: int = 2
    tree_keywords_top_n: int = 8

    # hybrid fusion (linear; not RRF)
    alpha_dense: float = 0.75

    # query views (templated only; LLM-generated views are M7's job)
    max_query_views: int = 4
    enable_query_view_generation: bool = False

    # candidate selection
    top_docs_after_tree: int = 5
    top_chunks_per_doc_for_context: int = 3
    context_neighbor_radius: int = 1
    rerank_top_n: int = 30
    top_k_final: int = FINAL_CONTEXT_CHUNKS

    # abstention (sigmoid of cross-encoder logit)
    abstention_threshold: float = 0.35

    # TF-IDF keyword extractor fix
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95


@dataclass(frozen=True)
class M5Config:
    """M5 — Microsoft GraphRAG baseline (entity-relation graph + community
    detection/summarisation).

    A non-RAPTOR hierarchical paradigm — a baseline for M7 to beat, not an
    M7 component (evaluation_plan.pdf §3). The authors' implementation is
    wrapped, not reimplemented.

    Experimental controls (evaluation_plan.pdf §7):
      - Embedder held at bge-m3 across M2/M3/M4/M7; M5 matches it for a
        clean comparison. GraphRAG's text-embedding-3-small default would
        confound any M5-vs-others delta with the embedding model, so
        bge-m3 is injected via GraphRAG's library embedding-model
        registry instead.
      - Generator stays Qwen2.5-3B-Instruct (HarnessConfig.generation);
        GraphRAG's own answer LLM is not used. M5 differs from the other
        systems only in retrieval and embedding.
      - index_llm_model (entity extraction + community summaries) is
        gpt-4o-mini, matching the M4/M7 index-time LLM decision.

    graphrag_version is pinned and folded into the M5 cache key so a
    library bump invalidates cached graph artifacts cleanly.

    Note: the bge-m3 parity rationale above predates the per-paper
    component rule. Under the per-paper rule M5 would use the components
    its own paper specifies (paid OpenAI embeddings + GPT-4-class LLM),
    which is why it was archived — too costly for the eval grid.
    """

    graphrag_version: str = "3.0.9"
    embedder_model: str = EMBEDDER_MODEL      # bge-m3 — parity, eval-plan §7
    index_llm_model: str = JUDGE_MODEL        # gpt-4o-mini
    chunk_size: int = 1200                    # GraphRAG text-unit size (tokens)
    chunk_overlap: int = 100
    community_level: int = 2                  # local-search community depth
    retrieval_mode: str = "local"             # local search + community orientation
    top_k_final: int = FINAL_CONTEXT_CHUNKS
    trace: bool = False


__all__ = [
    "M5Config",
    "M8Config",
    "M8_LOW_CONFIDENCE_ANSWER",
]
