"""Shared configuration for the thesis RAG harness.

Values here are defaults for every system in the benchmark (M1-M8).
System-specific knobs live next to their implementation.
The PIPELINE_DESIGN.md M7 spec is the source of truth — if a constant
here disagrees with that doc, the doc wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# --- Filesystem ------------------------------------------------------------
# Path roles (INPUT_DIR / CACHE_DIR / OUTPUT_DIR / HF_CACHE_DIR) are resolved
# at runtime by src/paths.py — they depend on whether Drive is mounted and
# on environment-variable overrides. Do not hardcode any of those here.

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Shared models ---------------------------------------------------------

EMBEDDER_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL = "gpt-4o-mini"


# --- Retrieval defaults ---------------------------------------------------

FINAL_CONTEXT_CHUNKS = 15
FIRST_STAGE_TOP_K = 50
RRF_K = 60                       # Cormack et al. (2009)


# --- Chunking -------------------------------------------------------------
# Two strategies, selected per HarnessConfig.chunking.strategy:
#   "semantic"    — sentence-buffered embeddings + percentile breakpoints
#                   (Greek-aware: . ! ? ; as terminators, · excluded).
#                   Will be the production default once M4/M7 land; for
#                   now nothing in the harness uses it and the default
#                   stays word_window so M1/M2/M3 behaviour is unchanged.
#   "word_window" — fixed word window + overlap. Used in smoke tests and
#                   as the current default while baselines are stabilising.

ChunkingStrategy = Literal["semantic", "word_window"]


# --- Generation -----------------------------------------------------------

GEN_MAX_NEW_TOKENS = 512
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0
LOAD_GENERATOR_IN_4BIT = True


# --- Prompts --------------------------------------------------------------

BASE_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided evidence. "
    "If the evidence is insufficient, say so explicitly rather than fabricate. "
    "Be concise and factual."
)

CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the user's question from your own knowledge. "
    "If you do not know the answer, say so explicitly rather than fabricate."
)

M8_LOW_CONFIDENCE_ANSWER = (
    "I do not have sufficient evidence in the provided documents to answer "
    "this question reliably."
)


# --- Typed configs --------------------------------------------------------

NodeType = Literal["chunk", "summary_low", "summary_mid", "summary_high"]


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = FINAL_CONTEXT_CHUNKS
    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: ChunkingStrategy = "word_window"

    # --- semantic parameters (notebook chunk_text_semantic defaults) ---
    breakpoint_percentile: float = 90.0
    absolute_threshold: float = 0.5
    min_words: int = 80
    max_words: int = 400
    max_if_min_words: int = 500
    buffer_size: int = 1

    # --- word-window parameters ---
    chunk_words: int = 200
    overlap_words: int = 50

    # --- both ---
    min_chars_per_doc: int = 200


@dataclass(frozen=True)
class GenerationConfig:
    model: str = GENERATOR_MODEL
    max_new_tokens: int = GEN_MAX_NEW_TOKENS
    temperature: float = GEN_TEMPERATURE
    top_p: float = GEN_TOP_P
    load_in_4bit: bool = LOAD_GENERATOR_IN_4BIT


@dataclass(frozen=True)
class RaptorBuildParams:
    """RAPTOR cluster-tree topology (PIPELINE_DESIGN.md section 3.4).

    Defaults match the document. Smoke overrides them to produce a
    tree on the small fixture corpus.
    """
    branching_factor: int = 4
    min_cluster_size: int = 24
    max_depth: int = 4


@dataclass(frozen=True)
class ExpansionParams:
    """Per-node-type expansion (PIPELINE_DESIGN.md section 4.4)."""
    max_children_to_follow_from_broad_summary: int = 2
    summary_expansion_top_k_chunks: int = 3
    max_descendant_chunks_for_direct_expansion: int = 50
    max_expansion_recursion_depth: int = 2
    # Depth boundaries: 0-1 high (root excluded from flat index), 2 mid, 3+ low.
    high_level_max_depth: int = 1
    mid_level_depth: int = 2
    low_level_min_depth: int = 3


@dataclass(frozen=True)
class M4Config:
    """M4-specific knobs.

    M4 is the official-RAPTOR collapsed-retrieval baseline. No
    cross-encoder rerank (matches the published paper; rerank is M7's
    contribution, not M4's). Trace is opt-in: smoke flips it on for
    routing-path sanity checks; production benchmarks leave it off.
    """
    build: RaptorBuildParams = field(default_factory=RaptorBuildParams)
    expansion: ExpansionParams = field(default_factory=ExpansionParams)
    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K
    include_root_in_flat_index: bool = False
    summary_model: str = JUDGE_MODEL  # gpt-4o-mini by project decision
    top_k_final: int = FINAL_CONTEXT_CHUNKS
    trace: bool = False


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


# --- M7 sub-configs (PIPELINE_DESIGN.md §5 CONFIG, verbatim) --------------


@dataclass(frozen=True)
class AspectParams:
    """§4.1 query decomposition."""
    max_aspects: int = 3
    min_aspect_importance: float = 0.25
    drop_low_importance_aspects: bool = True


@dataclass(frozen=True)
class AspectScoringParams:
    """§4.2 aspect scoring = w_i·importance + w_c·retrieval_confidence.

    retrieval_confidence = sigmoid(top-1 cross-encoder logit) over the
    preliminary-rerank top-K first-stage hits of the aspect's paraphrase
    view. Source is fixed (cross_encoder_top1) by design — not a knob.
    """
    importance_weight: float = 0.5
    retrieval_confidence_weight: float = 0.5
    preliminary_rerank_top_k: int = 10


@dataclass(frozen=True)
class BudgetParams:
    """§4.3 final-context budget allocation."""
    final_context_chunks: int = FINAL_CONTEXT_CHUNKS  # 15 = 13 aspect + 2 global
    global_view_quota: int = 2
    min_chunks_per_aspect: int = 2
    max_chunks_per_aspect: int = 8


@dataclass(frozen=True)
class MultiBranchParams:
    """§4.4 Axis-1 Part B multi-branch tree traversal."""
    top_k_depth_1: int = 3
    top_k_per_level: int = 2
    max_depth: int = 4
    leaves_per_path: int = 5


@dataclass(frozen=True)
class StructuralAxisParams:
    """§4.4 Axis-2 Docling structural rerank/diversification."""
    section_diversity_cap: int = 3
    neighbor_radius: int = 1
    aspect_section_bias_factor: float = 1.15
    include_section_title_header: bool = True


@dataclass(frozen=True)
class DiversityParams:
    """§4.5 anti-redundancy caps (cluster ancestry tagged at index time)."""
    max_chunks_per_raptor_cluster: int = 4


@dataclass(frozen=True)
class ContextPackingParams:
    """§4.8 parent-summary orientation context packing."""
    max_ancestor_summaries_per_chunk_group: int = 2
    max_parent_summaries_per_chunk_group: int = 2
    max_parent_summary_tokens: int = 80
    summary_context_token_ratio: float = 0.15
    chunk_context_token_ratio: float = 0.85
    include_root_summary: bool = False


@dataclass(frozen=True)
class AbstentionParams:
    """§4.9 retrieval-side abstention signal."""
    retrieval_confidence_threshold: float = 0.40


@dataclass(frozen=True)
class M7Config:
    """M7 — three-axis hybrid over RAPTOR (the thesis contribution).

    Reuses the shared RAPTOR substrate: `build` / `expansion` /
    `summary_model` / `rrf_k` / `include_root_in_flat_index` MUST keep
    M4's defaults so M7 and M4 land on the same RAPTOR/<substrate_hash>/
    cache directory (see raptor.raptor_substrate_extra). Changing them
    forks the substrate and forces a rebuild.

    The eight ablation switches are top-level fields so the eval grid
    (evaluation_plan.pdf §4, A1-A8) flips exactly one off per row via
    config, never a code change. Six are pure toggles; A3 (view_types)
    and A6 (quota_preserving_rerank) gate explicit code branches in the
    orchestrator.
    """
    # --- shared RAPTOR substrate (keep == M4 defaults) ---
    build: RaptorBuildParams = field(default_factory=RaptorBuildParams)
    expansion: ExpansionParams = field(default_factory=ExpansionParams)
    summary_model: str = JUDGE_MODEL  # gpt-4o-mini
    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K
    include_root_in_flat_index: bool = False

    # --- M7 query-time sub-configs ---
    aspects: AspectParams = field(default_factory=AspectParams)
    scoring: AspectScoringParams = field(default_factory=AspectScoringParams)
    budget: BudgetParams = field(default_factory=BudgetParams)
    multi_branch: MultiBranchParams = field(default_factory=MultiBranchParams)
    structural: StructuralAxisParams = field(default_factory=StructuralAxisParams)
    diversity: DiversityParams = field(default_factory=DiversityParams)
    packing: ContextPackingParams = field(default_factory=ContextPackingParams)
    abstention: AbstentionParams = field(default_factory=AbstentionParams)

    # --- ablation switches (evaluation_plan.pdf §4) ---
    use_docling_structural_axis: bool = True          # A1
    use_intent_decomposition: bool = True             # A2
    view_types: tuple[str, ...] = ("paraphrase", "hyde")  # A3 -> (..,"paraphrase2")
    use_bm25: bool = True                             # A4
    include_parent_summaries: bool = True             # A5
    quota_preserving_rerank: bool = True              # A6
    pass_retrieval_confidence_to_llm: bool = True     # A7
    always_include_global_query_view: bool = True     # A8

    # --- diagnostics (smoke flips on for sanity checks; eval leaves off) ---
    trace: bool = False


@dataclass(frozen=True)
class HarnessConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    m4: M4Config = field(default_factory=M4Config)
    m7: M7Config = field(default_factory=M7Config)
    m8: M8Config = field(default_factory=M8Config)


DEFAULT_CONFIG = HarnessConfig()
