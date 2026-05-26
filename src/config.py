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

# CK-4 (shared context-budget machinery, OPT-IN). The packer
# (src.prompt_packing.pack_context) supports a token budget over the
# evidence block, but the DEFAULT IS OFF per professor's directive:
# baselines must run at their natural strength, unconstrained. The
# budget exists as an ABLATION tool for post-hoc context-volume
# studies, NOT as an imposed control on the main eval.
#
# EVIDENCE_TOKEN_BUDGET = None — no budget enforcement by default; each
# system's natural full retrieval (post-RETRIEVAL_RANKING_DEPTH deep
# pull and M7's quota un-cap) flows into the generator unchanged.
#
# To opt-in for an ablation run, set this constant at runtime via the
# CLI flag `python -m src.eval.runner --evidence-budget 3000`, which
# monkey-patches `src.config.EVIDENCE_TOKEN_BUDGET` for the duration
# of the process. The packer then enforces; the analyser's
# --check-budget-equality assertion becomes meaningful.
EVIDENCE_TOKEN_BUDGET: int | None = None
EVIDENCE_TOKEN_BUDGET_TOKENIZER = "gpt-4o-mini"

# RETRIEVAL_RANKING_DEPTH: deeper candidate pool from retrieve(). NOT a
# constraint — it gives the packer (when opt-in) a larger menu, and
# under the no-budget default it lets M7's un-cap quota fill freely.
# Doesn't limit any system's natural strength: bumping FAISS top-K
# extends the tail; the head of the ranking is unchanged.
RETRIEVAL_RANKING_DEPTH = 50


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

# Canonical abstention response. The reader prompt instructs the model
# to output exactly this string when the evidence does not contain the
# answer. The eval-layer unanswerable scorer detects it deterministically
# in addition to a fuzzy phrase-match fallback. Keep this string short,
# distinctive (no leading article so it doesn't trip the QASPER token-F1
# normaliser by accident), and stable: changing it invalidates the
# scorer's exact-match branch.
ABSTENTION_RESPONSE = "No answer available."

BASE_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided evidence. "
    f"If the evidence does not contain the answer, respond with exactly: "
    f"'{ABSTENTION_RESPONSE}'. "
    "Do not fabricate; do not guess. Be concise and factual."
)

CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the user's question from your own knowledge. "
    f"If you do not know the answer, respond with exactly: "
    f"'{ABSTENTION_RESPONSE}'. "
    "Do not fabricate; do not guess."
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

    The optional `embedder`, `chunker`, `reranker` fields are component
    overrides resolved by `src.components.resolve_components`. None = use
    the shared default (EMBEDDER_MODEL for embedder, HarnessConfig.chunking
    for chunker, no reranker for M4). The index-time LLM is the existing
    `summary_model` field; the resolver normalises that to its
    index_llm_id slot. There is no `final_generator` field by design —
    the answer generator is harness-level (HarnessConfig.generation),
    held constant across systems.
    """
    build: RaptorBuildParams = field(default_factory=RaptorBuildParams)
    expansion: ExpansionParams = field(default_factory=ExpansionParams)
    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K
    include_root_in_flat_index: bool = False
    summary_model: str = JUDGE_MODEL  # gpt-4o-mini by project decision
    top_k_final: int = FINAL_CONTEXT_CHUNKS

    # --- component overrides (per-paper assignment) ---
    # Per-paper rule (professor-approved): each system uses the
    # components its own paper specifies. The RAPTOR paper uses
    # multi-qa-mpnet-base-cos-v1 as its SBERT checkpoint (verified
    # against the official repo's EmbeddingModels.py default and the
    # ICLR 2024 paper text). multi-qa-mpnet-base-cos-v1 is 768-dim
    # vs bge-m3's 1024-dim, so M4's substrate cache forks off the
    # previously-shared M4/M7 RAPTOR substrate directory under a new
    # hash. Chunker and reranker stay None (= shared default chunker,
    # no reranker — RAPTOR has no cross-encoder rerank). The index-
    # time summariser stays as `summary_model` (gpt-4o-mini,
    # modernised from the paper's GPT-3.5-turbo per professor
    # direction "preserve architecture, modernize deprecated model").
    # English-centric mpnet vs multilingual bge-m3 means M4 retrieval
    # degrades on Greek queries — documented as a paper-faithfulness
    # limitation in the methods section.
    embedder: str | None = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    chunker: ChunkingConfig | None = None
    reranker: str | None = None  # M4 does not rerank

    trace: bool = False


# M5Config (GraphRAG) and M8Config (hierarchical cluster-tree port) were
# dropped from the active roster when the per-paper-component rule was
# adopted. Their dataclasses are preserved verbatim in
# `src/retrievers/deprecated/_archived_config.py` for resurrection.


@dataclass(frozen=True)
class M6Config:
    """M6 — HippoRAG 1 (legacy, NeurIPS'24), single-step retrieval.

    Faithful port of the OSU-NLP-Group/HippoRAG legacy branch. Components
    per the paper's main-experiment shell scripts (not the constructor
    defaults, which are stubs — see notes below).

    Damping convention: igraph's `personalized_pagerank(damping=...)` is
    the continue-walk probability (1 - restart probability). The legacy
    main_exps + ablations + case_study + ircot_main_exps shell scripts
    all use `--damping 0.5`, overriding the constructor default of 0.1.
    M6Config.damping = 0.5 is byte-for-byte faithful at the call site.

    Empty-NER policy: when query NER returns zero entities, the legacy
    code falls back to uniform doc_prob (random top-k). M6 preserves
    that fallback for paper faithfulness and logs every empty-NER event
    so the impact can be quantified in analysis (do NOT silently fix).

    Multilingual: Contriever is English-centric; processing_phrases
    drops non-ASCII characters. Greek queries degrade — documented as a
    paper-faithfulness limitation alongside M4's mpnet.
    """

    # --- index-time LLM (paper used gpt-3.5-turbo-1106; modernised) ---
    openie_llm: str = JUDGE_MODEL  # gpt-4o-mini
    openie_prompt_version: str = "v1"  # bumped when any prompt string changes

    # --- graph build (paper defaults) ---
    sim_threshold: float = 0.8       # synonymy edge cutoff
    synonym_top_k_cap: int = 100     # per-source-phrase neighbour cap (legacy line 280)
    node_specificity: bool = True

    # --- PPR (paper main-experiment values, NOT constructor defaults) ---
    damping: float = 0.5             # continue-walk probability; see class docstring
    doc_ensemble: bool = False       # paper headline = single-step + no DPR ensemble
    dpr_only: bool = False

    # --- query ---
    max_query_ner: int = 8           # bound personalisation-vector size (soft cap, not in paper)
    top_k_final: int = FINAL_CONTEXT_CHUNKS

    # --- component overrides (per-paper assignment) ---
    # Contriever (768-dim) per the HippoRAG paper / legacy
    # main_exps.sh:5. Same dim as M4's mpnet but different model;
    # M4 and M6 cache namespaces are entirely separate (no shared
    # substrate — M6 has its own graph-based artifacts in cache/M6/).
    # Chunker falls through to the harness default (paper takes
    # pre-chunked passages; chunking is OUR choice, faithful to no
    # specific paper choice). Reranker stays None (HippoRAG has no
    # cross-encoder rerank).
    embedder: str | None = "facebook/contriever"
    chunker: ChunkingConfig | None = None
    reranker: str | None = None

    trace: bool = False


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
    """§4.3 final-context budget allocation.

    CK-4 update: defaults bumped from {final_context_chunks=15,
    max_chunks_per_aspect=8} to {50, 25} so M7's quota machinery
    distributes over the shared RETRIEVAL_RANKING_DEPTH=50 budget the
    rest of the systems also retrieve. The shared packer
    (src.prompt_packing.pack_context) then enforces the token-level
    EVIDENCE_TOKEN_BUDGET=3000 at prompt-build time uniformly across
    all systems. The quota algorithm itself is unchanged — same
    proportional split, same per-aspect clamps, just sized to the
    deeper budget so M7 doesn't self-handicap to 8-12 chunks while
    baselines feed 15.
    """
    final_context_chunks: int = RETRIEVAL_RANKING_DEPTH  # 50 post-CK-4
    global_view_quota: int = 2
    min_chunks_per_aspect: int = 2
    max_chunks_per_aspect: int = 25


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

    # --- component overrides (None = shared default) ---
    # Resolved by src.components.resolve_components. The reranker default
    # for M7 is RERANKER_MODEL (bge-reranker-v2-m3), passed by the M7
    # resolve_components(..., default_reranker=RERANKER_MODEL) call site;
    # `reranker=None` here means "use that per-system default", not "no
    # reranker". Embedder/chunker fall back to HarnessConfig defaults.
    # No `final_generator` field — final generator is harness-level
    # (HarnessConfig.generation), held constant across systems.
    embedder: str | None = None
    chunker: ChunkingConfig | None = None
    reranker: str | None = None

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
    m6: M6Config = field(default_factory=M6Config)
    m7: M7Config = field(default_factory=M7Config)


DEFAULT_CONFIG = HarnessConfig()
