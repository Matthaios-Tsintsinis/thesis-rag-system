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

# Safe: raptor_paper imports only the stdlib and numpy, never src.config,
# so there is no cycle. (chunking.py late-imports raptor_paper for the
# same reason.)
from .raptor_paper import PaperTreeParams


# --- Filesystem ------------------------------------------------------------
# Path roles (INPUT_DIR / CACHE_DIR / OUTPUT_DIR / HF_CACHE_DIR) are resolved
# at runtime by src/paths.py — they depend on whether Drive is mounted and
# on environment-variable overrides. Do not hardcode any of those here.

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Shared models ---------------------------------------------------------

EMBEDDER_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# Final-answer generator, shared across ALL systems. Held constant so
# per-system answer-quality deltas attribute to retrieval, not to reader
# capacity.
#
# LOCAL as of 2026-08-02. gpt-4o-mini is removed from the project
# entirely; `src.models.generate` routes any non-OpenAI id to
# `_generate_local`, and `generate_batch` batches it through HF
# transformers. Serving is HF transformers rather than vLLM, which is
# unusable on Colab (CUDA-13 wheel against a CUDA-12 runtime, and its
# install cycle breaks torch badly enough to need a runtime delete).
#
# Measured on an L4: fp16, 15.2 GB after load, 0.52 req/s at 4k-in
# batch 8 (batch 12 OOMs), 2.07 req/s at 800-in batch 32.
#
# ⚠ THE READER-COMPARISON FRAMING IS DEAD (reversed 2026-08-06).
# Llama-3.1-8B-Instruct now runs the FULL MATRIX as a second, wholly
# INDEPENDENT replication: its own trees, its own summaries, its own
# caches. Nothing model-dependent is shared with the Qwen column.
# The consequence, stated rather than dropped: M4's two columns differ in
# TWO variables (who summarised and who read), so an M4 Qwen-vs-Llama
# difference cannot be attributed to either alone. M1/M2/M3/M9 have no
# index-time LLM, so they are unconfounded. See
# docs/PREREGISTRATION.md and the audit.
GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Index-time LLM (RAPTOR tree summaries; retired-M6 OpenIE).
#
# ⚠ THE PERMANENT PIN IS REVERSED (2026-08-06). It read: pinned to
# Qwen2.5-7B permanently, so a later second-reader pass would REUSE M4's
# trees and vary exactly one thing. The design is now FULL INDEPENDENT
# REPLICATION, so each column builds its own trees with its own
# summariser. Set per run with `--generator`, which moves this AND the
# reader together; it is in M4's substrate key, so the columns cannot
# collide. The cost of the reversal is the M4 two-variable confound,
# which is accepted and documented, not mitigated.
#
# KNOWN LIMITATION, recorded rather than mitigated: if this summariser's
# output is systematically poor, M4 is handicapped identically in every
# column and no later column can reveal it. The answer, if one is ever
# needed, is the NarrativeQA-10 sensitivity check with a second
# summariser — not a second full tree set.
#
# Kept as a separate constant from GENERATOR_MODEL so the two roles can
# diverge without touching every call site, even though they hold the
# same value today.
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Tiktoken ENCODING SELECTOR for evidence-token accounting. NOT a
# generator, despite the name — it is passed to
# tiktoken.encoding_for_model and only picks a byte-pair table. Left
# pointing at an OpenAI id deliberately: changing it would change
# evidence_tokens for every cell and break comparability with anything
# already banked. Token counting is a measurement convention here, not a
# model choice.


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

# DEPTH AT WHICH RETRIEVAL IS *MEASURED*, decoupled from what the reader
# sees. Rank-aware metrics used to be computed over whatever documents
# happened to surface in the reader's top-15 CHUNKS, so the document
# ranking's depth varied per query and per system: a system whose 15
# chunks collapsed into 4 articles was scored at Hit@10 over 4
# candidates, while one spreading over 15 got 10. That is not the
# published Hit@K, and it penalised M4 twice over, since its summary
# nodes carry no provenance and rank no document at all.
#
# Every system now returns a ranking of this depth FOR SCORING ONLY.
# Generation input is untouched: the reader still gets top-15
# (M2/M3/M9) or M4's 2,000-token budget fill.
SCORING_RANKING_DEPTH = 50


# --- Chunking -------------------------------------------------------------
# Two strategies, selected per HarnessConfig.chunking.strategy:
#   "semantic"    — sentence-buffered embeddings + percentile breakpoints
#                   (Greek-aware: . ! ? ; as terminators, · excluded).
#                   Will be the production default once M4/M7 land; for
#                   now nothing in the harness uses it and the default
#                   stays word_window so M1/M2/M3 behaviour is unchanged.
#   "word_window" — fixed word window + overlap. Used in smoke tests and
#                   as the current default while baselines are stabilising.
#   "raptor_100tok" — M4 ONLY. Paper-faithful RAPTOR leaves: contiguous,
#                   sentence-preserving, ~100 tiktoken cl100k_base tokens,
#                   NO overlap (src/raptor_paper.py). Reads chunk_words as
#                   a TOKEN count and requires overlap_words == 0.
#
# CACHE DISCIPLINE: a new strategy is a new VALUE of an existing field,
# so `asdict(ChunkingConfig())` is unchanged and no system's cache key
# moves. Adding a new FIELD to ChunkingConfig would move every system's
# key at once (compute_cache_key folds the whole asdict) — that is why
# raptor_100tok reuses chunk_words/overlap_words instead of declaring
# its own size field. tests/test_raptor_chunking.py pins the schema.

ChunkingStrategy = Literal["semantic", "word_window", "raptor_100tok"]


# --- Generation -----------------------------------------------------------

GEN_MAX_NEW_TOKENS = 512

# THE ONE BATCH SIZE, used by every cell in the matrix.
#
# Batch COMPOSITION can change generated text at temperature 0 — padding
# plus batched-matmul reduction order can flip an argmax on a near-tie,
# measured once at 8e-5 on a MultiHop answer mean. That is small, but it
# is only harmless while it is CONSTANT: cells generated at different
# batch sizes are not strictly comparable to each other. Fixing it here
# rather than passing it per invocation is what makes "the same batch
# size in every cell" a property of the code instead of a habit.
#
# 16 is the value the banked M1 cell used and the value the L4 headroom
# measurements were taken at; --batch-size still overrides for probes.
MATRIX_BATCH_SIZE = 16
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0
# fp16, NOT 4-bit. This default was True and it is the b6e35c6 failure
# mode waiting to recur: a silently quantized model is NOT the model the
# thesis names, and nothing in the output would have said so. Under the
# local-generator design the reported models (Qwen2.5-7B / Llama-3.1-8B,
# ~15-16GB fp16) fit an L4's 24GB without quantization, so there is no
# reason to quantize and every reason not to. `models.load_generator`
# now REFUSES to return a model whose realised quantization or dtype
# disagrees with what was requested.
LOAD_GENERATOR_IN_4BIT = False

# Placement budget handed to accelerate. "cpu": "0GiB" leaves NOWHERE to
# spill, which converts a silent offload into a load-time exception.
#
# Measured 2026-08-16, and the reason this exists. A second generator
# load landed in VRAM already holding the first copy plus the embedder;
# device_map="auto" did what it is designed to do and placed 62% of the
# weights off-device. The build did not fail. It produced correct
# summaries, a correct tree and a plausible results row, while every
# decode step streamed 4.74 B parameters across PCIe — a flat ~230 s per
# generate() call against a healthy 6.9 s, 33x, for twenty hours.
#
# 20GiB of a 22.03GiB L4 leaves headroom for KV and activations while
# still admitting the ~14.2GB fp16 weights. Raise it only with a measured
# reason; lowering it below the weight size makes every load raise.
GENERATOR_MAX_MEMORY = {0: "20GiB", "cpu": "0GiB"}


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
    # --- LEGACY: the top-down MiniBatchKMeans substrate (src/raptor.py) ---
    # Unused by the paper-faithful path, which builds bottom-up via
    # src/raptor_paper.py and reads `paper` below. Retained so that
    # reverting to the pre-fidelity M4 is a code revert against a
    # preserved cache (RAPTOR/bfc50c2...), not a config redesign, and so
    # existing callers that construct M4Config(build=..., expansion=...)
    # keep working.
    build: RaptorBuildParams = field(default_factory=RaptorBuildParams)
    expansion: ExpansionParams = field(default_factory=ExpansionParams)

    # --- paper-faithful tree (src/raptor_paper.py) ---
    paper: PaperTreeParams = field(default_factory=PaperTreeParams)
    # Reference TreeBuilderConfig.summarization_length. Ruling 4: this is
    # the ONLY concrete specification anywhere; the paper's reported 131
    # tokens is a MEASUREMENT, not a parameter, and is not to be
    # reverse-engineered into a cap. Expect visible truncation near 100 —
    # that discrepancy is itself a finding about the reference.
    summary_max_tokens: int = 100
    # M4-LOCAL prompt id. Deliberately NOT summarization.SUMMARY_PROMPT_VERSION:
    # that is a module-level constant the FROZEN M7 also reads, so bumping
    # it would move M7's substrate key. Proven, not assumed — see the
    # frozen-M7 key landmine table in CLAUDE.md.
    summary_prompt_version: str = "raptor_paper_v1"

    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K
    # Paper §3 Querying collapses the ENTIRE tree into one layer. There is
    # no synthetic all-corpus root to exclude under bottom-up
    # construction, so unlike the legacy path this is True.
    include_root_in_flat_index: bool = True
    # Index-time summariser. Pinned to ONE model permanently (2026-07-29):
    # M4's trees are always built by this model, so a later second-reader
    # pass reuses them rather than forking a second tree set. That makes
    # the later pass a READER comparison — same retrieval, same index,
    # different model reading it — NOT a generator comparison. Known
    # limitation, recorded in the audit: if this summariser's output is
    # systematically poor, M4 is handicapped identically in every column
    # and no later column can reveal it; the NarrativeQA-10 sensitivity
    # check with a second summariser is the answer if one is ever needed.
    summary_model: str = JUDGE_MODEL
    top_k_final: int = FINAL_CONTEXT_CHUNKS

    # Escape hatch for the doomed-build guard in m4_raptor.index(). A
    # RAPTOR tree build is the most expensive thing in the harness and
    # its summariser is baked into the substrate cache key, so building
    # one with the WRONG index-time LLM produces an artifact that is
    # thrown away entirely. Default False: refuse to start such a build.
    # Also settable per-run via the M4_ALLOW_API_INDEX_LLM env var, since
    # the runner constructs systems from DEFAULT_CONFIG and has no CLI
    # path to this field.
    allow_api_index_llm: bool = False

    # --- paper retrieval budget (professor-approved 2026-08-02) ---
    # Paper §3 Querying: "we use the collapsed tree with 2000 maximum
    # tokens, which approximately equates to retrieving the top-20
    # nodes", selecting by "Keep adding nodes to the result set until you
    # reach a predefined maximum number of tokens".
    #
    # M4 ONLY. M1/M2/M3/M9 stay at natural top-15 (locked decision #3).
    # The professor's ruling was that he does not mind top-15 versus a
    # token budget — what he wants is the model matching its paper. Those
    # four are METHOD baselines whose papers specify no budget, so moving
    # them would be a feasibility change wearing a fidelity
    # justification. Only RAPTOR changes, and only because its paper
    # says so.
    #
    # ASYMMETRY THIS CREATES, stated rather than left to be discovered:
    # M4 then answers from ~2,000 evidence tokens while M2/M3/M9 answer
    # from ~3,900 — M4 competes on roughly half the context. It compounds
    # with a measured side effect of the fidelity rebuild: the paper
    # chunker had already collapsed M4's context from ~3,900 to ~1,700
    # (top-15 of ~110-token units instead of ~260-token ones), so the
    # budget fill actually raises M4's context slightly rather than
    # lowering it.
    #
    # PAPER vs REFERENCE CODE, an observed divergence: the code applies
    # `indices[:top_k]` with top_k=10 BEFORE its 3500-token cap, so the
    # reference retrieves ~10 nodes (~1,000 tokens) and the token cap
    # never binds. We follow the PAPER TEXT, which is what the thesis
    # cites and competes against.
    #
    # Query-time only: NOT part of paper_substrate_extra, so changing it
    # never moves the substrate cache key.
    # None restores plain top_k_final selection.
    retrieval_budget_tokens: int | None = 2000

    # --- DIAGNOSTIC TWIN: leaf-expanded retrieval (default OFF) ---
    # Retrieved SUMMARY nodes carry an empty gold_provenance, so CK-2
    # cannot credit them — a summary is abstractive text with no gold
    # span. That is honest, and it has a consequence that must be
    # reported rather than hidden: 18.5-57% of M4's retrieved units
    # (paper App. I) are unscoreable BY CONSTRUCTION, so M4's retrieval
    # F1 is not directly comparable to a system returning only leaves.
    #
    # Turning this on replaces each retrieved summary with its top-N
    # descendant LEAVES (ranked against the query), which are scoreable,
    # producing a diagnostic twin that quantifies exactly that gap.
    #
    # NEVER A REPORTED M4 NUMBER. Expansion is applied POST-SELECTION, so
    # the retrieval decision is identical to real M4 — but the evidence
    # text changes, so answers change too. A run with this on is a
    # different system and its JSONL says so (every row carries
    # metadata m4_summary_expansion=true).
    #
    # Query-time only: deliberately NOT in raptor_paper.
    # paper_substrate_extra, so toggling it never moves the substrate key
    # and the diagnostic twin reuses the same tree.
    expand_summary_nodes: bool = False
    summary_expansion_leaves: int = 3

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
    # PER-SYSTEM chunker override — the paper's 100-token, sentence-
    # preserving, non-overlapping leaves. This is set HERE and never on
    # HarnessConfig.chunking: the harness default is inherited by M2, M3,
    # M9 and the frozen M7 (all of whose `chunker` is None), so changing
    # it would move four other systems' substrate keys at once. Note also
    # that `chunk_words` is read as a TOKEN budget under this strategy —
    # a new FIELD on ChunkingConfig would likewise move every key, since
    # compute_cache_key folds the whole asdict.
    chunker: ChunkingConfig | None = field(
        default_factory=lambda: ChunkingConfig(
            strategy="raptor_100tok", chunk_words=100, overlap_words=0
        )
    )
    reranker: str | None = None  # M4 does not rerank

    # --- index-time summariser batching (local model) ---
    # These REPLACE the previous `summary_max_workers` thread pool, which
    # was correct against an API and unsafe against a local one: threads
    # contend on the GIL, serialise onto one CUDA stream, each
    # `model.generate` allocates its own KV cache, and — decisively —
    # `models.load_generator` is lru_cached, so every thread shares one
    # tokenizer object that `generate_batch` MUTATES (padding side). A
    # thread observing a right-padded tokenizer mid-call does not crash;
    # it emits fluent text continued from PAD.
    #
    # BOTH ARE CACHE-KEY INPUTS (raptor_paper.paper_substrate_extra).
    # Batch composition can change generated text at temperature 0, and
    # summaries are CACHED — they are the artifact M4's substrate key
    # names — so the batch shape is named rather than assumed inert.
    # Accepted cost: retuning either invalidates every tree built at the
    # old value. Node ids and tree shape are NOT affected by either (they
    # are computed before any call is dispatched); a test pins that.
    #
    # summary_max_padded_tokens bounds n * longest-prompt in a batch.
    # It is not optional: cluster contexts range from ~110 tokens to
    # PaperTreeParams.max_length_in_cluster (3500), so a fixed count
    # sized for short clusters OOMs on a layer of long ones.
    #
    # THIS IS A HARNESS BATCHING KNOB, NOT A RAPTOR PARAMETER. The paper
    # specifies no such value, so no setting of it is "paper-faithful" in
    # the sense M4 ruling 4 uses for the 100-token summary cap. The
    # argument for preferring a HIGHER cap is one step removed and still
    # real: the cap bounds n * longest_prompt, so a lower cap constrains
    # which contexts may share a batch, and batch composition can move
    # generated text at temperature 0. A higher cap therefore leaves
    # summarisation less constrained BY THE HARNESS. Do not restate that
    # as fidelity to the paper.
    #
    # 16000 IS MEASURED, on story d431326b under the FIXED generator
    # loading (2026-08-16):
    #
    #   cap 16000 -> build 79.6 s, peak 17.44 GB, free 4.17 GB at the
    #                tightest call, 3 calls, mean width 11.33, max 25
    #   cap  8000 -> build 82.4 s, 4 calls, mean width 8.5, max 16
    #
    # Identical tree either way: 176 nodes, 34 summary nodes, layers
    # 142 -> 30 -> 4. Wider batches, fewer calls, slightly faster, and
    # 4.17 GB still free at the tightest moment.
    #
    # IT WAS NEVER A MEMORY LIMIT. 16000 had been demoted to 8000 because
    # it OOM'd after 370 s at a 21.62 GB peak — but that peak included a
    # PHANTOM SECOND COPY of the 15 GB generator, loaded because
    # load_generator was lru_cached on its argument tuple and two call
    # sites spelled the defaults differently. With one copy resident the
    # ceiling that forced 8000 does not exist.
    #
    # EVERY ABSOLUTE NUMBER IN THE OLD SWEEP IS VOID. It ran under the
    # double load, so each point paid a ~33x tax on every generate()
    # call. Its RELATIVE ordering survived (all points paid the same
    # tax) which is why 8000 was the right call at the time, but nothing
    # in it may be quoted as a cost.
    #
    # Keep summary_batch_size at 32: at cap 16000 the effective width
    # reaches 25, so the cap still binds and the nominal count does not.
    summary_batch_size: int = 32
    summary_max_padded_tokens: int = 16000

    # UNREAD by the paper-faithful path, retained so existing callers
    # construct. The rebuild dropped BM25 entirely: the paper's collapsed
    # retrieval is dense cosine only, and the sparse index existed only
    # because M4 used to SHARE a substrate with M7, which needs it. M4
    # now owns its namespace, so carrying non-paper machinery would be
    # keeping it for no reason. See m4_raptor.py DEVIATIONS item 7.
    hybrid_first_stage: bool = False

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
class CorrectiveConfig:
    """M9 — CorrectiveRAG (Yan et al., 2024), corpus-internal variant.

    M9 composes over the M3 hybrid substrate (no index-time artifacts
    of its own, no substrate cache namespace) and adds a query-time
    corrective loop: bge-reranker evaluator -> two-threshold action
    decision -> optional gpt-4o-mini query rewrite + re-retrieval ->
    strip refinement. All fields here are query-time parameters; none
    enter any cache key. Deviations from the paper are documented in
    the module comment block (src/retrievers/m9_corrective.py).

    THRESHOLD PROVENANCE (baked 2026-06-12). Derived empirically by
    scripts/derive_corrective_thresholds.py on the QASPER VALIDATION
    20-paper small sample (55 queries; 72 gold / 708 non-gold chunks,
    base rate 0.0923), artifact
    derivation_validation_20260612-014811.json. Criterion v2
    (non-gold percentile / FPR control):
      tau_high = 0.6395 — 90th percentile of the NON-GOLD confidence
        distribution (a chunk above it scores higher than ~all
        known-irrelevant chunks; FPR 0.100 by construction; precision
        0.193, lift 2.09x over base rate at the cut).
      tau_low  = 0.5001 — 5th percentile of the GOLD confidence
        distribution (discarding below it loses ~5% of gold).
    Derivation-time action mix: 49.1% correct / 50.9% ambiguous /
    0.0% incorrect; strip survival at tau_strip(=tau_low): 57.2%.
    The realized mix of every M9 run must roughly match (analyse.py
    prints it); large drift = miscalibration. Derived ONCE — no
    per-benchmark tuning; MultiHop transfer is checked via the
    action-mix logging, not re-derivation. The paper's published
    thresholds (0.59 / -0.99) live on its fine-tuned-T5 score scale
    and do not transfer; an absolute-precision criterion (v1) was
    retired after measuring a 0.50 precision ceiling at every cut —
    see the CALIBRATION FINDINGS block in m9_corrective.py.
    """

    tau_high: float = 0.6395  # max conf >= tau_high -> CORRECT (non-gold p90)
    tau_low: float = 0.5001   # max conf <  tau_low  -> INCORRECT (gold p5)
    # Strip-refinement threshold; None -> use tau_low (one fewer free
    # parameter — revisit only if refinement degenerates).
    tau_strip: float | None = None
    refine: bool = True             # strip refinement; flag kept for ablation
    strip_sentences: int = 2        # sentences per refinement strip
    rewrite_prompt_version: str = "v1"  # names REWRITE_PROMPT_V{n} in the module

    # --- component overrides (per-paper assignment) ---
    # Embedder/chunker stay None = shared defaults (the substrate IS
    # M3's: bge-m3 + harness chunker; overriding either here would
    # fork the inner M3 cache key and rebuild). The reranker default
    # for M9 is RERANKER_MODEL, passed by the M9
    # resolve_components(..., default_reranker=RERANKER_MODEL) call
    # site — reranker=None here means "use that per-system default".
    # No final_generator field — harness-level, held constant.
    embedder: str | None = None
    chunker: ChunkingConfig | None = None
    reranker: str | None = None

    trace: bool = False


@dataclass(frozen=True)
class HarnessConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    m4: M4Config = field(default_factory=M4Config)
    m6: M6Config = field(default_factory=M6Config)
    m7: M7Config = field(default_factory=M7Config)
    corrective: CorrectiveConfig = field(default_factory=CorrectiveConfig)


DEFAULT_CONFIG = HarnessConfig()
