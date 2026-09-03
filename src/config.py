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

# CK-4 evidence budget: NONE at the harness level (locked decision 4:
# baselines run at their natural strength). The packer implements no
# budget since the repo reduction; the constant records the decision.
# M4's 2,000-token paper budget is M4Config.retrieval_budget_tokens and
# is applied by M4 itself, before packing.
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
#   "word_window" — fixed word window + overlap (200 / 50): the harness
#                   default, i.e. M2 and M3's chunker on every cell.
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
# The embedding-similarity "semantic" strategy left in the repo
# reduction (never selected on any cell); its six fields below STAY for
# exactly the reason above — removing a field would move every key.

ChunkingStrategy = Literal["word_window", "raptor_100tok"]


# --- Generation -----------------------------------------------------------

GEN_MAX_NEW_TOKENS = 512

# ANSWER GENERATION IS SEQUENTIAL, by construction: the answer path has
# no batch knob since the repo reduction. Measured 2026-08-18, M2 x
# MultiHop, 64 queries, L4: sequential 4.2558 s/query against 5.1654 at
# the best batched cap — a batch runs until its LONGEST member stops,
# and at answer time that cap is GEN_MAX_NEW_TOKENS = 512 while a
# typical answer is far shorter. Every banked cell answered
# sequentially. M4's 100-token tree summaries are the one place batching
# wins; that path has its own knobs, `M4Config.summary_batch_size` and
# `M4Config.summary_max_padded_tokens`, both inside M4's substrate key.
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0
# fp16, NEVER quantized. The 4-bit load option left in the repo
# reduction (it was False on every path); `models.load_generator`
# REFUSES a quantized model or an unexpected dtype, so a silently
# quantized model cannot become the model the thesis names.

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

    # --- KEEP-BY-CONSTRAINT: the former semantic chunker's parameters ---
    # The strategy itself is gone, but these six fields sit inside every
    # M2/M3/M4 substrate key (compute_cache_key folds the whole asdict),
    # so they stay at their historical defaults; deleting or renaming one
    # would move every banked key. TestCacheDiscipline pins the schema.
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


@dataclass(frozen=True)
class M4Config:
    """M4-specific knobs.

    M4 is the official-RAPTOR collapsed-retrieval baseline. No
    cross-encoder rerank (matches the published paper). Trace is
    opt-in; production benchmarks leave it off.

    The optional `embedder`, `chunker`, `reranker` fields are component
    overrides resolved by `src.components.resolve_components`. None = use
    the shared default (EMBEDDER_MODEL for embedder, HarnessConfig.chunking
    for chunker, no reranker for M4). The index-time LLM is the existing
    `summary_model` field; the resolver normalises that to its
    index_llm_id slot. There is no `final_generator` field by design —
    the answer generator is harness-level (HarnessConfig.generation),
    held constant across systems.
    """
    # --- paper-faithful tree (src/raptor_paper.py) ---
    paper: PaperTreeParams = field(default_factory=PaperTreeParams)
    # Reference TreeBuilderConfig.summarization_length. Ruling 4: this is
    # the ONLY concrete specification anywhere; the paper's reported 131
    # tokens is a MEASUREMENT, not a parameter, and is not to be
    # reverse-engineered into a cap. Expect visible truncation near 100 —
    # that discrepancy is itself a finding about the reference.
    summary_max_tokens: int = 100
    # M4-LOCAL prompt id. Deliberately its own field rather than a shared
    # module constant: a shared constant would have moved every reader's
    # substrate key at once (the frozen-M7 key landmine, now historical).
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


@dataclass(frozen=True)
class HarnessConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    m4: M4Config = field(default_factory=M4Config)


DEFAULT_CONFIG = HarnessConfig()
