"""Shared defaults for the harness: models, retrieval depths, chunking,
generation, prompts, and the typed configs that carry them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# raptor_paper imports only the stdlib and numpy, so this cannot cycle.
from .raptor_paper import PaperTreeParams


# --- Filesystem ------------------------------------------------------------
# Input, cache, output and HF-cache roots depend on the host and on env
# overrides, so src/paths.py resolves them at runtime. Only the repo root
# lives here.

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Shared models ---------------------------------------------------------

# Embedder for M2 and M3; M4 overrides it in M4Config.embedder.
# harness choice: per-paper-components rule (METHODS §A.2)
EMBEDDER_MODEL = "BAAI/bge-m3"
# Output width of bge-m3.
EMBEDDING_DIM = 1024
# Reader that answers every query for every system. Any id that is not an
# OpenAI name loads locally through HF transformers in src.models.
# harness choice: one reader across all systems (METHODS §D)
GENERATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Index-time LLM: the RAPTOR tree summariser. --generator moves this and
# the reader together, and it sits in M4's substrate key, so each reader
# column builds its own trees and M4's columns differ in both summariser
# and reader. Kept apart from GENERATOR_MODEL so the two roles can
# diverge without touching call sites.
# deviation from paper (gpt-3.5-turbo is retired; one local summariser per reader column): see METHODS §A.4.2
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# --- Retrieval defaults ---------------------------------------------------

# Chunks handed to the reader by M2 and M3; M4 fills a token budget.
# harness choice: baselines at natural strength, no imposed budget (METHODS §A.2)
FINAL_CONTEXT_CHUNKS = 15
# Candidates per leg (dense, BM25) that M3 fuses.
# harness choice: 50 per leg (METHODS §A.3)
FIRST_STAGE_TOP_K = 50
# RRF (Cormack et al. 2009): score = sum 1/(k + rank), rank 1-based, k = 60
RRF_K = 60

# No shared evidence budget: the packer keeps every chunk it is given.
# M4's 2,000-token budget is M4Config.retrieval_budget_tokens and M4
# applies it inside retrieve(), so the packer never sees a cap.
# harness choice: no shared evidence budget (METHODS §D)
EVIDENCE_TOKEN_BUDGET: int | None = None
# Tiktoken table that counts evidence tokens. Not a generator: the name
# only selects the byte-pair encoding, and every cell counts with it.
EVIDENCE_TOKEN_BUDGET_TOKENIZER = "gpt-4o-mini"

# Candidate pool M4 pulls in budget mode, where the stopping point is not
# known in advance. The head of the ranking is unchanged.
RETRIEVAL_RANKING_DEPTH = 50

# Depth of the ranking every retriever returns for metrics only. The
# reader still sees top-15 (M2/M3) or M4's budget fill.
# harness choice: one scoring depth for every system (METHODS §D)
SCORING_RANKING_DEPTH = 50


# --- Chunking -------------------------------------------------------------
# Two strategies, chosen by HarnessConfig.chunking.strategy:
#   "word_window"   sliding word window plus overlap (200 / 50): the harness
#                   default, so M2 and M3's chunker on every cell.
#   "raptor_100tok" M4 only: contiguous, sentence-preserving leaves of
#                   about 100 cl100k_base tokens, no overlap
#                   (src/raptor_paper.py). Reads chunk_words as a token
#                   count and requires overlap_words == 0.
# A new strategy is a new value of an existing field, so no cache key
# moves. A new field on ChunkingConfig would move every system's key,
# because compute_cache_key folds the whole asdict; that is why
# raptor_100tok reuses chunk_words instead of adding a size field.
# tests/test_raptor_chunking.py pins the schema.

ChunkingStrategy = Literal["word_window", "raptor_100tok"]


# --- Generation -----------------------------------------------------------
# Reader decoding: greedy (temperature 0, top_p 1), 512 new tokens, one
# query per generate() call. M4's tree summaries batch through their own
# knobs on M4Config, both inside M4's substrate key.
# harness choice: one reader across all systems (METHODS §D)

# Cap on generated answer tokens.
GEN_MAX_NEW_TOKENS = 512

# Greedy decoding.
GEN_TEMPERATURE = 0.0
# No nucleus truncation.
GEN_TOP_P = 1.0

# Placement budget handed to accelerate. The reader runs in fp16, never
# quantised (models.load_generator refuses a quantised model or an
# unexpected dtype). "cpu": "0GiB" leaves nowhere to spill, so a model
# that does not fit raises at load instead of running partly offloaded.
# 20GiB of the L4's 22.03GiB admits the ~14.2GB fp16 weights with
# headroom for the KV cache and activations.
GENERATOR_MAX_MEMORY = {0: "20GiB", "cpu": "0GiB"}


# --- Prompts --------------------------------------------------------------

# Refusal string the prompts ask for and the null-rule scorer matches
# exactly. Keep it short, distinctive and stable.
# harness choice: the string the null rule recognises (METHODS §C.9)
ABSTENTION_RESPONSE = "No answer available."

# Reader prompt for every retrieval system: answer from the evidence only.
BASE_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using only the provided evidence. "
    f"If the evidence does not contain the answer, respond with exactly: "
    f"'{ABSTENTION_RESPONSE}'. "
    "Do not fabricate; do not guess. Be concise and factual."
)

# M1's prompt: answer from parametric knowledge, same refusal string.
# harness choice: same refusal string the null rule recognises (METHODS §A.1)
CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the user's question from your own knowledge. "
    f"If you do not know the answer, respond with exactly: "
    f"'{ABSTENTION_RESPONSE}'. "
    "Do not fabricate; do not guess."
)

# --- Typed configs --------------------------------------------------------

# Node kinds a RAPTOR tree can hold.
NodeType = Literal["chunk", "summary_low", "summary_mid", "summary_high"]


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval depths shared by M2 and M3."""
    top_k: int = FINAL_CONTEXT_CHUNKS
    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunker parameters; the whole dataclass is in every substrate key."""
    strategy: ChunkingStrategy = "word_window"

    # No strategy reads these six fields, but compute_cache_key folds the
    # whole asdict, so they stay at their defaults.
    # kept: part of every substrate cache key
    breakpoint_percentile: float = 90.0
    absolute_threshold: float = 0.5
    min_words: int = 80
    max_words: int = 400
    max_if_min_words: int = 500
    buffer_size: int = 1

    # Word window of 200 words with 50 overlap (a token count under
    # raptor_100tok).
    # harness choice: shared default for M2/M3 (METHODS §A.2)
    chunk_words: int = 200
    overlap_words: int = 50

    # Documents shorter than this are dropped by both strategies.
    min_chars_per_doc: int = 200


@dataclass(frozen=True)
class GenerationConfig:
    """Reader model and decoding settings."""
    model: str = GENERATOR_MODEL
    max_new_tokens: int = GEN_MAX_NEW_TOKENS
    temperature: float = GEN_TEMPERATURE
    top_p: float = GEN_TOP_P


@dataclass(frozen=True)
class M4Config:
    """Knobs for M4, the paper-faithful RAPTOR collapsed-tree system."""
    # Tree construction parameters (src/raptor_paper.py).
    paper: PaperTreeParams = field(default_factory=PaperTreeParams)
    # Completion cap for each cluster summary.
    # ref: raptor/tree_builder.py::TreeBuilderConfig @ 7da1d48a (summarization_length=100); the paper's 131 is a measured mean (App. C)
    summary_max_tokens: int = 100
    # M4-local id of the summary prompt pair, kept in M4's own substrate
    # extras rather than a module constant shared with other systems.
    # RAPTOR paper App. D Table 11 (paper over repo): see METHODS §A.4.3
    summary_prompt_version: str = "raptor_paper_v1"

    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K
    # Querying collapses the whole tree into one flat index; bottom-up
    # construction has no synthetic all-corpus root to exclude.
    # RAPTOR paper §3: collapsed tree, the paper's main-results strategy
    include_root_in_flat_index: bool = True
    # Index-time summariser; the runner sets it from --generator together
    # with the reader. It is in the substrate key, so a poor summariser
    # handicaps M4 identically in every column.
    # deviation from paper (gpt-3.5-turbo is retired; one local summariser per reader column): see METHODS §A.4.2
    summary_model: str = JUDGE_MODEL
    # Node count when the budget is None; an explicit k from the caller
    # wins over both.
    top_k_final: int = FINAL_CONTEXT_CHUNKS

    # Escape hatch for the guard in m4_raptor.index(): a tree is the most
    # expensive artifact in the harness and its summariser is in the
    # cache key, so a build with an API index-time LLM is refused unless
    # this is True or M4_ALLOW_API_INDEX_LLM is set in the environment.
    allow_api_index_llm: bool = False

    # Retrieval budget: keep adding nodes by cosine rank until the token
    # total overflows. M4 only; M2/M3 stay at natural top-15 because
    # their papers specify no budget. M4 then answers from ~2,000
    # evidence tokens against M2/M3's ~3,900. The paper chunker already
    # brings M4's natural top-15 to ~1,700 (~110-token units), so the
    # budget fill slightly raises M4's context rather than lowering it.
    # RAPTOR paper §3: "2000 maximum tokens ... top-20 nodes" (paper over repo): see METHODS §A.4.3
    # Query-time only: not in paper_substrate_extra, so changing it never
    # moves the substrate key. None restores plain top_k_final selection.
    retrieval_budget_tokens: int | None = 2000

    # Component overrides: each system uses its own paper's components.
    # mpnet is 768-d against bge-m3's 1024-d, so M4's substrate lives
    # under its own hash.
    # RAPTOR paper §3: SBERT multi-qa-mpnet-base-cos-v1 (paper over repo): see METHODS §A.4.3
    embedder: str | None = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    # M4's own chunker: 100-token, sentence-preserving, non-overlapping
    # leaves. Set here and never on HarnessConfig.chunking, which M2 and
    # M3 inherit; chunk_words is a token count under this strategy.
    # RAPTOR paper §3: "short, contiguous texts of length 100"
    # ref: raptor/utils.py::split_text @ 7da1d48a (overlap never passed, 0)
    chunker: ChunkingConfig | None = field(
        default_factory=lambda: ChunkingConfig(
            strategy="raptor_100tok", chunk_words=100, overlap_words=0
        )
    )
    reranker: str | None = None  # the paper has no reranking step

    # Summariser batching. A batch holds up to summary_batch_size prompts
    # and at most summary_max_padded_tokens of n * longest prompt, because
    # cluster contexts range from ~110 tokens to max_length_in_cluster
    # (3500) and one count sized for short clusters OOMs on long ones.
    # Both are in paper_substrate_extra: batch composition can change the
    # generated text at temperature 0 and summaries are the cached
    # artifact, so retuning either invalidates every tree built at the
    # old value. Node ids and tree shape do not depend on either. Neither
    # is a RAPTOR parameter; the paper specifies no batch shape. At cap
    # 16000 the effective width reaches 25, so the cap binds and the
    # nominal 32 does not.
    # harness choice: batch shape is in the cache key because it can move text at temperature 0
    summary_batch_size: int = 32
    summary_max_padded_tokens: int = 16000

    # Unread by the paper-faithful path, kept so callers construct.
    # Collapsed retrieval is dense cosine only, so M4 has no sparse index.
    hybrid_first_stage: bool = False

    # Opt-in retrieval trace; benchmark runs leave it off.
    trace: bool = False


@dataclass(frozen=True)
class HarnessConfig:
    """Top-level config: retrieval, chunking, generation and M4."""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    m4: M4Config = field(default_factory=M4Config)


DEFAULT_CONFIG = HarnessConfig()
