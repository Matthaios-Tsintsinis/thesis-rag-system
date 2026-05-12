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

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"
RESULTS_DIR = REPO_ROOT / "outputs"


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


# --- Chunking (word-window placeholder; semantic chunker added later) ----

CHUNK_WORDS = 200
CHUNK_OVERLAP_WORDS = 50
MIN_CHARS_PER_DOC = 200


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


# --- Typed configs --------------------------------------------------------

NodeType = Literal["chunk", "summary_low", "summary_mid", "summary_high"]


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = FINAL_CONTEXT_CHUNKS
    first_stage_top_k: int = FIRST_STAGE_TOP_K
    rrf_k: int = RRF_K


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_words: int = CHUNK_WORDS
    overlap_words: int = CHUNK_OVERLAP_WORDS
    min_chars_per_doc: int = MIN_CHARS_PER_DOC


@dataclass(frozen=True)
class GenerationConfig:
    model: str = GENERATOR_MODEL
    max_new_tokens: int = GEN_MAX_NEW_TOKENS
    temperature: float = GEN_TEMPERATURE
    top_p: float = GEN_TOP_P
    load_in_4bit: bool = LOAD_GENERATOR_IN_4BIT


@dataclass(frozen=True)
class HarnessConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    cache_dir: Path = CACHE_DIR
    results_dir: Path = RESULTS_DIR


DEFAULT_CONFIG = HarnessConfig()
