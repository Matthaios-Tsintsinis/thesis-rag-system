"""Chunking: parsed documents to Chunk objects.

Two strategies: the M2/M3 word window and M4's RAPTOR leaf (raptor_100tok).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .config import ChunkingConfig
from .parsing import ParsedDocument


# --- Chunk -----------------------------------------------------------------


@dataclass
class Chunk:
    """One retrieval unit of a document with its position and metadata."""
    chunk_id: str
    doc_id: str
    text: str
    n_words: int
    position: int                              # ordinal within doc
    metadata: dict = field(default_factory=dict)

    # Gold-passage atoms this chunk touches, as (parent_id, span_id) pairs,
    # e.g. ("https://example.com/article", "<whole>") for MultiHop-RAG or
    # ("story_id", "<whole>") for NarrativeQA. A chunk spanning two
    # paragraphs carries two pairs. Empty outside evaluation.
    # Eval-time metadata only: the cache key hashes the chunking config,
    # embedder, parser identity, corpus hash and extra, never a Chunk.
    # harness choice: content-addressed substrates (METHODS §D)
    gold_provenance: tuple = field(default_factory=tuple)


# --- Word window (M2/M3) ---------------------------------------------------


# Window 200 words, overlap 50, from ChunkingConfig.
# harness choice: shared default for M2/M3 (METHODS §A.2)
def _chunk_doc_word_window(
    doc: ParsedDocument,
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    """Slide a constant-size word window with overlap across the document."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    # Words are maximal non-whitespace runs, the same split str.split()
    # gives, with char offsets kept for the metadata below.
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", doc.text)]
    words = [doc.text[a:b] for a, b in spans]
    if not words:
        return []

    # One chunk per stride. start_char/end_char span the window's first
    # and last word; index_items uses them to derive gold_provenance when
    # a parent holds several items, and raises without them.
    stride = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for position, start in enumerate(range(0, len(words), stride)):
        window = words[start : start + chunk_words]
        if not window:
            break
        last = min(start + chunk_words, len(words)) - 1
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::{position:04d}",
            doc_id=doc.doc_id,
            text=" ".join(window),
            n_words=len(window),
            position=position,
            metadata={
                "start_char": spans[start][0],
                "end_char": spans[last][1],
            },
        ))
        if start + chunk_words >= len(words):
            break
    return chunks


# --- RAPTOR leaves (M4 only) -----------------------------------------------


def _chunk_doc_raptor_100tok(
    doc: ParsedDocument,
    cfg: ChunkingConfig,
) -> list[Chunk]:
    """Map RAPTOR sentence-preserving ~100-token leaves onto Chunk objects."""
    # cfg.chunk_words is a tiktoken token budget here, not a word count.
    # RAPTOR paper §3: "short, contiguous texts of length 100"
    # Leaves never overlap, so overlap_words must be 0.
    # ref: raptor/utils.py::split_text @ 7da1d48a (overlap never passed, 0)
    if cfg.overlap_words != 0:
        raise ValueError(
            "raptor_100tok is a non-overlapping strategy "
            f"(reference overlap=0); got overlap_words={cfg.overlap_words}"
        )

    # Import here: raptor_paper needs tiktoken, and cache.py imports this
    # module on paths that have no tokenizer.
    from .raptor_paper import split_text_raptor

    # Segmentation lives in split_text_raptor. Carry its offsets and token
    # count in metadata; M4's index_items derives gold_provenance from them.
    spans = split_text_raptor(doc.text, max_tokens=cfg.chunk_words)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i:04d}",
            doc_id=doc.doc_id,
            text=s.text,
            n_words=len(s.text.split()),
            position=i,
            metadata={
                "start_char": s.start_char,
                "end_char": s.end_char,
                "n_tokens": s.n_tokens,
            },
        )
        for i, s in enumerate(spans)
    ]


# --- Public entrypoints ----------------------------------------------------


# The strategy comes from HarnessConfig.chunking or a system's chunker
# override; M4 uses the override so its leaf shape moves only its own key.
def chunk_document(
    doc: ParsedDocument,
    cfg: ChunkingConfig,
) -> list[Chunk]:
    """Chunk one document with the strategy named in cfg."""
    if cfg.strategy == "word_window":
        return _chunk_doc_word_window(doc, cfg.chunk_words, cfg.overlap_words)
    if cfg.strategy == "raptor_100tok":
        return _chunk_doc_raptor_100tok(doc, cfg)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy!r}")


def chunk_corpus(
    docs: Iterable[ParsedDocument],
    cfg: ChunkingConfig,
) -> list[Chunk]:
    """Chunk every document with the strategy named in cfg."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg))
    return out
