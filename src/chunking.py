"""Chunking.

Two strategies share the Chunk dataclass and `chunk_corpus` entrypoint:

  * "word_window" — fixed-size word window with overlap (200 / 50):
                    M2 and M3's chunker, the harness default.
  * "raptor_100tok" — M4 ONLY. Paper-faithful RAPTOR leaves: contiguous,
                    sentence-preserving, ~100 tiktoken tokens, no overlap.
                    Segmentation lives in `src/raptor_paper.py`; this
                    module only adapts it to the Chunk dataclass.

Strategy is selected by HarnessConfig.chunking.strategy, or per-system
via the `chunker` override on a system config (M4 uses that route so
its chunker change cannot move any other system's cache key). The
embedding-similarity "semantic" strategy left in the repo reduction; its
six ChunkingConfig fields stay because they sit inside every substrate
key.
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
    chunk_id: str
    doc_id: str
    text: str
    n_words: int
    position: int                              # ordinal within doc
    metadata: dict = field(default_factory=dict)

    # CK-2 retrieval-recall provenance for benchmark eval. Each pair is
    # (parent_id, span_id) identifying a gold-passage atom this chunk
    # touches — e.g. ("https://example.com/article", "<whole>") for
    # MultiHop-RAG or ("story_id", "<whole>") for NarrativeQA. A chunk
    # produced from one paragraph carries one pair; a chunk spanning two
    # paragraphs carries two. Empty tuple for non-eval indexing (default;
    # preserves existing chunk behaviour byte-for-byte).
    #
    # CACHE DISCIPLINE: this field is eval-time metadata, NOT retrieval
    # content. It is intentionally excluded from every cache key — the
    # cache key (compute_cache_key) hashes chunking_config + embedder +
    # parsing_identity + corpus_hash + extra, none of which read the
    # Chunk dataclass itself. Adding this field therefore does NOT
    # invalidate any existing substrate (RAPTOR 78fb239..., M4 mpnet
    # bfc50c2..., M6 graph, etc.). save_chunks/load_chunks survives
    # back-compat because the field has a default factory: old cached
    # chunks lack the key in their on-disk JSON and Chunk(**d) falls
    # back to the empty tuple.
    gold_provenance: tuple = field(default_factory=tuple)


# --- Word-window (cheap, embedder-free) -----------------------------------


def _chunk_doc_word_window(
    doc: ParsedDocument,
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    """Fixed-size word window with overlap.

    `start_char` / `end_char` land in `Chunk.metadata` — offsets into
    `doc.text`, spanning from the first character of the window's first
    word to the last character of its last word. `index_items` consumes
    them to derive gold_provenance by span intersection when a parent
    holds several CorpusItems; without them that path raises rather than
    stamping empty provenance.

    The chunk TEXT is unchanged by their addition, and deliberately so:
    `re.finditer(r"\\S+")` yields exactly the sequence `str.split()`
    yields (both split on whitespace runs and drop them), so the joined
    window is byte-identical to what this function produced before, and
    cached chunk sets stay valid. `Chunk.metadata` is not a cache-key
    input, so nothing here moves a substrate hash.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", doc.text)]
    words = [doc.text[a:b] for a, b in spans]
    if not words:
        return []

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


# --- RAPTOR paper-faithful (M4 only) ---------------------------------------


def _chunk_doc_raptor_100tok(
    doc: ParsedDocument,
    cfg: ChunkingConfig,
) -> list[Chunk]:
    """Paper-faithful RAPTOR leaves: ~100 tokens, sentence-preserving, no overlap.

    Delegates the actual segmentation to `src.raptor_paper.
    split_text_raptor` (which carries the fidelity notes and the one
    documented divergence from the reference implementation). This
    wrapper only maps TextSpans onto the shared Chunk dataclass.

    `cfg.chunk_words` is read as a TOKEN budget here, not a word count —
    see the strategy note in config.py. `cfg.overlap_words` must be 0;
    the reference never overlaps and a non-zero value would silently
    misrepresent the strategy.

    The span offsets land in `Chunk.metadata` (start_char / end_char /
    n_tokens). Chunk.metadata is NOT part of any cache key, so carrying
    them is free. M4's per-parent `index_items` override consumes them
    to derive gold_provenance by offset intersection.
    """
    if cfg.overlap_words != 0:
        raise ValueError(
            "raptor_100tok is a non-overlapping strategy "
            f"(reference overlap=0); got overlap_words={cfg.overlap_words}"
        )

    # Local import: raptor_paper pulls tiktoken lazily, and chunking.py
    # is imported by cache.py on every path including ones with no
    # tokenizer available.
    from .raptor_paper import split_text_raptor

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


def chunk_document(
    doc: ParsedDocument,
    cfg: ChunkingConfig,
) -> list[Chunk]:
    if cfg.strategy == "word_window":
        return _chunk_doc_word_window(doc, cfg.chunk_words, cfg.overlap_words)
    if cfg.strategy == "raptor_100tok":
        return _chunk_doc_raptor_100tok(doc, cfg)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy!r}")


def chunk_corpus(
    docs: Iterable[ParsedDocument],
    cfg: ChunkingConfig,
) -> list[Chunk]:
    """Chunk every doc using the configured strategy."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg))
    return out
