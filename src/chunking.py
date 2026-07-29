"""Chunking.

Two strategies share the Chunk dataclass and `chunk_corpus` entrypoint:

  * "semantic"    — sentence-buffered embedding similarity breaks, with
                    Greek-aware sentence splitting (`.`, `!`, `?`, `;`
                    as terminators; `·` excluded; ports
                    `chunk_text_semantic` from the existing notebook).
                    Used for production benchmarks.
  * "word_window" — fixed-size word window with overlap. Cheap;
                    used in the smoke test and when no embedder is loaded.
  * "raptor_100tok" — M4 ONLY. Paper-faithful RAPTOR leaves: contiguous,
                    sentence-preserving, ~100 tiktoken tokens, no overlap.
                    Segmentation lives in `src/raptor_paper.py`; this
                    module only adapts it to the Chunk dataclass.

Strategy is selected by HarnessConfig.chunking.strategy, or per-system
via the `chunker` override on a system config (M4 uses that route so
its chunker change cannot move any other system's cache key).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

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
    # touches — e.g. ("paper_1909.00694", "sec3.para0") for QASPER or
    # ("https://example.com/article", "<whole>") for MultiHop-RAG. A
    # chunk produced from one paragraph carries one pair; a semantic-
    # chunker chunk spanning two paragraphs carries two. Empty tuple
    # for smoke / non-eval indexing (default; preserves existing chunk
    # behaviour byte-for-byte).
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
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    words = doc.text.split()
    if not words:
        return []

    stride = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for position, start in enumerate(range(0, len(words), stride)):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::{position:04d}",
            doc_id=doc.doc_id,
            text=" ".join(window),
            n_words=len(window),
            position=position,
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


# --- Semantic --------------------------------------------------------------

# Greek-aware sentence terminators. `.`, `!`, `?` (English) plus `;`
# (Greek question mark). `·` (Greek áno teleía) is *excluded* — it's a
# clause separator, not a full stop, and including it produces sliver
# chunks on Greek text. Verified empirically by the existing notebook.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _embed_for_chunking(texts: list[str], embedder: Any) -> np.ndarray:
    """Run the bge-m3 embedder over buffered sentences for chunking.

    Accepts either a SentenceTransformer-like object (has .encode) or
    None. Returns L2-normalised float32; raises if embedder is None.
    """
    if embedder is None:
        raise RuntimeError(
            "semantic chunking requires an embedder; pass one or set "
            "ChunkingConfig.strategy='word_window'"
        )
    embs = embedder.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embs.astype("float32", copy=False)


def _chunk_text_semantic(
    text: str,
    embedder: Any,
    cfg: ChunkingConfig,
) -> list[dict]:
    """Port of `chunk_text_semantic` from the notebook. Returns dicts
    with keys text/n_words/start_word/end_word; callers wrap to Chunk.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if len(sentences) < 2:
        words = text.split()
        return [{
            "text": text,
            "n_words": len(words),
            "start_word": 0,
            "end_word": len(words),
        }]

    # Buffer ±buffer_size sentences for stable embeddings
    buffered: list[str] = []
    for i in range(len(sentences)):
        lo = max(0, i - cfg.buffer_size)
        hi = min(len(sentences), i + cfg.buffer_size + 1)
        buffered.append(" ".join(sentences[lo:hi]))

    embs = _embed_for_chunking(buffered, embedder)

    # cosine distance between consecutive buffered sentences
    dots = (embs[:-1] * embs[1:]).sum(axis=1)
    distances = 1.0 - dots

    perc_threshold = float(np.percentile(distances, cfg.breakpoint_percentile))
    final_threshold = min(perc_threshold, cfg.absolute_threshold)

    breakpoints = [i + 1 for i, d in enumerate(distances) if d > final_threshold]

    raw_chunks: list[str] = []
    start = 0
    for bp in breakpoints + [len(sentences)]:
        seg = sentences[start:bp]
        if seg:
            raw_chunks.append(" ".join(seg))
        start = bp

    # Force-split overlong chunks
    sized: list[str] = []
    for ch in raw_chunks:
        words = ch.split()
        if len(words) > cfg.max_words:
            for j in range(0, len(words), cfg.max_words):
                sized.append(" ".join(words[j : j + cfg.max_words]))
        else:
            sized.append(ch)

    # Merge undersized chunks into previous, but only if it won't blow
    # past max_if_min_words. This prevents accumulator snowballing on
    # corpora of mostly-short sentences.
    merged: list[str] = []
    for ch in sized:
        ch_len = len(ch.split())
        if merged and ch_len < cfg.min_words:
            prev_len = len(merged[-1].split())
            if prev_len + ch_len <= cfg.max_if_min_words:
                merged[-1] = merged[-1] + " " + ch
            else:
                merged.append(ch)
        else:
            merged.append(ch)

    out: list[dict] = []
    word_cursor = 0
    for ch_text in merged:
        n_words = len(ch_text.split())
        out.append({
            "text": ch_text,
            "n_words": n_words,
            "start_word": word_cursor,
            "end_word": word_cursor + n_words,
        })
        word_cursor += n_words
    return out


def _chunk_doc_semantic(
    doc: ParsedDocument,
    cfg: ChunkingConfig,
    embedder: Any,
) -> list[Chunk]:
    raw = _chunk_text_semantic(doc.text, embedder, cfg)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i:04d}",
            doc_id=doc.doc_id,
            text=r["text"],
            n_words=r["n_words"],
            position=i,
            metadata={
                "start_word": r["start_word"],
                "end_word": r["end_word"],
            },
        )
        for i, r in enumerate(raw)
    ]


# --- Public entrypoints ----------------------------------------------------


def chunk_document(
    doc: ParsedDocument,
    cfg: ChunkingConfig,
    embedder: Any | None = None,
) -> list[Chunk]:
    if cfg.strategy == "semantic":
        return _chunk_doc_semantic(doc, cfg, embedder)
    if cfg.strategy == "word_window":
        return _chunk_doc_word_window(doc, cfg.chunk_words, cfg.overlap_words)
    if cfg.strategy == "raptor_100tok":
        return _chunk_doc_raptor_100tok(doc, cfg)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy!r}")


def chunk_corpus(
    docs: Iterable[ParsedDocument],
    cfg: ChunkingConfig,
    embedder: Any | None = None,
) -> list[Chunk]:
    """Chunk every doc using the configured strategy.

    Semantic strategy requires an embedder. Callers in retriever code
    pass `models.load_embedder()`; smoke tests configured with
    strategy='word_window' do not.
    """
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, cfg, embedder))
    return out
