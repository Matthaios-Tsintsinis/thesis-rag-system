"""Chunking.

Word-window strategy only at this commit: fixed-size window with overlap.
The semantic chunker (sentence-embedding breakpoints, Greek-aware) ports
from the existing notebook in a later commit. The Chunk dataclass is the
stable interface across both implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .parsing import ParsedDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    n_words: int
    position: int                              # ordinal within doc
    metadata: dict = field(default_factory=dict)


def _split_words(text: str) -> list[str]:
    return text.split()


def chunk_document(
    doc: ParsedDocument,
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    words = _split_words(doc.text)
    if not words:
        return []

    stride = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for position, start in enumerate(range(0, len(words), stride)):
        window = words[start : start + chunk_words]
        if not window:
            break
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{position:04d}",
                doc_id=doc.doc_id,
                text=text,
                n_words=len(window),
                position=position,
            )
        )
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_corpus(
    docs: Iterable[ParsedDocument],
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_words, overlap_words))
    return out
