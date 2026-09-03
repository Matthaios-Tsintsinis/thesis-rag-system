"""Document parsing for the corpus layouts the systems write.

Every benchmark corpus reaches a system as `.txt` files written by
`BaseSystem._write_corpus_layout`, so the only live parser is the text
one; the PDF/DOCX/HTML/CSV/JSON/XLSX backends left in the repo
reduction. Each document carries a minimal "sections" payload (single
section, depth 0, title = filename) so downstream code treats every doc
uniformly.

Cache identity: `parsing_identity()` is folded into every retriever
cache key via cache.compute_cache_key. It is a LITERAL — the
`pdf_parser: docling` value names the identity under which every banked
substrate was keyed — and it stays byte-identical: changing it would
move every key in the matrix.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# --- Identity for cache invalidation --------------------------------------

# Bump this whenever the parser identity or output schema changes so
# cached embeddings/indexes from older runs are invalidated cleanly.
PARSING_VERSION = "docling-v1"


def parsing_identity() -> dict:
    return {"pdf_parser": "docling", "parsing_version": PARSING_VERSION}


# --- Supported formats ----------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class ParsedDocument:
    doc_id: str
    path: Path
    text: str
    metadata: dict = field(default_factory=dict)


# --- Helpers --------------------------------------------------------------


def safe_read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _fallback_sections(filename: str, text: str) -> list[dict]:
    """Minimal one-section payload: section_title = file_name,
    section_depth = 0, one section spanning the whole document."""
    return [{
        "section_title": filename,
        "section_depth": 0,
        "section_path": [filename],
        "page_start": None,
        "page_end": None,
        "order_in_document": 0,
        "text": text,
    }]


# --- Per-format parsers ---------------------------------------------------


def parse_txt(path: Path) -> tuple[str, dict]:
    text = safe_read_text(path)
    return text, {"sections": _fallback_sections(Path(path).name, text)}


# --- Cleaning + dispatch --------------------------------------------------


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ", " ", text)
    return text.strip()


def extract_text(path: Path) -> tuple[str, dict]:
    """Dispatch by extension. Returns (cleaned_text, metadata).

    Metadata always contains a one-entry `sections` list.
    """
    ext = Path(path).suffix.lower()
    metadata: dict = {"path": str(path), "ext": ext}
    if ext in {".txt", ".md"}:
        text, extra = parse_txt(path)
    else:
        raise ValueError(f"Unsupported extension {ext!r} for {path}")
    metadata.update(extra)
    return clean_text(text), metadata


def parse_file(path: Path) -> str:
    """Back-compat: cleaned text only. Use extract_text for metadata."""
    text, _ = extract_text(path)
    return text


def detect_page_refs(chunk_text: str) -> list[int]:
    return [int(m) for m in re.findall(r"\[PAGE (\d+)\]", chunk_text)]


def list_files_recursive(root: Path) -> list[Path]:
    files: list[Path] = []
    for path, _, fnames in os.walk(str(root)):
        for f in fnames:
            full = Path(path) / f
            if full.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(full)
    return sorted(files)


def walk_corpus(folder: Path, min_chars: int = 0) -> Iterable[ParsedDocument]:
    """Yield ParsedDocument for every supported file under `folder`.

    Docs whose cleaned text is below `min_chars` are skipped.
    """
    folder = Path(folder)
    for path in list_files_recursive(folder):
        try:
            text, meta = extract_text(path)
        except Exception as e:
            print(f"[parsing] skip {path}: {type(e).__name__}: {e}")
            continue
        if len(text) < min_chars:
            continue
        doc_id = str(path.relative_to(folder)).replace("\\", "/")
        yield ParsedDocument(doc_id=doc_id, path=path, text=text, metadata=meta)
