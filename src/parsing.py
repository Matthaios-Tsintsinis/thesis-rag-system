"""Read a corpus of `.txt` / `.md` files into cleaned documents.
Every document gets a one-section `sections` payload so downstream code
treats all documents the same way."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# --- Identity for cache invalidation --------------------------------------

# Names the parser version; a new value gives every substrate a new key.
PARSING_VERSION = "docling-v1"


def parsing_identity() -> dict:
    """Return the parser identity that cache.compute_cache_key folds in."""
    # kept: part of every substrate cache key
    return {"pdf_parser": "docling", "parsing_version": PARSING_VERSION}


# --- Supported formats ----------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class ParsedDocument:
    """One parsed corpus file: id, path, cleaned text, metadata."""
    doc_id: str
    path: Path
    text: str
    metadata: dict = field(default_factory=dict)


# --- Helpers --------------------------------------------------------------


def safe_read_text(path: Path) -> str:
    """Read a file as UTF-8, dropping undecodable bytes."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _fallback_sections(filename: str, text: str) -> list[dict]:
    """Build the one-section payload: title = filename, depth 0, whole text."""
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
    """Read a text file and return (raw_text, {"sections": [...]})."""
    text = safe_read_text(path)
    return text, {"sections": _fallback_sections(Path(path).name, text)}


# --- Cleaning + dispatch --------------------------------------------------


def clean_text(text: str) -> str:
    """Drop NULs, collapse runs of spaces and blank lines, strip the ends."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ", " ", text)
    return text.strip()


def extract_text(path: Path) -> tuple[str, dict]:
    """Parse one file by extension and return (cleaned_text, metadata)."""
    # Only .txt and .md are parsed; anything else is an error.
    ext = Path(path).suffix.lower()
    metadata: dict = {"path": str(path), "ext": ext}
    if ext in {".txt", ".md"}:
        text, extra = parse_txt(path)
    else:
        raise ValueError(f"Unsupported extension {ext!r} for {path}")
    metadata.update(extra)
    return clean_text(text), metadata


def parse_file(path: Path) -> str:
    """Return the cleaned text of one file, without metadata."""
    text, _ = extract_text(path)
    return text


def detect_page_refs(chunk_text: str) -> list[int]:
    """Return the page numbers of every `[PAGE n]` marker in the text."""
    return [int(m) for m in re.findall(r"\[PAGE (\d+)\]", chunk_text)]


def list_files_recursive(root: Path) -> list[Path]:
    """Return every supported file under `root`, sorted by path."""
    files: list[Path] = []
    for path, _, fnames in os.walk(str(root)):
        for f in fnames:
            full = Path(path) / f
            if full.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(full)
    return sorted(files)


def walk_corpus(folder: Path, min_chars: int = 0) -> Iterable[ParsedDocument]:
    """Yield a ParsedDocument for every supported file under `folder`."""
    folder = Path(folder)
    # Skip files that fail to parse or whose cleaned text is too short.
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
