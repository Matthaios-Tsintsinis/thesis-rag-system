"""Document parsing.

This commit supports txt / md / pdf only. Additional formats and a
proper structural PDF parser arrive in later commits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class ParsedDocument:
    doc_id: str
    path: Path
    text: str


def parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pypdf is required to parse PDFs (pip install pypdf)") from e

    reader = pypdf.PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            parts.append(text)
    return "\n".join(parts)


def parse_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return parse_txt(path)
    if ext == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"Unsupported extension {ext!r} for {path}")


def walk_corpus(folder: Path, min_chars: int = 0) -> Iterable[ParsedDocument]:
    """Yield ParsedDocument for every supported file under `folder`.

    Docs whose cleaned text is below `min_chars` are skipped.
    """
    folder = Path(folder)
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = parse_file(path).strip()
        if len(text) < min_chars:
            continue
        doc_id = str(path.relative_to(folder)).replace("\\", "/")
        yield ParsedDocument(doc_id=doc_id, path=path, text=text)
