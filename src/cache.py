"""Cache-key computation and a small save/load helper for retriever indexes.

Key design:

    cache_key = sha256(
        chunking_config_json
      + embedder_model
      + corpus_content_hash
      + extra_json
    )

Switching chunking strategy, embedding model, or corpus content
invalidates the cache automatically — same pattern as the existing
notebook's `embeddings_{strategy}.npy` keying, but content-addressed.

Cache layout on disk:

    <cache_dir>/<system_id>/<cache_key>/
        manifest.json         # bookkeeping (chunking cfg, model, n_chunks, ...)
        chunks.jsonl          # one Chunk per line, json-serialised
        embeddings.npy        # float32 matrix, L2-normalised
        faiss.index           # FAISS native serialisation (optional)
        bm25.pkl              # pickled BM25Okapi (optional)

Each retriever picks which optional files it writes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .chunking import Chunk


# --- Hashing --------------------------------------------------------------


def corpus_content_hash(corpus_path: Path) -> str:
    """SHA-256 over sorted (rel_path, bytes) tuples for every file in dir."""
    h = hashlib.sha256()
    root = Path(corpus_path).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(path.read_bytes())
        h.update(b"\xff")
    return h.hexdigest()


def _json_repr(obj: Any) -> str:
    if is_dataclass(obj):
        return json.dumps(asdict(obj), sort_keys=True)
    return json.dumps(obj, sort_keys=True, default=str)


def compute_cache_key(
    *,
    chunking_config: Any,
    embedder_model: str,
    corpus_hash: str,
    extra: dict[str, Any] | None = None,
    parsing_identity: dict[str, Any] | None = None,
) -> str:
    """Stable hex key from chunking + embedder + parser + corpus + extras.

    Parser identity is folded in automatically (default-loaded from
    parsing.parsing_identity()) so that swapping PDF backends — e.g.
    PyMuPDF → Docling — invalidates every cached index across systems.
    Pass `parsing_identity={}` explicitly to opt out (tests only).
    """
    if parsing_identity is None:
        # Late import to avoid a chunking → parsing → cache cycle at module load.
        from .parsing import parsing_identity as _parsing_identity
        parsing_identity = _parsing_identity()

    payload = "\n".join([
        f"chunking={_json_repr(chunking_config)}",
        f"embedder={embedder_model}",
        f"parsing={_json_repr(parsing_identity)}",
        f"corpus={corpus_hash}",
        f"extra={_json_repr(extra or {})}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# --- Cache directory layout -----------------------------------------------


@dataclass
class CacheDir:
    root: Path
    system_id: str
    cache_key: str

    @property
    def path(self) -> Path:
        return self.root / self.system_id / self.cache_key

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.path / "chunks.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.path / "embeddings.npy"

    @property
    def faiss_path(self) -> Path:
        return self.path / "faiss.index"

    @property
    def bm25_path(self) -> Path:
        return self.path / "bm25.pkl"

    def is_complete(self, required: Iterable[str]) -> bool:
        if not self.manifest_path.exists():
            return False
        for name in required:
            if not (self.path / name).exists():
                return False
        return True


# --- Chunk serialisation --------------------------------------------------


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks


# --- Manifest -------------------------------------------------------------


@dataclass
class Manifest:
    system_id: str
    cache_key: str
    chunking_config: dict
    embedder_model: str
    corpus_hash: str
    n_chunks: int
    files: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls(**json.loads(path.read_text()))


# --- Embedding / BM25 helpers ---------------------------------------------


def save_embeddings(emb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)
