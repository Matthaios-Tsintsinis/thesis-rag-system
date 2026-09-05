"""Content-addressed cache keys and the on-disk substrate layout:
<cache_dir>/<system_id>/<cache_key>/ holds the manifest, chunks,
embeddings and any index files the retriever writes."""

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
    """SHA-256 over every file under the corpus dir, in sorted path order."""
    h = hashlib.sha256()
    root = Path(corpus_path).resolve()
    # Feed each file as its relative path, then its bytes, with separators.
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
    """Sorted-key JSON for a dataclass or any JSON-able object."""
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
    """32-hex key over chunking, embedder, parser, corpus hash and extras."""
    # harness choice: content-addressed substrates (METHODS §D)
    # Default to the live parser identity; pass {} to leave it out.
    if parsing_identity is None:
        # Import here to avoid a chunking -> parsing -> cache import cycle.
        from .parsing import parsing_identity as _parsing_identity
        parsing_identity = _parsing_identity()

    # Hash the five inputs as one sorted-key payload; keep 32 hex chars.
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
    """Paths of one substrate under <root>/<system_id>/<cache_key>/."""
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
        """True when the manifest and every required file are on disk."""
        # The manifest is written last, so its presence means a finished
        # build; then check each file the retriever needs.
        if not self.manifest_path.exists():
            return False
        for name in required:
            if not (self.path / name).exists():
                return False
        return True


# --- Chunk serialisation --------------------------------------------------


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write chunks as JSONL, one Chunk per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Read a JSONL file back into Chunk objects."""
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks


# --- Manifest -------------------------------------------------------------


@dataclass
class Manifest:
    """Bookkeeping for one substrate: what built it and which files exist."""
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
        """Write the manifest as sorted, indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Read a manifest back from its JSON file."""
        return cls(**json.loads(path.read_text()))


# --- Embedding / BM25 helpers ---------------------------------------------


def save_embeddings(emb: np.ndarray, path: Path) -> None:
    """Save an embedding matrix as .npy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)


def load_embeddings(path: Path) -> np.ndarray:
    """Load an embedding matrix from .npy."""
    return np.load(path)


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle an object (the BM25 index) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load a pickled object from disk."""
    with path.open("rb") as f:
        return pickle.load(f)
