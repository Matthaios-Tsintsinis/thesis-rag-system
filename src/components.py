"""Resolve each system's retrieval-side components over shared defaults.

The reader is never a per-system field; it lives on HarnessConfig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import (
    ChunkingConfig,
    EMBEDDER_MODEL,
    HarnessConfig,
)


@dataclass(frozen=True)
class ResolvedComponents:
    """Component identities a system uses at index and retrieve time."""

    # embedder_id and chunker_config feed the substrate cache key.
    # reranker_id is None for every system in the matrix.
    # index_llm_id is None where a system runs no index-time LLM; M4
    # resolves it from summary_model. There is no reader slot here.
    # harness choice: one reader across all systems (METHODS §D)
    embedder_id: str
    chunker_config: ChunkingConfig
    reranker_id: str | None
    index_llm_id: str | None


def _resolve_field(cfg: Any | None, name: str, fallback: Any) -> Any:
    """Read cfg.name; use the fallback when it is missing or None."""
    val = getattr(cfg, name, None) if cfg is not None else None
    return val if val is not None else fallback


def resolve_components(
    system_cfg: Any | None,
    harness_cfg: HarnessConfig,
    *,
    default_reranker: str | None = None,
    default_index_llm: str | None = None,
) -> ResolvedComponents:
    """Fold a system config's optional overrides over the shared defaults."""
    # system_cfg=None gives the shared defaults (the M2/M3 path).
    # default_reranker and default_index_llm are per-system fallbacks; M4
    # passes default_reranker=None so "M4 does not rerank" stays visible.
    # embedder fallback: BAAI/bge-m3
    # harness choice: per-paper-components rule (METHODS §A.2)
    # chunker fallback: harness_cfg.chunking
    # harness choice: shared default for M2/M3 (METHODS §A.2)
    embedder = _resolve_field(system_cfg, "embedder", EMBEDDER_MODEL)
    chunker = _resolve_field(system_cfg, "chunker", harness_cfg.chunking)
    reranker = _resolve_field(system_cfg, "reranker", default_reranker)

    # Pick one index-time LLM id from the paper-side field names: an
    # explicit index_llm wins, then summary_model (M4), then openie_llm,
    # then the caller's default. A missing field counts as None.
    explicit = (
        getattr(system_cfg, "index_llm", None) if system_cfg is not None else None
    )
    summary_model = (
        getattr(system_cfg, "summary_model", None)
        if system_cfg is not None
        else None
    )
    openie_llm = (
        getattr(system_cfg, "openie_llm", None) if system_cfg is not None else None
    )
    index_llm = explicit or summary_model or openie_llm or default_index_llm

    return ResolvedComponents(
        embedder_id=embedder,
        chunker_config=chunker,
        reranker_id=reranker,
        index_llm_id=index_llm,
    )


def format_components_log(system_id: str, r: ResolvedComponents) -> str:
    """Return the one-line JSON `[components]` record for a system."""
    # The chunker is named by its strategy string; the full chunker
    # parameters are in the substrate manifest.
    return json.dumps(
        {
            "system": system_id,
            "embedder_id": r.embedder_id,
            "chunker": r.chunker_config.strategy,
            "reranker_id": r.reranker_id,
            "index_llm_id": r.index_llm_id,
        },
        ensure_ascii=False,
    )


__all__ = [
    "ResolvedComponents",
    "resolve_components",
    "format_components_log",
]
