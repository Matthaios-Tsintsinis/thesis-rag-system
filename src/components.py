"""Per-system component resolver.

Each active system can override its retrieval-side components (embedder,
chunker, reranker, index-time LLM) on its per-system Config dataclass.
Any field left None falls back to a shared default. Today no system
overrides anything — this module is pure capability, defaults are
preserved everywhere — but the plumbing is in place so the per-paper
component audit can flip values without touching call sites.

The FINAL GENERATOR is intentionally NOT a per-system field. It lives at
the harness level (HarnessConfig.generation) and is held constant across
all systems, so per-paper generator choices cannot leak in and confound
the retrieval comparison. ResolvedComponents therefore carries no
generator slot — the invariant is enforced structurally.

Index-time LLM names differ per paper (RAPTOR calls it the summariser
and stores it as M4Config.summary_model). The resolver normalises the
paper-side name to a single index_llm_id for the per-system audit log.

The resolver also produces the structured log line that smoke + harness
emit per system at index time, which feeds the per-paper audit table
mechanically.
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
    """Concrete component identities a system uses at index/retrieve time.

    embedder_id and chunker_config feed compute_cache_key directly (the
    cache key has always carried embedder_model and chunking_config as
    top-level fields; the resolver just centralises which value lands
    there).

    reranker_id is None for systems that do not rerank (M2, M3, M4 today).
    For systems that do rerank (M7), the value is folded into the
    system-specific cache key extras, NOT into the shared RAPTOR
    substrate key — reranking is query-time and the substrate is rerank-
    independent.

    index_llm_id is None for systems with no index-time LLM (M2, M3).
    For M4/M7 it normalises to the value of `summary_model`; a future M6
    will normalise from `openie_llm`. Used today only for the audit log.
    """

    embedder_id: str
    chunker_config: ChunkingConfig
    reranker_id: str | None
    index_llm_id: str | None


def _resolve_field(cfg: Any | None, name: str, fallback: Any) -> Any:
    """Read `cfg.name`, falling back when the field is absent OR present-but-None.

    Both branches collapse to the same outcome by design: a system that
    declares `embedder: str | None = None` to opt in to the override
    capability while not yet overriding must behave identically to a
    system that does not declare the field at all.
    """
    val = getattr(cfg, name, None) if cfg is not None else None
    return val if val is not None else fallback


def resolve_components(
    system_cfg: Any | None,
    harness_cfg: HarnessConfig,
    *,
    default_reranker: str | None = None,
    default_index_llm: str | None = None,
) -> ResolvedComponents:
    """Fold a per-system config's optional component overrides over shared defaults.

    `system_cfg=None` returns pure shared defaults (the M2/M3 path
    today — neither has a per-system config namespace yet, and neither
    overrides anything).

    `default_reranker` and `default_index_llm` are caller-supplied
    per-system fallbacks. They differ from harness-level defaults
    because not every system reranks or runs an index-time LLM — M4
    passes default_reranker=None to keep "M4 does not rerank" visible
    in the resolved bundle.
    """
    embedder = _resolve_field(system_cfg, "embedder", EMBEDDER_MODEL)
    chunker = _resolve_field(system_cfg, "chunker", harness_cfg.chunking)
    reranker = _resolve_field(system_cfg, "reranker", default_reranker)

    # Index-time LLM normalisation: paper-faithful field names on each
    # system config, single id at the resolver. RAPTOR-family configs
    # carry `summary_model` (M4); `openie_llm` was HippoRAG's name and is
    # read through getattr, so a config without it resolves to None. An
    # explicit `index_llm` field overrides both if set.
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
    """One-line JSON for the per-system audit log.

    Emitted by every active system at index time so the per-paper
    component audit table can be assembled mechanically from smoke or
    eval-grid logs (just grep for `[components]` and json.loads each
    line). The chunker is summarised by its strategy string; the full
    chunker params are already in the manifest if needed.
    """
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
