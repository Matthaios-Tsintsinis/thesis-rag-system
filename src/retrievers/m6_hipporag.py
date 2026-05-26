"""M6 — HippoRAG 1 (legacy NeurIPS'24), single-step retrieval.

Faithful port of the OSU-NLP-Group/HippoRAG legacy branch, NOT a wrapper
of the original repo (the legacy code pins numpy==1.26.4 / torch==1.13.1,
which conflict ABI-hard with our harness). Same approach as M4, where a
RAPTOR-paper port lives in `src/raptor.py` rather than a wrapped repo.

Components per the paper / legacy main-experiment scripts:

  * Contriever embedder (facebook/contriever, 768-dim)
  * gpt-4o-mini for OpenIE + query NER (modernised from gpt-3.5-turbo-1106)
  * Synonymy edges with sim_threshold = 0.8
  * Personalised PageRank with damping = 0.5 (continue-walk probability)
  * Node specificity ON
  * doc_ensemble OFF, dpr_only OFF, single-step (no IRCoT)

Cache: M6 keeps a system-specific cache namespace `M6/<m6_hash>/`. The
RAPTOR substrate is unrelated and not shared. See `src/hipporag_graph.py`
for the on-disk layout.

THIS FILE IS A C4a SKELETON — `index` and `retrieve` raise
NotImplementedError. The system is NOT registered in
`smoke_test/run_smoke.py` until C4b lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..components import (
    ResolvedComponents,
    format_components_log,
    resolve_components,
)
from ..config import DEFAULT_CONFIG, HarnessConfig
from .base import AnswerResult, BaseSystem, RetrievedChunk


class HippoRAGSystem(BaseSystem):
    """HippoRAG 1 — entity-graph PPR retrieval over OpenIE-extracted facts."""

    system_id = "M6"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        # Populated at the top of index(); used at query time by retrieve().
        self._resolved: ResolvedComponents | None = None
        # Graph artefacts. Set by index() / load_graph().
        self._graph: Any | None = None  # hipporag_graph.HippoGraph
        # Indexed chunk list (1:1 with HippoRAG's "passages" — each chunk = one passage).
        self.chunks: list = []
        # Index-time stats (n_phrases, n_facts, n_edges, n_synonymy_edges,
        # openie_failed_count, etc.). Populated after build / load.
        self._index_stats: dict = {}
        # Per-query trace (empty-NER flag, n_linked_phrases, etc.). Surfaced
        # in smoke for sanity checks and in analysis to quantify empty-NER
        # impact.
        self._last_trace: dict = {}

    @property
    def resolved_components(self) -> ResolvedComponents | None:
        return self._resolved

    @property
    def index_stats(self) -> dict:
        return dict(self._index_stats)

    @property
    def last_trace(self) -> dict:
        return dict(self._last_trace)

    def index(self, corpus_path: Path) -> None:
        """Parse corpus -> chunk -> OpenIE -> graph build -> Contriever phrase
        embed -> synonymy edges -> persist.

        Resolves components first (Contriever + gpt-4o-mini per M6Config
        defaults under the per-paper rule), logs the components line,
        cache-keys on (chunking, embedder, openie_llm, sim_threshold,
        prompt_version, ...), short-circuits to load_graph on cache hit.

        NOT IMPLEMENTED in C4a — function signature only. C4b lands the
        full pipeline + the Contriever pooling HARD GATE (cosine >= 0.99
        vs masked-mean reference, see hipporag_graph.embed_phrases).
        """
        raise NotImplementedError("C4b: implement HippoRAG index pipeline.")

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Query NER -> dense link -> PPR -> phrase->fact->doc -> top-k.

        Empty-NER fallback: uniform doc_prob per paper, with prominent
        logging of every empty-NER event into self._last_trace and a
        running counter in self._index_stats.

        NOT IMPLEMENTED in C4a.
        """
        raise NotImplementedError("C4b: implement HippoRAG query pipeline.")

    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        """Standard retrieve-then-read using the harness-level final generator.

        Reader is HarnessConfig.generation (held constant across systems
        per the controlled-variable invariant — no per-system generator
        field on M6Config or any other system config).

        NOT IMPLEMENTED in C4a.
        """
        raise NotImplementedError("C4b: implement HippoRAG answer (retrieve + generate).")


__all__ = ["HippoRAGSystem"]
