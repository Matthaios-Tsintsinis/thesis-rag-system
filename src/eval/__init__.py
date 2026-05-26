"""Benchmark evaluation layer.

Loaders + scorers + retrieval-recall (CK-2) alignment for the eval grid.

This package is read-only with respect to the retrieval systems
themselves — it consumes the existing BaseSystem interface and adds a
benchmark-driven driver that feeds in per-benchmark corpora and queries.
The two structural changes to the retrieval side that the eval layer
needs are minimal:

  * `Chunk.gold_provenance` (added in C5a) — eval-time provenance for
    each chunk, allowing chunker-agnostic CK-2 retrieval-recall.
  * `BaseSystem.index_items` (added in C5b) — accept an in-memory list
    of CorpusItem rather than a filesystem path, so QASPER's 888
    per-paper corpora do not require 888 temp-dir writes per system.

Pass-1 scope: types + benchmark protocol + alignment + QASPER loader
and scorer + MultiHop-RAG loader (skeleton scorer) + CLI runner.
Pass-2: LLM-judge for abstractive answers and the MultiHop free-form
scorer. Pass-3: the three answer-only benchmarks (NarrativeQA,
QuALITY, CRAG) once their schemas have been audited the same way.
"""
