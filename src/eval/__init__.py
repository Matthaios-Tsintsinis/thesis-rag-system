"""Benchmark evaluation layer.

Loaders + scorers + retrieval-recall (CK-2) alignment for the 4 x 4
matrix: MultiHop-RAG, NarrativeQA, HotpotQA-distractor and HotpotQA-pooled
across M1-M4, driven by `src.eval.runner`.

This package is read-only with respect to the retrieval systems
themselves — it consumes the existing BaseSystem interface and adds a
benchmark-driven driver that feeds in per-benchmark corpora and queries.
The two structural hooks on the retrieval side are `Chunk.gold_provenance`
(eval-time provenance for chunker-agnostic CK-2 retrieval recall) and
`BaseSystem.index_items` (an in-memory list of CorpusItem rather than a
filesystem path, so per-question corpora need no temp-dir writes).
"""
