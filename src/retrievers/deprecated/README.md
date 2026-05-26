# Archived retrievers

Systems in this directory were built (some smoke-verified) but are no
longer part of the active evaluation roster. The code is preserved here
for thesis discussion, potential resurrection, and historical reference.
Nothing in this directory is registered in `smoke_test/run_smoke.py` or
imported by the active harness.

## M5 — GraphRAG (Microsoft)

`m5_graphrag.py`, `graphrag_backend.py`

Archived under the per-paper-component rule for the evaluation grid:
each system uses the components its own paper specifies. GraphRAG's
paper mandates paid OpenAI embeddings (`text-embedding-3-small`) and a
GPT-4-class LLM for entity extraction and community summarisation.
Across the 5-benchmark × 8-system × M7-ablation grid the cost is
prohibitive on a thesis budget, so M5 is dropped from the active roster.

A prior version of M5 ran with the shared `bge-m3` embedder injected
into GraphRAG's library embedding-model registry, but that parity choice
was reversed when the per-paper rule was adopted (it would no longer be
faithful to the GraphRAG paper).

The code reached green end-to-end and is preserved verbatim. To
resurrect, restore `graphrag==3.0.9` from the deprecated section of
`requirements.txt`, re-register the system in `smoke_test/run_smoke.py`,
and update imports to point at the new module locations.

## M8 — Hierarchical cluster-tree RAG (ported)

`m8_hierarchical.py`

Archived because there is no clean published non-agentic, non-graph
above-RAPTOR hierarchical system to instantiate it as under the
per-paper rule. The implementation was ported from an earlier Colab
notebook (MiniBatchKMeans tree with TF-IDF per-node keywords, linear
`alpha_dense` fusion, neighbor expansion within docs, sigmoid-gated
abstention) and predates the per-paper directive. Rather than
retrofitting it to a specific paper's component set, it is archived.

The code is smoke-verified and preserved verbatim. To resurrect, move
the file back to `src/retrievers/`, restore `M8Config` and
`M8_LOW_CONFIDENCE_ANSWER` to `src/config.py` (or import them from
`_archived_config.py`), re-register in `smoke_test/run_smoke.py`, and
re-add the `m8: M8Config` field to `HarnessConfig`.

## `_archived_config.py`

Holds the `M5Config` and `M8Config` dataclasses, plus the
`M8_LOW_CONFIDENCE_ANSWER` constant, that previously lived in
`src/config.py`. The archived modules expect to import these from
`src.config`; that import path is preserved-as-was inside the archived
files (they will not resolve until resurrected), but the definitions
themselves live here.
