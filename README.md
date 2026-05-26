# Three-Axis Hybrid Retrieval over RAPTOR

Undergraduate thesis project — *Development of a Retrieval-Augmented Generation System with Hierarchical Clustering Methods and Large Language Models*.

This repository contains the implementation and evaluation harness for **M7**, a retrieval-augmented generation system that diversifies retrieval along three independent axes — semantic granularity, document structure, and query intent — layered on top of a RAPTOR substrate. The thesis examines whether a system designed to address three complementary failure modes simultaneously outperforms strong single-axis baselines on multi-aspect and multi-hop questions, and uses a structured ablation methodology to attribute observed gains to specific components.

## Thesis claim

> Retrieval failures break down into three independent categories — wrong-granularity retrieval, structurally-misplaced retrieval, and missed sub-question coverage. A system that diversifies along all three axes, combined with an ablation methodology that empirically measures the contribution of each axis, beats the strongest published hierarchical baseline (RAPTOR collapsed retrieval) on multi-aspect and multi-hop questions.

The claim is comparative and the comparison is empirical. The contribution is the three-axis taxonomy, the specific combination of techniques along each axis, and the ablation grid that attributes performance to each component.

## What is original to this work

This project combines existing published techniques into a new system. The components that are genuinely original to the thesis:

1. The three-axis taxonomy of retrieval failures (granularity / structure / intent) as a framing of the problem and a design principle for diversification.
2. A multi-branch RAPTOR tree traversal RRF-fused with the published collapsed-retrieval variant.
3. A quota-preserving cross-encoder rerank mechanism that runs inside each aspect's candidate pool rather than globally, with backfill on cross-aspect deduplication.
4. The empirical attribution methodology — eight ablations, five benchmarks, per-question-type slicing — designed so the contribution of each component is measured rather than asserted.

The underlying components (RAPTOR, HyDE, query decomposition, BM25, RRF, cross-encoder reranking, Docling structural extraction) are existing published techniques and are cited as such.

## System inventory

Each system is implemented as a `BaseSystem` subclass in `src/retrievers/` and registered in `smoke_test/run_smoke.py`.

| ID  | System                              | Status                            |
| --- | ----------------------------------- | --------------------------------- |
| M1  | Closed-book LLM (no retrieval)      | Implemented                       |
| M2  | Flat dense retrieval                | Implemented, smoke-verified       |
| M3  | Hybrid dense + BM25, RRF-fused      | Implemented, smoke-verified       |
| M4  | RAPTOR, collapsed retrieval         | Implemented, smoke-verified       |
| M6  | HippoRAG                            | Planned — not yet implemented     |
| M7  | Three-axis hybrid (thesis contribution) | Implemented, smoke-verified, 8 ablations runnable |

GraphRAG (formerly M5) and the hierarchical cluster-tree port (formerly M8) are archived under `src/retrievers/deprecated/`; see the README there for the per-paper-component rationale. They are no longer part of the active evaluation roster.

M4 and M7 share a RAPTOR substrate cache at `cache/RAPTOR/<substrate_hash>/`; the same tree and summaries are reused across the two systems and across the 8 M7 ablations.

## Pipeline overview

The full M7 pipeline is described in detail in `docs/PIPELINE_DESIGN.md` (local; not version-controlled). The compact version:

```
user query
  │
  ▼
Axis 3 — intent decomposition (LLM classifies + extracts up to 3 aspects)
  │
  ▼
view generation per aspect — paraphrase + HyDE
  │  (plus an always-on protected global query view)
  ▼
aspect scoring — 0.5 × LLM importance + 0.5 × cross-encoder retrieval confidence
  │
  ▼
budget allocation — 15-chunk final context, min 2 / max 8 per aspect, 2 reserved global
  │
  ▼
Axis 1a — collapsed retrieval over flat index (dense + BM25, RRF-fused)
Axis 1b — multi-branch tree traversal (top-k branches kept at each depth)
  │
  ▼
RRF-fuse 1a and 1b candidate pools
  │
  ▼
Axis 2 — Docling structural rerank (section diversity cap, aspect-section bias, neighbor expansion)
  │
  ▼
aspect-internal merge + quota-preserving cross-encoder rerank
  │
  ▼
global merge, dedup, backfill if any aspect is under-quota
  │
  ▼
parent-summary context packing (orientation summaries deduplicated by parent/section)
  │
  ▼
final prompt — aspect-segmented blocks + global view + confidence-tagged abstention signal
```

## Evaluation plan

The full methodology is in `docs/evaluation_plan.pdf`. Summary:

**Benchmarks (5):** NarrativeQA, QASPER, QuALITY (inherited from the RAPTOR paper for direct comparison), MultiHop-RAG, CRAG (added for stress-testing on multi-hop and comprehensive-RAG question types).

**Ablations (8) — each removes one M7 component:**

| ID  | Removed component                                | Question answered                                            |
| --- | ------------------------------------------------ | ------------------------------------------------------------ |
| A1  | Docling structural axis                          | Does Axis 2 contribute independently of Axes 1 and 3?        |
| A2  | Intent decomposition (single global view only)   | Does Axis 3 contribute independently of Axes 1 and 2?        |
| A3  | HyDE view (two paraphrase views instead)         | Does HyDE add over a second paraphrase?                      |
| A4  | BM25 (dense-only inside semantic axis)           | Does sparse retrieval contribute over dense alone?           |
| A5  | Parent-summary context packing                   | Does orientation context improve answer quality?             |
| A6  | Quota-preserving rerank (global rerank instead)  | Does the quota mechanism protect multi-aspect coverage?      |
| A7  | Abstention signal                                | Does the confidence signal reduce hallucination on low-conf? |
| A8  | Protected global query view                      | Does the always-on global view protect against aspect-extractor failures? |

Each ablation is a config-flag toggle in `M7Config`, not a code change. All eight are verified to disable their target component on every smoke run.

**Metric families:** dataset-native (accuracy, F1, EM), retrieval quality (Recall@k, context precision/recall via RAGAS), answer quality (faithfulness, answer relevancy, answer correctness via RAGAS, plus LLM-judge scoring), efficiency (latency, tokens, retrieval calls), calibration (abstention rate, hallucination rate on false-premise slices).

**Reporting standards:** bootstrap 95% confidence intervals on all comparisons; per-question-type slicing (specific-fact, abstractive single-aspect, multi-aspect, multi-hop, comparison, false-premise/no-answer, structure-heavy); ablation deltas reported alongside aggregate results; negative results reported with the same weight as positive results.

## Stack

- **Embedder:** `BAAI/bge-m3` (multilingual; supports Greek + English)
- **Reranker:** `BAAI/bge-reranker-v2-m3` (cross-encoder; outputs raw logits, sigmoid applied for confidence scores)
- **Answer generator:** `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4 quantization (fits a Colab T4)
- **Index-time / query-time LLM calls:** `gpt-4o-mini` (summarization, aspect extraction, paraphrase, HyDE)
- **Judge LLM (evaluation only):** `gpt-4o-mini`
- **Vector index:** FAISS `IndexFlatIP`, L2-normalized embeddings
- **Sparse index:** `rank-bm25` (BM25Okapi)
- **PDF parsing:** Docling
- **Chunking:** semantic chunker with sentence buffering and percentile breakpoints (Greek punctuation aware)

## Repository layout

```
thesis-rag-system/
├── docs/
│   └── evaluation_plan.pdf       # full evaluation methodology
├── src/
│   ├── config.py                 # typed configs incl. M4Config, M7Config, ablation flags
│   ├── parsing.py                # Docling integration + fallback section assignment
│   ├── chunking.py               # semantic chunker
│   ├── models.py                 # embedder/reranker/generator lazy loaders
│   ├── paths.py                  # Drive-aware path resolution
│   ├── cache.py                  # content-addressed cache keys
│   ├── summarization.py          # gpt-4o-mini wrapper, prompt templates, key resolver
│   ├── raptor.py                 # RAPTOR tree primitives + flat collapsed index
│   ├── structural.py             # Axis 2 — post-hoc section attachment and rerank
│   ├── intent.py                 # Axis 3 — decomposition, scoring, budget allocation
│   ├── multibranch.py            # Axis 1b — multi-branch traversal + RRF merge
│   ├── retrievers/
│   │   ├── base.py
│   │   ├── m1_closedbook.py
│   │   ├── m2_flat_dense.py
│   │   ├── m3_hybrid.py
│   │   ├── m4_raptor.py
│   │   ├── m7_three_axis.py
│   │   └── deprecated/          # archived: m5_graphrag, m8_hierarchical
│   └── harness.py
├── smoke_test/
│   ├── corpus/                   # 8-document smoke corpus
│   └── run_smoke.py              # runs all systems, verifies sanity checks + 8 ablations
└── requirements.txt
```

## Running the smoke test

The smoke test is a self-contained verification that exercises every implemented system against an 8-document corpus and five sample questions, plus runs the full 8-ablation sweep for M7. It catches integration regressions but does not produce benchmark-quality results.

Tested on Google Colab with a T4 GPU runtime. A local run requires GPU access (CPU-only execution is functional but the cross-encoder rerank becomes the bottleneck — single queries take minutes rather than seconds).

```bash
# Clone the feature branch (the development branch for this project)
git clone -b claude/reverent-chaplygin-42b2c0 https://github.com/Matthaios-Tsintsinis/thesis-rag-system.git
cd thesis-rag-system

# Install dependencies (Python 3.12)
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

# Provide an OpenAI API key (used by gpt-4o-mini for summarization and aspect extraction)
export OPENAI_API_KEY=sk-...

# On Colab specifically, also export the key into the subprocess environment:
# (resolver in src/summarization.py reads os.environ first, then google.colab.userdata)

# Run the smoke test
python -m smoke_test.run_smoke --no-generate
```

Expected output on a fresh run: cache miss on every system, RAPTOR substrate built into `cache/RAPTOR/<hash>/` (16 gpt-4o-mini summary calls, roughly $0.003), per-query retrieval traces for five questions across M2/M3/M4/M7, 8 ablation runs each reporting `effect_observed=True`, and a final block of `[smoke] OK:` sanity check lines covering tree shape, routing coverage, view generation, multi-aspect decomposition, multi-branch exploration, deduplication, and quota preservation.

A second invocation should report cache hits on all systems with zero new summary calls; query-time LLM calls (aspect extraction, paraphrase, HyDE) run on every invocation and are not cached.

## Smoke baselines

The smoke test produces JSONL output files in `outputs/smoke_results_<timestamp>.jsonl`. Two reference baselines from verified runs of the post-cleanup `f9f65c7` commit:

- Cache-miss run: `smoke_results_20260516-175201.jsonl`
- Cache-hit run: `smoke_results_20260516-175518.jsonl`

These are local artifacts and are not committed; they are produced fresh on each verified run.

## Project status

**Implemented and smoke-verified on Colab T4:** M1, M2, M3, M4, M7. All 8 M7 ablations confirmed to disable their target component.

**Archived:** M5 (GraphRAG) and M8 (hierarchical cluster-tree port), under `src/retrievers/deprecated/`. See that directory's README for the per-paper-component rationale.

**Pending:** M6 (HippoRAG wrapper), benchmark loaders for the five datasets in the evaluation plan, RAGAS-based evaluation pipeline, full eval grid execution, analysis and write-up.

The smoke-verified state confirms the systems are individually correct and integrated correctly. It does not yet provide benchmark-scale evidence for the thesis claim — that depends on the full eval grid, which is the next major phase of work.

## References

The full reference list is in `docs/PIPELINE_DESIGN.md` and `docs/evaluation_plan.pdf`. The most directly relevant works:

- Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., Manning, C. D. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*. ICLR 2024. [arXiv:2401.18059](https://arxiv.org/abs/2401.18059).
- Gao, L., Ma, X., Lin, J., Callan, J. (2023). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. ACL 2023. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496).
- Ammann, P. J. L., Golde, J., Akbik, A. (2025). *Question Decomposition for Retrieval-Augmented Generation*. ACL SRW 2025.
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009.
- Auer, C., Lysak, M., Nassar, A., et al. (2024). *Docling Technical Report*. [arXiv:2408.09869](https://arxiv.org/abs/2408.09869).
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2024). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. EACL 2024 System Demonstrations. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217).

## Author

Matthaios Tsintsinis — undergraduate thesis, computer science.

## License

MIT — see [LICENSE](LICENSE).
