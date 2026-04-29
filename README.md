# Hierarchical RAG System

A multilingual (Greek + English) Retrieval-Augmented Generation pipeline. Supports two chunking strategies (fixed-size and semantic), hierarchical document retrieval via a semantic tree, hybrid BM25 + dense search, and RAGAS-based evaluation.

Runs entirely on Google Colab (tested on T4 GPU).

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Models](#models)
- [Notebook Structure](#notebook-structure)
- [Getting Started](#getting-started)
- [Supported File Formats](#supported-file-formats)
- [Configuration Reference](#configuration-reference)
- [Chunking Strategies](#chunking-strategies)
- [Query View Generation](#query-view-generation)
- [Hybrid Retrieval](#hybrid-retrieval)
- [Reranking](#reranking)
- [Confidence Threshold](#confidence-threshold)
- [Generation](#generation)
- [Corpus Analytics](#corpus-analytics)
- [Cache System](#cache-system)
- [Utility Cells](#utility-cells)
- [RAGAS Evaluation](#ragas-evaluation)
- [Comparing Chunking Strategies](#comparing-chunking-strategies)
- [Outputs](#outputs)

---

## Pipeline Overview

```
PDFs / DOCX / HTML
       │
       ▼
  1. Parsing & Chunking ──── fixed (220 words) or semantic (bge-m3 similarity)
       │
       ▼
  2. Embedding (BAAI/bge-m3)
       │
       ▼
  3. FAISS index + BM25 index
       │
       ▼
  4. Semantic Tree (hierarchical clustering → tree of centroids)
       │
  At query time:
       │
       ▼
  5. Query View Generation (Qwen2.5 paraphrases + template views)
       │
       ▼
  6. Tree Traversal (top-k branches per level → candidate chunks)
       │
       ▼
  7. Hybrid Scoring (dense cosine + BM25, alpha-weighted)
       │
       ▼
  8. Context Expansion (neighbour chunks within document)
       │
       ▼
  9. Reranking (BAAI/bge-reranker-v2-m3)
       │
       ▼
 10. Grounded Answer Generation (Qwen2.5-3B-Instruct, 4-bit NF4)
       │
       ▼
 11. RAGAS Evaluation (gpt-4o-mini as judge)
```

### How retrieval works

When a query comes in, the pipeline generates several semantic "views" of it (paraphrases and template variations) to increase recall. Each view is used to traverse the semantic tree — a hierarchical clustering of all chunk embeddings — narrowing down candidate chunks efficiently without scanning the entire FAISS index. Candidates are then scored with a hybrid of cosine similarity and BM25, the top documents are selected, and their neighbouring chunks are pulled in for context. Finally the reranker re-orders everything by cross-encoder relevance before the generator produces a grounded, cited answer.

---

## Models

| Role | Model | Notes |
|---|---|---|
| Embedder | `BAAI/bge-m3` | Multilingual (Greek + English), used for chunking, FAISS, and RAGAS |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder, optional via `use_reranker` flag |
| Generator | `Qwen/Qwen2.5-3B-Instruct` | 4-bit NF4 quantization via bitsandbytes, fits T4 alongside embedder + reranker |
| RAGAS judge | `gpt-4o-mini` (OpenAI) | External API, only needed for evaluation cells |

### VRAM footprint (T4, 15 GB)

| Component | ~VRAM |
|---|---|
| bge-m3 | 2.3 GB |
| bge-reranker-v2-m3 | 1.1 GB |
| Qwen2.5-3B 4-bit | 2.0 GB |
| **Total** | **~5.4 GB** |

---

## Notebook Structure

| Cell | Section | What it does |
|---|---|---|
| — | Clear uploads | Optional: wipe `/content/uploads` |
| — | Clear cache | Optional: delete cached embeddings/index |
| 1 | Environment setup | `pip install` all dependencies |
| 2 | Configuration | All tunable parameters in one `CONFIG` dict |
| 3 | Utilities & parsers | File parsers (PDF, DOCX, HTML, XLSX), `chunk_text_words`, `chunk_text_semantic` |
| 4 | Load models | bge-m3, reranker, Qwen2.5 (4-bit) |
| 5 | Ingestion & chunking | Parse files, chunk by strategy, build `documents` and `chunks` lists |
| 6 | Embeddings & FAISS | Embed chunks, build FAISS index and BM25 index, cache everything |
| 7 | Semantic tree | Hierarchical agglomerative clustering → tree of centroids |
| 8 | Retrieval functions | `retrieve()`, tree traversal, hybrid scoring, context expansion, reranking |
| 9 | Answer generation | `ask()` — grounded generation with citations, confidence scoring |
| 10 | Corpus analytics | Stats report + JSON saved to `OUTPUT_DIR` |
| 11 | RAGAS setup | Install RAGAS, configure gpt-4o-mini judge, wrap bge-m3 for embeddings |
| 11a | Testset generation | Synthetic QA pairs from your corpus via RAGAS knowledge graph |
| 11b | RAGAS functions | `generate_predictions()`, `run_ragas()`, metric selection |
| 11c | RAGAS run | Score predictions, print results, save report JSON |
| 12 | Example usage | Sample queries demonstrating the full pipeline |
| 13 | Download outputs | Zip and download outputs + cache to local machine |

---

## Getting Started

### Requirements

- Google Colab with a T4 GPU runtime (or better)
- OpenAI API key (only for RAGAS evaluation cells 11–11c)
- Your documents (PDF, DOCX, HTML, or XLSX)

### Steps

1. Open the notebook in Colab and set the runtime to GPU.
2. Run **cell 1** (environment setup) — installs all dependencies.
3. Run **cell 2** (configuration) — you will be prompted to upload your PDF/DOCX files here. They are saved to `/content/uploads`.
4. Run **cells 3–10** in order. This parses your documents, embeds them, builds the FAISS and BM25 indexes, and constructs the semantic tree. Progress bars show embedding status.
5. Run **cell 12** to test the pipeline with example queries.
6. Optionally run **cells 11–11c** to evaluate with RAGAS (requires OpenAI API key).

---

## Supported File Formats

The parser handles the following formats automatically based on file extension:

| Format | Extension(s) | Notes |
|---|---|---|
| PDF | `.pdf` | Text-based PDFs only; scanned images without OCR will produce poor chunks |
| Word | `.docx` | Full text extraction including paragraphs and tables |
| HTML | `.html`, `.htm` | Body text extracted, tags stripped |
| Plain text | `.txt`, `.md` | Read as-is |
| CSV | `.csv` | Rows concatenated as text |
| JSON | `.json` | Serialized to text |
| Excel | `.xlsx` | Sheet content extracted as text |

Files shorter than `min_chars_per_doc` (default 200 characters) after parsing are skipped and logged in `skipped`.

All parameters live in the `CONFIG` dict in cell 2. Key options:

### Chunking

```python
"chunking_strategy": "fixed",        # "fixed" | "semantic"

# Fixed chunking
"chunk_size_words": 220,
"chunk_overlap_words": 40,

# Semantic chunking
"semantic_breakpoint_percentile": 90,    # split at top N% similarity drops
"semantic_min_words_per_chunk": 80,      # merge chunks smaller than this
"semantic_max_words_per_chunk": 400,     # force-split chunks larger than this
"semantic_max_words_if_min_chunk": 500,  # max allowed size when merging a small chunk into previous
"semantic_buffer_size": 1,               # neighbour sentences used for embedding context
"semantic_absolute_threshold": 0.5,     # hard cap on breakpoint threshold (prevents over-merging on homogeneous text)
```

### Retrieval

```python
"top_k_flat": 12,                    # candidates from FAISS
"top_k_final": 6,                    # final fragments returned
"top_docs_after_tree": 8,            # documents selected after tree traversal
"top_chunks_per_doc_for_context": 3, # anchor chunks per document for context expansion
"context_neighbor_radius": 1,        # ±N chunks expanded around each anchor
"tree_branching_factor": 4,
"tree_top_branches_per_level": 2,
"tree_min_cluster_size": 24,
"tree_max_depth": 4,
"max_query_views": 5,                # total views including original query
"rerank_top_n": 24,                  # how many candidates the reranker sees
"use_reranker": True,                # enable/disable cross-encoder reranking
"use_bm25_hybrid": True,             # enable/disable BM25 hybrid scoring
"alpha_dense": 0.75,                 # weight for dense vs BM25 (0.75 = 75% dense)
```

### Query augmentation

```python
"enable_query_view_generation": True,  # LLM-generated paraphrases via Qwen
```

### Generation

```python
"enable_local_generation": True,   # load Qwen and generate answers locally
                                   # set False to retrieve only (no generation)
"max_new_tokens": 512,
"temperature": 0.1,
"do_sample": False,
```

### Corpus analytics

```python
"analysis_queries": [              # queries run automatically by the corpus analytics cell
    "Ποια είναι τα βασικά θέματα που καλύπτουν τα έγγραφα;",
    "What are the main topics in the document collection?",
    "Give a summary of the most important entities, concepts, and recurring themes."
]
```

### Evaluation

```python
"evaluation_json_path": None,        # path to testset JSON; set after first 11a run
```

---

## Chunking Strategies

### Fixed-size chunking (default)

Splits documents into windows of `chunk_size_words` words with `chunk_overlap_words` overlap. Fast, deterministic, works well as a baseline. Overlap ensures concepts split at a boundary still appear in two chunks.

### Semantic chunking

Uses the already-loaded `bge-m3` embedder to find natural topic boundaries:

1. **Sentence splitting** — text is split on `.`, `!`, `?`, `;` (the last covers Greek question marks)
2. **Buffered embedding** — each sentence is embedded together with its ±`buffer_size` neighbours, giving the model local context and smoothing out single-sentence noise
3. **Cosine distance** — distance between consecutive buffered embeddings is computed (1 − cosine similarity)
4. **Breakpoints** — distances above the `semantic_breakpoint_percentile` threshold AND above `semantic_absolute_threshold` become split points
5. **Size enforcement** — chunks above `max_words` are force-split; chunks below `min_words` are merged into the previous chunk, unless that would exceed `max_words_if_min_chunk`

The absolute threshold (default 0.5) prevents the percentile from going too high on homogeneous text — without it, the algorithm would still split even when all sentences are about the same topic.

### Switching strategies

Change one line in CONFIG and re-run from cell 5 downward:

```python
"chunking_strategy": "semantic",   # or "fixed"
```

The cache system handles the rest automatically — no manual file deletion needed.

---

## Query View Generation

To improve recall, every query is expanded into multiple "views" before hitting the retrieval pipeline. The pipeline always generates up to `max_query_views` total views (default 5), combining two sources:

**1. Template views (always on)**

Four hardcoded reformulations targeting different semantic angles:

```
main topic of: {query}
key entities and concepts in: {query}
evidence and passages relevant to: {query}
sections discussing: {query}
```

These run regardless of any flag.

**2. LLM-generated paraphrases (optional)**

If template views don't fill the `max_query_views` budget, Qwen generates additional retrieval-oriented reformulations. Controlled by two flags that must both be true:

```python
"enable_query_view_generation": True,   # master switch for LLM paraphrases
"enable_local_generation": True,        # Qwen must be loaded
```

If `enable_query_view_generation` is `False`, or if `enable_local_generation` is `False` (so Qwen is never loaded), the LLM step is silently skipped and only template views are used.

All views (original + template + LLM) are deduplicated before retrieval. Each view independently queries the FAISS index, and scores are aggregated by max and mean across views before ranking.

---

## Hybrid Retrieval

Candidate chunks are scored by combining dense cosine similarity (bge-m3 embeddings via FAISS) and sparse keyword matching (BM25). The final hybrid score is:

```
score_hybrid = alpha_dense × score_dense_norm + (1 − alpha_dense) × score_bm25_norm
```

Both scores are min-max normalised before combining. Default `alpha_dense = 0.75` weights dense retrieval higher, which works well for semantic queries. Raise it toward 1.0 for purely semantic queries; lower it toward 0.5 if your corpus has a lot of specific terminology or acronyms that BM25 handles better.

BM25 can be disabled entirely:

```python
"use_bm25_hybrid": False,   # pure dense retrieval
```

---

## Reranking

After hybrid scoring, the top `rerank_top_n` candidates (default 24) are passed to a cross-encoder reranker (`BAAI/bge-reranker-v2-m3`). Unlike the bi-encoder (which embeds query and chunk independently), the cross-encoder sees both together and produces a more accurate relevance score. The final `top_k_final` results are taken from the reranked list.

```python
"use_reranker": True,      # enable cross-encoder reranking
"rerank_top_n": 24,        # how many candidates the reranker scores
```

Disabling the reranker reduces VRAM usage by ~1.1 GB and speeds up retrieval, at the cost of lower precision. Useful if you hit memory limits or want faster iteration during development.

---

## Confidence Threshold

After retrieval, the top result's score is checked against a minimum threshold:

```python
CONFIDENCE_THRESHOLD = 0.35
```

If the best score (reranker score if available, otherwise hybrid score) falls below this, the pipeline flags the result as `low_confidence` and returns a fallback answer ("I could not find sufficient evidence...") instead of hallucinating. The retrieved fragments are still returned in the output for inspection.

This threshold is hardcoded in cell 9 (answer generation) and can be adjusted there directly.

---

## Generation

The generator (`Qwen2.5-3B-Instruct`, 4-bit NF4) receives a structured prompt containing the query and the top retrieved fragments as numbered sources. It is instructed to:

- Answer only from the provided sources
- Cite sources inline using `[SOURCE N]` notation
- Reply in the same language as the question (supports Greek and English)
- Explicitly say so if evidence is insufficient

```python
"enable_local_generation": True,   # set False to skip generation entirely
"max_new_tokens": 512,
"temperature": 0.1,                # low temperature for factual grounding
"do_sample": False,
```

Setting `enable_local_generation: False` skips loading Qwen entirely, saving ~2 GB VRAM. The pipeline still retrieves and returns fragments — only the final answer generation step is skipped. Useful for retrieval-only benchmarking or when VRAM is tight.

---

## Corpus Analytics

Cell 10 runs automatically at the end of the main pipeline and saves a `corpus_report_<RUN_ID>.json` to `OUTPUT_DIR`. It includes:

- Document and chunk counts, skipped file list
- Word and character statistics per document (min/max/mean/median)
- Chunks per document distribution
- Semantic tree structure (root ID, largest leaves, sample nodes)
- Full answers to the `analysis_queries` defined in CONFIG

The `analysis_queries` are run through the full `ask()` pipeline and their answers are embedded in the report — useful for a quick sanity check that retrieval and generation are working correctly on your corpus. Edit them in CONFIG to match your domain:

```python
"analysis_queries": [
    "Ποια είναι τα βασικά θέματα που καλύπτουν τα έγγραφα;",
    "What are the main topics in the document collection?",
]
```

---

## Cache System

Embeddings, the FAISS index, and chunks are cached to `/content/cache/` so re-runs don't re-embed the entire corpus. Cache filenames include the chunking strategy:

```
cache/
  embeddings_fixed.npy
  faiss_fixed.index
  chunks_fixed.json
  embeddings_semantic.npy
  faiss_semantic.index
  chunks_semantic.json
```

Switching `chunking_strategy` in CONFIG automatically uses a separate cache. Both strategies can coexist — switching back and forth is instant after the first run of each.

---

## Utility Cells

Two optional cells sit at the top of the notebook, above the main pipeline. They are **disabled by default** (flags set to `False`) and do nothing unless you explicitly enable them.

### Clear uploads

```python
RUN_CLEAR_UPLOADS = True   # set to True to wipe /content/uploads
```

Deletes all files in `/content/uploads`. Useful when starting a fresh experiment with different documents.

### Clear cache

```python
RUN_CLEAR_CACHE = True   # set to True to delete cached embeddings and index
```

Deletes the cached `.npy`, `.index`, and `.json` files in `/content/cache`. Use this if you want to force a full re-embed — for example after changing the embedding model or modifying the chunking logic in a way that isn't reflected in the strategy name.

> Note: switching `chunking_strategy` does **not** require clearing the cache manually. The per-strategy filenames handle this automatically.

---

## RAGAS Evaluation

RAGAS scores the full RAG pipeline end-to-end using an LLM as judge. The judge is `gpt-4o-mini` (external API). The embedder is `bge-m3`, reused from the retrieval pipeline — no second model is loaded.

### Metrics

| Metric | Needs `gold_answer`? | What it measures |
|---|---|---|
| Faithfulness | No | Is the answer grounded in the retrieved contexts? |
| Response relevancy | No | Does the answer address the question? |
| LLM context precision (no ref) | No | Are the retrieved chunks on-topic? |
| LLM context precision (with ref) | Yes | Same, validated against the gold answer |
| LLM context recall | Yes | Did retrieval surface enough relevant content? |
| Factual correctness | Yes | Semantic + factual match against gold answer |

If `gold_answer` is present in the testset JSON, all six metrics run. Otherwise only the top three (reference-free) run.

### Cell 11 — Setup

Installs RAGAS and configures the judge. You will be prompted for your `OPENAI_API_KEY` when this cell runs. The key is stored only in session memory and is never written to disk or notebook output.

```python
judge_chat = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=4096,
    max_retries=10,
    request_timeout=120,
)
```

### Cell 11a — Testset generation

Generates synthetic questions from your own corpus using a RAGAS knowledge graph. Run this **once per corpus** — it produces a JSON file and sets `CONFIG["evaluation_json_path"]` automatically.

```python
TESTSET_SIZE = 20          # number of questions to generate
REGEN_TESTSET = False      # set True to regenerate (overwrites existing testset)
```

**`REGEN_TESTSET`**: defaults to `False`. If `evaluation_json_path` already points to an existing file, the cell skips generation entirely. Set to `True` only when you want to regenerate with a different corpus or size.

The cell also saves a `_raw.csv` alongside the JSON — open this to review and optionally delete low-quality questions before running the eval.

> Testset generation makes many API calls and can take 20–40 minutes for a large corpus. It hits the daily RPD limit on low-tier OpenAI accounts — reduce `TESTSET_SIZE` or corpus size if this is a concern.

### Cell 11b — Functions

Defines `generate_predictions()` and `run_ragas()`. Nothing runs. Also sets RAGAS concurrency:

```python
run_config.max_workers = 1   # sequential requests, avoids rate limit cascades
```

### Cell 11c — Run

Generates answers for every question using the local Qwen model, then scores them with RAGAS. Saves two files per run:

- `ragas_predictions_<RUN_ID>.csv` — raw per-question answers and context counts
- `ragas_evaluation_report_<RUN_ID>.json` — mean scores + per-question breakdown

---

## Comparing Chunking Strategies

The workflow for comparing chunking strategies:

**Run 1 — fixed chunking:**

```python
"chunking_strategy": "fixed",
"evaluation_json_path": None,   # let 11a generate the testset
```

Run all cells. Note the testset path printed by cell 11a (e.g. `/content/outputs/20250101_120000/ragas_testset_20250101_120000.json`).

**Run 2 — semantic chunking:**

```python
"chunking_strategy": "semantic",
"evaluation_json_path": "/content/outputs/20250101_120000/ragas_testset_20250101_120000.json",
```

Run all cells. Cell 11a skips (path already set). Cell 11c produces a second report.

**Compare:**

Both `ragas_evaluation_report_*.json` files have the same structure:

```json
{
  "run_id": "...",
  "summary": {
    "scores": {
      "faithfulness": 0.74,
      "response_relevancy": 0.81,
      "llm_context_precision_without_reference": 0.69
    }
  }
}
```

Diff the `scores` dicts between the two files. The delta is your result.

---

## Outputs

Each run produces a timestamped subfolder at `/content/outputs/<RUN_ID>/`:

```
outputs/
  20250101_120000/
    corpus_report_20250101_120000.json     # document + chunk statistics
    ragas_testset_20250101_120000.json     # evaluation questions (reuse across runs)
    ragas_testset_20250101_120000_raw.csv  # raw testset for manual review
    ragas_predictions_20250101_120000.csv  # per-question answers
    ragas_evaluation_report_20250101_120000.json  # RAGAS scores
```

**Cell 13 (Download)** zips both the outputs folder and the cache folder and downloads them to your local machine via the Colab file download API.
