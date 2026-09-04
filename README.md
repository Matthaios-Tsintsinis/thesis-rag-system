# thesis-rag-system

The benchmark harness behind an undergraduate CS thesis: a comparative
evaluation of published retrieval-augmented generation architectures
under one verified harness. Four systems, four benchmarks, two readers.

| id | system | retrieval | embedder |
|----|--------|-----------|----------|
| M1 | closed-book LLM | none | — |
| M2 | flat dense | FAISS over 200-word chunks, top-15 | bge-m3 |
| M3 | hybrid | dense + BM25, RRF (k=60), top-15 | bge-m3 |
| M4 | RAPTOR (paper-faithful, collapsed tree) | 2,000-token evidence budget | multi-qa-mpnet |

Benchmarks: MultiHop-RAG, NarrativeQA (seeded 40-story draw), HotpotQA
distractor (registered 1,000-question sample), HotpotQA pooled (the same
questions, paragraphs pooled per 100-question shard). Readers:
`Qwen/Qwen2.5-7B-Instruct` (bank `p10`) and
`meta-llama/Llama-3.1-8B-Instruct` (bank `p11`), each a full
independent replication (its own M4 trees and summaries). 32 cells.

**This repository is the reproduction path and nothing else.** It runs
a cell, replays the retrieval rankings behind the R@5 column, and
exports the three tables (F1 | EM | R@5 per benchmark and reader). The
analysis, audit and probe tooling, the withdrawn systems and the
off-matrix benchmarks were removed in September 2026; the complete tree
lives at the annotated tag **`thesis-full-2026-09-03`** (`git show
thesis-full-2026-09-03:<path>` reads any deleted file). The fidelity
record, the methods document and the results are disk-only working
documents (`docs/` is gitignored) and cite that tag.

## What you need

- **An NVIDIA L4 (24 GB).** Every banked cell ran on an L4; the runner
  refuses to add a cell to a bank whose summaries record different
  hardware, so a reproduction runs on an L4 or into its own bank.
- **CPython 3.12.13, exactly.** The environment gate compares the full
  interpreter string recorded in the lock.
- **`requirements.lock`** — the banked environment: the pinned versions
  of the topology stack, torch and transformers, faiss, rank-bm25,
  tiktoken, datasets and huggingface-hub, plus the `# python=3.12.13`
  line (`lockfile_hash 17878bc8740173be`). It is not in this repository;
  it sits at the root of the Drive folder beside the banks. The runner
  checks it before any model loads.
- **The Drive folder** `/content/drive/MyDrive/thesis_rag/`:

  ```
  thesis_rag/
    requirements.lock                 the environment authority (see above)
    cache/                            substrate caches: M2/, M3/, M4_RAPTOR/, ...
    outputs/
      p10/                            the Qwen bank: <benchmark>_<system>_validation.jsonl
      p11/                            the Llama bank:   + .summary.json per cell,
                                      + rankings.<stem>.jsonl / .json sidecars (replay)
      COMPARISON.csv, COMPARISON.md   the exported tables
  ```

  `src/paths.py` resolves the cache and output roots to that folder
  automatically once Drive is mounted; `THESIS_CACHE_DIR` and
  `THESIS_OUTPUT_DIR` override them (a re-derivation ALWAYS uses a
  throwaway `THESIS_CACHE_DIR` and a throwaway output directory — the
  cold-tree gate refuses an M4 cell whose trees already exist, and the
  bank gates refuse a cell written beside cells from another reader or
  another GPU).

## The Colab session, block by block

The run host is a Colab notebook whose cells call a **separate CPython
3.12.13 interpreter as a subprocess** (`/content/py312/bin/python`).
Nothing is imported into the notebook kernel itself, so the numpy
splice that bites in-kernel installs cannot occur; if you ever `pip
install` into the notebook kernel instead, RESTART the runtime before
importing anything (installing umap-learn upgrades numpy under an
already-imported copy and the failure is `ImportError: cannot import
name '_center' from 'numpy._core.umath'`, which takes down faiss, torch
and sentence-transformers).

Set the runtime accelerator to **L4** before Block 1; Colab assigns a
CPU or a T4 silently otherwise, and the GPU gate will refuse the bank.

**Block 1 — clone, mount Drive, HF cache local.** The HF cache must be
local disk: Drive's FUSE layer corrupts large `.safetensors` mid-download.
The env vars must be exported BEFORE any process imports transformers.

```bash
git clone -b claude/reverent-chaplygin-42b2c0 https://github.com/Matthaios-Tsintsinis/thesis-rag-system.git /content/thesis-rag-system
cd /content/thesis-rag-system && git fetch --tags
```

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ.update({
    "HF_HOME": "/content/hf_cache",
    "TRANSFORMERS_CACHE": "/content/hf_cache",
    "HF_DATASETS_CACHE": "/content/hf_cache/datasets",
    "SENTENCE_TRANSFORMERS_HOME": "/content/hf_cache/sentence-transformers",
})
```

**Block E — the interpreter.** The operator's block, verbatim: CPython
3.12.13 from conda-forge at `/content/py312`, the interpreter every
later command runs as a subprocess.

```bash
cd /content && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
/content/bin/micromamba create -y -p /content/py312 python=3.12.13 -c conda-forge
/content/py312/bin/python --version                      # Python 3.12.13, exactly
```

**Block F — the locked environment and the pin gate: the PROVEN
sequence.** The order is the result of an incident, not a preference:
torch FIRST from the PyTorch CUDA-12.8 index at the lock's exact
version (PyPI does not serve the `+cu128` build), then
`requirements.txt` (the reduced import graph), then the lock's pins
with the same index as an extra, then `pip check`, then the gate. The
commented uninstall line is the record of an incident: installing over
a fresh environment once dragged a `torchvision` wheel in over the
locked torch and broke `PreTrainedModel` AFTER the pin had printed OK
(2026-08-24). torchvision, torchaudio and the docling family are in no
lockfile line, so absence is the clean state. The line was retired on
2026-09-04 by the ruled proof: the fresh-clone smoke on an L4 ran `pip
check` BEFORE it and was clean — the reduced `requirements.txt` pulls
none of them. Re-enable it only if `pip check` names one of those
packages. The gate must print
`[pin] lockfile_hash=17878bc8740173be`, `[pin] python=3.12.13 (locked
3.12.13)`, `[pin] checked N pinned package(s)` and `[pin] OK —
environment matches the lockfile.`; it screens `pip check` itself and
FAILS on any conflict naming a locked package.

```bash
cd /content/thesis-rag-system
cp /content/drive/MyDrive/thesis_rag/requirements.lock requirements.lock
/content/py312/bin/python -m pip install $(grep -E "^torch==" requirements.lock) --index-url https://download.pytorch.org/whl/cu128
/content/py312/bin/python -m pip install -r requirements.txt
/content/py312/bin/python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
/content/py312/bin/python -m pip check
# retired 2026-09-04 (fresh-clone smoke: pip check clean before this line) -- re-enable only if pip check names one of these:
# /content/py312/bin/python -m pip uninstall -y torchvision torchaudio docling-ibm-models docling docling-core docling-parse docling-slim
/content/py312/bin/python -m scripts.pin_environment check --lockfile requirements.lock
```

The stack check, once the gate is green (numba >= 0.66 is load-bearing:
older numba pins numpy < 2.1; fix by upgrading numba, never by lowering
numpy):

```bash
cd /content/thesis-rag-system && /content/py312/bin/python -c "import numpy, numba, umap, faiss, torch, sentence_transformers, tiktoken, sklearn; import importlib.metadata as m; assert tuple(int(x) for x in m.version('numpy').split('.')[:2]) >= (2, 1) and tuple(int(x) for x in m.version('numba').split('.')[:2]) >= (0, 66), (m.version('numpy'), m.version('numba')); print('stack OK')"
```

**Block F2a — the Hugging Face token (Llama column only).** The Llama
repo is gated: accept the license on the model page with the SAME
account as the token, create a READ token at
`huggingface.co/settings/tokens`, store it in Colab Secrets as
`HF_TOKEN`, and export it into the environment the subprocess inherits.
The runner's own preflight message refers to this step as "Block F2".

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

**Block F2b — prove the files are served, not just the metadata.** The
runner does exactly this before any GPU time (repo metadata is public
on gated repos; only the files are gated, and a metadata probe once
printed "verified" over a 403-bound run). Two seconds now saves a
session later.

```bash
cd /content/thesis-rag-system && /content/py312/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('meta-llama/Llama-3.1-8B-Instruct', 'config.json'))"
```

## The two commands

**Run one cell** (here M4 x HotpotQA distractor, the Qwen reader).
This is the command shape that produced every banked cell; the banks
are complete, so a reproduction runs into a THROWAWAY cache and output
directory (pointing `--output` at a banked cell answers nothing and is
refused by the population gate before any summary is written — the
banked summary is never rewritten). `--output` is always passed (the
runner's default name carries a timestamp, not the bank stem). Gates
before any model loads: the pin, the bank's reader, the bank's GPU,
hub file access, the benchmark preflight, and the cold-tree preflight
over every unit (an M4 cell refuses to serve a warm substrate; there is
no flag that allows it).

```bash
cd /content/thesis-rag-system && THESIS_CACHE_DIR=/content/repro_cache /content/py312/bin/python -m src.eval.runner --system M4 --benchmark hotpotqa --split validation --output /content/drive/MyDrive/thesis_rag/outputs/repro/hotpotqa_M4_validation.jsonl
```

For the Llama reader add `--generator meta-llama/Llama-3.1-8B-Instruct`
and use a separate output directory (the bank-reader gate refuses two
readers in one directory). Every cell is the full declared population;
there is no small-sample mode. `--resume` appends to an interrupted
cell and skips its banked queries; the summary written at the end
covers the rows the finishing process scored and the units it indexed,
so a cell is completed within one process where possible. On M4, a
unit interrupted after its tree was flushed is warm on resume and the
cold-tree preflight lists its directory: delete it and re-run (the
rebuild is deterministic under the pinned stack, up to the measured
floor of one 16-leaf unit in a thousand).

**Export the three tables** against both banks. Refuses on any missing
or partial cell, on any population mismatch, on a recomputed
credited-refusal count that disagrees with the recorded battery, on a
HotpotQA EM that disagrees with the banked per-row value, and on a
missing rankings sidecar (it never falls back to hit@5).

```bash
cd /content/thesis-rag-system && /content/py312/bin/python -m scripts.export_comparison --p10 /content/drive/MyDrive/thesis_rag/outputs/p10 --p11 /content/drive/MyDrive/thesis_rag/outputs/p11 --out /content/drive/MyDrive/thesis_rag/outputs
```

**Re-derive the R@5 sidecars** (only if they are absent — they are on
Drive). The replay re-runs retrieval over the warm substrates for the 18
ranked cells and writes `rankings.<stem>.jsonl` beside each; every
replayed row must reproduce the banked set-F1, hit@K, MAP@K and MRR
bit-for-bit or the cell refuses. An existing sidecar means done only
after it is verified (the summary names the cell, the rows file parses
and holds the summary's row count, and the embedded `rows_sha256`
matches where present); a verified cell is named on one line and never
rewritten, so against the Drive banks the command is a no-op that
prints eighteen verified lines, and a corrupt or foreign sidecar
refuses the run. Delete a cell's two sidecar files by hand to
regenerate it deliberately. Same GPU class as the bank, for the gate's
sake.

```bash
cd /content/thesis-rag-system && /content/py312/bin/python -m scripts.replay_retrieval --p10 /content/drive/MyDrive/thesis_rag/outputs/p10 --p11 /content/drive/MyDrive/thesis_rag/outputs/p11
```

## Acceptance

A fresh clone at the current HEAD, against the existing banks, must
produce `COMPARISON.csv` with md5 **`ba08898a57f586dfb255e04304ff91d5`**
— byte-identical to the placed edition. Anything else is a defect in the
tree, never a finding.

```bash
md5sum /content/drive/MyDrive/thesis_rag/outputs/COMPARISON.csv
```

Two further checks a stranger can run without the banks:

- **CPU:** `python -m unittest discover -s tests -t .` is green (483
  tests; no GPU, no model — the suite fakes generation). The three CLIs
  print their reduced surfaces: `python -m src.eval.runner --help`
  (`--lockfile --system --benchmark --split --output --generator --resume`),
  `python -m scripts.replay_retrieval --help` (`--p10 --p11`),
  `python -m scripts.export_comparison --help` (`--p10 --p11 --out`).
- **GPU:** one cell, M4 x HotpotQA distractor, into throwaway locations
  so no gate refuses and nothing banked is touched (1,000 cold trees,
  1,000 answers, about 2.7 h on an L4); the banked cell's rows and
  `mean_answer_score` reproduce, up to the measured within-stack floor
  of one 16-leaf unit in a thousand that may flip its layer-1 count.

  ```bash
  cd /content/thesis-rag-system && THESIS_CACHE_DIR=/content/smoke_cache /content/py312/bin/python -m src.eval.runner --system M4 --benchmark hotpotqa --split validation --output /content/drive/MyDrive/thesis_rag/outputs/smoke_reduced/hotpotqa_M4_validation.jsonl
  ```

## Repository layout

```
src/
  config.py              every constant and dataclass the path reads (ChunkingConfig is inside every substrate key)
  models.py              embedder, the local fp16 generator (one load per model name, residency asserted), generate / generate_batch
  cache.py               compute_cache_key, corpus hash, substrate directories and manifests
  chunking.py            word_window (M2/M3) and raptor_100tok (M4) chunkers
  parsing.py             the .txt corpus reader; parsing_identity() is a key input and a literal
  components.py          per-system embedder / chunker resolution
  prompt_packing.py      the evidence block and token counting (no budget)
  raptor_paper.py        M4's chunker, bottom-up UMAP+GMM tree, collapsed index, substrate-key extras
  paths.py               Drive-aware cache / output roots
  retrievers/            base.py (index_items layout, prepare, retrieve_for_scoring), m1..m4
  eval/                  runner.py (CLI + gates), base.py (BenchmarkRunner), the four loaders and their scorers,
                         alignment.py (set-F1 and the rank-aware metrics), sampling.py (the one dated seed), types.py
scripts/
  pin_environment.py     write / check the lock; the pip-check screen; GPU string; provenance block
  replay_retrieval.py    the R@5 producer (gated retrieval replay -> sidecars)
  export_comparison.py   the one export: COMPARISON.csv / COMPARISON.md
  verify_provenance_citations.py   documentation tooling, off the output path: checks the disk-only
                         fidelity documents' citations against the tag
tests/                   483 tests; python -m unittest discover -s tests -t .
requirements.txt         the reduced import graph (the lock on Drive is the version authority)
```

## What is deliberately NOT here

The five audit documents, the living fidelity record, the results and
the thesis map are working documents under the gitignored `docs/`
directory and are delivered separately. The withdrawn systems (M6
HippoRAG, M7 three-axis, M9 CorrectiveRAG), the archived ones (M5, M8),
the QASPER / QuALITY loaders, ROUGE-L, the analysis and significance
tooling, every probe and cost script, the smoke corpus and the
prototype notebook are at `thesis-full-2026-09-03`.

## Author

Matthaios Tsintsinis — undergraduate thesis, computer science. MIT
licence, see [LICENSE](LICENSE).
