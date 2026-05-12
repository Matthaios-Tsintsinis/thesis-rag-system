"""Smoke test runner.

Indexes M1/M2/M3/M8 on the tiny smoke corpus and answers 5 questions
with each. Writes a single combined JSONL to
    <OUTPUT_DIR>/smoke_results_<timestamp>.jsonl
(via src.paths; OUTPUT_DIR is Drive when mounted, else /content/, else
<repo>/local_runs/outputs).

Per-system chunking: M1/M2/M3 use word-window with smoke-tuned params so
behaviour stays identical to previous smoke runs. M8 uses semantic
chunking with smoke-tuned min/max words because semantic chunking is
part of what M8 is testing.

Run from repo root:
    python -m smoke_test.run_smoke                  # full: retrieval + generation
    python -m smoke_test.run_smoke --no-generate    # CPU-only: retrieval only
    python -m smoke_test.run_smoke --systems M2 M3  # subset

Sanity checks (exit code 2 on fail):
  * M1 returns 0 chunks on every question
  * M3 ranking differs from M2 on at least one question (RRF actually fires)
  * M8 returns >0 chunks on at least one question (tree traversal isn't empty)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from src import paths
from src.config import DEFAULT_CONFIG, ChunkingConfig, HarnessConfig
from src.harness import JSONLBenchmark, _record
from src.retrievers.base import AnswerResult, BaseSystem
from src.retrievers.m1_closedbook import ClosedBookSystem
from src.retrievers.m2_flat_dense import FlatDenseSystem
from src.retrievers.m3_hybrid import HybridRRFSystem
from src.retrievers.m8_hierarchical import HierarchicalSystem


SMOKE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SMOKE_DIR / "corpus"
QUESTIONS_PATH = SMOKE_DIR / "questions.json"


SYSTEM_REGISTRY: dict[str, type[BaseSystem]] = {
    "M1": ClosedBookSystem,
    "M2": FlatDenseSystem,
    "M3": HybridRRFSystem,
    "M8": HierarchicalSystem,
}


# --- Smoke-specific chunking overrides ------------------------------------

# Word-window (small chunks so M3 has room to reorder M2's dense ranking).
_SMOKE_WW = ChunkingConfig(
    strategy="word_window",
    chunk_words=80,
    overlap_words=20,
    min_chars_per_doc=200,
)

# Semantic (small min/max so semantic chunker actually splits the 3 short
# smoke docs into multiple chunks instead of falling back to one each).
_SMOKE_SEM = ChunkingConfig(
    strategy="semantic",
    min_words=30,
    max_words=120,
    max_if_min_words=180,
    breakpoint_percentile=85.0,
    absolute_threshold=0.5,
    buffer_size=1,
    min_chars_per_doc=200,
)


def _config_for(sid: str) -> HarnessConfig:
    if sid == "M8":
        return replace(DEFAULT_CONFIG, chunking=_SMOKE_SEM)
    return replace(DEFAULT_CONFIG, chunking=_SMOKE_WW)


def _retrieval_only_answer(system: BaseSystem, query: str) -> AnswerResult:
    t0 = time.perf_counter()
    retrieved = system.retrieve(query)
    return AnswerResult(
        query=query,
        answer="[generation skipped]",
        retrieved=retrieved,
        latency_s=time.perf_counter() - t0,
        n_retrieval_calls=1 if retrieved else 0,
    )


def _check_m2_m3_divergence(
    m2_rankings: dict[str, list[str]],
    m3_rankings: dict[str, list[str]],
) -> bool:
    if not m2_rankings or not m3_rankings:
        return True
    shared = set(m2_rankings) & set(m3_rankings)
    return any(m2_rankings[q] != m3_rankings[q] for q in shared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="*",
                        default=["M1", "M2", "M3", "M8"])
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip LLM generation; report retrieval only.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    bench = JSONLBenchmark(name="smoke", corpus_path=CORPUS_DIR,
                           questions_path=QUESTIONS_PATH)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = paths.output_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"smoke_results_{stamp}.jsonl"
    print(f"[smoke] writing to {out_path}")

    m2_rankings: dict[str, list[str]] = {}
    m3_rankings: dict[str, list[str]] = {}
    m8_chunk_counts: list[int] = []

    with out_path.open("w", encoding="utf-8") as f:
        for sid in args.systems:
            if sid not in SYSTEM_REGISTRY:
                raise SystemExit(
                    f"Unknown system {sid!r}; known: {list(SYSTEM_REGISTRY)}"
                )
            if sid == "M1" and args.no_generate:
                print("[smoke] skipping M1 (retrieval-only mode; "
                      "M1 has no retrieval)")
                continue

            system = SYSTEM_REGISTRY[sid](config=_config_for(sid))
            print(f"[smoke] === {sid}: {system.__class__.__name__} ===")
            system.index(CORPUS_DIR)

            for n, q in enumerate(bench.questions()):
                if args.limit is not None and n >= args.limit:
                    break
                if args.no_generate:
                    ar = _retrieval_only_answer(system, q.question)
                else:
                    ar = system.answer(q.question)

                rec = _record(sid, bench.name, q, ar)
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                f.flush()

                ids = [r.chunk.chunk_id for r in ar.retrieved]
                if sid == "M2":
                    m2_rankings[q.question_id] = ids
                elif sid == "M3":
                    m3_rankings[q.question_id] = ids
                elif sid == "M8":
                    m8_chunk_counts.append(len(ar.retrieved))

                preview = ar.answer[:120].replace("\n", " ")
                extras = ""
                if ar.extra:
                    extras = (
                        f" abstained={ar.extra.get('abstained')} "
                        f"conf={ar.extra.get('confidence', float('nan')):.3f}"
                    )
                print(f"  q={q.question_id} retrieved={len(ar.retrieved)} "
                      f"lat={ar.latency_s:.2f}s ids={ids[:3]}{extras} "
                      f"ans={preview!r}")

    print(f"\n[smoke] wrote {out_path}")

    # --- Sanity checks ----------------------------------------------------

    fail = False

    if "M1" in args.systems and not args.no_generate:
        m1_chunk_counts: list[int] = []
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["system_id"] == "M1":
                m1_chunk_counts.append(len(rec["retrieved_chunk_ids"]))
        if any(c != 0 for c in m1_chunk_counts):
            print("[smoke] FAIL: M1 returned non-empty retrieved_chunk_ids")
            fail = True
        else:
            print("[smoke] OK: M1 returned 0 chunks on every question")

    if "M2" in args.systems and "M3" in args.systems:
        if not _check_m2_m3_divergence(m2_rankings, m3_rankings):
            print(
                "[smoke] FAIL: M2 and M3 produced identical rankings on every "
                "question — RRF fusion may be broken (BM25 should reorder)."
            )
            fail = True
        else:
            print("[smoke] OK: M3 ranking differs from M2 on at least one question")

    if "M8" in args.systems:
        if not m8_chunk_counts or all(c == 0 for c in m8_chunk_counts):
            print(
                "[smoke] FAIL: M8 returned 0 chunks on every question — "
                "tree traversal collapsed or rerank dropped all candidates."
            )
            fail = True
        else:
            n_hits = sum(c > 0 for c in m8_chunk_counts)
            print(f"[smoke] OK: M8 returned chunks on {n_hits}/"
                  f"{len(m8_chunk_counts)} questions")

    if fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
