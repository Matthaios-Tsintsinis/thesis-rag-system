"""Measure the two costs the matrix has never measured, on the pinned stack.

WHAT IS MISSING FROM EVERY FORECAST SO FAR

  1. M4 COLD TREE-BUILD TIME, isolated from query cost. Paid ONCE PER
     CELL and invisible to `s_per_query`, which is wall-clock over
     queries only. It has never been measured under the local summariser,
     and it has never been paid at all on MultiHop or NarrativeQA,
     because those cells ran against trees built in earlier sessions. The
     cold-tree lever (`cf8a7b8`) now forces a rebuild, so this cost is
     new to the matrix rather than merely unmeasured.

     NarrativeQA is the concern: FORTY separate story-trees, each built
     cold. If one story's build is a meaningful fraction of the session
     guard, that cell needs intra-cell tree checkpointing or it is a
     re-run risk — `--resume` protects the remainder of a cell's QUERIES,
     not a tree that was half-built when the session died.

  2. FIVE-SYSTEM `s_per_query` ON MULTIHOP, the largest benchmark at
     2,556 queries. M9 has never run a timed slice under the local
     generator; M4's per-query cost was derived, not measured.

THE PROBE RULES THIS FILE OBEYS (earned five times over in this project)

  * It ASSERTS that it measured what it claims. A tree "build" that was
    served from cache is a cache read, and reporting its 0.4 s as a build
    time would be worse than not measuring at all — so a cache hit
    ABORTS.
  * A VACUOUS PASS IS A FAILURE. A degenerate tree (no summary layer)
    exits non-zero rather than reporting a suspiciously small number.
  * It PREWARMS. The first system measured otherwise pays a ~15 GB model
    load the second does not, which has already inverted the sign of one
    measurement in this project.

USAGE (Colab, pinned stack, GPU attached)

    python -m scripts.probe_cell_costs --mode tree
    python -m scripts.probe_cell_costs --mode queries --n 50
    python -m scripts.probe_cell_costs --mode both --out /content/cell_costs.json

`--mode tree` builds ONE unit per benchmark. Expect it to spend real
money on summaries; that is the point, and it is a rounding error against
the run it de-risks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _fail(msg: str) -> None:
    print(f"[probe] FAILED: {msg}")
    sys.exit(1)


def _prewarm() -> float:
    """Load the generator BEFORE any timing. Returns load seconds."""
    from src.config import DEFAULT_CONFIG
    from src.models import load_generator

    t0 = time.perf_counter()
    load_generator(DEFAULT_CONFIG.generation.model)
    dt = time.perf_counter() - t0
    print(f"[probe] prewarm: generator resident in {dt:.1f}s (EXCLUDED)")
    return dt


def _one_unit(benchmark_id: str):
    """The smallest honest unit of each benchmark's corpus shape."""
    from src.eval.runner import BENCHMARK_REGISTRY

    bench = BENCHMARK_REGISTRY[benchmark_id]()
    split = "validation"
    for unit in bench.iter_eval_units(split=split, max_units=1):
        return bench, unit
    _fail(f"{benchmark_id}: loader yielded no unit")


def probe_tree_builds(benchmarks: list[str]) -> list[dict]:
    """Wall-time for M4 tree construction alone, one unit per benchmark."""
    from src.config import DEFAULT_CONFIG
    from src.retrievers.m4_raptor import RaptorSystem

    rows: list[dict] = []
    for benchmark_id in benchmarks:
        bench, unit = _one_unit(benchmark_id)
        system = RaptorSystem(DEFAULT_CONFIG)

        t0 = time.perf_counter()
        system.index_items(unit.corpus)
        build_s = time.perf_counter() - t0

        # THE ASSERTIONS. A cache read is not a build.
        if system.tree_cache_hit is None:
            _fail(f"{benchmark_id}: index() never recorded a cache verdict")
        if system.tree_cache_hit:
            _fail(
                f"{benchmark_id}: substrate was served WARM, so {build_s:.1f}s "
                "is a cache read, not a build. Clear the cache dir for this "
                "corpus, or the cold-tree lever did not take."
            )
        stats = dict(system.index_stats)
        n_leaves = int(stats.get("n_leaves", 0) or 0)
        if n_leaves <= 0:
            _fail(f"{benchmark_id}: build produced no leaves; nothing was measured")
        if int(stats.get("n_summary_calls_at_index", 0) or 0) <= 0:
            _fail(
                f"{benchmark_id}: zero summary calls, so no summariser work "
                "was timed. That is not a tree build."
            )
        if stats.get("degenerate_no_tree"):
            _fail(
                f"{benchmark_id}: FLAT INDEX — the corpus fell below the layer "
                "stop condition, so this is not a tree-build time and must not "
                "be extrapolated as one."
            )

        row = {
            "benchmark": benchmark_id,
            "corpus_id": unit.corpus_id,
            "n_corpus_items": len(unit.corpus),
            "n_queries_in_unit": len(unit.queries),
            "build_s": round(build_s, 2),
            "n_leaves": n_leaves,
            "n_summary_nodes": int(stats.get("flat_n_summaries", 0) or 0),
            "n_summary_calls": int(stats.get("n_summary_calls_at_index", 0) or 0),
            "layer_sizes": stats.get("layer_sizes"),
            "tree_cache_hit": system.tree_cache_hit,
        }
        rows.append(row)
        print(f"[probe] tree {benchmark_id:<16} build={build_s:8.2f}s  "
              f"items={row['n_corpus_items']:<6} leaves={n_leaves:<6} "
              f"summary_nodes={row['n_summary_nodes']:<5} "
              f"summary_calls={row['n_summary_calls']}")
    return rows


def probe_query_slice(system_ids: list[str], benchmark_id: str, n: int) -> list[dict]:
    """Timed slice: s_per_query net of indexing and of model load."""
    from src.config import DEFAULT_CONFIG
    from src.eval.runner import SYSTEM_REGISTRY

    rows: list[dict] = []
    bench, unit = _one_unit(benchmark_id)
    queries = list(unit.queries)[:n]
    if len(queries) < n:
        _fail(f"{benchmark_id}: only {len(queries)} queries available, wanted {n}")

    for system_id in system_ids:
        system = SYSTEM_REGISTRY[system_id](DEFAULT_CONFIG)

        t_index = time.perf_counter()
        system.index_items(unit.corpus)
        index_s = time.perf_counter() - t_index

        t0 = time.perf_counter()
        n_done = 0
        for q in queries:
            system.answer(q.question_text)
            n_done += 1
        answer_s = time.perf_counter() - t0

        if n_done != n:
            _fail(f"{system_id}: answered {n_done} of {n} queries")

        row = {
            "system": system_id,
            "benchmark": benchmark_id,
            "n": n_done,
            "index_s": round(index_s, 2),
            "answer_s": round(answer_s, 2),
            "s_per_query": round(answer_s / n_done, 4),
            "tree_cache_hit": getattr(system, "tree_cache_hit", None),
        }
        rows.append(row)
        print(f"[probe] queries {system_id:<4} {benchmark_id:<14} "
              f"index={index_s:7.1f}s  {row['s_per_query']:.3f} s/query "
              f"over n={n_done}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("tree", "queries", "both"), default="both")
    ap.add_argument("--n", type=int, default=50, help="queries in the timed slice")
    ap.add_argument("--tree-benchmarks", default="narrativeqa,multihop_rag,hotpotqa")
    ap.add_argument("--query-systems", default="M4,M9")
    ap.add_argument("--query-benchmark", default="multihop_rag")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from scripts.pin_environment import environment_provenance

    report: dict[str, Any] = {
        "environment": environment_provenance(Path("requirements.lock")),
        "prewarm_s": None,
        "tree_builds": [],
        "query_slices": [],
    }
    print(f"[probe] gpu={report['environment'].get('gpu')}")

    report["prewarm_s"] = round(_prewarm(), 1)

    if args.mode in ("tree", "both"):
        report["tree_builds"] = probe_tree_builds(
            [b.strip() for b in args.tree_benchmarks.split(",") if b.strip()])
    if args.mode in ("queries", "both"):
        report["query_slices"] = probe_query_slice(
            [s.strip() for s in args.query_systems.split(",") if s.strip()],
            args.query_benchmark, args.n)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[probe] wrote {args.out}")
    print()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
