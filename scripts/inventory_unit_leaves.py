"""Leaf count per EvalUnit, for any benchmark. CPU only, no GPU.

WHY IT EXISTS. A cell's tree-build cost is a SUM over units, and unit
sizes are wildly uneven — 37x across the NarrativeQA draw. This supplies
the per-unit leaf count that `project_matrix_cost.estimate_cell_builds`
sums over, and it reports the distribution, which is what a heterogeneous
cell needs before anyone can declare what it contains.

IT COUNTS WHAT THE BUILD COUNTS, by calling the build's own functions.
Two transforms sit between a CorpusItem and a leaf, and BOTH were got
wrong here before:

  1. `parsing.clean_text` collapses runs of spaces/tabs and runs of 3+
     newlines. `split_text_raptor` treats a newline run as a boundary, so
     the cleaning MOVES chunk boundaries. Measured: story 961902ae
     counted 519 leaves raw against 481 in the build.
  2. `index_items` GROUPS ITEMS BY PARENT and joins each parent's members
     into ONE document (`build_parent_payload`), then chunks that. Summing
     per-item counts coincides only when a unit has exactly one item —
     true of NarrativeQA, FALSE of HotpotQA-distractor, whose ~10
     paragraphs per question can pack across item boundaries.

So this calls `group_items_by_parent` and `build_parent_payload`
directly. Re-deriving either would reintroduce the drift the moment one
of them changed.

THE DEGENERACY THRESHOLD is read from the config, not hardcoded: RAPTOR
stops when `len(current) <= reduction_dimension + 1`, so a unit at or
below that builds NO tree and runs as flat dense retrieval.

USAGE

    python -m scripts.inventory_unit_leaves --benchmark hotpotqa \\
        --out /content/hotpot_units.json
    python -m scripts.inventory_unit_leaves --benchmark narrativeqa \\
        --out /content/nqa_units.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def degeneracy_threshold() -> int:
    """Leaves at or below which RAPTOR builds no tree.

    `len(current) <= reduction_dimension + 1`, checked BEFORE the first
    clustering pass, so a unit at the threshold breaks on iteration 0.
    """
    from src.config import DEFAULT_CONFIG

    return int(DEFAULT_CONFIG.m4.paper.reduction_dimension) + 1


def unit_leaf_count(unit) -> int:  # noqa: ANN001
    """Leaves the BUILD would produce for one EvalUnit.

    Groups by parent and builds each parent's payload with the SAME
    functions `index_items` uses, then chunks each payload. Not a sum of
    per-item counts: a parent's members are joined into one document
    first, and chunks can pack across member boundaries.
    """
    from src.config import DEFAULT_CONFIG
    from src.raptor_paper import split_text_raptor
    from src.retrievers.base import build_parent_payload, group_items_by_parent

    chunker = DEFAULT_CONFIG.m4.chunker
    max_tokens = getattr(chunker, "chunk_words", 100) if chunker else 100

    total = 0
    for members in group_items_by_parent(list(unit.corpus)).values():
        payload, _spans = build_parent_payload(members)
        if not payload:
            continue
        total += len(split_text_raptor(payload, max_tokens=max_tokens))
    return total


def inventory(benchmark_id: str, max_units: int | None = None) -> dict[str, Any]:
    """Per-unit leaf counts for the POPULATION THE CELL BUILDS.

    `max_units` defaults to the benchmark's declared `cell_units`, so the
    inventory describes the cell rather than the whole split. For
    NarrativeQA that is load-bearing — its draw is seeded and a different
    n selects a different SET, not a subset.
    """
    from src.eval.runner import BENCHMARK_REGISTRY

    bench = BENCHMARK_REGISTRY[benchmark_id]()
    declared = getattr(bench, "cell_units", None)
    requested = declared if max_units is None else max_units

    if requested is None:
        # Same principle as the runner's population gate: an undeclared
        # population is the condition that let the 115-vs-40 inventory
        # bug through, so it is announced rather than assumed.
        print(f"[inv] WARNING: {benchmark_id} declares no cell_units and no "
              "--max-units was given. Enumerating whatever the loader "
              "yields; this may not be the population a cell builds.")

    units = list(bench.iter_eval_units(split="validation",
                                       max_units=requested))
    if not units:
        raise RuntimeError(f"{benchmark_id}: loader yielded no unit")
    if requested is not None and len(units) != requested:
        raise RuntimeError(
            f"{benchmark_id}: asked for {requested} units, got {len(units)}. "
            "This inventory would describe a population the cell does not "
            "build; refusing to report it."
        )

    threshold = degeneracy_threshold()
    rows = []
    for i, unit in enumerate(units, 1):
        leaves = unit_leaf_count(unit)
        rows.append({
            "corpus_id": str(unit.corpus_id),
            "n_corpus_items": len(unit.corpus),
            "n_queries": len(unit.queries),
            "chars": sum(len(it.text) for it in unit.corpus),
            "leaves": leaves,
            "degenerate": leaves <= threshold,
        })
        if i % 100 == 0 or len(units) <= 50:
            print(f"[inv] {i:>5}/{len(units)}  {unit.corpus_id}  "
                  f"leaves={leaves:>6,}"
                  f"{'  DEGENERATE' if leaves <= threshold else ''}")

    leaves = sorted(r["leaves"] for r in rows)
    n_degenerate = sum(1 for r in rows if r["degenerate"])
    by_leaves = sorted(rows, key=lambda r: r["leaves"])
    median_row = by_leaves[len(by_leaves) // 2]

    return {
        "benchmark": benchmark_id,
        "population": "declared cell_units" if max_units is None
        else f"explicit max_units={max_units}",
        "n_units": len(rows),
        "degeneracy_threshold": threshold,
        # THE NUMBER THE DECLARATION TURNS ON.
        "n_degenerate": n_degenerate,
        "fraction_degenerate": round(n_degenerate / len(rows), 4),
        "n_with_tree": len(rows) - n_degenerate,
        "units": rows,
        "leaves_per_unit": [r["leaves"] for r in rows],
        "total_leaves": sum(leaves),
        "min_leaves": leaves[0],
        "p25_leaves": leaves[len(leaves) // 4],
        "median_leaves": statistics.median(leaves),
        "p75_leaves": leaves[(3 * len(leaves)) // 4],
        "max_leaves": leaves[-1],
        "spread_max_over_min": round(leaves[-1] / max(1, leaves[0]), 1),
        "measure_these": {
            "largest": {"corpus_id": by_leaves[-1]["corpus_id"],
                        "leaves": by_leaves[-1]["leaves"]},
            "median": {"corpus_id": median_row["corpus_id"],
                       "leaves": median_row["leaves"]},
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="narrativeqa")
    ap.add_argument("--max-units", type=int, default=None,
                    help="Override the declared cell_units. NOT a plain cap "
                         "for NarrativeQA, whose seeded draw selects a "
                         "different SET for each n.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    report = inventory(args.benchmark, args.max_units)

    print()
    print(f"[inv] {report['benchmark']}: {report['n_units']} units, "
          f"{report['total_leaves']:,} leaves total")
    print(f"[inv] leaves min={report['min_leaves']:,} "
          f"p25={report['p25_leaves']:,} median={report['median_leaves']:,} "
          f"p75={report['p75_leaves']:,} max={report['max_leaves']:,}")
    print(f"[inv] DEGENERACY (<= {report['degeneracy_threshold']} leaves, "
          f"no tree, flat dense): {report['n_degenerate']}/"
          f"{report['n_units']} units = "
          f"{report['fraction_degenerate']:.1%}")
    print(f"[inv]   with a tree: {report['n_with_tree']} units")
    m = report["measure_these"]
    print(f"[inv] largest: {m['largest']['corpus_id']} "
          f"({m['largest']['leaves']:,} leaves)")
    print(f"[inv] median : {m['median']['corpus_id']} "
          f"({m['median']['leaves']:,} leaves)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[inv] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
