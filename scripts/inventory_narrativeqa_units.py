"""Leaf count for every story in the NarrativeQA draw. CPU only.

WHY THIS EXISTS. M4's NarrativeQA cell is 40 separate per-unit tree
builds and the draw spans 37x in story size, so the cell cannot be
projected by multiplying one measured story by forty. It has to be a SUM
over units. Two measured builds anchor a line in leaves
(`project_matrix_cost.estimate_cell_builds`) and this supplies the leaf
count for every unit so the sum has something to sum over.

It also names the two stories worth measuring: the LARGEST, which is the
only unit that could individually threaten a session and whose time is
worth knowing on its own, and one near the MEDIAN, which anchors the
other end of the line where most of the mass sits.

NO GPU, NO GENERATION, NO CACHE WRITES. It chunks with the same
`split_text_raptor` the build uses — the leaf count IS the chunk count —
and touches nothing else. Safe to run in a CPU session while a GPU
session does real work.

USAGE

    python -m scripts.inventory_narrativeqa_units --out /content/nqa_units.json

The JSON drops straight into a measurements file as
`m4_leaves_per_unit.narrativeqa`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from src.eval.narrativeqa import CELL_UNITS


def leaf_count(text: str) -> int:
    """Leaves the BUILD would produce for one document.

    NOT `split_text_raptor(item.text)`. The build does not chunk the raw
    item text: `index_items` writes each parent to a temp file and reads
    it back through `walk_corpus` -> `extract_text` -> `parsing.clean_text`,
    which collapses runs of spaces and tabs to a single space, and runs
    of three or more newlines to two. Because `split_text_raptor` treats
    a newline RUN as a boundary and folds a lone newline to a space, that
    cleaning moves chunk boundaries.

    MEASURED DIVERGENCE, and it is why this function exists: story
    961902ae counted 519 leaves raw against 481 in the build, while
    57523a48 and d431326b matched exactly. Story-dependent, because it
    depends on how much whitespace the text happens to carry — which is
    the worst shape of error, since two agreeing samples read as proof.

    The token budget comes from the configured chunker rather than the
    function default, for the same reason: an inventory that hardcodes
    100 silently stops describing the build the moment the config moves.
    """
    from src.config import DEFAULT_CONFIG
    from src.parsing import clean_text
    from src.raptor_paper import split_text_raptor

    chunker = DEFAULT_CONFIG.m4.chunker
    max_tokens = getattr(chunker, "chunk_words", 100) if chunker else 100
    return len(split_text_raptor(clean_text(text), max_tokens=max_tokens))


def inventory(max_units: int | None = CELL_UNITS) -> dict[str, Any]:
    """Leaf counts for the POPULATION THE CELL BUILDS — the seeded draw.

    THE BUG THIS SIGNATURE FIXES. The default used to be None, which
    makes `iter_eval_units` yield the FULL 115-story validation split
    instead of the 40-story draw a cell runs. Every number the inventory
    produced was then correct about the wrong population, and a cell
    projected from it would have overstated the build term by ~3x.

    `max_units` is not a cap, it is part of the DRAW:
    `subsample_indices(115, n)` selects a different SET for each n, so
    `subsample_indices(115, 1)` is not the first story of
    `subsample_indices(115, 40)`. Passing a different number here does
    not narrow the sample, it takes a different one — which is why the
    default is the cell's own constant rather than None.
    """
    from src.eval.runner import BENCHMARK_REGISTRY

    bench = BENCHMARK_REGISTRY["narrativeqa"]()
    units = list(bench.iter_eval_units(split="validation",
                                       max_units=max_units))
    if not units:
        raise RuntimeError("narrativeqa loader yielded no unit")

    # ASSERT THE POPULATION, because the loader's own header printing
    # "115 stories" beside a 40-story build log is exactly how this got
    # through the first time. A count that disagrees with the request is
    # a wrong inventory, not a note.
    if max_units is not None and len(units) != max_units:
        raise RuntimeError(
            f"asked the loader for {max_units} units and it yielded "
            f"{len(units)}. This inventory would describe a population "
            "the cell does not build; refusing to report it."
        )

    rows = []
    for i, unit in enumerate(units, 1):
        chars = sum(len(item.text) for item in unit.corpus)
        leaves = sum(leaf_count(item.text) for item in unit.corpus)
        rows.append({
            "corpus_id": str(unit.corpus_id),
            "n_corpus_items": len(unit.corpus),
            "n_queries": len(unit.queries),
            "chars": chars,
            "leaves": leaves,
        })
        print(f"[nqa] {i:>3}/{len(units)}  {unit.corpus_id}  "
              f"chars={chars:>9,}  leaves={leaves:>6,}")

    by_leaves = sorted(rows, key=lambda r: r["leaves"])
    leaves = [r["leaves"] for r in by_leaves]
    median_row = by_leaves[len(by_leaves) // 2]
    largest = by_leaves[-1]

    return {
        "population": "seeded cell draw" if max_units is not None
        else "FULL validation split (NOT the cell population)",
        "max_units_requested": max_units,
        "n_units": len(rows),
        "units": rows,
        "leaves_per_unit": [r["leaves"] for r in rows],
        "total_leaves": sum(leaves),
        "min_leaves": leaves[0],
        "median_leaves": statistics.median(leaves),
        "max_leaves": leaves[-1],
        "spread_max_over_min": round(leaves[-1] / max(1, leaves[0]), 1),
        # The two stories to measure, named rather than left to judgement.
        "measure_these": {
            "largest": {"corpus_id": largest["corpus_id"],
                        "leaves": largest["leaves"]},
            "median": {"corpus_id": median_row["corpus_id"],
                       "leaves": median_row["leaves"]},
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-units", type=int, default=CELL_UNITS,
                    help=f"Units to draw. DEFAULTS TO {CELL_UNITS}, the "
                         "cell's own seeded draw. This is NOT a cap: "
                         "subsample_indices(115, n) picks a different SET "
                         "for each n, so changing it inventories different "
                         "stories, not fewer of the same ones. Pass 0 for "
                         "the full 115-story split as CONTEXT ONLY - those "
                         "numbers do not describe any cell.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    report = inventory(args.max_units or None)

    print()
    print(f"[nqa] population: {report['population']}")
    print(f"[nqa] {report['n_units']} units, "
          f"{report['total_leaves']:,} leaves total")
    print(f"[nqa] leaves min={report['min_leaves']:,} "
          f"median={report['median_leaves']:,} max={report['max_leaves']:,} "
          f"(spread {report['spread_max_over_min']}x)")
    m = report["measure_these"]
    print(f"[nqa] MEASURE THESE TWO:")
    print(f"[nqa]   largest: {m['largest']['corpus_id']} "
          f"({m['largest']['leaves']:,} leaves)")
    print(f"[nqa]   median : {m['median']['corpus_id']} "
          f"({m['median']['leaves']:,} leaves)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[nqa] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
