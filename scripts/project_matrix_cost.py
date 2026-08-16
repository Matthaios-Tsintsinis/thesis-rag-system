"""Project the 20-cell wall time from MEASURED inputs. No guessing.

WHY THIS IS A SCRIPT AND NOT A SPREADSHEET. Every earlier estimate in
this project was scaled from one number and was wrong by 33x, because the
number it scaled from was measured under a defect. This takes each input
explicitly, refuses to invent a missing one, and marks an M4 cell whose
build was never measured as UNMEASURED rather than projecting zero — a
zero build term reads as "no build needed", which for M4 is the most
expensive possible misreading.

WHAT IT COMPUTES, per cell:

    total_s = build_s + s_per_query x n_queries

`build_s` is zero for M1/M2/M3/M9, which index cheaply, and for M4 is the
SUM over units of that unit's measured tree build. Summed, never averaged:
story sizes across the NarrativeQA draw span 37x, and an average would
both understate the cell and hide the single unit most likely to overrun
a session.

`n_queries` must come from the loader, not from a literal — P8's rule.
The CLI reads it from a measurements JSON so the caller can derive it
however the loader does.

USAGE

    python -m scripts.project_matrix_cost --measurements /content/measured.json

where the JSON is, at minimum:

    {
      "s_per_query": {"M1": 0.5, "M2": 3.7, "M3": 4.0, "M4": 5.0, "M9": 5.8},
      "n_queries": {"multihop_rag": 2556, "narrativeqa": 1208,
                    "hotpotqa": 1000, "hotpotqa_pooled": 1000},
      "m4_build_s_per_unit": {"narrativeqa": [82.4, 91.0, ...]}
    }

`s_per_query` comes from `probe_cell_costs --mode queries`;
`m4_build_s_per_unit` from `--mode tree` or a real build's per-unit log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SYSTEMS = ("M1", "M2", "M3", "M4", "M9")
BENCHMARKS = ("multihop_rag", "narrativeqa", "hotpotqa", "hotpotqa_pooled")

# Systems that pay an index-time tree build. M2/M3/M9 build embedding
# substrates too, but those are cheap and already inside s_per_query's
# measured envelope for a warm cell; M4's cold tree is not.
TREE_SYSTEMS = ("M4",)

# The Colab session a cell has to fit inside, in seconds. ~5 hours.
SESSION_GUARD_S = 5 * 3600

# A cell above this fraction of the guard is flagged: it leaves too little
# room for the variance a real session carries (reconnects, a slow unit,
# a Drive stall).
WARN_FRACTION = 0.60


def project_cell(
    *,
    system: str,
    benchmark: str,
    n_queries: int,
    s_per_query: float | None,
    build_s_per_unit: list[float] | None = None,
) -> dict:
    """One cell. Raises rather than defaulting on a missing input."""
    if s_per_query is None:
        raise ValueError(
            f"{system}/{benchmark}: no measured s_per_query. Run "
            "`probe_cell_costs --mode queries` for this system; a default "
            "here would silently become a session plan."
        )
    if not n_queries:
        raise ValueError(
            f"{system}/{benchmark}: n_queries is {n_queries!r}. A cell with "
            "no queries is a configuration error, not a free cell."
        )

    builds = list(build_s_per_unit or [])
    build_s = float(sum(builds))
    query_s = float(s_per_query) * int(n_queries)
    total_s = build_s + query_s
    return {
        "system": system,
        "benchmark": benchmark,
        "n_queries": int(n_queries),
        "s_per_query": float(s_per_query),
        "n_units_with_measured_build": len(builds),
        "build_s": build_s,
        "max_unit_build_s": max(builds) if builds else 0.0,
        "query_s": query_s,
        "total_s": total_s,
        "total_h": round(total_s / 3600, 2),
        "fraction_of_session": round(total_s / SESSION_GUARD_S, 3),
        "over_warn_fraction": total_s > SESSION_GUARD_S * WARN_FRACTION,
        # Categorically worse than merely large: this cell cannot finish
        # in one session and needs --resume planning, not a warning.
        "exceeds_session": total_s > SESSION_GUARD_S,
        # A tree system with no measured build is NOT a zero-build cell.
        "build_unmeasured": system in TREE_SYSTEMS and not builds,
    }


def project_matrix(
    *,
    s_per_query: dict[str, float],
    n_queries: dict[str, int],
    m4_build_s_per_unit: dict[str, list[float]] | None = None,
) -> dict:
    """All 20 cells, plus the rollup and the flagged list."""
    missing_sys = [s for s in SYSTEMS if s_per_query.get(s) is None]
    if missing_sys:
        raise ValueError(
            f"no measured s_per_query for: {', '.join(missing_sys)}. "
            "Every system in the matrix needs a timed slice."
        )
    missing_bench = [b for b in BENCHMARKS if not n_queries.get(b)]
    if missing_bench:
        raise ValueError(
            f"no loader-derived query count for: {', '.join(missing_bench)}"
        )

    builds = m4_build_s_per_unit or {}
    cells = [
        project_cell(
            system=system,
            benchmark=benchmark,
            n_queries=n_queries[benchmark],
            s_per_query=s_per_query[system],
            build_s_per_unit=(
                builds.get(benchmark) if system in TREE_SYSTEMS else None
            ),
        )
        for system in SYSTEMS
        for benchmark in BENCHMARKS
    ]
    total_s = sum(c["total_s"] for c in cells)
    return {
        "cells": cells,
        "total_s": total_s,
        "total_h": round(total_s / 3600, 2),
        "session_guard_s": SESSION_GUARD_S,
        "warn_fraction": WARN_FRACTION,
        "flagged": [c for c in cells if c["over_warn_fraction"]],
        "exceeding": [c for c in cells if c["exceeds_session"]],
        "unmeasured_builds": [c for c in cells if c["build_unmeasured"]],
        "n_sessions_at_guard": round(total_s / SESSION_GUARD_S, 2),
    }


def _render(m: dict) -> str:
    rows = [
        f"{'system':<6} {'benchmark':<18} {'build_h':>8} {'query_h':>8} "
        f"{'total_h':>8} {'frac':>6}  flags",
        "-" * 74,
    ]
    for c in m["cells"]:
        flags = []
        if c["exceeds_session"]:
            flags.append("EXCEEDS-SESSION")
        elif c["over_warn_fraction"]:
            flags.append("over-60%")
        if c["build_unmeasured"]:
            flags.append("BUILD-UNMEASURED")
        rows.append(
            f"{c['system']:<6} {c['benchmark']:<18} "
            f"{c['build_s'] / 3600:>8.2f} {c['query_s'] / 3600:>8.2f} "
            f"{c['total_h']:>8.2f} {c['fraction_of_session']:>6.2f}  "
            f"{', '.join(flags)}"
        )
    rows.append("-" * 74)
    rows.append(
        f"TOTAL {m['total_h']:.2f} h = {m['n_sessions_at_guard']} sessions "
        f"at the {SESSION_GUARD_S / 3600:.0f} h guard"
    )
    if m["unmeasured_builds"]:
        rows.append(
            f"WARNING: {len(m['unmeasured_builds'])} M4 cell(s) carry NO "
            "measured tree build. Their totals are query time only and "
            "UNDERSTATE the cell."
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurements", type=Path, required=True,
                    help="JSON with s_per_query, n_queries, and optionally "
                         "m4_build_s_per_unit")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    data: dict[str, Any] = json.loads(
        args.measurements.read_text(encoding="utf-8")
    )
    m = project_matrix(
        s_per_query=data["s_per_query"],
        n_queries=data["n_queries"],
        m4_build_s_per_unit=data.get("m4_build_s_per_unit"),
    )
    print(_render(m))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
