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
      "n_queries": {"multihop_rag": 2556, "narrativeqa": 1230,
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


def estimate_build_s_from_leaves(
    points: list[tuple[float, float]],
    leaves: float,
) -> tuple[float, bool]:
    """Estimate one unit's tree-build seconds from its leaf count.

    WHY TWO MEASURED STORIES AND NOT ONE. The NarrativeQA draw spans 37x
    in story size, so a single sample cannot project the cell — it can
    only be multiplied, and a multiply against a story no unit resembles
    is how a session plan goes wrong. Two points anchor a line and every
    other story INTERPOLATES between them.

    WHY LINEAR IN LEAVES. Build time is `n_calls x per_call`, `n_calls ~
    n_summary_nodes / batch_width`, and summary nodes scale with leaves.
    Per-call cost is flat across widths (measured), so the whole thing is
    near-linear in leaves. That is a MODEL, not a measurement, which is
    why every value derived from it is marked.

    Returns `(seconds, extrapolated)`. `extrapolated` is True outside the
    measured range, where the line is no longer anchored on both sides.
    """
    xs = sorted({float(x) for x, _ in points})
    if len(points) < 2 or len(xs) < 2:
        raise ValueError(
            "need at least TWO measured stories with DIFFERENT leaf counts: "
            "one point defines no slope, and two identical ones define no "
            "line. Measure the largest story and one near the median."
        )

    # Least squares through however many points were supplied; with the
    # expected two, this is exactly the line through them.
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    intercept = mean_y - slope * mean_x

    est = intercept + slope * float(leaves)
    extrapolated = not (min(xs) <= float(leaves) <= max(xs))
    if est < 0:
        # A fitted intercept can go negative below the measured range. A
        # negative build time is not a projection, it is a bug leaking
        # into a session plan.
        est = 0.0
        extrapolated = True
    return est, extrapolated


def estimate_cell_builds(
    points: list[tuple[float, float]],
    leaves_per_unit: list[float],
) -> tuple[list[float], int]:
    """Per-unit build seconds for a whole cell, one entry per unit.

    Returns the list plus how many entries fell outside the measured
    range. The list feeds `build_s_per_unit`, which the projector SUMS —
    so the cell stays an addition over real units rather than a mean
    multiplied by a count, which is the property the 37x spread makes
    load-bearing.
    """
    out: list[float] = []
    n_extrapolated = 0
    for leaves in leaves_per_unit:
        est, extrapolated = estimate_build_s_from_leaves(points, leaves)
        out.append(est)
        n_extrapolated += int(extrapolated)
    return out, n_extrapolated


def project_cell(
    *,
    system: str,
    benchmark: str,
    n_queries: int,
    s_per_query: float | None,
    build_s_per_unit: list[float] | None = None,
    s_per_query_source: str | None = None,
) -> dict:
    """One cell. Raises rather than defaulting on a missing input.

    `build_s_per_unit` distinguishes **None** (not measured) from **[]**
    (measured, and there is no build). M4 on HotpotQA-distractor has NO
    TREE — ~10 leaves per question falls below the layer stop condition,
    so it degenerates to flat retrieval — and reporting that as
    BUILD-UNMEASURED would send someone to measure a thing that does not
    exist.

    `s_per_query_source` names a DIFFERENT benchmark when the rate was
    borrowed rather than measured here, so the cell can carry the fact.
    """
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

    measured_build = build_s_per_unit is not None
    builds = list(build_s_per_unit or [])
    build_s = float(sum(builds))
    query_s = float(s_per_query) * int(n_queries)
    total_s = build_s + query_s

    # PER-UNIT CHECKPOINTING CHANGES WHAT AN OVERRUN COSTS, and that
    # drives session packing more than the raw percentage does.
    # `index_items` runs once per EvalUnit and flushes each story's tree
    # to its own cache dir, manifest written last, before any query for
    # that story is answered. A multi-unit M4 cell that dies mid-build
    # therefore loses ONE story; a single-unit cell loses the whole build.
    # MultiHop is one shared corpus, so its M4 build has no granularity to
    # fall back on. A cell at 90% of a session WITH checkpointing is safer
    # than one at 70% without it.
    checkpointed = system in TREE_SYSTEMS and len(builds) > 1
    fraction = total_s / SESSION_GUARD_S
    # The penalty applies ONLY where there is a build to lose. A
    # query-only cell has nothing at risk beyond one batch — answers are
    # flushed per batch and `--resume` skips what is banked — so charging
    # it here would rank M9/MultiHop, which builds no tree at all, above
    # the M4 cell whose whole hour of tree work is genuinely exposed.
    # That is the inversion this column exists to prevent.
    at_risk = build_s > 0 and not checkpointed
    risk_rank = round(fraction + (0.5 if at_risk else 0.0), 3)

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
        "fraction_of_session": round(fraction, 3),
        # What an interrupted build costs: one unit, or all of it.
        "build_checkpointed": checkpointed,
        "build_loss_on_interrupt_s": (
            max(builds) if (checkpointed and builds) else build_s
        ),
        "risk_rank": risk_rank,
        "over_warn_fraction": total_s > SESSION_GUARD_S * WARN_FRACTION,
        # Categorically worse than merely large: this cell cannot finish
        # in one session and needs --resume planning, not a warning.
        "exceeds_session": total_s > SESSION_GUARD_S,
        # A tree system with no measured build is NOT a zero-build cell.
        # An EMPTY list is a measured zero (no tree here) and is not
        # flagged.
        "build_unmeasured": system in TREE_SYSTEMS and not measured_build,
        "s_per_query_source": s_per_query_source,
        "s_per_query_extrapolated": bool(s_per_query_source)
        and s_per_query_source != benchmark,
    }


def _rate_table(s_per_query: dict) -> dict[str, dict[str, float]]:
    """Accept either {system: rate} or {benchmark: {system: rate}}.

    The flat form is legacy and means "one rate everywhere", which is the
    assumption this function exists to make visible: rates do NOT transfer
    across benchmarks. M4 measured 1.920 s/query on MultiHop — the fastest
    of the five, because its 2,000-token budget means less to read — and
    that ratio cannot carry to NarrativeQA, where it builds 40 per-unit
    trees against MultiHop's single shared corpus.
    """
    if s_per_query and all(
        isinstance(v, dict) for v in s_per_query.values()
    ):
        return {k: dict(v) for k, v in s_per_query.items()}
    return {b: dict(s_per_query) for b in BENCHMARKS}


def project_matrix(
    *,
    s_per_query: dict,
    n_queries: dict[str, int],
    m4_build_s_per_unit: dict[str, list[float]] | None = None,
    extrapolate_from: str | None = None,
) -> dict:
    """All 20 cells, plus the rollup and the flagged lists.

    `extrapolate_from` names a benchmark whose measured rates fill cells
    that have none. It is OPT-IN and every cell it touches is marked, so
    a borrowed rate can never be read as a measured one. Without it, a
    missing rate raises.
    """
    rates = _rate_table(s_per_query)
    missing_bench = [b for b in BENCHMARKS if not n_queries.get(b)]
    if missing_bench:
        raise ValueError(
            f"no loader-derived query count for: {', '.join(missing_bench)}"
        )

    if extrapolate_from and extrapolate_from not in rates:
        raise ValueError(
            f"extrapolate_from={extrapolate_from!r} has no measured rates"
        )

    gaps: list[str] = []
    for benchmark in BENCHMARKS:
        for system in SYSTEMS:
            if rates.get(benchmark, {}).get(system) is None and not (
                extrapolate_from
                and rates[extrapolate_from].get(system) is not None
            ):
                gaps.append(f"{system}/{benchmark}")
    if gaps:
        raise ValueError(
            f"no measured s_per_query for: {', '.join(gaps)}. Run the "
            "timed slice for those cells, or pass extrapolate_from=<a "
            "benchmark> to borrow its rates — every borrowed cell is "
            "marked, because rates do NOT transfer across benchmarks."
        )

    builds = m4_build_s_per_unit or {}
    cells = []
    for system in SYSTEMS:
        for benchmark in BENCHMARKS:
            rate = rates.get(benchmark, {}).get(system)
            source = benchmark
            if rate is None:
                rate = rates[extrapolate_from][system]
                source = extrapolate_from
            cells.append(project_cell(
                system=system,
                benchmark=benchmark,
                n_queries=n_queries[benchmark],
                s_per_query=rate,
                build_s_per_unit=(
                    builds.get(benchmark) if system in TREE_SYSTEMS else None
                ),
                s_per_query_source=source,
            ))
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
        # Packing order: an unprotected build outranks a larger protected
        # one, because the question is what an interrupted session COSTS,
        # not how long it takes.
        "by_risk": sorted(cells, key=lambda c: -c["risk_rank"]),
        "unprotected_builds": [
            c for c in cells if c["build_s"] and not c["build_checkpointed"]
        ],
        # Borrowed rates, listed so they cannot be read as measured.
        "extrapolated": [c for c in cells if c["s_per_query_extrapolated"]],
        "n_sessions_at_guard": round(total_s / SESSION_GUARD_S, 2),
    }


def _render(m: dict) -> str:
    rows = [
        f"{'system':<6} {'benchmark':<18} {'build_h':>8} {'query_h':>8} "
        f"{'total_h':>8} {'frac':>6} {'ckpt':>6}  flags",
        "-" * 86,
    ]
    for c in m["cells"]:
        flags = []
        if c["exceeds_session"]:
            flags.append("EXCEEDS-SESSION")
        elif c["over_warn_fraction"]:
            flags.append("over-60%")
        if c["build_unmeasured"]:
            flags.append("BUILD-UNMEASURED")
        # Only meaningful where there IS a build to lose.
        if c["build_s"] and not c["build_checkpointed"]:
            flags.append("BUILD-ALL-OR-NOTHING")
        if c["s_per_query_extrapolated"]:
            flags.append(f"RATE-FROM-{c['s_per_query_source']}")
        ckpt = (
            "per-unit" if c["build_checkpointed"]
            else ("whole" if c["build_s"] else "-")
        )
        rows.append(
            f"{c['system']:<6} {c['benchmark']:<18} "
            f"{c['build_s'] / 3600:>8.2f} {c['query_s'] / 3600:>8.2f} "
            f"{c['total_h']:>8.2f} {c['fraction_of_session']:>6.2f} "
            f"{ckpt:>6}  {', '.join(flags)}"
        )
    rows.append("-" * 86)
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
    ap.add_argument("--extrapolate-from", default=None,
                    help="Benchmark whose measured rates fill cells that "
                         "have none. OPT-IN, and every borrowed cell is "
                         "marked RATE-FROM-<benchmark>. Rates do NOT "
                         "transfer across benchmarks.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    data: dict[str, Any] = json.loads(
        args.measurements.read_text(encoding="utf-8")
    )
    m = project_matrix(
        s_per_query=data["s_per_query"],
        n_queries=data["n_queries"],
        m4_build_s_per_unit=data.get("m4_build_s_per_unit"),
        extrapolate_from=args.extrapolate_from or data.get("extrapolate_from"),
    )
    print(_render(m))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
