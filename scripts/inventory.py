"""P0 — structural inventory and integrity gate over a matrix output directory.

WHAT THIS REPORTS, AND WHAT IT DELIBERATELY DOES NOT. Per file: the
system and benchmark it claims to be, the row count, the distinct
`query_id` count, every id appearing more than once, the `answer.method`
values with counts, and the number of skipped-retrieval rows. Then a
coverage matrix over the expected cells.

**STRUCTURE ONLY. NO METRIC MEANS.** Not an oversight: the correction
plan's standing constraint is that the previous numbers are void and no
before/after comparison is produced anywhere. An inventory that printed
answer or retrieval means would invite exactly the comparison the plan
forbids, and would do it in the one document a reader opens first.

WHY IT IS ALSO A GATE. The same three properties this reports —
no duplicate `query_id`, every expected cell present, every row parseable
— are the preconditions P10's preflight has to check before a 20-cell run
starts. Rather than write that logic twice, this exits non-zero when any
of them fails, so `python -m scripts.inventory ... --require-complete`
is the preflight's cell-coverage step.

A NOTE ON `--require-complete`. Without it, missing cells are reported
but do not fail the run, because the first use of this script is on a
directory that is EXPECTED to be incomplete (the old, void outputs).
With it, a missing cell is an error. Duplicate ids and unparseable lines
always fail: those are corruption, not incompleteness.

    python -m scripts.inventory --dir <outputs/matrix> --out docs/INVENTORY.md
    python -m scripts.inventory --dir <outputs/matrix_v2> --require-complete
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# The corrected-run roster. M7 is DEFERRED to Phase 2 and M9 is
# REINSTATED — see docs/PREREGISTRATION.md ADDENDUM 5. Enumerating what
# is expected (rather than globbing whatever exists) is what makes a
# MISSING cell visible instead of silently absent.
EXPECTED_SYSTEMS = ("M1", "M2", "M3", "M4", "M9")
EXPECTED_BENCHMARKS = (
    "multihop_rag",
    "narrativeqa",
    "hotpotqa",
    "hotpotqa_pooled",
)


def _scan_file(path: Path) -> dict[str, Any]:
    """Structural facts about one JSONL cell. Reads no score values."""
    ids: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    systems: Counter[str] = Counter()
    benchmarks: Counter[str] = Counter()
    n_rows = 0
    n_skipped = 0
    n_unparseable = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_unparseable += 1
                continue
            qid = r.get("query_id")
            if qid is not None:
                ids[str(qid)] += 1
            systems[str(r.get("system_id", "?"))] += 1
            benchmarks[str(r.get("benchmark", "?"))] += 1
            methods[str((r.get("answer") or {}).get("method", "?"))] += 1
            if (r.get("retrieval") or {}).get("skipped"):
                n_skipped += 1

    return {
        "path": path,
        "n_rows": n_rows,
        "n_distinct_ids": len(ids),
        "duplicates": sorted(q for q, c in ids.items() if c > 1),
        "methods": dict(methods),
        "n_skipped": n_skipped,
        "n_unparseable": n_unparseable,
        # A cell whose rows disagree about their own identity is
        # corrupt in a way a filename cannot reveal.
        "systems": dict(systems),
        "benchmarks": dict(benchmarks),
    }


def _cell_identity(rec: dict) -> tuple[str | None, str | None]:
    """The (system, benchmark) a file's ROWS claim, not its filename.

    Filenames are a convention; the rows carry the provenance the runner
    stamped. Reading identity from the rows is what catches a cell
    written to the wrong path — the single destructive mistake available
    when a second model column shares an output root.
    """
    sysm = max(rec["systems"], key=rec["systems"].get) if rec["systems"] else None
    bench = max(rec["benchmarks"], key=rec["benchmarks"].get) if rec["benchmarks"] else None
    return sysm, bench


def build_report(directory: Path) -> tuple[list[dict], dict, list[str]]:
    files = sorted(p for p in directory.glob("*.jsonl") if p.is_file())
    records = [_scan_file(p) for p in files]

    found: dict[tuple[str, str], dict] = {}
    problems: list[str] = []

    for rec in records:
        sysm, bench = _cell_identity(rec)
        if len(rec["systems"]) > 1 or len(rec["benchmarks"]) > 1:
            problems.append(
                f"{rec['path'].name}: rows disagree about identity "
                f"(systems={rec['systems']}, benchmarks={rec['benchmarks']})"
            )
        if rec["duplicates"]:
            problems.append(
                f"{rec['path'].name}: {len(rec['duplicates'])} duplicate "
                f"query_id(s), first few: {rec['duplicates'][:5]}"
            )
        if rec["n_unparseable"]:
            problems.append(
                f"{rec['path'].name}: {rec['n_unparseable']} unparseable line(s)"
            )
        if sysm and bench:
            key = (bench, sysm)
            if key in found:
                problems.append(
                    f"two files claim the same cell {bench}/{sysm}: "
                    f"{found[key]['path'].name} and {rec['path'].name}"
                )
            found[key] = rec

    return records, found, problems


def render_markdown(directory: Path, records: list[dict], found: dict,
                    problems: list[str], missing: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# Banked output inventory")
    lines.append("")
    lines.append(f"Directory: `{directory}`")
    lines.append("")
    lines.append(
        "STRUCTURE ONLY — no metric means are reported here by design; "
        "the previous numbers are void and this document must not invite "
        "a comparison against them."
    )
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("| file | system | benchmark | rows | distinct ids | dup ids | skipped retr | unparseable |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in records:
        sysm, bench = _cell_identity(rec)
        lines.append(
            f"| `{rec['path'].name}` | {sysm or '?'} | {bench or '?'} | "
            f"{rec['n_rows']} | {rec['n_distinct_ids']} | "
            f"{len(rec['duplicates'])} | {rec['n_skipped']} | "
            f"{rec['n_unparseable']} |"
        )
    if not records:
        lines.append("| _(no .jsonl files found)_ | | | | | | | |")
    lines.append("")

    lines.append("## answer.method values per file")
    lines.append("")
    for rec in records:
        methods = ", ".join(f"`{k}` x{v}" for k, v in sorted(rec["methods"].items()))
        lines.append(f"- `{rec['path'].name}`: {methods or '_none_'}")
    lines.append("")

    lines.append("## Coverage matrix")
    lines.append("")
    lines.append("Expected roster: " + ", ".join(EXPECTED_SYSTEMS)
                 + " (M7 deferred to Phase 2 per ADDENDUM 5).")
    lines.append("")
    header = "| benchmark | " + " | ".join(EXPECTED_SYSTEMS) + " |"
    lines.append(header)
    lines.append("|---" * (len(EXPECTED_SYSTEMS) + 1) + "|")
    for bench in EXPECTED_BENCHMARKS:
        cells = []
        for sysm in EXPECTED_SYSTEMS:
            rec = found.get((bench, sysm))
            cells.append(f"{rec['n_rows']} rows" if rec else "**MISSING**")
        lines.append(f"| `{bench}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Missing cells")
    lines.append("")
    if missing:
        for bench, sysm in missing:
            lines.append(f"- **MISSING** `{bench}` x `{sysm}`")
    else:
        lines.append("_none — all expected cells present._")
    lines.append("")

    lines.append("## Integrity problems")
    lines.append("")
    if problems:
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("_none._")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path,
                    help="directory holding the banked {benchmark}_{system}.jsonl files")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the markdown report here (e.g. docs/INVENTORY.md)")
    ap.add_argument("--require-complete", action="store_true",
                    help="treat a missing expected cell as an error (P10 preflight)")
    args = ap.parse_args(argv)

    if not args.dir.is_dir():
        print(f"ERROR: not a directory: {args.dir}")
        return 2

    records, found, problems = build_report(args.dir)
    missing = [
        (b, s)
        for b in EXPECTED_BENCHMARKS
        for s in EXPECTED_SYSTEMS
        if (b, s) not in found
    ]

    md = render_markdown(args.dir, records, found, problems, missing)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(md)

    total_dupes = sum(len(r["duplicates"]) for r in records)
    print()
    print(f"[inventory] {len(records)} file(s), "
          f"{sum(r['n_rows'] for r in records)} row(s), "
          f"{len(found)}/{len(EXPECTED_BENCHMARKS) * len(EXPECTED_SYSTEMS)} "
          f"expected cells present")
    # THE ONE-LINE VERDICT the plan asks for.
    if total_dupes:
        print(f"[inventory] DUPLICATE query_id: YES — {total_dupes} id(s) "
              "appear more than once")
    else:
        print("[inventory] DUPLICATE query_id: NONE")

    fatal = bool(problems)
    if missing:
        print(f"[inventory] MISSING cells: {len(missing)} — "
              + ", ".join(f"{b}/{s}" for b, s in missing))
        if args.require_complete:
            fatal = True
    if fatal:
        print("[inventory] FAILED")
        return 1
    print("[inventory] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
