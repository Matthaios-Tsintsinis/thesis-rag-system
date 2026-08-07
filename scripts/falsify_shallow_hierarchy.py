"""Falsification test for docs/FINDING_SHALLOW_HIERARCHY.md. Read-only.

⚠ THIS TEST WAS MIS-SPECIFIED IN ITS FIRST FORM AND RETURNED A FALSE
FALSIFICATION. Read docs/FINDING_SHALLOW_HIERARCHY.md §0 first.

The tier labels are INVERTED between the two M4 eras. Old top-down M4
maps depth 0-1 (the BROADEST, shallowest nodes) to `summary_high`; new
bottom-up M4 maps the TOP layer to `summary_high`. So `summary_high` in
a pre-rebuild cell is evidence FOR a shallow tree, and the original
prediction read it as evidence against. This script now REFUSES to judge
pre-rebuild rows rather than inverting them silently -- an era-detection
bug that produced a plausible verdict is exactly what it exists to
prevent.

THE ACCOUNT. On document-scale corpora a faithful RAPTOR build produces
exactly ONE summary layer: ~70-90 leaves cluster to ~10-14 parents, which
already satisfies the stop condition (`layer <= reduction_dimension + 1`
= 11), so the build halts. `_layer_to_unit_type` maps layer 1 to
`summary_low`, and layers 2 and 3+ to `summary_mid` / `summary_high`.

THE PREDICTION. If the account holds, no QASPER or QuALITY row can carry
a `summary_mid` or `summary_high` retrieved unit, because those layers
were never built. A single one falsifies it.

WHY THIS IS WORTH RUNNING. It costs nothing -- the evidence is already
banked in `retrieved_unit_types`, which every row has carried since CK-4
-- and it either strengthens the account or kills it BEFORE it reaches a
thesis chapter.

    python -m scripts.falsify_shallow_hierarchy <dir-or-glob> [...]
    python -m scripts.falsify_shallow_hierarchy   # defaults to OUTPUT_DIR/eval

THE VACUOUS PASS IS TREATED AS A FAILURE. If no eligible cell is found,
or every eligible cell retrieved zero summary units of any kind, this
EXITS NON-ZERO rather than reporting "no violations". A test that cannot
observe the thing it is testing has not passed -- that mistake has
already cost this project three measurements.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

# Benchmarks whose EvalUnit is ONE document, i.e. where the account
# predicts a single summary layer. MultiHop (609-article shared corpus)
# and NarrativeQA (~62.5k-token books) are excluded on purpose: they are
# large enough to build a real hierarchy, and finding summary_mid there
# CONFIRMS the account rather than falsifying it.
PER_DOCUMENT = ("qasper", "quality")
DEEP_TIERS = ("summary_mid", "summary_high")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", default=[])
    args = ap.parse_args()

    specs = args.inputs
    if not specs:
        from src import paths

        specs = [str(paths.output_dir() / "eval")]

    files: list[Path] = []
    for spec in specs:
        p = Path(spec)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        else:
            files.extend(Path(x) for x in glob.glob(spec))
    files = [f for f in files if f.is_file()]

    # (benchmark, system) -> unit-type totals
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    pre_rebuild: set[tuple[str, str]] = set()
    n_rows = 0
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bench = (r.get("benchmark") or "").lower()
            if bench not in PER_DOCUMENT:
                continue
            sid = r.get("system_id", "?")
            md = r.get("metadata") or {}
            # ERA GATE. The m4_* diagnostics landed with the rebuilt M4
            # (commit 5). An M4 row without them predates the rebuild, so
            # its tier labels carry the OPPOSITE orientation and the
            # prediction below does not apply to it.
            if sid == "M4" and "m4_tree_degenerate" not in md:
                pre_rebuild.add((bench, sid))
                continue
            n_rows += 1
            key = (bench, sid)
            for ut, n in (r.get("retrieved_unit_types") or {}).items():
                cells[key][ut] += int(n)

    print(f"scanned {len(files)} files, {n_rows} eligible rows from "
          f"{PER_DOCUMENT}")
    if pre_rebuild:
        print("")
        print("SKIPPED as PRE-REBUILD M4 (inverted tier labels; the "
              "prediction does not apply):")
        for bench, sid in sorted(pre_rebuild):
            print(f"  {bench}/{sid}")
        print("  -> see docs/FINDING_SHALLOW_HIERARCHY.md §0 and §7. Those "
              "cells are stale anyway (their substrate key moved).")
    if not cells:
        print("\nINCONCLUSIVE: no QASPER or QuALITY rows found. The test "
              "did not run.\nPoint it at the banked cells, e.g.:\n"
              "  python -m scripts.falsify_shallow_hierarchy "
              "/content/drive/MyDrive/thesis_rag/outputs/matrix_baseline")
        return 2

    violations: list[str] = []
    observed_any_summary = False
    print(f"\n{'benchmark':<10} {'system':<7} {'chunk':>8} {'s_low':>8} "
          f"{'s_mid':>8} {'s_high':>8}  verdict")
    print("-" * 68)
    for (bench, sid), counts in sorted(cells.items()):
        low = counts.get("summary_low", 0)
        mid = counts.get("summary_mid", 0)
        high = counts.get("summary_high", 0)
        if low or mid or high:
            observed_any_summary = True
        deep = mid + high
        if deep:
            verdict = "*** FALSIFIES ***"
            violations.append(f"{bench}/{sid}: {mid} summary_mid, "
                              f"{high} summary_high")
        elif low:
            verdict = "consistent"
        else:
            verdict = "no summary units (uninformative)"
        print(f"{bench:<10} {sid:<7} {counts.get('chunk', 0):>8} {low:>8} "
              f"{mid:>8} {high:>8}  {verdict}")

    print()
    if violations:
        print("ACCOUNT FALSIFIED. Deeper summary layers were retrieved on a "
              "per-document benchmark:")
        for v in violations:
            print(f"  {v}")
        print("\ndocs/FINDING_SHALLOW_HIERARCHY.md is WRONG as written and "
              "must be corrected before it goes any further.")
        return 1

    if not observed_any_summary:
        # The trap this script exists to avoid. Zero summary units
        # everywhere is consistent with the prediction and ALSO with
        # having scanned rows from a system that never retrieves summary
        # nodes at all (M1/M2/M3/M9). That is not evidence.
        print("INCONCLUSIVE: no summary units of ANY tier were retrieved in "
              "the rows scanned, so the prediction was never actually "
              "tested. Include M4 (and M7 if present) cells.")
        return 2

    print("ACCOUNT SURVIVES. Every per-document cell that retrieved summary "
          "units retrieved ONLY summary_low, i.e. layer 1 -- exactly what a "
          "one-summary-layer tree can produce.")
    print("Note this is corroboration, not proof: it is also what a run "
          "would show if deeper nodes existed but never ranked top-k.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
