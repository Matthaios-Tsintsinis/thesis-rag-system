"""PROOF PROBE — does promoting the per-parent `index_items` layout from
`RaptorSystem` into `BaseSystem` move any banked substrate cache key?

WHY THIS EXISTS. `corpus_content_hash` hashes sorted (rel_path, bytes)
over the TEMP DIRECTORY `index_items` writes, so the temp-dir LAYOUT is a
cache-key input for every system that uses the base — M2, M3, M9 and the
FROZEN M7. Eight banked matrix cells (M1/M2/M3/M4 x MultiHop/NarrativeQA)
depend on those keys not moving. Rule B (a single-item parent is written
with the BASE filename and the BASE raw bytes) is the argument that they
cannot move on a 1:1 benchmark, and `tests/test_m4_index_items.py` pins
that property on fixtures.

A PROPERTY PINNED ON FIXTURES IS NOT THE CLAIM. The claim is about the
hash of the REAL MultiHop and NarrativeQA corpora under both layouts, so
this probe computes exactly that, on the real corpora, through the real
code path.

HOW IT MEASURES. `CacheDir.is_complete` is replaced with a sentinel that
raises the moment a system asks whether its cache is warm. Every system
computes its cache key immediately before that call, so the key that
comes back is computed BY THE SYSTEM'S OWN `index()` — not re-derived
here — and nothing downstream (embedder load, chunking, tree build) ever
runs. `corpus_content_hash` is wrapped in each retriever module so the
corpus hash and the realised temp-dir FILE COUNT are recorded too; the
real function still does the hashing.

THE VACUOUS-PASS TRAP (standing rule: a check that could not observe its
subject has not passed). "No key moved" is worthless unless the probe
could have seen a key move. HotpotQA is therefore run as a POSITIVE
CONTROL: its parents hold many sentence items, so the promotion MUST
change its file count and MUST move M2/M3/M7's keys. `--compare` exits
non-zero if the control fails to fire, if any expected key is missing, or
if a 1:1 assumption turns out to be false in the data.

⚠ ON WINDOWS, PASS `--posix-newlines` OR THE KEYS ARE FICTION. This was
learned the hard way: the first run of this probe produced an internally
consistent before/after pair in which NEITHER column matched the keys the
banked cells actually used. `Path.write_text` opens in text mode, so on
Windows every "\\n" in the temp corpus becomes CRLF — 56,804 extra bytes
across MultiHop's 609 articles — and every system's corpus_hash differs
from the Colab (Linux) run for a reason unrelated to the layout under
test. With the flag, this probe reproduces the banked MultiHop keys
EXACTLY (M2 51a2e3f9…, M3 0c3c65e2…, M4 a7276135…), which is what makes
the invariance claim about the banked cells rather than about some
neighbouring configuration. No-op on Linux.

USAGE

    # at HEAD, before the change
    python -m scripts.prove_index_layout_key_invariance \
        --posix-newlines --out before.json

    # with the promotion applied to the working tree
    python -m scripts.prove_index_layout_key_invariance \
        --posix-newlines --out after.json

    python -m scripts.prove_index_layout_key_invariance \
        --compare before.json after.json

The before column MUST be checked against real cache directory names or
`cache hit:` lines from the run logs. An invariance proof between two
computed columns says nothing if neither column is the banked one.

No GPU, no network beyond the HuggingFace dataset download, no model
load. Nothing is written to the cache directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src import cache as cache_mod
from src.config import DEFAULT_CONFIG
from src.retrievers import (
    m2_flat_dense,
    m3_hybrid,
    m4_raptor,
    m7_three_axis,
    m9_corrective,
)
from src.retrievers.base import BaseSystem


# System id -> class. M9 is included even though it is out of the matrix:
# it indexes through an inner HybridRRFSystem, and its banked MultiHop
# cell is the evidence base for FINDING_M9_CLOSED_CORPUS.md. Its key is
# expected to equal M3's, which is itself a check on the composition.
# M1 is absent BY CONSTRUCTION — it never computes a cache key (its
# index() discards the corpus path), so there is no key to move.
SYSTEMS: dict[str, Any] = {
    "M2": m2_flat_dense.FlatDenseSystem,
    "M3": m3_hybrid.HybridRRFSystem,
    "M4": m4_raptor.RaptorSystem,
    "M7": m7_three_axis.ThreeAxisSystem,
    "M9": m9_corrective.CorrectiveRAGSystem,
}

# Modules that imported `corpus_content_hash` into their own namespace.
_HASH_CALLERS = (m2_flat_dense, m3_hybrid, m4_raptor, m7_three_axis)

# The banked cells this probe is protecting, as run.
NARRATIVEQA_MAX_UNITS = 40
HOTPOTQA_CONTROL_UNITS = 3


class _KeyCaptured(Exception):
    """Raised at the cache check to stop before any build work."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


class _Observation(dict):
    pass


def _capture(system: BaseSystem, items: Any) -> dict:
    """Run the real index_items -> index() path, stopping at the cache check.

    Returns {key, corpus_hash, n_files} — all three MEASURED by the code
    under test, not reconstructed here.
    """
    seen: dict[str, Any] = {}

    real_hash = cache_mod.corpus_content_hash

    def wrapped(corpus_path: Path) -> str:
        root = Path(corpus_path)
        seen["n_files"] = sum(1 for p in root.rglob("*") if p.is_file())
        seen["n_bytes"] = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        value = real_hash(corpus_path)
        seen["corpus_hash"] = value
        return value

    real_is_complete = cache_mod.CacheDir.is_complete

    def stop(self: Any, required: Any) -> bool:  # noqa: ANN401
        raise _KeyCaptured(self.cache_key)

    patched = []
    for mod in _HASH_CALLERS:
        if getattr(mod, "corpus_content_hash", None) is real_hash:
            mod.corpus_content_hash = wrapped
            patched.append(mod)
    cache_mod.CacheDir.is_complete = stop
    try:
        system.index_items(items)
    except _KeyCaptured as exc:
        seen["key"] = exc.key
    else:
        raise AssertionError(
            f"{system.system_id}: index() returned without reaching the cache "
            "check — the probe measured nothing"
        )
    finally:
        cache_mod.CacheDir.is_complete = real_is_complete
        for mod in patched:
            mod.corpus_content_hash = real_hash

    for field in ("key", "corpus_hash", "n_files"):
        if field not in seen:
            raise AssertionError(
                f"{system.system_id}: probe did not observe {field!r}; the "
                "instrumentation missed the code path it claims to measure"
            )
    return seen


def _parent_stats(items: Any) -> dict:
    """MEASURE the items-per-parent shape rather than assuming 1:1."""
    counts: dict[str, int] = {}
    id_is_parent = 0
    for item in items:
        counts[item.parent_id] = counts.get(item.parent_id, 0) + 1
        if item.item_id == item.parent_id:
            id_is_parent += 1
    return {
        "n_items": len(list(items)),
        "n_parents": len(counts),
        "max_items_per_parent": max(counts.values()) if counts else 0,
        "n_item_id_equals_parent_id": id_is_parent,
    }


def _units_for(benchmark_id: str) -> tuple[list[tuple[str, list, dict]], Any]:
    """((corpus_id, corpus items, per-unit stats) per unit, loader)."""
    if benchmark_id == "multihop_rag":
        from src.eval.multihop import MultiHopBenchmark

        bench = MultiHopBenchmark()
        units = list(bench.iter_eval_units(split="validation"))
    elif benchmark_id == "narrativeqa":
        from src.eval.narrativeqa import NarrativeQABenchmark

        bench = NarrativeQABenchmark()
        units = list(
            bench.iter_eval_units(split="validation", max_units=NARRATIVEQA_MAX_UNITS)
        )
    elif benchmark_id == "hotpotqa":
        from src.eval.hotpotqa import HotpotQABenchmark

        bench = HotpotQABenchmark()
        units = list(
            bench.iter_eval_units(split="validation", max_units=HOTPOTQA_CONTROL_UNITS)
        )
    else:
        raise ValueError(f"unknown benchmark {benchmark_id!r}")
    return [
        (u.corpus_id, list(u.corpus), _parent_stats(u.corpus)) for u in units
    ], bench


def _force_posix_newlines() -> None:
    """Write temp-corpus files with LF, as the matrix runs did.

    `Path.write_text` opens in TEXT mode with `newline=None`, which
    translates every "\\n" to `os.linesep` — CRLF on Windows. Every
    banked cell was produced on Colab (Linux), so a Windows probe writes
    a temp corpus that is byte-for-byte LARGER than the one the matrix
    hashed (MultiHop alone carries 56,804 newlines across 609 articles),
    and EVERY system's key differs for a reason that has nothing to do
    with the layout under test.

    This does not "fix" anything in the harness: on Linux the two are
    already identical. It exists so a Windows probe can compute the key
    the Linux run computed.
    """
    real = Path.write_text

    def posix_write_text(self, data, encoding=None, errors=None, newline=None):
        return real(self, data, encoding=encoding, errors=errors, newline="\n")

    Path.write_text = posix_write_text  # type: ignore[method-assign]
    print("[probe] POSIX newline mode: temp-corpus files written with LF")


def run(benchmarks: list[str], out_path: Path, posix_newlines: bool = False) -> dict:
    if posix_newlines:
        _force_posix_newlines()
    report: dict[str, Any] = {"posix_newlines": bool(posix_newlines), "benchmarks": {}}
    for benchmark_id in benchmarks:
        units, bench = _units_for(benchmark_id)
        print(f"\n=== {benchmark_id}: {len(units)} unit(s) ===")
        rows: list[dict] = []
        for corpus_id, items, stats in units:
            row: dict[str, Any] = {"corpus_id": corpus_id, "corpus": stats}
            for system_id, cls in SYSTEMS.items():
                system = cls(DEFAULT_CONFIG)
                row[system_id] = _capture(system, items)
            rows.append(row)
            print(
                f"  {corpus_id[:44]:<44} items={stats['n_items']:>5} "
                f"parents={stats['n_parents']:>5} "
                f"max/parent={stats['max_items_per_parent']:>3} "
                f"files(M2)={row['M2']['n_files']:>5} "
                f"files(M4)={row['M4']['n_files']:>5}"
            )
        report["benchmarks"][benchmark_id] = {
            "n_units": len(rows),
            "loader_stats": dict(getattr(bench, "stats", {})),
            "units": rows,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_path}")
    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    print("\n--- keys ---")
    for benchmark_id, block in report["benchmarks"].items():
        for system_id in SYSTEMS:
            keys = [u[system_id]["key"] for u in block["units"]]
            uniq = sorted(set(keys))
            head = uniq[0] if len(uniq) == 1 else f"{len(uniq)} distinct"
            print(f"  {benchmark_id:<14} {system_id}  {head}")


def compare(before_path: Path, after_path: Path) -> int:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    control_fired = False
    print(f"{'benchmark':<14} {'system':<7} {'unit':<40} verdict")
    print("-" * 84)

    for benchmark_id, b_block in before["benchmarks"].items():
        a_block = after["benchmarks"].get(benchmark_id)
        if a_block is None:
            failures.append(f"{benchmark_id}: absent from the after run")
            continue
        if len(a_block["units"]) != len(b_block["units"]):
            failures.append(
                f"{benchmark_id}: unit count changed "
                f"{len(b_block['units'])} -> {len(a_block['units'])}; the two "
                "runs did not load the same corpora"
            )
            continue
        for b_unit, a_unit in zip(b_block["units"], a_block["units"]):
            if b_unit["corpus_id"] != a_unit["corpus_id"]:
                failures.append(
                    f"{benchmark_id}: unit order changed "
                    f"({b_unit['corpus_id']} != {a_unit['corpus_id']})"
                )
                continue
            for system_id in SYSTEMS:
                b_row, a_row = b_unit[system_id], a_unit[system_id]
                moved = b_row["key"] != a_row["key"]
                if benchmark_id == "hotpotqa" and system_id in ("M2", "M3", "M7", "M9"):
                    # POSITIVE CONTROL. These MUST move: the promotion is
                    # what changes their layout on a multi-item-parent
                    # corpus. If they hold still, the probe is measuring a
                    # change that did not take effect and every "unmoved"
                    # verdict elsewhere is vacuous.
                    if moved:
                        control_fired = True
                        verdict = "MOVED (expected — positive control)"
                    else:
                        verdict = "UNMOVED — CONTROL FAILED"
                        failures.append(
                            f"positive control did not fire: {benchmark_id} "
                            f"{system_id} {b_unit['corpus_id']} key unchanged"
                        )
                else:
                    verdict = "MOVED — BANKED CELL INVALIDATED" if moved else "unmoved"
                    if moved:
                        failures.append(
                            f"{benchmark_id} {system_id} {b_unit['corpus_id']}: "
                            f"{b_row['key']} -> {a_row['key']}"
                        )
                print(
                    f"{benchmark_id:<14} {system_id:<7} "
                    f"{b_unit['corpus_id'][:40]:<40} {verdict}"
                )

    # 1:1-ness is an ASSUMPTION about the data, so check it in the data.
    for source, label in ((before, "before"), (after, "after")):
        for benchmark_id in ("multihop_rag", "narrativeqa"):
            block = source["benchmarks"].get(benchmark_id)
            if block is None:
                continue
            worst = max(
                (u["corpus"]["max_items_per_parent"] for u in block["units"]),
                default=0,
            )
            if worst != 1:
                failures.append(
                    f"{label}: {benchmark_id} is NOT 1:1 — a parent holds "
                    f"{worst} items, so rule B's degeneracy argument does not "
                    "apply to it"
                )

    if not control_fired:
        failures.append(
            "the positive control never fired — no HotpotQA key moved, so "
            "this comparison could not have detected a moved key at all"
        )

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: every banked key unmoved; positive control moved as expected.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the measurement JSON here")
    parser.add_argument(
        "--benchmarks",
        default="multihop_rag,narrativeqa,hotpotqa",
        help="comma-separated benchmark ids to measure",
    )
    parser.add_argument(
        "--posix-newlines",
        action="store_true",
        help=(
            "Write the temp corpus with LF instead of the platform "
            "default. REQUIRED on Windows to reproduce a key computed by "
            "a Colab run: Path.write_text translates newlines in text "
            "mode, so a Windows temp corpus is a different byte sequence "
            "from the one the matrix hashed. No-op on Linux."
        ),
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        help="compare two measurement files and exit non-zero on any failure",
    )
    args = parser.parse_args(argv)

    if args.compare:
        return compare(*args.compare)
    if not args.out:
        parser.error("--out is required unless --compare is given")
    run(
        [b.strip() for b in args.benchmarks.split(",") if b.strip()],
        args.out,
        posix_newlines=args.posix_newlines,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
