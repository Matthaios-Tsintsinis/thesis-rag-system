"""Read-only aggregator for ScoredQuery JSONL files.

Parses one or more JSONL files produced by `src.eval.runner` and prints
per-system aggregates: query count, chunk-count distribution, retrieval
F1 / recall / precision, answer score, abstention rate, plus a per-
(system, question_type) slice. Optionally dumps the aggregates to JSON.

USAGE:
    python -m src.eval.analyse <OUTPUT_DIR>/eval/qasper_*_validation_*.jsonl
    python -m src.eval.analyse --inputs file1.jsonl file2.jsonl --output aggregates.json
    python -m src.eval.analyse --inputs <dir>/*.jsonl --by-type

This script reads ONLY what the current ScoredQuery captures: per-query
chunk count (n_retrieved), retrieval/answer scores, predicted_answer
text, question_type. It does NOT have token volume or retrieved unit
types — those land in a follow-up instrumentation commit if the chunk-
count analysis is inconclusive (M4/M7 retrieve summary nodes mixed
with raw chunks, so count is a noisy proxy for token volume for those
two specifically).

ABSTENTION DETECTION uses the same `is_abstention` predicate the eval
scorers use, so the abstention-rate number here matches what the
scorers see at scoring time.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .scorers import is_abstention


def _iter_records(paths: Iterable[Path]) -> Iterable[dict]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[analyse] WARN: bad JSONL line in {p}: {e}")


def _expand_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for spec in inputs:
        if any(ch in spec for ch in "*?["):
            out.extend(Path(p) for p in glob.glob(spec))
        else:
            out.append(Path(spec))
    return [p for p in out if p.is_file()]


def _safe_stats(xs: list[float]) -> dict[str, float | int]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def _aggregate(records: Iterable[dict]) -> dict[str, Any]:
    """Per-system aggregates. Records read in any order; system_id keys the rollup."""
    by_system: dict[str, dict[str, list]] = defaultdict(
        lambda: {
            "chunk_counts": [],
            "packed_counts": [],
            "evidence_tokens": [],
            "input_tokens": [],
            "retr_f1": [],
            "retr_recall": [],
            "retr_precision": [],
            "retr_skipped": 0,
            "ans_score": [],
            "abstained": 0,
            "latency": [],
            "retrieved_unit_types_agg": defaultdict(int),
            "packed_unit_types_agg": defaultdict(int),
            "by_type_chunk_counts": defaultdict(list),
            "by_type_retr_f1": defaultdict(list),
            "by_type_ans_score": defaultdict(list),
            "by_type_n": defaultdict(int),
        }
    )

    n_total = 0
    for r in records:
        n_total += 1
        sid = r.get("system_id", "?")
        bucket = by_system[sid]

        bucket["chunk_counts"].append(int(r.get("n_retrieved", 0)))
        bucket["packed_counts"].append(int(r.get("n_packed", 0)))
        bucket["evidence_tokens"].append(int(r.get("evidence_tokens", 0)))
        bucket["input_tokens"].append(int(r.get("n_input_tokens", 0)))
        bucket["latency"].append(float(r.get("latency_s", 0.0)))

        for ut, n in (r.get("retrieved_unit_types") or {}).items():
            bucket["retrieved_unit_types_agg"][ut] += int(n)
        for ut, n in (r.get("packed_unit_types") or {}).items():
            bucket["packed_unit_types_agg"][ut] += int(n)

        retr = r.get("retrieval") or {}
        if retr.get("skipped"):
            bucket["retr_skipped"] += 1
        else:
            bucket["retr_f1"].append(float(retr.get("f1", 0.0)))
            bucket["retr_recall"].append(float(retr.get("recall", 0.0)))
            bucket["retr_precision"].append(float(retr.get("precision", 0.0)))

        ans = r.get("answer") or {}
        bucket["ans_score"].append(float(ans.get("value", 0.0)))

        predicted = r.get("predicted_answer", "") or ""
        if is_abstention(predicted):
            bucket["abstained"] += 1

        qtype = r.get("question_type", "?")
        bucket["by_type_n"][qtype] += 1
        bucket["by_type_chunk_counts"][qtype].append(int(r.get("n_retrieved", 0)))
        if not retr.get("skipped"):
            bucket["by_type_retr_f1"][qtype].append(float(retr.get("f1", 0.0)))
        bucket["by_type_ans_score"][qtype].append(float(ans.get("value", 0.0)))

    # Reshape to a serialisable rollup.
    out: dict[str, Any] = {"n_total_records": n_total, "systems": {}}
    for sid, b in sorted(by_system.items()):
        n_q = len(b["chunk_counts"])
        out["systems"][sid] = {
            "n_queries": n_q,
            "chunk_count": _safe_stats(b["chunk_counts"]),
            "packed_count": _safe_stats(b["packed_counts"]),
            "evidence_tokens": _safe_stats(b["evidence_tokens"]),
            "input_tokens": _safe_stats(b["input_tokens"]),
            "retrieved_unit_types": dict(b["retrieved_unit_types_agg"]),
            "packed_unit_types": dict(b["packed_unit_types_agg"]),
            "retr_f1_mean": (statistics.mean(b["retr_f1"]) if b["retr_f1"] else None),
            "retr_recall_mean": (
                statistics.mean(b["retr_recall"]) if b["retr_recall"] else None
            ),
            "retr_precision_mean": (
                statistics.mean(b["retr_precision"]) if b["retr_precision"] else None
            ),
            "retr_n_scored": len(b["retr_f1"]),
            "retr_n_skipped": b["retr_skipped"],
            "ans_score_mean": (statistics.mean(b["ans_score"]) if b["ans_score"] else None),
            "abstention_rate": b["abstained"] / max(1, n_q),
            "latency_s_mean": (statistics.mean(b["latency"]) if b["latency"] else None),
            "by_question_type": {
                qt: {
                    "n": b["by_type_n"][qt],
                    "chunk_count_mean": (
                        statistics.mean(b["by_type_chunk_counts"][qt])
                        if b["by_type_chunk_counts"][qt]
                        else None
                    ),
                    "retr_f1_mean": (
                        statistics.mean(b["by_type_retr_f1"][qt])
                        if b["by_type_retr_f1"][qt]
                        else None
                    ),
                    "ans_score_mean": (
                        statistics.mean(b["by_type_ans_score"][qt])
                        if b["by_type_ans_score"][qt]
                        else None
                    ),
                }
                for qt in sorted(b["by_type_n"])
            },
        }
    return out


def _fmt(x: float | int | None, places: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{places}f}"
    return str(x)


def _print_text(rollup: dict[str, Any], *, by_type: bool) -> None:
    print(f"[analyse] {rollup['n_total_records']} ScoredQuery records across "
          f"{len(rollup['systems'])} systems\n")

    # Top table.
    cols = [
        ("system", 8),
        ("n_q", 5),
        ("chunks", 8),
        ("packed", 8),
        ("ev_tok", 8),
        ("in_tok", 8),
        ("retr_f1", 8),
        ("ans", 7),
        ("abstain%", 9),
        ("skip", 5),
        ("lat_s", 7),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for sid in rollup["systems"]:
        s = rollup["systems"][sid]
        row = [
            (sid, 8),
            (str(s["n_queries"]), 5),
            (_fmt(s["chunk_count"].get("mean"), places=1), 8),
            (_fmt(s["packed_count"].get("mean"), places=1), 8),
            (_fmt(s["evidence_tokens"].get("mean"), places=0), 8),
            (_fmt(s["input_tokens"].get("mean"), places=0), 8),
            (_fmt(s["retr_f1_mean"]), 8),
            (_fmt(s["ans_score_mean"]), 7),
            (_fmt(s["abstention_rate"] * 100, places=1) + "%", 9),
            (str(s["retr_n_skipped"]), 5),
            (_fmt(s["latency_s_mean"], places=2), 7),
        ]
        print("  ".join(val.ljust(w) for val, w in row))

    # Unit-type distribution (per-system).
    print("\n  --- retrieved unit-type distribution ---")
    for sid in rollup["systems"]:
        ut = rollup["systems"][sid].get("retrieved_unit_types") or {}
        if ut:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(ut.items()))
            print(f"  {sid}: {parts}")
    print("  --- packed unit-type distribution ---")
    for sid in rollup["systems"]:
        ut = rollup["systems"][sid].get("packed_unit_types") or {}
        if ut:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(ut.items()))
            print(f"  {sid}: {parts}")

    if not by_type:
        return

    # Per-(system, question_type) slice.
    print("\n  --- per question_type ---\n")
    qtype_cols = [
        ("system", 8),
        ("qtype", 16),
        ("n", 5),
        ("chunks_mean", 11),
        ("retr_f1", 8),
        ("ans", 7),
    ]
    header2 = "  ".join(name.ljust(w) for name, w in qtype_cols)
    print(header2)
    print("-" * len(header2))
    for sid in rollup["systems"]:
        s = rollup["systems"][sid]
        for qt, qs in s["by_question_type"].items():
            row = [
                (sid, 8),
                (qt, 16),
                (str(qs["n"]), 5),
                (_fmt(qs.get("chunk_count_mean")), 11),
                (_fmt(qs.get("retr_f1_mean")), 8),
                (_fmt(qs.get("ans_score_mean")), 7),
            ]
            print("  ".join(val.ljust(w) for val, w in row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate ScoredQuery JSONL files from src.eval.runner."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="JSONL paths or glob patterns. Combined with --inputs.",
    )
    parser.add_argument(
        "--inputs",
        dest="inputs_named",
        nargs="*",
        default=[],
        help="Alternate way to pass inputs (paths or glob patterns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full rollup.",
    )
    parser.add_argument(
        "--by-type",
        action="store_true",
        help="Also print per-question-type rows under each system.",
    )
    parser.add_argument(
        "--check-budget-equality",
        action="store_true",
        help="Assert all systems' mean evidence_tokens fall within "
        "±5%% of the cross-system mean. CK-4 invariant check — the "
        "point of the whole change. Exits non-zero on violation.",
    )
    parser.add_argument(
        "--budget-equality-tolerance",
        type=float,
        default=0.05,
        help="Fractional tolerance for --check-budget-equality. Default 0.05 (±5%%).",
    )
    args = parser.parse_args()

    raw_inputs = (args.inputs or []) + (args.inputs_named or [])
    if not raw_inputs:
        parser.error("at least one input path or glob is required")

    paths = _expand_inputs(raw_inputs)
    if not paths:
        parser.error(f"no files matched the input(s): {raw_inputs!r}")

    print(f"[analyse] reading {len(paths)} JSONL file(s):")
    for p in paths:
        print(f"  {p}")

    rollup = _aggregate(_iter_records(paths))
    _print_text(rollup, by_type=args.by_type)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(rollup, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[analyse] rollup -> {args.output}")

    if args.check_budget_equality:
        # Compare mean evidence_tokens across systems; flag outliers.
        # Skip systems with 0 mean (M1 closed-book legitimately has 0
        # evidence). Assertion: every non-zero system's mean falls
        # within (1 ± tol) × cross-system mean.
        means = []
        for sid, s in rollup["systems"].items():
            m = s["evidence_tokens"].get("mean")
            if m and m > 0:
                means.append((sid, float(m)))
        if not means:
            print("\n[analyse] --check-budget-equality: no system reported "
                  "non-zero evidence_tokens (CK-4 not applied?).")
            raise SystemExit(2)
        avg = sum(m for _, m in means) / len(means)
        tol = args.budget_equality_tolerance
        lo, hi = avg * (1.0 - tol), avg * (1.0 + tol)
        print(f"\n[analyse] --check-budget-equality (tol ±{tol*100:.1f}%):")
        print(f"  cross-system mean evidence_tokens = {avg:.0f}")
        print(f"  acceptable band: [{lo:.0f}, {hi:.0f}]")
        violations = []
        for sid, m in means:
            ok = lo <= m <= hi
            marker = "OK" if ok else "VIOLATION"
            print(f"    {sid}: {m:.0f}  [{marker}]")
            if not ok:
                violations.append((sid, m))
        if violations:
            print(f"\n[analyse] FAIL: {len(violations)} system(s) outside "
                  f"±{tol*100:.1f}% band.")
            raise SystemExit(2)
        print("\n[analyse] PASS: all systems within budget-equality band.")


if __name__ == "__main__":
    main()
