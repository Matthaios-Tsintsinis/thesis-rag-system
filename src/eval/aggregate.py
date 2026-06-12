"""Results aggregator: every *.summary.json -> one conditions-annotated table.

Scans one or more roots for runner-produced summary files and renders a
single cross-benchmark results table (markdown + CSV + stdout) whose
rows are SELF-DESCRIBING: benchmark, system, split, query count,
chunking strategy, context budget, GENERATOR, retrieval F1, answer
score, skipped/unalignable counts, git commit, date.

The generator column is mandatory and never guessed. Resolution order:
  1. the summary's own "generator" field (runs after the provenance
     enrichment commit);
  2. stash-era inference — a summary living under a directory whose
     name contains "qwen_era" is labelled "qwen2.5-3b (stash-era)";
  3. otherwise "unknown (pre-enrichment)" with a footnote. The
     pre-enrichment gpt-4o-mini re-baseline summaries fall here by
     design — gate numbers, not matrix numbers.

Benchmark-specific extras are tolerated, not required: with --deep
(default ON) the aggregator re-reads each summary's JSONL — when the
file still exists at the recorded path or as a sibling — through
analyse._aggregate (pure reuse, no duplicated metric logic) and
renders extras sections only where present: rank-aware retrieval for
MultiHop rows, extraction-method stats for QuALITY rows, the
corrective action mix for M9 rows.

Rows are never merged or overwritten: multiple runs of the same
(benchmark, system, split) cell stay as separate rows — the era,
commit, and date columns disambiguate them.

USAGE (run before/while the matrix accumulates):
    python -m src.eval.aggregate
    python -m src.eval.aggregate <OUTPUT_DIR>/eval <OUTPUT_DIR>/eval_qwen_era_pre_b6e35c6
    python -m src.eval.aggregate --no-deep --output-dir local_runs/aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from .. import paths
from .analyse import _aggregate, _iter_records


DISCLAIMER = (
    "NOTE: validation-split / small-sample numbers are development "
    "diagnostics, NOT conclusions. Final thesis numbers come from the "
    "reserved test split, single run, after the pipeline locks."
)

_BUDGET_RE = re.compile(r"_budget(\d+)")

CORE_COLUMNS = [
    "benchmark", "system", "split", "n_q", "chunking", "budget",
    "generator", "retr_f1", "ans", "skipped", "commit", "date",
]
EXTRA_COLUMNS = [
    "mrr", "hit@10", "map@10",
    "mc_unparseable", "mc_abstained",
    "m9_correct", "m9_ambiguous", "m9_incorrect", "m9_overlap",
]


def _resolve_generator(summary: dict, summary_path: Path) -> str:
    gen = summary.get("generator")
    if gen:
        return str(gen)
    if "qwen_era" in str(summary_path).lower():
        return "qwen2.5-3b (stash-era)"
    return "unknown (pre-enrichment)"


def _resolve_budget(summary: dict, summary_path: Path) -> str:
    if "evidence_budget" in summary:
        b = summary["evidence_budget"]
        return "off" if not b else str(b)
    m = _BUDGET_RE.search(summary_path.name)
    return m.group(1) if m else "off"


def _resolve_jsonl(summary: dict, summary_path: Path) -> Path | None:
    """Recorded path first (same machine), then the sibling file (the
    runner always writes foo.jsonl next to foo.summary.json — survives
    Drive-to-local moves)."""
    recorded = summary.get("output_path")
    if recorded:
        p = Path(recorded)
        if p.is_file():
            return p
    name = summary_path.name
    if name.endswith(".summary.json"):
        sibling = summary_path.with_name(name[: -len(".summary.json")] + ".jsonl")
        if sibling.is_file():
            return sibling
    return None


def _fmt(x: Any, places: int = 3) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


def build_row(summary_path: Path, *, deep: bool) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    n_q = int(summary.get("n_queries_scored") or 0)
    n_skipped = int(summary.get("n_retrieval_skipped") or 0)

    # Retrieval F1 is meaningless when every query skipped retrieval
    # scoring (answer-only benchmarks) — blank beats a misleading 0.0.
    retr_f1: float | None = summary.get("mean_retrieval_f1")
    if n_q > 0 and n_skipped >= n_q:
        retr_f1 = None

    skipped_str = str(n_skipped)
    unalignable = (summary.get("benchmark_stats") or {}).get("n_evidence_unalignable")
    if unalignable:
        skipped_str += f" (unalign {unalignable})"

    row: dict[str, Any] = {
        "benchmark": summary.get("benchmark", "?"),
        "system": summary.get("system", "?"),
        "split": summary.get("split", "?"),
        "n_q": n_q,
        "chunking": summary.get("chunking_strategy") or "word_window (default-era)",
        "budget": _resolve_budget(summary, summary_path),
        "generator": _resolve_generator(summary, summary_path),
        "retr_f1": retr_f1,
        "ans": summary.get("mean_answer_score"),
        "skipped": skipped_str,
        "commit": summary.get("git_commit") or "-",
        "date": summary.get("timestamp", "?"),
        "_path": str(summary_path),
    }

    if deep:
        jsonl = _resolve_jsonl(summary, summary_path)
        if jsonl is not None:
            rollup = _aggregate(_iter_records([jsonl]))
            s = (rollup.get("systems") or {}).get(row["system"]) or {}
            if s.get("mrr_mean") is not None:
                row["mrr"] = s["mrr_mean"]
                row["hit@10"] = (s.get("hit_at_k_mean") or {}).get(10)
                row["map@10"] = (s.get("map_at_k_mean") or {}).get(10)
            mc = s.get("mc_extraction")
            if mc:
                row["mc_unparseable"] = mc.get("unparseable_rate")
                row["mc_abstained"] = mc.get("abstained_rate")
            m9 = s.get("m9_corrective")
            if m9:
                mix = m9.get("action_mix") or {}
                row["m9_correct"] = mix.get("correct")
                row["m9_ambiguous"] = mix.get("ambiguous")
                row["m9_incorrect"] = mix.get("incorrect")
                row["m9_overlap"] = m9.get("overlap_jaccard_mean")
    return row


def collect_rows(roots: list[Path], *, deep: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            print(f"[aggregate] WARN: root does not exist, skipping: {root}")
            continue
        for summary_path in sorted(root.rglob("*.summary.json")):
            try:
                rows.append(build_row(summary_path, deep=deep))
            except (json.JSONDecodeError, OSError, KeyError) as e:
                print(f"[aggregate] WARN: bad summary {summary_path}: {e}")
    rows.sort(key=lambda r: (r["benchmark"], r["system"], r["date"]))
    return rows


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for r in rows:
        lines.append(
            "| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |"
        )
    return "\n".join(lines)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    out = [f"# Eval results rollup ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    out.append(f"> {DISCLAIMER}")
    out.append("")
    out.append(_md_table(rows, CORE_COLUMNS))

    if any(r.get("generator", "").startswith("unknown") for r in rows):
        out.append("")
        out.append(
            "Footnote: rows with generator `unknown (pre-enrichment)` predate "
            "the summary-provenance fields; their generator was not recorded "
            "and is NOT guessed here. Re-run for an attributed row."
        )

    rank_rows = [r for r in rows if r.get("mrr") is not None]
    if rank_rows:
        out += ["", "## Rank-aware retrieval (MultiHop rows)", "",
                _md_table(rank_rows, ["benchmark", "system", "split", "date",
                                      "mrr", "hit@10", "map@10"])]
    mc_rows = [r for r in rows if r.get("mc_unparseable") is not None]
    if mc_rows:
        out += ["", "## Multiple-choice extraction (QuALITY rows)", "",
                _md_table(mc_rows, ["benchmark", "system", "split", "date",
                                    "mc_unparseable", "mc_abstained"])]
    m9_rows = [r for r in rows if r.get("m9_correct") is not None]
    if m9_rows:
        out += ["", "## M9 corrective action mix", "",
                _md_table(m9_rows, ["benchmark", "system", "split", "date",
                                    "m9_correct", "m9_ambiguous",
                                    "m9_incorrect", "m9_overlap"])]
    out.append("")
    return "\n".join(out)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = CORE_COLUMNS + EXTRA_COLUMNS + ["_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate runner *.summary.json files into one "
        "conditions-annotated results table (markdown + CSV)."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Directories to scan recursively for *.summary.json. "
        "Default: <OUTPUT_DIR>/eval. Pass stash dirs explicitly to "
        "include archived eras.",
    )
    parser.add_argument(
        "--no-deep",
        action="store_true",
        help="Skip re-reading JSONLs for benchmark-specific extras "
        "(rank-aware / extraction stats / M9 action mix).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write results_{stamp}.md/.csv. "
        "Default: <OUTPUT_DIR>/aggregate.",
    )
    args = parser.parse_args()

    roots = args.roots or [paths.output_dir() / "eval"]
    rows = collect_rows(roots, deep=not args.no_deep)
    if not rows:
        raise SystemExit(f"[aggregate] no *.summary.json found under: "
                         f"{[str(r) for r in roots]}")

    md = render_markdown(rows)
    print(md)

    out_dir = args.output_dir or (paths.output_dir() / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"results_{stamp}.md"
    csv_path = out_dir / f"results_{stamp}.csv"
    md_path.write_text(md, encoding="utf-8")
    write_csv(rows, csv_path)
    print(f"[aggregate] {len(rows)} rows -> {md_path}")
    print(f"[aggregate] csv -> {csv_path}")


if __name__ == "__main__":
    main()
