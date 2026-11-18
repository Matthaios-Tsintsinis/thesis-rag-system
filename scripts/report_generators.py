"""Ours across generators (v20.2): the v20 model with each sweep generator as its one model,
reader and tree summariser alike, on NarrativeQA and MultiHop, beside the reader's own cells
from round 2. Read-only on every bank. Descriptive: the sweep is not one of the declared
comparisons, so no family, no Holm and no verdict, only differences and their intervals.

  table  GENERATORS_OURS.{csv,md} into the sweep's directory. For each generator with a
         finished cell (smallest first, the reader last): F1 (MultiHop: answerable, with the
         supplementary beside), EM, the refused share, MultiHop's null score, R@5 and hit@5
         from ours' document ranking, ours' mechanics and seconds per question, and the cell
         against the reader's on the same questions (mean difference, 95% bootstrap CI).
         With --comparison, the matrix's M1-M4 rows of each generator that has a column
         (Qwen2.5-7B and the reader); with --p10, ours under Qwen2.5-7B against the Qwen
         column's M1-M4 on the same questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ours import config as settings
from scripts.dev_ours import SWEEP_BENCHMARKS, golds_of, ours_numbers, read_rows, sweep_bank
from scripts.export_comparison import DATASET_LABEL, QWEN, _em_and_hit5, _fail, read_cell
from scripts.report_ours import (
    FROZEN_CODE,
    SAME_OUTPUTS,
    SEED,
    SYSTEMS,
    bootstrap_ci,
    check_doc_ranking,
    credited,
    paired,
    row_scores,
)

# The sweep's directory name -> the frozen model it runs (with the reader, its rows would be
# that round's bit for bit). The sweep: v20, round 2's model.
SWEEP_CODE: dict = {"ours_v20_generators": "e77ef1d48101c109"}
# Round 2's bank, which holds the reader's own cells of the same model.
READER_BANK = "ours_v20"
# Billions of parameters, from the hub's safetensors totals (2026-11-19); run order, the reader last.
PARAMS_B = {**dict(zip(settings.SWEEP_GENERATORS, (1.24, 1.54, 3.09, 3.21, 3.84, 7.62, 14.77))),
            settings.READER: 8.03}

CSV_COLUMNS = ["generator", "params_b", "benchmark", "n_queries", "f1", "f1_supplementary", "em",
               "refused_pct", "n_credited", "null_score", "recall_at_5", "hit_at_5", "map_at_10",
               "parse_failure_rate", "mean_llm_calls", "mean_aspects", "mean_evidence_tokens",
               "s_per_question", "gpu", "answer_delta_vs_reader", "answer_ci_lo", "answer_ci_hi",
               "retrieval_delta_vs_reader", "retrieval_ci_lo", "retrieval_ci_hi", "n_paired"]


def short(model: str) -> str:
    return model.split("/")[-1]


def finished(bank: Path, benchmark: str) -> bool:
    return (bank / f"{benchmark}_OURS_validation.summary.json").is_file()


def check_cell_code(rows, frozen: str, exact: bool, where: str) -> str:
    """One fingerprint in a cell: the frozen one (the reader's round), or code that gives it."""
    codes = {r["metadata"]["ours_code"] for r in rows}
    if len(codes) != 1:
        _fail(f"{where}: rows carry codes {sorted(codes)}")
    code = codes.pop()
    if (code if exact else SAME_OUTPUTS.get(code, code)) != frozen:
        _fail(f"{where}: rows of code {code}, not the sweep's model {frozen}")
    return code


def cell_numbers(bank: Path, generator: str, benchmark: str, golds, docs) -> dict:
    """One finished ours cell through the exporter's functions, plus ours' own measures."""
    rows = read_rows(bank / f"{benchmark}_OURS_validation.jsonl")
    # Descriptive: the battery is counted from the rows, not gated against a record.
    battery = credited(rows)
    cell = read_cell(bank, generator, benchmark, "OURS", {(generator, benchmark, "OURS"): battery})
    summary = json.loads((bank / f"{benchmark}_OURS_validation.summary.json").read_text(encoding="utf-8"))
    extra = _em_and_hit5(bank, benchmark, "OURS", golds, summary)
    if benchmark != "narrativeqa":
        check_doc_ranking(rows, docs)
    mech = ours_numbers(rows, golds, docs)
    null = cell["mean_answer_score_null"]
    return {
        "rows": rows, "generator": generator, "params_b": PARAMS_B.get(generator),
        "benchmark": benchmark, "n_queries": cell["n_queries"],
        "f1": float(cell["mean_answer_score_answerable"]),
        "f1_supplementary": float(cell["supplementary_mean"]),
        "em": extra["em"], "refused_pct": float(cell["abstain_pct"]),
        "n_credited": cell["n_credited"], "null_score": float(null) if null else None,
        "recall_at_5": mech["recall_at_5"], "hit_at_5": extra["hit_at_5"],
        "map_at_10": mech["map_at_10"],
        "parse_failure_rate": mech["parse_failure_rate"], "mean_llm_calls": mech["mean_llm_calls"],
        "mean_aspects": mech["mean_aspects"], "mean_evidence_tokens": mech["mean_evidence_tokens"],
        "s_per_question": mech["s_per_query"],
        "gpu": (summary.get("environment") or {}).get("gpu"),
    }


def against(base_path: Path, other_path: Path, metric: str, rng) -> tuple:
    """other minus base on the questions both score: (n, mean, CI low, CI high)."""
    d = paired(row_scores(base_path, metric), row_scores(other_path, metric))
    if not len(d):
        return 0, None, None, None
    lo, hi = bootstrap_ci(d, rng)
    return len(d), float(np.mean(d)), lo, hi


def collect(sweep: Path, reader_bank: Path, loaders=None) -> list[dict]:
    """Every finished cell of the sweep, smallest generator first, then the reader's."""
    from src.eval.runner import BENCHMARK_REGISTRY

    frozen = SWEEP_CODE.get(sweep.name)
    if frozen is None:
        _fail(f"{sweep} is not a registered sweep ({', '.join(SWEEP_CODE)})")
    if reader_bank.name != READER_BANK or FROZEN_CODE.get(READER_BANK) != frozen:
        _fail(f"the reader's cells are round 2's bank, {READER_BANK}, frozen at {frozen}")
    loaders = loaders or {b: BENCHMARK_REGISTRY[b]() for b in SWEEP_BENCHMARKS}
    rng = np.random.default_rng(SEED)
    out = []
    for benchmark in SWEEP_BENCHMARKS:
        golds, docs = golds_of(loaders[benchmark])
        reader_path = reader_bank / f"{benchmark}_OURS_validation.jsonl"
        reader = cell_numbers(reader_bank, settings.READER, benchmark, golds, docs)
        check_cell_code(reader["rows"], frozen, exact=True, where=f"{READER_BANK}/{benchmark}")
        for generator in settings.SWEEP_GENERATORS:
            bank = sweep_bank(sweep, generator)
            if not finished(bank, benchmark):
                out.append({"generator": generator, "params_b": PARAMS_B[generator],
                            "benchmark": benchmark, "not_run": True})
                continue
            row = cell_numbers(bank, generator, benchmark, golds, docs)
            check_cell_code(row["rows"], frozen, exact=False, where=f"{bank.name}/{benchmark}")
            path = bank / f"{benchmark}_OURS_validation.jsonl"
            n, row["answer_delta_vs_reader"], row["answer_ci_lo"], row["answer_ci_hi"] = against(
                reader_path, path, "answer", rng)
            row["n_paired"] = n
            if benchmark != "narrativeqa":
                (_, row["retrieval_delta_vs_reader"], row["retrieval_ci_lo"],
                 row["retrieval_ci_hi"]) = against(reader_path, path, "retrieval", rng)
            out.append(row)
        out.append(reader)
    return out


def baselines(sweep: Path, p10: Path, rows) -> list[dict]:
    """Ours under Qwen2.5-7B against the Qwen column's M1-M4, paired, where both have run."""
    rng = np.random.default_rng(SEED)
    out = []
    for benchmark in SWEEP_BENCHMARKS:
        if not finished(sweep_bank(sweep, QWEN), benchmark):
            continue
        path = sweep_bank(sweep, QWEN) / f"{benchmark}_OURS_validation.jsonl"
        metrics = ("answer",) if benchmark == "narrativeqa" else ("answer", "retrieval")
        for system in SYSTEMS:
            for metric in metrics:
                if system == "M1" and metric == "retrieval":
                    continue
                n, delta, lo, hi = against(p10 / f"{benchmark}_{system}_validation.jsonl", path, metric, rng)
                out.append({"benchmark": benchmark, "system": system, "metric": metric,
                            "n": n, "delta": delta, "lo": lo, "hi": hi})
    return out


def _fmt(v, dp=4):
    return "n/a" if v in ("", None) else f"{float(v):.{dp}f}"


def _ci(row, key):
    if row.get(f"{key}_delta_vs_reader") is None:
        return "—"
    return (f"{row[f'{key}_delta_vs_reader']:+.4f} "
            f"[{row[f'{key}_ci_lo']:+.4f}, {row[f'{key}_ci_hi']:+.4f}]")


def write_outputs(rows, out_dir: Path, comparison: Path | None = None, paired_baselines=None) -> None:
    with (out_dir / "GENERATORS_OURS.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            if not r.get("not_run"):
                writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})
    reader = short(settings.READER)
    lines = ["# GENERATORS_OURS — ours (v20) across generators", "",
             "Each generator is ours' one model: the reader, every query-side call, and the tree "
             f"summariser, with trees of its own. {reader} is round 2's cells of the same model. "
             "Descriptive: Δ is the generator's cell minus the reader's on the same questions, "
             "with a 95% bootstrap interval; no test, no verdict. Ours' R@5 ranks only its final "
             "context (at most 15 chunks).", ""]
    for benchmark in SWEEP_BENCHMARKS:
        mhr = benchmark == "multihop_rag"
        lines += [f"## {DATASET_LABEL[benchmark]}", "",
                  "| generator | params (B) | F1 | EM | refused | " + ("null | R@5 | " if mhr else "")
                  + f"Δ F1 vs {reader} [95% CI] | " + (f"Δ retrieval F1 vs {reader} [95% CI] | " if mhr else "")
                  + "parse failures | evidence tokens | s / question | GPU |",
                  "|---|---|---|---|---|" + ("---|---|" if mhr else "") + "---|" + ("---|" if mhr else "")
                  + "---|---|---|---|"]
        for r in rows:
            if r["benchmark"] != benchmark:
                continue
            if r.get("not_run"):
                lines.append(f"| {short(r['generator'])} | {r['params_b']} | not run yet |")
                continue
            f1 = _fmt(r["f1"]) + (f" ({_fmt(r['f1_supplementary'])})" if mhr else "")
            cells = [short(r["generator"]), f"{r['params_b']}", f1, _fmt(r["em"]),
                     f"{r['refused_pct']:.1f}%"]
            if mhr:
                cells += [_fmt(r["null_score"]), _fmt(r["recall_at_5"], 3)]
            cells.append(_ci(r, "answer") if r["generator"] != settings.READER else "(reference)")
            if mhr:
                cells.append(_ci(r, "retrieval") if r["generator"] != settings.READER else "(reference)")
            cells += [f"{r['parse_failure_rate']:.1%}", f"{r['mean_evidence_tokens']:.0f}",
                      _fmt(r["s_per_question"], 1), str(r["gpu"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if comparison is not None:
        matrix = [r for r in csv.DictReader(comparison.open(encoding="utf-8"))
                  if r["benchmark"] in SWEEP_BENCHMARKS and r["generator"] in (QWEN, settings.READER)]
        lines += ["## The matrix's baselines under the same generator", "",
                  "| generator | benchmark | system | F1 | EM | R@5 |", "|---|---|---|---|---|---|"]
        for r in matrix:
            f1 = _fmt(r["f1_primary"]) + (f" ({_fmt(r['f1_supplementary'])})"
                                          if r["benchmark"] == "multihop_rag" else "")
            lines.append(f"| {short(r['generator'])} | {DATASET_LABEL[r['benchmark']]} | {r['label']} | "
                         f"{f1} | {_fmt(r['em'])} | {_fmt(r['recall_at_5'], 3)} |")
        lines.append("")
    if paired_baselines:
        lines += [f"## Ours under {short(QWEN)} against the Qwen column, same questions", "",
                  "| benchmark | against | measure | n | Δ (ours − baseline) [95% CI] |",
                  "|---|---|---|---|---|"]
        for p in paired_baselines:
            delta = "—" if p["delta"] is None else f"{p['delta']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
            lines.append(f"| {DATASET_LABEL[p['benchmark']]} | {p['system']} | {p['metric']} F1 | "
                         f"{p['n']} | {delta} |")
        lines.append("")
    (out_dir / "GENERATORS_OURS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[generators] wrote {out_dir / 'GENERATORS_OURS.csv'} and .md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("table", help="GENERATORS_OURS.csv and .md into the sweep's directory.")
    t.add_argument("--sweep", type=Path, required=True, help=".../outputs/ours_v20_generators")
    t.add_argument("--reader", type=Path, required=True, help="Round 2's bank, .../outputs/ours_v20")
    t.add_argument("--comparison", type=Path, default=None, help="The matrix's COMPARISON.csv.")
    t.add_argument("--p10", type=Path, default=None, help="The Qwen column's bank, .../outputs/p10.")
    args = ap.parse_args()
    rows = collect(args.sweep, args.reader)
    paired_baselines = baselines(args.sweep, args.p10, rows) if args.p10 else None
    write_outputs(rows, args.sweep, args.comparison, paired_baselines)


if __name__ == "__main__":
    main()
