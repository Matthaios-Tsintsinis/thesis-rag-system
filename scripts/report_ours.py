"""Ours against the Llama cells (p11), read-only on every bank but ours.

  bar    the pooled MAP@10 Phase-2 target, from the six p11 pairs, after the
         same code reproduces the seven published Llama targets
  table  outputs/ours/COMPARISON_OURS.{csv,md}: F1 | EM | R@5 per benchmark,
         Llama M1-M4 through the existing exporter's functions (checked
         against outputs/COMPARISON.csv) plus ours, and ours' mechanics
  stats  Family O (the three declared comparisons, Holm, m = 3) and Family
         O-x (ours against M1-M4 on the grid's seven families, Holm, m = 28)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dev_ours import golds_of, ours_numbers, read_rows, recall_at
from scripts.export_comparison import (
    BENCHMARKS,
    CANONICAL_REFUSAL,
    CSV_COLUMNS,
    DATASET_LABEL,
    LLAMA,
    NULL_METHOD,
    RECORDED_CREDITED,
    ROW_ORDER,
    _em_and_hit5,
    _fail,
    _norm_pred,
    _recall_from_sidecar,
    checksum_line,
    read_cell,
)

SYSTEMS = ("M1", "M2", "M3", "M4")
GRID_FAMILIES = (("multihop_rag", "retrieval"), ("multihop_rag", "answer"),
                 ("narrativeqa", "answer"), ("hotpotqa", "retrieval"),
                 ("hotpotqa", "answer"), ("hotpotqa_pooled", "retrieval"),
                 ("hotpotqa_pooled", "answer"))

# The Llama column's Phase-2 targets as published (PROVENANCE_TABLE §5c).
PUBLISHED_TARGETS = {
    ("multihop_rag", "retrieval"): 0.0103, ("multihop_rag", "answer"): 0.0214,
    ("narrativeqa", "answer"): 0.0214, ("hotpotqa", "retrieval"): 0.0029,
    ("hotpotqa", "answer"): 0.0304, ("hotpotqa_pooled", "retrieval"): 0.0024,
    ("hotpotqa_pooled", "answer"): 0.0300,
}
# From `bar` before any ours row existed (ADDENDUM 7): 0.023978705859676434,
# rounded to 4 dp like the published targets.
POOLED_MAP10_BAR: float | None = 0.0240

# Family O: comparator, metric and bar, as declared in ADDENDUM 7.
FAMILY_O = (("multihop_rag", "retrieval", "M3"),
            ("narrativeqa", "answer", "M2"),
            ("hotpotqa_pooled", "map10", "M3"))

# Credited refusals per ours cell: recorded at the first export, then gated.
# HotpotQA cells credit nothing: the yes/no guard scores a refusal 0.
RECORDED_OURS: dict = {"hotpotqa": (0, 0.0), "hotpotqa_pooled": (0, 0.0)}
# Ours' code fingerprint at the freeze; every scored row must carry it.
# Frozen 2026-10-27 at 26a28e3: development v2, the model v2's held-out
# rows measured.
FROZEN_CODE: str | None = "d975038e7513cf11"

# The grid's constants (significance_diagnostic at the tag).
N_PERM = 10_000
N_BOOT = 10_000
MDE_Z = 2.80
SEED = 12345
ALPHA = 0.05


# --- per-row scores -----------------------------------------------------------


def row_scores(path: Path, metric: str) -> dict:
    """query id -> score for a metric; None where the row is not scored on it."""
    out = {}
    for r in read_rows(path):
        retrieval = r.get("retrieval") or {}
        skipped = retrieval.get("skipped", True)
        if metric == "answer":
            value = float(r["answer"]["value"])
        elif metric == "retrieval":
            value = None if skipped else float(retrieval.get("f1", 0.0))
        else:
            value = None if skipped else float(retrieval["map_at_k"]["10"])
        out[str(r["query_id"])] = value
    return out


def paired(a: dict, b: dict) -> np.ndarray:
    """b minus a over the queries both score."""
    keys = sorted(a.keys() & b.keys())
    return np.asarray([b[k] - a[k] for k in keys
                       if a[k] is not None and b[k] is not None], dtype=float)


# --- the Phase-2 target -----------------------------------------------------


def target(deltas) -> float:
    """2.8 x median SD of the paired differences / sqrt(median n)."""
    sds = [float(np.std(d, ddof=1)) for d in deltas]
    ns = [len(d) for d in deltas]
    return MDE_Z * float(np.median(sds)) / float(np.sqrt(int(np.median(ns))))


def family_target(bank: Path, benchmark: str, metric: str) -> float:
    """The target over the six pairs of a bank's four cells."""
    scores = {s: row_scores(bank / f"{benchmark}_{s}_validation.jsonl", metric)
              for s in SYSTEMS}
    deltas = [paired(scores[a], scores[b])
              for a, b in itertools.combinations(SYSTEMS, 2)]
    return target([d for d in deltas if len(d)])


def check_published(bank: Path, families=PUBLISHED_TARGETS) -> None:
    """Refuse unless this code reproduces every published target from the bank."""
    for (benchmark, metric), published in families.items():
        got = round(family_target(bank, benchmark, metric), 4)
        if got != published:
            raise SystemExit(f"[report] REFUSED: {benchmark}/{metric} target "
                             f"{got} != published {published}")
        print(f"[report] {benchmark}/{metric} target {got:.4f} reproduces")


# --- paired tests and Holm (the grid's rules) -------------------------------


def permutation_p(d: np.ndarray, rng) -> float:
    """Two-sided paired sign-flip permutation test on the mean difference."""
    obs = abs(float(np.mean(d)))
    signs = rng.choice((-1.0, 1.0), size=(N_PERM, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (N_PERM + 1))


def bootstrap_ci(d: np.ndarray, rng) -> tuple[float, float]:
    """Percentile 95% CI on the mean difference, resampling queries."""
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order kept."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def run_family(tests, rng, *, directional: bool) -> list[dict]:
    """Paired tests for (name, comparator scores, ours scores, bar), Holm across them."""
    rows = []
    for name, comparator, ours, bar in tests:
        d = paired(comparator, ours)
        sd = float(np.std(d, ddof=1))
        lo, hi = bootstrap_ci(d, rng)
        rows.append({"name": name, "n": len(d), "delta": float(np.mean(d)),
                     "sd": sd, "mde": MDE_Z * sd / np.sqrt(len(d)), "lo": lo,
                     "hi": hi, "p": permutation_p(d, rng), "bar": bar})
    for row, p_holm in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p_holm
        significant = p_holm < ALPHA
        if directional:
            # Declared direction: ours higher, so the CI must sit above zero.
            row["verdict"] = "SUPPORTED" if significant and row["lo"] > 0 else "NOT SUPPORTED"
        else:
            excludes = row["lo"] > 0 or row["hi"] < 0
            row["verdict"] = ("SUPPORTED" if significant and excludes
                              else "marginal" if excludes or abs(row["delta"]) >= row["mde"]
                              else "NOT SUPPORTED")
        row["below_bar"] = bar is not None and abs(row["delta"]) < bar
    return rows


# --- gates on ours' cells -----------------------------------------------------


def check_doc_ranking(rows, gold_docs) -> None:
    """Every row's document ranking must give the scorer's own hit@K and MRR."""
    from scripts.replay_retrieval import rank_stats_from_ranking

    for r in rows:
        gold = gold_docs.get(str(r["query_id"]))
        if not gold:
            continue
        banked = {int(k): v for k, v in r["retrieval"]["hit_at_k"].items()}
        stats = rank_stats_from_ranking(r["metadata"]["ours_doc_ranking"], gold,
                                        tuple(banked))
        if stats["hit_at_k"] != banked or stats["mrr"] != r["retrieval"]["mrr"]:
            raise SystemExit(f"[report] REFUSED: {r['query_id']}: the document "
                             "ranking does not reproduce the row's hit@K / MRR")


def check_code(cells: dict, frozen: str | None) -> None:
    """Every row of every ours cell carries one code fingerprint (the frozen one, once set)."""
    codes = {r["metadata"]["ours_code"] for rows in cells.values() for r in rows}
    if len(codes) != 1:
        raise SystemExit(f"[report] REFUSED: ours cells carry codes {sorted(codes)}")
    if frozen is not None and codes != {frozen}:
        raise SystemExit(f"[report] REFUSED: ours code {codes.pop()} != frozen {frozen}")


def credited(rows) -> tuple[int, float]:
    """Credited refusals: the canonical refusal scoring above zero, as the exporter counts them."""
    n, mass = 0, 0.0
    for r in rows:
        a = r["answer"]
        if a.get("method") == NULL_METHOD:
            continue
        if ((a.get("metadata") or {}).get("abstained")
                and _norm_pred(r.get("predicted_answer")) == CANONICAL_REFUSAL
                and float(a["value"]) > 0):
            n += 1
            mass += float(a["value"])
    return n, mass


def recorded_battery(benchmark, rows, recorded=RECORDED_OURS) -> tuple[int, float]:
    """The recorded credited-refusal count for an ours cell; refuse until it is recorded."""
    if benchmark not in recorded:
        raise SystemExit(f"[report] REFUSED: no recorded battery for ours/{benchmark}; "
                         f"record RECORDED_OURS[{benchmark!r}] = {credited(rows)} "
                         "in scripts/report_ours.py and export again")
    return recorded[benchmark]


# --- the table ------------------------------------------------------------------


def _csv_row(benchmark, system, label, cell, extra, recall) -> dict:
    """A COMPARISON-shaped row, formatted as the exporter formats it."""
    def fmt(v):
        return "" if v is None else repr(round(v, 10))
    return {"generator": LLAMA, "benchmark": benchmark, "system": system,
            "label": label, "f1_primary": cell["mean_answer_score_answerable"],
            "f1_supplementary": cell["supplementary_mean"],
            "em": fmt(extra["em"]), "n_em_population": extra["n_em_population"],
            "hit_at_5": fmt(extra["hit_at_5"]),
            "recall_at_1": fmt(recall and recall["1"]),
            "recall_at_5": fmt(recall and recall["5"]),
            "recall_at_10": fmt(recall and recall["10"]),
            "n_rank_population": ("" if extra["n_rank_population"] is None
                                  else extra["n_rank_population"])}


def llama_row(p11: Path, benchmark, system, label, gold_map) -> dict:
    """One p11 cell through the existing exporter's functions."""
    stem = f"{benchmark}_{system}_validation"
    cell = read_cell(p11, LLAMA, benchmark, system, RECORDED_CREDITED)
    summary = json.loads((p11 / f"{stem}.summary.json").read_text(encoding="utf-8"))
    extra = _em_and_hit5(p11, benchmark, system, gold_map, summary)
    ranked = system != "M1" and benchmark != "narrativeqa"
    recall = _recall_from_sidecar(p11, stem, extra["n_rank_population"]) if ranked else None
    return _csv_row(benchmark, system, label, cell, extra, recall)


def ours_row(ours_dir: Path, benchmark, gold_map, gold_docs, rows) -> dict:
    """Ours' cell: the exporter's F1 and EM, R@K from the rows' document ranking."""
    battery = recorded_battery(benchmark, rows)
    cell = read_cell(ours_dir, LLAMA, benchmark, "OURS", {(LLAMA, benchmark, "OURS"): battery})
    summary = json.loads((ours_dir / f"{benchmark}_OURS_validation.summary.json")
                         .read_text(encoding="utf-8"))
    extra = _em_and_hit5(ours_dir, benchmark, "OURS", gold_map, summary)
    recall = None
    if benchmark != "narrativeqa":
        check_doc_ranking(rows, gold_docs)
        ranked = [r for r in rows if gold_docs.get(str(r["query_id"]))]
        if len(ranked) != extra["n_rank_population"]:
            _fail(f"ours/{benchmark}: {len(ranked)} ranked rows != rank population "
                  f"{extra['n_rank_population']}")
        recall = {str(k): sum(recall_at(r["metadata"]["ours_doc_ranking"],
                                        gold_docs[str(r["query_id"])], k)
                              for r in ranked) / len(ranked) for k in (1, 5, 10)}
        # recall@5 never exceeds hit@5; with two gold titles it is at least half.
        if recall["5"] > extra["hit_at_5"] + 1e-12 or (
                benchmark != "multihop_rag" and recall["5"] < extra["hit_at_5"] / 2 - 1e-12):
            _fail(f"ours/{benchmark}: recall@5 {recall['5']} outside its bounds "
                  f"against hit@5 {extra['hit_at_5']}")
    return _csv_row(benchmark, "OURS", "ours", cell, extra, recall)


def table(p11: Path, ours_dir: Path, comparison: Path) -> None:
    """Write COMPARISON_OURS.csv and .md into the ours bank."""
    from scripts.export_comparison import _gold_maps_from_loaders
    from src.eval.runner import BENCHMARK_REGISTRY

    if ours_dir.resolve() == p11.resolve() or ours_dir.name in ("p10", "p11"):
        _fail(f"the ours bank {ours_dir} is a banked column")
    published = {(r["benchmark"], r["system"]): r
                 for r in csv.DictReader(comparison.open(encoding="utf-8"))
                 if r["generator"] == LLAMA}
    gold_maps = _gold_maps_from_loaders()
    rows_out, mechanics, cells = [], {}, {}
    for benchmark in BENCHMARKS:
        for label, system in ROW_ORDER:
            row = llama_row(p11, benchmark, system, label, gold_maps[benchmark])
            if any(str(row[c]) != published[(benchmark, system)][c] for c in CSV_COLUMNS):
                _fail(f"{benchmark}/{system}: recomputed row differs from {comparison}")
            rows_out.append(row)
        rows = read_rows(ours_dir / f"{benchmark}_OURS_validation.jsonl")
        cells[benchmark] = rows
        golds, docs = golds_of(BENCHMARK_REGISTRY[benchmark]())
        rows_out.append(ours_row(ours_dir, benchmark, gold_maps[benchmark], docs, rows))
        mechanics[benchmark] = ours_numbers(rows, golds, docs)
    check_code(cells, FROZEN_CODE)
    write_outputs(rows_out, mechanics, ours_dir)


def write_outputs(rows, mechanics, out_dir: Path) -> None:
    """The CSV holds every column; the markdown shows F1 | EM | R@5 and ours' mechanics."""
    with (out_dir / "COMPARISON_OURS.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v, dp=4):
        return "n/a" if v in ("", None) else f"{float(v):.{dp}f}"

    lines = ["# COMPARISON_OURS — ours against the Llama cells", "",
             "Llama M1-M4 rows come from the banked p11 cells through the existing "
             "exporter's functions and equal outputs/COMPARISON.csv. Ours' R@5 ranks "
             "only its final context (at most 15 chunks, ordered by the selection "
             "score) where M2-M4 rank 50 deep; a gold document outside the context "
             "counts as missed, and a context with fewer than five documents is "
             "scored over what it holds.", "", f"`{checksum_line(rows)}`", ""]
    for benchmark in BENCHMARKS:
        lines += [f"## {DATASET_LABEL[benchmark]} — {LLAMA}", "",
                  "| system | F1 | EM | R@5 |", "|---|---|---|---|"]
        for r in rows:
            if r["benchmark"] == benchmark:
                f1 = fmt(r["f1_primary"])
                if benchmark == "multihop_rag":
                    f1 += f" ({fmt(r['f1_supplementary'])})"
                lines.append(f"| {r['label']} | {f1} | {fmt(r['em'])} | "
                             f"{fmt(r['recall_at_5'], 3)} |")
        lines.append("")
    lines += ["## Ours' mechanics", "",
              "| benchmark | parse failures | LLM calls | aspects | low-confidence "
              "aspects | evidence chunks | evidence tokens | warm trees | section "
              "detectors |", "|---|---|---|---|---|---|---|---|---|"]
    for benchmark, m in mechanics.items():
        detectors = ", ".join(f"{k} {v:.1%}" for k, v in m["section_detector_share"].items())
        lines.append(f"| {DATASET_LABEL[benchmark]} | {m['parse_failure_rate']:.1%} | "
                     f"{m['mean_llm_calls']:.2f} | {m['mean_aspects']:.2f} | "
                     f"{m['low_confidence_share']:.1%} | {m['mean_evidence_chunks']:.1f} | "
                     f"{m['mean_evidence_tokens']:.0f} | {m['tree_cache_hit_share']:.1%} | "
                     f"{detectors} |")
    (out_dir / "COMPARISON_OURS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {out_dir / 'COMPARISON_OURS.csv'} and .md")


# --- the families ---------------------------------------------------------------


def stats(p11: Path, ours_dir: Path) -> list[dict]:
    """Family O (directional, Holm m = 3), then Family O-x (exploratory, Holm m = 28)."""
    if POOLED_MAP10_BAR is None:
        raise SystemExit("[report] REFUSED: POOLED_MAP10_BAR is unset; run `bar` "
                         "and record it before any ours row exists")
    bars = {("multihop_rag", "retrieval"): PUBLISHED_TARGETS[("multihop_rag", "retrieval")],
            ("narrativeqa", "answer"): PUBLISHED_TARGETS[("narrativeqa", "answer")],
            ("hotpotqa_pooled", "map10"): POOLED_MAP10_BAR}

    def scores(bank, benchmark, system, metric):
        return row_scores(bank / f"{benchmark}_{system}_validation.jsonl", metric)

    rng = np.random.default_rng(SEED)
    family_o = run_family(
        [(f"{b} {m}: ours vs {c}", scores(p11, b, c, m), scores(ours_dir, b, "OURS", m),
          bars[(b, m)]) for b, m, c in FAMILY_O], rng, directional=True)
    family_x = run_family(
        [(f"{b} {m}: ours vs {c}", scores(p11, b, c, m), scores(ours_dir, b, "OURS", m), None)
         for b, m in GRID_FAMILIES for c in SYSTEMS], rng, directional=False)
    lines = []
    for title, rows in (("Family O (declared, Holm m = 3)", family_o),
                        ("Family O-x (exploratory, Holm m = 28)", family_x)):
        lines += [f"=== {title} ===",
                  f"{'comparison':40s} {'n':>5s} {'delta':>8s} {'SD':>7s} {'MDE':>7s} "
                  f"{'95% CI':>19s} {'p':>7s} {'p_holm':>7s}  verdict"]
        for r in rows:
            note = "  (below its bar: undetectable at this n)" if r["below_bar"] else ""
            lines.append(f"{r['name']:40s} {r['n']:5d} {r['delta']:+8.4f} {r['sd']:7.4f} "
                         f"{r['mde']:7.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['p']:7.4f} "
                         f"{r['p_holm']:7.4f}  {r['verdict']}{note}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (ours_dir / "STATS_OURS.txt").write_text(text, encoding="utf-8")
    return family_o + family_x


def main() -> None:
    """Parse the subcommand and run it."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bar", help="Pooled MAP@10 target from the p11 pairs.")
    b.add_argument("--p11", type=Path, required=True)
    t = sub.add_parser("table", help="COMPARISON_OURS.csv and .md into the ours bank.")
    s = sub.add_parser("stats", help="Family O and Family O-x.")
    for p in (t, s):
        p.add_argument("--p11", type=Path, required=True)
        p.add_argument("--ours", type=Path, required=True)
    t.add_argument("--comparison", type=Path, default=None,
                   help="The existing COMPARISON.csv (default: beside p11).")
    args = ap.parse_args()
    if args.command == "bar":
        check_published(args.p11)
        bar = family_target(args.p11, "hotpotqa_pooled", "map10")
        print(f"[report] POOLED MAP@10 TARGET: {bar:.4f} (unrounded {bar!r})")
    elif args.command == "table":
        table(args.p11, args.ours, args.comparison or args.p11.parent / "COMPARISON.csv")
    else:
        stats(args.p11, args.ours)


if __name__ == "__main__":
    main()
