"""Export COMPARISON.md and COMPARISON.csv: per-dataset tables of F1, EM and
R@5 for M1/M2/M4/M3 under both generators, read from the banked cells, the
replay sidecars and the loaders' gold."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.scorers.extractive import normalize_qasper_answer

QWEN = "Qwen/Qwen2.5-7B-Instruct"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
BENCHMARKS = ("multihop_rag", "narrativeqa", "hotpotqa", "hotpotqa_pooled")
SYSTEMS = ("M1", "M2", "M3", "M4")
# harness choice: the string the null rule recognises (METHODS §C.9)
CANONICAL_REFUSAL = "no answer available"
NULL_METHOD = "unanswerable_rule"

# Recorded credited-refusal (count, mass) per cell. read_cell recomputes
# both from the rows and refuses the cell on any disagreement. NarrativeQA
# masses are recorded at 2 dp; MultiHop masses are exact halves.
RECORDED_CREDITED: dict[tuple[str, str, str], tuple[int, float]] = {
    (QWEN, "multihop_rag", "M1"): (315, 157.5),
    (QWEN, "multihop_rag", "M2"): (462, 231.0),
    (QWEN, "multihop_rag", "M3"): (458, 229.0),
    (QWEN, "multihop_rag", "M4"): (504, 252.0),
    (QWEN, "narrativeqa", "M1"): (7, 1.58),
    (QWEN, "narrativeqa", "M2"): (3, 0.68),
    (QWEN, "narrativeqa", "M3"): (3, 0.77),
    (QWEN, "narrativeqa", "M4"): (6, 1.36),
    (LLAMA, "multihop_rag", "M1"): (558, 279.0),
    (LLAMA, "multihop_rag", "M2"): (506, 253.0),
    (LLAMA, "multihop_rag", "M3"): (510, 255.0),
    (LLAMA, "multihop_rag", "M4"): (496, 248.0),
    (LLAMA, "narrativeqa", "M1"): (7, 1.58),
    (LLAMA, "narrativeqa", "M2"): (6, 1.30),
    (LLAMA, "narrativeqa", "M3"): (6, 1.44),
    (LLAMA, "narrativeqa", "M4"): (6, 1.36),
}
# HotpotQA cells credit nothing: the yes/no guard forces exact match on
# yes/no golds, so a refusal never scores above zero there.
# official: hotpot_evaluate_v1.py::f1_score @ 36358534 (both early returns)
for _gen in (QWEN, LLAMA):
    for _bench in ("hotpotqa", "hotpotqa_pooled"):
        for _sys in SYSTEMS:
            RECORDED_CREDITED[(_gen, _bench, _sys)] = (0, 0.0)

MASS_TOLERANCE = 0.005  # the record rounds NarrativeQA masses to 2 dp


def _fail(msg: str) -> None:
    """Stop the export with a refusal message."""
    raise SystemExit(f"[export] REFUSED: {msg}")


def _norm_pred(text: str) -> str:
    """Lower-case a prediction and drop its trailing full stop."""
    return (text or "").strip().lower().rstrip(".")


def read_cell(bank: Path, generator: str, benchmark: str, system: str,
              recorded: dict[tuple[str, str, str], tuple[int, float]]) -> dict:
    """Read one cell's summary and rows into a row; refuse on any gap."""
    stem = f"{benchmark}_{system}_validation"
    spath = bank / f"{stem}.summary.json"
    jpath = bank / f"{stem}.jsonl"
    if not spath.is_file():
        _fail(f"missing summary {spath}")
    if not jpath.is_file():
        _fail(f"missing rows {jpath}")
    s = json.loads(spath.read_text(encoding="utf-8"))

    # Refuse a summary that is partial, from the wrong bank or system, or
    # short of its expected population.
    if s.get("partial_run"):
        _fail(f"{stem}: partial_run is true")
    if s.get("generator") != generator:
        _fail(f"{stem}: summary generator {s.get('generator')!r} != bank's "
              f"{generator!r}")
    if s.get("system") != system or s.get("benchmark") != benchmark:
        _fail(f"{stem}: identity mismatch in summary")
    if s.get("n_queries_scored") != s.get("expected_n_queries"):
        _fail(f"{stem}: n_queries_scored {s.get('n_queries_scored')} != "
              f"expected {s.get('expected_n_queries')}")

    # Walk the answerable rows: split abstentions from plain answers and
    # total the credited refusals (the canonical refusal scoring above 0).
    n_answerable = int(s["n_answerable"])
    n_abstained = 0
    n_credited = 0
    credited_mass = 0.0
    plain_sum = 0.0
    n_plain = 0
    with jpath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            a = r["answer"]
            if a.get("method") == NULL_METHOD:
                continue  # null rows: their mean is a summary field
            value = float(a["value"])
            abstained = bool((a.get("metadata") or {}).get("abstained"))
            if abstained:
                n_abstained += 1
                if (_norm_pred(r.get("predicted_answer")) == CANONICAL_REFUSAL
                        and value > 0):
                    n_credited += 1
                    credited_mass += value
            else:
                n_plain += 1
                plain_sum += value

    # The row counts must match the summary and the recorded battery.
    if n_abstained + n_plain != n_answerable:
        _fail(f"{stem}: answerable rows {n_abstained + n_plain} != summary "
              f"n_answerable {n_answerable}")
    exp_n, exp_mass = recorded[(generator, benchmark, system)]
    if n_credited != exp_n or abs(credited_mass - exp_mass) > MASS_TOLERANCE:
        _fail(f"{stem}: recomputed credited ({n_credited}, "
              f"{credited_mass:.4f}) disagrees with the recorded battery "
              f"({exp_n}, {exp_mass}) -- wrong rows or drifted rule; a "
              "matrix over unverified cells must not exist")

    # harness choice: primary minus credited mass over the same denominator (METHODS §C.1)
    primary = float(s["mean_answer_score_answerable"])
    supplementary = primary - (credited_mass / n_answerable if n_answerable else 0.0)

    # No retrieval figure for M1 (no retriever) or NarrativeQA.
    # dataset: no passage annotation, retrieval never scored
    if system == "M1":
        retr, absence = "", "no_retrieval"
    elif benchmark == "narrativeqa":
        retr, absence = "", "no_gold"
    else:
        retr, absence = repr(float(s["mean_retrieval_f1"])), ""

    null_mean = s.get("mean_answer_score_null")
    return {
        "generator": generator,
        "benchmark": benchmark,
        "system": system,
        "mean_retrieval_f1": retr,
        "retrieval_absence": absence,
        "mean_answer_score": repr(float(s["mean_answer_score"])),
        "mean_answer_score_answerable": repr(primary),
        "supplementary_mean": repr(supplementary),
        "mean_answer_score_null": "" if null_mean is None else repr(float(null_mean)),
        "n_queries": int(s["n_queries_scored"]),
        "n_credited": n_credited,
        "credited_mass": repr(round(credited_mass, 10)),
        "abstain_pct": repr(round(100.0 * n_abstained / n_answerable, 4)),
        "mean_plain": repr(round(plain_sum / n_plain, 10)) if n_plain else "",
        "elapsed_s": repr(float(s["elapsed_s"])),
        "git_commit": s.get("git_commit", ""),
        "timestamp": s.get("timestamp", ""),
        "python": (s.get("environment") or {}).get("python", ""),
        "lockfile_hash": (s.get("environment") or {}).get("lockfile_hash", ""),
        "generator_revision": ((s.get("model_revisions") or {})
                               .get("revisions") or {}).get("generator", ""),
    }


def _generating_commit() -> str:
    """Short hash of the checked-out commit, or "unknown" outside git."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[1]
                              ).stdout.strip()
    except Exception:
        return "unknown"


# Row labels in table order; bm25-hybrid is M3, the dense+BM25 RRF system.
ROW_ORDER = (("LLM", "M1"), ("flat", "M2"), ("raptor", "M4"),
             ("bm25-hybrid", "M3"))
DATASET_LABEL = {"multihop_rag": "MHR", "narrativeqa": "NQA",
                 "hotpotqa": "HPQA-distractor",
                 "hotpotqa_pooled": "HPQA-pooled"}
CSV_COLUMNS = ["generator", "benchmark", "system", "label",
               "f1_primary", "f1_supplementary", "em", "n_em_population",
               "hit_at_5", "recall_at_1", "recall_at_5", "recall_at_10",
               "n_rank_population"]


def _recall_from_sidecar(bank: Path, stem: str, expected_n: int) -> dict:
    """Mean recall@1/5/10 over the replay sidecar; refuse if it is absent."""
    side = bank / f"rankings.{stem}.jsonl"
    if not side.is_file():
        _fail(f"{stem}: sidecar {side} is MISSING -- run "
              "scripts.replay_retrieval first; by ruling this exporter "
              "refuses rather than falling back to hit@5")
    # Sum recall@K over the rows and check the row count against the cell.
    sums = {"1": 0.0, "5": 0.0, "10": 0.0}
    n = 0
    with side.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k in sums:
                sums[k] += float(r["recall_at_k"][k])
            n += 1
    if n != expected_n:
        _fail(f"{stem}: sidecar holds {n} rows, expected {expected_n}")
    return {k: sums[k] / n for k in sums}


def _em_and_hit5(bank: Path, benchmark: str, system: str,
                 gold_map: dict[str, tuple[str, ...]],
                 summary: dict) -> dict:
    """EM and hit@5 for one cell from its rows; EM is gated on HotpotQA."""
    jpath = bank / f"{benchmark}_{system}_validation.jsonl"
    em_sum = 0
    n_em = 0
    hit_sum = 0.0
    n_rank = 0
    # Recompute EM per answerable row: normalise both sides, max over
    # references.
    # official: hotpot_evaluate_v1.py::normalize_answer @ 36358534
    # harness extension (inert on ASCII): see METHODS §C.11
    # NarrativeQA paper: max over the two references
    with jpath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["answer"].get("method") != NULL_METHOD:
                qid = str(r["query_id"])
                if qid not in gold_map:
                    _fail(f"{benchmark}/{system}: query {qid!r} has no gold "
                          "in the loader map -- loader/bank mismatch")
                pred = normalize_qasper_answer(r.get("predicted_answer") or "")
                em_row = max(int(pred == normalize_qasper_answer(g))
                             for g in gold_map[qid])
                if benchmark.startswith("hotpotqa"):
                    # Every HotpotQA row banks exact_match; the recomputed
                    # EM must equal it row by row or the cell is refused.
                    # harness choice: recomputation must reproduce the banked value
                    banked = (r["answer"].get("metadata") or {}).get(
                        "exact_match")
                    if banked is None:
                        _fail(f"{benchmark}/{system}: row {qid!r} carries "
                              "no answer.metadata.exact_match -- the "
                              "recomputed EM cannot be gated; refusing")
                    if int(round(float(banked))) != em_row:
                        _fail(f"{benchmark}/{system}: row {qid!r}: "
                              f"recomputed EM {em_row} != banked "
                              f"exact_match {banked!r} -- drifted "
                              "normaliser or wrong rows; refusing")
                em_sum += em_row
                n_em += 1
            # hit@5 from the banked retrieval block, ranked cells only.
            # official: retrieval_evaluate.py @ cde8e844 (Hits@4, Hits@10); K = 1, 5 are ours
            if system != "M1" and benchmark != "narrativeqa":
                retr = r.get("retrieval") or {}
                hk = retr.get("hit_at_k") or {}
                if not retr.get("skipped") and ("5" in hk or 5 in hk):
                    hit_sum += float(hk.get("5", hk.get(5)))
                    n_rank += 1
    # The EM population matches the F1 column beside it: answerable rows on
    # MultiHop, all rows elsewhere.
    expected_em = (int(summary["n_answerable"])
                   if benchmark == "multihop_rag"
                   else int(summary["n_queries_scored"]))
    if n_em != expected_em:
        _fail(f"{benchmark}/{system}: EM population {n_em} != expected "
              f"{expected_em}")
    out = {"em": em_sum / n_em, "n_em_population": n_em}
    if system != "M1" and benchmark != "narrativeqa":
        # Rank-aware rows are exactly the answerable rows on a ranked
        # benchmark, so n_answerable sets the expected count. A summary
        # that also declares n_retrieval_scored is cross-checked against
        # it; the row-side count is the hard gate either way.
        expected_rank = int(summary["n_answerable"])
        declared = summary.get("n_retrieval_scored")
        if declared is not None and int(declared) != expected_rank:
            _fail(f"{benchmark}/{system}: summary declares "
                  f"n_retrieval_scored {declared} against n_answerable "
                  f"{expected_rank} -- self-inconsistent summary")
        if n_rank != expected_rank:
            _fail(f"{benchmark}/{system}: rank population {n_rank} != "
                  f"expected {expected_rank} (derived from n_answerable)")
        out.update({"hit_at_5": hit_sum / n_rank, "n_rank_population": n_rank})
    else:
        out.update({"hit_at_5": None, "n_rank_population": None})
    return out


def build_comparison(p10: Path, p11: Path,
                     gold_maps: dict[str, dict[str, tuple[str, ...]]],
                     recorded: dict | None = None) -> list[dict]:
    """Assemble the CSV rows, one per generator x benchmark x system."""
    rows = []
    # One row per cell: F1 from read_cell, EM and hit@5 from the rows,
    # recall@K from the sidecar on ranked cells.
    for generator, bank in ((QWEN, p10), (LLAMA, p11)):
        for benchmark in BENCHMARKS:
            for label, system in ROW_ORDER:
                cell = read_cell(bank, generator, benchmark, system,
                                 recorded if recorded is not None
                                 else RECORDED_CREDITED)
                stem = f"{benchmark}_{system}_validation"
                summary = json.loads((bank / f"{stem}.summary.json")
                                     .read_text(encoding="utf-8"))
                extra = _em_and_hit5(bank, benchmark, system,
                                     gold_maps[benchmark], summary)
                ranked_cell = system != "M1" and benchmark != "narrativeqa"
                recall = (_recall_from_sidecar(
                              bank, stem, extra["n_rank_population"])
                          if ranked_cell else None)
                # recall@5 never exceeds hit@5, and on a two-gold benchmark
                # it is at least half of it.
                if ranked_cell:
                    h5 = float(extra["hit_at_5"])
                    if recall["5"] > h5 + 1e-12:
                        _fail(f"{stem}: recall@5 {recall['5']} > hit@5 "
                              f"{h5} -- impossible; a bug, not a finding")
                    if (benchmark != "multihop_rag"
                            and recall["5"] < h5 / 2 - 1e-12):
                        _fail(f"{stem}: recall@5 {recall['5']} < hit@5/2 "
                              "on a two-gold benchmark -- impossible")
                rows.append({
                    "generator": generator, "benchmark": benchmark,
                    "system": system, "label": label,
                    "f1_primary": cell["mean_answer_score_answerable"],
                    "f1_supplementary": cell["supplementary_mean"],
                    "em": repr(round(extra["em"], 10)),
                    "n_em_population": extra["n_em_population"],
                    "hit_at_5": ("" if extra["hit_at_5"] is None
                                 else repr(round(extra["hit_at_5"], 10))),
                    "recall_at_1": ("" if recall is None
                                    else repr(round(recall["1"], 10))),
                    "recall_at_5": ("" if recall is None
                                    else repr(round(recall["5"], 10))),
                    "recall_at_10": ("" if recall is None
                                     else repr(round(recall["10"], 10))),
                    "n_rank_population": ("" if extra["n_rank_population"]
                                          is None
                                          else extra["n_rank_population"]),
                })
    return rows


def checksum_line(rows: list[dict]) -> str:
    """Row count plus the non-null count per column, printed and embedded."""
    nn = {c: sum(1 for r in rows if str(r[c]) != "") for c in CSV_COLUMNS}
    return ("COMPARISON checksum: rows=%d non-null " % len(rows)
            + " ".join(f"{c}:{nn[c]}" for c in CSV_COLUMNS))


def _fmt(v: str, dp: int = 4) -> str:
    """Format a CSV number to dp places, or n/a when the field is blank."""
    return f"{float(v):.{dp}f}" if v != "" else "n/a"


def write_outputs(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """Write COMPARISON.csv and COMPARISON.md to out_dir; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "COMPARISON.csv"
    md_path = out_dir / "COMPARISON.md"
    # The CSV holds every field in CSV_COLUMNS order, one row per cell.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    line = checksum_line(rows)
    # The markdown carries a provenance header, the checksum line, then one
    # F1 | EM | R@5 table per generator and benchmark.
    with md_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# COMPARISON — per-dataset tables, both generators\n\n")
        f.write(f"Generated by `scripts/export_comparison.py` @ "
                f"`{_generating_commit()}` on {date.today().isoformat()}, "
                "from the banked summaries and rows. **EM is POST-HOC and "
                "was never a pre-registered metric**; frozen normaliser "
                "(official composition + NFKC), max over references; "
                "population matches the F1 column beside it (MultiHop: "
                "answerable rows only). **EM is immune to the "
                "credited-refusal artifact by construction** — a bare "
                "\"no\" gold against the canonical refusal string is NOT "
                "an EM match. **R@5 = recall@5 from replayed "
                "rankings, gated against banked hit@K/MRR/set-F1** "
                "(scripts/replay_retrieval.py sidecars; the exporter "
                "REFUSES when a sidecar is missing, never falls back to "
                "hit@5, which now lives in the CSV only). \"bm25\" = M3, the hybrid dense+BM25 RRF "
                "system. Retrieval figures are generator-identical by "
                "construction for M2/M3 (bit-identity proven). M4's SET-F1 "
                "is tree-dependent (summary nodes consume its 2,000-token "
                "evidence budget, so the packed leaf set differs per "
                "generator); M4's RANK metrics are generator-invariant up "
                "to window truncation (summary nodes carry no provenance "
                "and rank no document; leaf order depends only on leaf and "
                "query embeddings, shared across columns; the only tree "
                "effect is summary-node intrusion shortening the 50-node "
                "scoring window's document tail — compare the CSV's rank "
                "columns across generators per benchmark). "
                "All comparisons EXPLORATORY under Declaration 3.\n\n")
        f.write(f"`{line}`\n\n")
        for generator in (QWEN, LLAMA):
            for benchmark in BENCHMARKS:
                sub = [r for r in rows if r["generator"] == generator
                       and r["benchmark"] == benchmark]
                f.write(f"## {DATASET_LABEL[benchmark]} — {generator}\n\n")
                f.write("| system | F1 | EM | R@5 |\n|---|---|---|---|\n")
                for r in sub:
                    f1 = _fmt(r["f1_primary"])
                    if benchmark == "multihop_rag":
                        f1 += f" ({_fmt(r['f1_supplementary'])})"
                    f.write(f"| {r['label']} | {f1} | {_fmt(r['em'])} | "
                            f"{_fmt(r['recall_at_5'], 3)} |\n")
                f.write("\n")
        f.write("MultiHop F1 cells show primary (supplementary) per "
                "caption 7. R@5 n/a for LLM rows (no retrieval) and all "
                "NQA rows (no retrieval ground truth); hit@5 and "
                "recall@1/@10 are recorded in the CSV, not tabled.\n")
    return csv_path, md_path


def gold_texts(q) -> tuple[str, ...]:
    """Gold texts for one EvalQuery, read from GoldAnswer.free_form."""
    return tuple(g.free_form for g in (q.gold_answers or ())
                 if getattr(g, "free_form", ""))


def _gold_maps_from_loaders() -> dict[str, dict[str, tuple[str, ...]]]:
    """Map query id to gold texts per benchmark from the registered loaders."""
    from src.eval.runner import BENCHMARK_REGISTRY
    maps: dict[str, dict[str, tuple[str, ...]]] = {}
    # Keep only queries that carry at least one gold text.
    for benchmark in BENCHMARKS:
        m: dict[str, tuple[str, ...]] = {}
        loader = BENCHMARK_REGISTRY[benchmark]()
        for unit in loader.iter_eval_units(split="validation"):
            for q in unit.queries:
                golds = gold_texts(q)
                if golds:
                    m[str(q.query_id)] = golds
        maps[benchmark] = m
        print(f"[compare] {benchmark}: gold for {len(m)} queries")
    return maps


def main() -> None:
    """Build the rows, write both outputs and print the checksum."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p10", required=True)
    ap.add_argument("--p11", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = build_comparison(Path(args.p10), Path(args.p11),
                            _gold_maps_from_loaders())
    csv_path, md_path = write_outputs(rows, Path(args.out))
    print(f"[compare] wrote {csv_path}")
    print(f"[compare] wrote {md_path}")
    print(checksum_line(rows))


if __name__ == "__main__":
    main()
