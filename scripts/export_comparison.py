"""Export the per-dataset comparison tables (COMPARISON.md + .csv):
rows LLM/flat/raptor/bm25-hybrid per generator, columns F1 | EM | hit@5
(labelled hit@5 everywhere, by ruling -- never R@5).

POST-HOC, AND SAYS SO. EM was never a pre-registered metric; it is
computed here after the fact from the banked JSONL rows, under the
FROZEN normaliser (`normalize_qasper_answer`: the official
SQuAD/HotpotQA composition + the declared NFKC extension), max over
references where a benchmark ships several. Population per benchmark
matches the F1 column it sits beside: MultiHop answerable rows only
(null rows score under the pure-refusal rule and are excluded exactly
as the primary excludes them); NarrativeQA and both HotpotQA variants
all rows. THE CREDITED-REFUSAL ARTIFACT CANNOT PAY UNDER EM: the bare
"no" gold normalises to "no" and the canonical refusal to "no answer
available" — one shared token buys token-F1 0.5 and EM exactly 0
(verified against the normaliser in tests). So MultiHop EM is
artifact-free by construction, unlike the primary token-F1.

RANK@5 — WHAT THE BANK SUPPORTS, stated bluntly. The depth-50 scoring
ranking was consumed AT RUN TIME to compute hit@K / MAP@K / MRR at
preset K and the ranking itself was never written to any row (the
recurring shape: computed, correct, not banked). recall@5 — the
fraction of gold evidence units in the top-5 — is therefore NOT
derivable post-hoc from the JSONLs, and this exporter refuses to fake
it: the `recall_at_5` column is emitted EMPTY with this explanation,
and `hit_at_5` (banked per row, exact) is emitted beside it as the
rank@5 column the bank can actually stand behind. If true recall@5 is
ruled in, it requires a retrieval-replay pass over the warm substrates
with an identity gate against the banked per-row scores — a separate
tool, not a silent recomputation here.

Retrieval columns are GENERATOR-IDENTICAL BY CONSTRUCTION for M2/M3
(model-free substrates; bit-identity proven on all six cells) but NOT
for M4, whose trees differ by summariser — the per-generator tables
show M4's rank figures moving and that is real, not drift.

"bm25" in the requested row labels = M3, the hybrid dense+BM25 RRF
system (there is no sparse-only system in the matrix). Row order as
requested: LLM (M1), flat (M2), raptor (M4), bm25-hybrid (M3).

Same discipline as export_matrix: every number read from the banked
summaries or recomputed from the banked rows under gates (the shared
`read_cell` supplies F1/supplementary and the credited assertions);
refusal on any missing or partial cell and on any population mismatch;
checksum line printed and embedded; generating commit in the header.

Run-host invocation (loaders need the HF cache there):

    python -m scripts.export_comparison \\
      --p10 /content/drive/MyDrive/thesis_rag/outputs/p10 \\
      --p11 /content/drive/MyDrive/thesis_rag/outputs/p11 \\
      --out /content/drive/MyDrive/thesis_rag/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.scorers.extractive import normalize_qasper_answer
from scripts.export_matrix import (
    BENCHMARKS,
    LLAMA,
    QWEN,
    NULL_METHOD,
    RECORDED_CREDITED,
    _fail,
    _generating_commit,
    read_cell,
)

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
    """recall@K recomputed from the sidecar ROWS (the banked artifact,
    written by replay_retrieval behind its per-row gate). Refuses -- by
    ruling, NEVER falls back to hit@5 -- when the sidecar is missing."""
    side = bank / f"rankings.{stem}.jsonl"
    if not side.is_file():
        _fail(f"{stem}: sidecar {side} is MISSING -- run "
              "scripts.replay_retrieval first; by ruling this exporter "
              "refuses rather than falling back to hit@5")
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
    """EM + hit@5 for one cell, from its banked rows. Refuses loudly."""
    jpath = bank / f"{benchmark}_{system}_validation.jsonl"
    em_sum = 0
    n_em = 0
    hit_sum = 0.0
    n_rank = 0
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
                em_sum += max(int(pred == normalize_qasper_answer(g))
                              for g in gold_map[qid])
                n_em += 1
            if system != "M1" and benchmark != "narrativeqa":
                retr = r.get("retrieval") or {}
                hk = retr.get("hit_at_k") or {}
                if not retr.get("skipped") and ("5" in hk or 5 in hk):
                    hit_sum += float(hk.get("5", hk.get(5)))
                    n_rank += 1
    expected_em = (int(summary["n_answerable"])
                   if benchmark == "multihop_rag"
                   else int(summary["n_queries_scored"]))
    if n_em != expected_em:
        _fail(f"{benchmark}/{system}: EM population {n_em} != expected "
              f"{expected_em}")
    out = {"em": em_sum / n_em, "n_em_population": n_em}
    if system != "M1" and benchmark != "narrativeqa":
        # The expected rank population is DERIVED from n_answerable --
        # a field hard-indexed by read_cell and therefore proven present
        # in all 32 summaries -- because rank-aware rows are exactly the
        # answerable rows on every ranked benchmark. The first version
        # read summary.get("n_retrieval_scored") or 0: the oldest banked
        # summary (ab0c7c0 era) predates that key, and the coerced zero
        # refused a correct cell -- a value that does not exist,
        # consumed as if it did (the recurring shape, inverted). Absent
        # fields are never defaulted: where the key exists it is
        # CROSS-CHECKED, where it does not the derived population
        # stands, and the row-side count is the hard gate either way.
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
    rows = []
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
    nn = {c: sum(1 for r in rows if str(r[c]) != "") for c in CSV_COLUMNS}
    return ("COMPARISON checksum: rows=%d non-null " % len(rows)
            + " ".join(f"{c}:{nn[c]}" for c in CSV_COLUMNS))


def _fmt(v: str, dp: int = 4) -> str:
    return f"{float(v):.{dp}f}" if v != "" else "n/a"


def write_outputs(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "COMPARISON.csv"
    md_path = out_dir / "COMPARISON.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    line = checksum_line(rows)
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
                "construction for M2/M3 (bit-identity proven); M4's are "
                "tree-dependent and legitimately differ per generator. "
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
    """Gold reference TEXTS for one EvalQuery.

    `gold_answers` holds GoldAnswer dataclasses, not strings — every
    live benchmark writes the reference text into `free_form` (HotpotQA
    stores its yes/no golds there as text too; see hotpotqa.py's own
    score_answer), so this mirrors the frozen scorers' read
    (`gold.free_form`) rather than inventing a second convention. The
    first shipped version passed the OBJECTS through and TypeError-ed on
    the first real cell while the tests fed strings — the fixture now
    goes through this function against real GoldAnswer objects so that
    class cannot recur.
    """
    return tuple(g.free_form for g in (q.gold_answers or ())
                 if getattr(g, "free_form", ""))


def _gold_maps_from_loaders() -> dict[str, dict[str, tuple[str, ...]]]:
    from scripts.report_children_per_parent import _benchmark
    maps: dict[str, dict[str, tuple[str, ...]]] = {}
    for benchmark in BENCHMARKS:
        m: dict[str, tuple[str, ...]] = {}
        for unit in _benchmark(benchmark).iter_eval_units(split="validation"):
            for q in unit.queries:
                golds = gold_texts(q)
                if golds:
                    m[str(q.query_id)] = golds
        maps[benchmark] = m
        print(f"[compare] {benchmark}: gold for {len(m)} queries")
    return maps


def main() -> None:
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
