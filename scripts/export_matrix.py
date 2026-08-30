"""Export the 32-cell matrix to MATRIX.csv + MATRIX.md — the thesis's
supplementary-data file and the single plotting source for every figure.

ONE SOURCE, NO DRIFT. Every value is read from the banked artifacts:
summary fields come from each cell's `.summary.json` verbatim; the
battery-derived columns (credited refusals, abstention rate, plain-answer
mean, the supplementary mean) are recomputed from the banked JSONL rows
using the same rules the recorded battery used, and the credited counts
are ASSERTED per cell against the values the living record §5c banked
(`RECORDED_CREDITED` below) — the record is the gate, the bank is the
source, nothing is retyped into the output.

THE SUPPLEMENTARY MEAN, defined (and verified on the canary):
    supplementary = mean_answer_score_answerable - credited_mass / n_answerable
i.e. the credited-refusal mass is removed while the denominator stays the
full answerable population (caption 5: name the denominator — it is the
same denominator as the primary). Credited rows are identified exactly as
`inspect_abstentions` counts them: `answer.metadata.abstained` is set, the
prediction is the bare canonical refusal string, and the score is > 0.

RETRIEVAL ABSENCE — two different absences, two markers (caption 6):
`mean_retrieval_f1` is left EMPTY for both, and `retrieval_absence` says
which: "no_retrieval" (M1 performs no retrieval — the table dash) versus
"no_gold" (NarrativeQA ships no retrieval ground truth — the n/a string).
M1-on-NarrativeQA carries "no_retrieval" (the system-level absence wins).

REFUSALS. The exporter refuses (exit non-zero, nothing written) if any of
the 32 summaries is missing, any `partial_run` is true, any summary's
generator disagrees with its bank, any cell's scored count misses its
expected count, or any cell's recomputed credited (n, mass) disagrees
with the recorded battery — a matrix over the wrong cells must not exist.

A CHECKSUM LINE (row count + per-column non-null counts) is printed and
embedded in MATRIX.md so a regenerated file is comparable at a glance.

Run-host invocation (both banks on Drive):

    python -m scripts.export_matrix \\
      --p10 /content/drive/MyDrive/thesis_rag/outputs/p10 \\
      --p11 /content/drive/MyDrive/thesis_rag/outputs/p11 \\
      --out /content/drive/MyDrive/thesis_rag/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

QWEN = "Qwen/Qwen2.5-7B-Instruct"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
BENCHMARKS = ("multihop_rag", "narrativeqa", "hotpotqa", "hotpotqa_pooled")
SYSTEMS = ("M1", "M2", "M3", "M4")
CANONICAL_REFUSAL = "no answer available"
NULL_METHOD = "unanswerable_rule"

COLUMNS = [
    "generator", "benchmark", "system",
    "mean_retrieval_f1", "retrieval_absence",
    "mean_answer_score", "mean_answer_score_answerable",
    "supplementary_mean", "mean_answer_score_null",
    "n_queries", "n_credited", "credited_mass",
    "abstain_pct", "mean_plain",
    "elapsed_s", "git_commit", "timestamp", "python",
    "lockfile_hash", "generator_revision",
]

# The battery record (living record §5c / §4.1): per-cell credited-refusal
# count and mass. These are RECORDED COMMITMENTS the recomputation must
# reproduce — the gate, not the source. Masses recorded at 2 dp where the
# record rounds (NarrativeQA); MultiHop masses are exact halves.
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
# HotpotQA-family cells (both variants, both generators): the official
# sentinel guard forces exact match on yes/no golds, so credited is zero
# everywhere — recorded in §4.1 and §5c.
for _gen in (QWEN, LLAMA):
    for _bench in ("hotpotqa", "hotpotqa_pooled"):
        for _sys in SYSTEMS:
            RECORDED_CREDITED[(_gen, _bench, _sys)] = (0, 0.0)

MASS_TOLERANCE = 0.005  # the record rounds NarrativeQA masses to 2 dp


def _fail(msg: str) -> None:
    raise SystemExit(f"[export] REFUSED: {msg}")


def _norm_pred(text: str) -> str:
    return (text or "").strip().lower().rstrip(".")


def read_cell(bank: Path, generator: str, benchmark: str, system: str,
              recorded: dict[tuple[str, str, str], tuple[int, float]]) -> dict:
    """One matrix row, from one cell's summary + rows. Refuses loudly."""
    stem = f"{benchmark}_{system}_validation"
    spath = bank / f"{stem}.summary.json"
    jpath = bank / f"{stem}.jsonl"
    if not spath.is_file():
        _fail(f"missing summary {spath}")
    if not jpath.is_file():
        _fail(f"missing rows {jpath}")
    s = json.loads(spath.read_text(encoding="utf-8"))

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

    if n_abstained + n_plain != n_answerable:
        _fail(f"{stem}: answerable rows {n_abstained + n_plain} != summary "
              f"n_answerable {n_answerable}")
    exp_n, exp_mass = recorded[(generator, benchmark, system)]
    if n_credited != exp_n or abs(credited_mass - exp_mass) > MASS_TOLERANCE:
        _fail(f"{stem}: recomputed credited ({n_credited}, "
              f"{credited_mass:.4f}) disagrees with the recorded battery "
              f"({exp_n}, {exp_mass}) -- wrong rows or drifted rule; a "
              "matrix over unverified cells must not exist")

    primary = float(s["mean_answer_score_answerable"])
    supplementary = primary - (credited_mass / n_answerable if n_answerable else 0.0)

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


def build_rows(p10: Path, p11: Path,
               recorded: dict | None = None) -> list[dict]:
    recorded = RECORDED_CREDITED if recorded is None else recorded
    rows = []
    for generator, bank in ((QWEN, p10), (LLAMA, p11)):
        for benchmark in BENCHMARKS:
            for system in SYSTEMS:
                rows.append(read_cell(bank, generator, benchmark, system,
                                      recorded))
    return rows


def checksum_line(rows: list[dict]) -> str:
    nn = {c: sum(1 for r in rows if str(r[c]) != "") for c in COLUMNS}
    parts = " ".join(f"{c}:{nn[c]}" for c in COLUMNS)
    return f"MATRIX checksum: rows={len(rows)} non-null {parts}"


def _generating_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parents[1]
                              ).stdout.strip()
    except Exception:
        return "unknown"


def write_outputs(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "MATRIX.csv"
    md_path = out_dir / "MATRIX.md"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    line = checksum_line(rows)
    commit = _generating_commit()
    with md_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# MATRIX — the 32-cell supplementary-data table\n\n")
        f.write(f"Generated by `scripts/export_matrix.py` @ `{commit}` on "
                f"{date.today().isoformat()}, from the banked cell summaries "
                "and rows (nothing retyped; credited columns recomputed from "
                "the JSONLs and asserted against the living record's battery "
                "values). `supplementary_mean` = answerable primary minus "
                "credited mass over the UNCHANGED answerable denominator. "
                "`retrieval_absence`: no_retrieval = the system performs "
                "none (M1); no_gold = the benchmark ships no retrieval "
                "ground truth (NarrativeQA).\n\n")
        f.write(f"`{line}`\n\n")
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("|" + "---|" * len(COLUMNS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |\n")
    return csv_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p10", required=True, help="Qwen bank directory")
    ap.add_argument("--p11", required=True, help="Llama bank directory")
    ap.add_argument("--out", required=True, help="output directory for "
                    "MATRIX.csv / MATRIX.md")
    args = ap.parse_args()
    rows = build_rows(Path(args.p10), Path(args.p11))
    csv_path, md_path = write_outputs(rows, Path(args.out))
    print(f"[export] wrote {csv_path}")
    print(f"[export] wrote {md_path}")
    print(checksum_line(rows))


if __name__ == "__main__":
    main()
