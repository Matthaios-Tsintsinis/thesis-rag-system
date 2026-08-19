"""Post-hoc ROUGE-L for NarrativeQA cells, at the official configuration.

WHY THIS IS A SEPARATE SCRIPT AND NOT A COLUMN IN `analyse`. The banked
rows carry `predicted_answer` and `n_references` but NOT the reference
TEXT, so ROUGE-L cannot be computed from a JSONL alone — it needs a join
back to the loader for the gold answers. `analyse` is benchmark-agnostic
and must not acquire a dataset download; this script owns that join and
asserts it covered every row.

WHY ROUGE-L IS LOAD-BEARING HERE, not a secondary nicety. Token-F1
against two short free-form references scores a CORRECT entity answer
0.0 whenever the reference paraphrases rather than names it. That is
precisely why NarrativeQA's own metrics are ROUGE-L and METEOR, and it
is why the NarrativeQA row is uninterpretable to a reader who knows the
benchmark unless both columns appear together.

CONFIGURATION comes from `src.eval.scorers.rouge_l`, which matches the de
facto official scorer (AllenNLP `narrativeqa.py`) exactly — 100-word
truncation, Porter stemming, alpha 0.5, weight_factor 1.2, max over
references — with one declared departure: no `round(..., 2)`. See that
module for the reasoning.

    pip install rouge     # deliberately NOT in requirements.lock

USAGE

    python -m scripts.score_rouge_l \\
        --input /content/drive/.../narrativeqa_M4_validation.jsonl

Reports overall mean ROUGE-L f/p/r and the SAME abstention split
`analyse` prints, because a refusal scores ~0 on ROUGE-L for the same
structural reason it scores 0.0 on token-F1.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable, Iterable

from src.eval.scorers.rouge_l import rouge_l_max_over_references


def load_gold(split: str = "validation") -> dict[str, tuple[str, ...]]:
    """query_id -> reference answers, from the NarrativeQA loader itself.

    Uses the loader rather than a re-implementation so the seeded draw,
    the story set and the query ids are the cell's own. A mismatch here
    would silently score a subset.
    """
    from src.eval.narrativeqa import NarrativeQABenchmark

    gold: dict[str, tuple[str, ...]] = {}
    bench = NarrativeQABenchmark()
    for unit in bench.iter_eval_units(split=split):
        for q in unit.queries:
            gold[q.query_id] = tuple(
                g.free_form for g in q.gold_answers if g.free_form
            )
    return gold


def iter_rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_file(
    path: Path,
    gold_by_id: dict[str, tuple[str, ...]],
    *,
    scorer: Callable[..., dict] = rouge_l_max_over_references,
) -> dict:
    """ROUGE-L over one cell, overall and split by abstention.

    ABORTS on any row whose query_id is missing from `gold_by_id`. A
    partial join would report a mean over a subset and nothing in the
    output would say which subset — the same defect class as a mean
    without its n.
    """
    f_all: list[float] = []
    f_abs: list[float] = []
    f_ans: list[float] = []
    p_all: list[float] = []
    r_all: list[float] = []
    missing: list[str] = []

    for row in iter_rows(path):
        qid = row.get("query_id")
        if qid not in gold_by_id:
            missing.append(str(qid))
            continue
        refs = gold_by_id[qid]
        scores = scorer(row.get("predicted_answer", "") or "", refs)
        f_all.append(scores["f"])
        p_all.append(scores["p"])
        r_all.append(scores["r"])
        md = (row.get("answer") or {}).get("metadata") or {}
        (f_abs if md.get("abstained") else f_ans).append(scores["f"])

    if missing:
        raise SystemExit(
            f"ROUGE-L join FAILED: {len(missing)} row(s) have a query_id the "
            f"loader did not produce (first: {missing[0]!r}). Scoring the "
            "remainder would report a mean over an unnamed subset."
        )

    mean = lambda xs: statistics.mean(xs) if xs else None  # noqa: E731
    return {
        "n": len(f_all),
        "rouge_l_f_mean": mean(f_all),
        "rouge_l_p_mean": mean(p_all),
        "rouge_l_r_mean": mean(r_all),
        "n_abstained": len(f_abs),
        "n_answered": len(f_ans),
        "rouge_l_f_abstained_mean": mean(f_abs),
        "rouge_l_f_answered_mean": mean(f_ans),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="cell JSONL")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    out = score_file(Path(args.input), load_gold(args.split))
    if args.json:
        print(json.dumps(out, indent=2))
        return

    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    print(f"\nROUGE-L (official AllenNLP config, unrounded) over {out['n']} rows")
    print(f"  f={fmt(out['rouge_l_f_mean'])}  "
          f"p={fmt(out['rouge_l_p_mean'])}  r={fmt(out['rouge_l_r_mean'])}")
    print(f"  abstained n={out['n_abstained']} "
          f"f={fmt(out['rouge_l_f_abstained_mean'])}")
    print(f"  answered  n={out['n_answered']} "
          f"f={fmt(out['rouge_l_f_answered_mean'])}")
    print("  report this BESIDE token-F1; neither alone reads correctly on "
          "NarrativeQA.")


if __name__ == "__main__":
    main()
