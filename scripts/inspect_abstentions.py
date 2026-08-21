"""What is the `abstained` flag ACTUALLY catching on this cell?

THE QUESTION THIS ANSWERS, and why it had to be asked. On M1 x MultiHop
the answerable-and-abstained rows scored a mean of 0.198 across 795 rows
against 0.020 for rows that did not abstain — ten times HIGHER. If
abstaining meant declining, every one of those rows would score 0.0.

On M4 x NarrativeQA the same flag gave 0.002 across 656 rows, which IS
what declining looks like.

So the flag may mean different things on different benchmarks, and if it
does then the abstention RATES are not comparable across them — which
would disqualify the 53.3% vs 35.8% "anchor" comparison entirely. This
script settles it from the banked rows instead of by argument.

THE CLASSIFIER IS NOT A HEURISTIC. It reuses the exact machinery the
null-query rule (P2) uses: `detect_abstention` returns the hedge-clause
span, the span is stripped, and `is_filler_only` decides whether what
remains asserts anything. Same definition of "pure refusal" the scorer
already applies, so this cannot disagree with it.

    PURE REFUSAL   remainder after the hedge is empty or filler
                   -> the system genuinely declined
    HEDGED CONTENT remainder carries content words
                   -> the flag caught VOCABULARY, not a refusal;
                      e.g. "I'm not certain, but it's Sam Bankman-Fried"

NULL ROWS ARE EXCLUDED (method == "unanswerable_rule"): a refusal there
is the correct answer and scores on a different scale.

    python -m scripts.inspect_abstentions --input <cell>.jsonl [--samples 12]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.eval.scorers.unanswerable import detect_abstention, is_filler_only

PURE_REFUSAL = "pure_refusal"
HEDGED_CONTENT = "hedged_content"
# The recorded flag says abstained; the CURRENT detector finds no hedge
# span at all. That is detector DRIFT between run time and analysis time,
# and it is the exact thing preferring the recorded flag protects
# against — so it is counted separately rather than folded into either
# real category. Non-zero here means banked rows were classified under a
# detector version this checkout no longer has.
DETECTOR_DISAGREES = "detector_disagrees"


def iter_rows(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def classify(predicted: str) -> tuple[str, str, str]:
    """(verdict, hedge_span_text, remainder) for one prediction."""
    match = detect_abstention(predicted)
    text = match.text
    if not match.span:
        return DETECTOR_DISAGREES, "", text
    lo, hi = match.span
    hedge = text[lo:hi]
    remainder = (text[:lo] + " " + text[hi:]).strip()
    verdict = PURE_REFUSAL if is_filler_only(remainder) else HEDGED_CONTENT
    return verdict, hedge, remainder


def inspect(path: Path, n_samples: int = 12) -> dict:
    keys = (PURE_REFUSAL, HEDGED_CONTENT, DETECTOR_DISAGREES)
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    samples: dict[str, list[dict]] = {k: [] for k in keys}
    hedge_counts: Counter[str] = Counter()
    hedge_scores: dict[str, list[float]] = {}
    n_answerable = n_flagged = 0
    all_answerable: list[float] = []

    for row in iter_rows(path):
        ans = row.get("answer") or {}
        if ans.get("method") == "unanswerable_rule":
            continue  # null row: a refusal there is CORRECT, different scale
        n_answerable += 1
        all_answerable.append(float(ans.get("value", 0.0)))
        md = ans.get("metadata") or {}
        if not md.get("abstained"):
            continue
        n_flagged += 1
        predicted = row.get("predicted_answer", "") or ""
        score = float(ans.get("value", 0.0))
        verdict, hedge, remainder = classify(predicted)
        buckets[verdict].append(score)
        hedge_counts[hedge or "<no span>"] += 1
        hedge_scores.setdefault(hedge or "<no span>", []).append(score)
        if len(samples[verdict]) < n_samples:
            samples[verdict].append({
                "query_id": row.get("query_id"),
                "score": round(score, 4),
                "hedge": hedge,
                "remainder": remainder[:160],
                "predicted": predicted[:200],
            })

    mean = lambda xs: statistics.mean(xs) if xs else None  # noqa: E731

    # CREDITED REFUSALS. A row classified PURE REFUSAL asserts nothing
    # once its hedge is stripped, so on an answerable question it should
    # score 0.0 every time. Any positive score means the answer scorer
    # gave a refusal credit, and the SCORE ALONE identifies why: on
    # MultiHop the canonical "No answer available." normalises to
    # ["no", "answer", "available"] and the only gold it overlaps is the
    # bare token "no", which yields EXACTLY 0.5. No join to the gold is
    # needed to diagnose it.
    refusal_scores = buckets[PURE_REFUSAL]
    credited = [x for x in refusal_scores if x > 0]

    # THE SUPPLEMENTARY ANSWER-QUALITY FIGURE, required beside the
    # primary on every MultiHop cell. The primary stays token-F1 exactly
    # as frozen; this is the same rows with the credited-refusal mass
    # removed, computed POST-HOC over stored predictions. No scorer
    # change, no freeze violation, no matrix split.
    #
    # IT IS NOT A CONSTANT OFFSET and must not be treated as one:
    # credited mass scales with a system's REFUSAL RATE, and refusals are
    # enriched 1.59x on "no"-gold queries, so systems collect this credit
    # preferentially where it pays. It is a CONFOUND in every
    # cross-system MultiHop comparison, not something that cancels.
    total_mass = sum(all_answerable)
    credited_mass = sum(credited)
    return {
        "n_answerable_rows": n_answerable,
        "n_flagged_abstained": n_flagged,
        "flagged_rate": (n_flagged / n_answerable) if n_answerable else None,
        "pure_refusal": {
            "n": len(buckets[PURE_REFUSAL]),
            "mean_score": mean(buckets[PURE_REFUSAL]),
            "n_credited": len(credited),
            "credited_share": (
                len(credited) / len(refusal_scores) if refusal_scores else None),
            "credited_score_mass": sum(credited),
            "score_histogram": dict(
                sorted(Counter(round(x, 4) for x in refusal_scores).items())),
        },
        "hedged_content": {
            "n": len(buckets[HEDGED_CONTENT]),
            "mean_score": mean(buckets[HEDGED_CONTENT]),
        },
        "detector_disagrees": {
            "n": len(buckets[DETECTOR_DISAGREES]),
            "mean_score": mean(buckets[DETECTOR_DISAGREES]),
        },
        "answerable_mean": (total_mass / n_answerable) if n_answerable else None,
        "answerable_mean_excluding_credited": (
            (total_mass - credited_mass) / n_answerable
            if n_answerable else None),
        "credited_share_of_answerable_mass": (
            credited_mass / total_mass if total_mass else None),
        "top_hedges": [
            {"hedge": h, "n": c, "mean_score": mean(hedge_scores[h])}
            for h, c in hedge_counts.most_common(10)
        ],
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = inspect(Path(args.input), args.samples)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    fmt = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    pr, hc = out["pure_refusal"], out["hedged_content"]
    dd = out["detector_disagrees"]
    print(f"\nWHAT THE `abstained` FLAG CAUGHT — answerable rows only")
    print(f"  answerable rows      {out['n_answerable_rows']}")
    print(f"  flagged abstained    {out['n_flagged_abstained']} "
          f"({(out['flagged_rate'] or 0):.1%})")
    print(f"\n  PURE REFUSAL    n={pr['n']:<6} mean={fmt(pr['mean_score'])}"
          "   <- genuinely declined")
    if pr.get("n_credited"):
        share = pr.get("credited_share") or 0.0
        print(f"      *** {pr['n_credited']} of these ({share:.1%}) scored "
              "ABOVE ZERO.")
        print("      A pure refusal asserts nothing, so on an ANSWERABLE "
              "question it")
        print("      must score 0.0. The answer scorer CREDITED a refusal.")
        print(f"      credited score mass : {pr['credited_score_mass']:.2f}")
        print(f"      score histogram     : {pr['score_histogram']}")
        print("      On MultiHop a score of exactly 0.500 means the gold was")
        print("      the bare token 'no' — 24.9% of answerable golds. See")
        print("      SESSION_HANDOFF 5c.")
    print(f"  HEDGED CONTENT  n={hc['n']:<6} mean={fmt(hc['mean_score'])}"
          "   <- vocabulary, NOT a refusal")

    if dd["n"]:
        print(f"  DETECTOR DRIFT   n={dd['n']:<6} mean={fmt(dd['mean_score'])}"
              "   *** recorded abstained, current detector finds no hedge")
        print("      Non-zero means these rows were classified under a "
              "detector version")
        print("      this checkout no longer has. Investigate before "
              "reporting any rate.")

    total = pr["n"] + hc["n"]
    if total:
        share = hc["n"] / total
        print(f"\n  {share:.1%} of flagged rows still ASSERT something after "
              "the hedge is stripped.")
        if share > 0.2:
            print("  => on this cell `abstained` is a HEDGING-VOCABULARY "
                  "marker, not a refusal rate.")
        else:
            print("  => on this cell `abstained` really does mean declined.")

    if pr.get("n_credited"):
        print("\n  ANSWER QUALITY, primary and supplementary:")
        print(f"    answerable mean (PRIMARY, frozen token-F1) : "
              f"{fmt(out['answerable_mean'])}")
        print(f"    excluding credited refusals (SUPPLEMENTARY): "
              f"{fmt(out['answerable_mean_excluding_credited'])}")
        print(f"    credited share of answerable score mass    : "
              f"{(out['credited_share_of_answerable_mass'] or 0):.1%}")
        print("    Report BOTH. The share differs BY SYSTEM because it "
              "scales with")
        print("    refusal rate, so it is a CONFOUND, not an offset that "
              "cancels.")

    print("\n  top matched hedges (n, mean score):")
    for h in out["top_hedges"]:
        print(f"    {h['n']:>6}  {fmt(h['mean_score'])}  {h['hedge'][:70]!r}")

    for verdict in (HEDGED_CONTENT, PURE_REFUSAL, DETECTOR_DISAGREES):
        rows = out["samples"][verdict]
        if not rows:
            continue
        print(f"\n  --- samples: {verdict} ---")
        for r in rows:
            print(f"    [{r['score']}] {r['predicted']!r}")
            print(f"        hedge={r['hedge']!r}  remainder={r['remainder']!r}")


if __name__ == "__main__":
    main()
