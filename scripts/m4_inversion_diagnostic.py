"""Why is M4 retrieval-worst and answer-best? Read-only, paired, per query.

THE OBSERVATION (MultiHop, Qwen): M4 retr 0.3763 / ans 0.2127 against
M3's 0.4450 / 0.2038 and M2's 0.4202 / 0.2011. Worst retrieval, best
answer.

DO NOT INTERPRET THIS YET. There are FOUR live mechanisms and the
headline numbers cannot separate them. Three of the four would make the
"hierarchy helps" reading wrong.

  A. THE RETRIEVAL DEFICIT MAY BE A MEASUREMENT ARTIFACT. Retrieved
     SUMMARY nodes carry an empty gold_provenance, so CK-2 cannot credit
     them at all. A large share of M4's returned units is unscoreable BY
     CONSTRUCTION, which depresses recall mechanically without M4 having
     retrieved worse. Settled by the leaf-expanded twin, NOT by this
     script: rerun M4 with expand_summary_nodes=True and compare.

  B. SUMMARIES GENUINELY HELP THE READER. RAPTOR's actual claim. If so,
     M4's answer advantage should CONCENTRATE on queries where it
     actually retrieved summary nodes.

  C. SHORTER CONTEXT HELPS. M4 feeds ~2,000 evidence tokens against
     M2/M3's ~3,900 (the M4-only paper budget). Less context can improve
     answers on its own -- fewer distractors, less lost-in-the-middle. If
     so, the advantage should be roughly UNIFORM in summary use and
     should track evidence length.

  D. ABSTENTION. With 301 null queries, a system that abstains more (or
     less) shifts mean answer score without answering anything better.

THE DESIGN. Compare M4 against a baseline PAIRED ON QUERY ID, then split
those pairs by whether M4 used summaries. B predicts a large gap between
the splits. C and D predict a small one.

    python -m scripts.m4_inversion_diagnostic <m4.jsonl> <baseline.jsonl>

Refuses to report when a split is too small to compare, rather than
returning a verdict the data cannot support.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MIN_GROUP = 30


def _load(p: Path) -> dict[str, dict]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("query_id") is not None:
                out[str(r["query_id"])] = r
    return out


def _ans(r: dict) -> float:
    return float((r.get("answer") or {}).get("value", 0.0))


def _summary_units(r: dict) -> int:
    ut = r.get("retrieved_unit_types") or {}
    return sum(v for k, v in ut.items() if k != "chunk")


def _mean(xs):
    return statistics.mean(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("m4", type=Path)
    ap.add_argument("baseline", type=Path)
    args = ap.parse_args()

    M, B = _load(args.m4), _load(args.baseline)
    shared = sorted(set(M) & set(B))
    print(f"M4 rows {len(M)}, baseline rows {len(B)}, shared {len(shared)}")
    if len(shared) < MIN_GROUP:
        print("\nINCONCLUSIVE: too few shared queries. Nothing compared.")
        return 2

    # Unscoreable share -- mechanism A's magnitude, reported not tested.
    tot_units = sum(sum((M[q].get("retrieved_unit_types") or {}).values())
                    for q in shared)
    tot_summary = sum(_summary_units(M[q]) for q in shared)
    print(f"\nM4 retrieved units: {tot_units}, of which SUMMARY "
          f"{tot_summary} ({tot_summary/max(1,tot_units):.1%}) - "
          "unscoreable by CK-2 BY CONSTRUCTION.")
    print("  -> mechanism A. Settle it with the leaf-expanded twin, not here.")

    with_s = [q for q in shared if _summary_units(M[q]) > 0]
    without_s = [q for q in shared if _summary_units(M[q]) == 0]
    print(f"\nqueries where M4 used summaries: {len(with_s)} "
          f"({len(with_s)/len(shared):.1%});  without: {len(without_s)}")

    def paired(qs):
        return [_ans(M[q]) - _ans(B[q]) for q in qs]

    all_d = paired(shared)
    print(f"\nPAIRED answer delta (M4 - baseline), all shared: "
          f"{_mean(all_d):+.4f}  (n={len(all_d)})")

    if len(with_s) < MIN_GROUP or len(without_s) < MIN_GROUP:
        print(f"\nINCONCLUSIVE for mechanism B/C: one split has under "
              f"{MIN_GROUP} queries, so the comparison that separates "
              "'summaries help' from 'shorter context helps' cannot be "
              "made. Report the paired delta only.")
        return 2

    d_with, d_without = paired(with_s), paired(without_s)
    mw, mo = _mean(d_with), _mean(d_without)
    sd = statistics.pstdev(d_with + d_without) or 1e-9
    gap = (mw - mo) / sd
    print(f"  with summaries   : {mw:+.4f}  (n={len(d_with)})")
    print(f"  without summaries: {mo:+.4f}  (n={len(d_without)})")
    print(f"  split gap        : {mw - mo:+.4f}   (d = {gap:+.2f})")

    ev_m4 = _mean([float(M[q].get("evidence_tokens", 0)) for q in shared])
    ev_b = _mean([float(B[q].get("evidence_tokens", 0)) for q in shared])
    print(f"\nevidence tokens: M4 {ev_m4:.0f} vs baseline {ev_b:.0f} "
          f"({ev_m4/max(1.0, ev_b):.0%} of baseline)  -> mechanism C's size")

    # MECHANISM D, DECOMPOSED BY ANSWERABILITY. A raw abstention gap
    # cannot say whether abstaining is CORRECT behaviour or a lost
    # answer. On a null query, abstaining is the right answer and scores
    # well; on an answerable one it is a forfeited point. If M4's answer
    # advantage is concentrated in the nulls, D is the whole story and C
    # is incidental. If it also holds on answerable queries, C is doing
    # real work and the two must be reported separately.
    from src.eval.scorers import is_abstention

    nulls = [q for q in shared if (M[q].get("retrieval") or {}).get("skipped")]
    answerable = [q for q in shared if q not in set(nulls)]
    print(f"\nMECHANISM D, split by answerability "
          f"(nulls {len(nulls)}, answerable {len(answerable)}):")
    print(f"{'group':<12} {'n':>6} {'M4 abst':>9} {'base abst':>10} "
          f"{'M4 ans':>8} {'base ans':>9} {'paired d':>9}")
    print("-" * 68)
    for label, qs in (("NULL", nulls), ("answerable", answerable),
                      ("all", shared)):
        if not qs:
            continue
        am = sum(is_abstention(M[q].get("predicted_answer", "")) for q in qs)
        ab = sum(is_abstention(B[q].get("predicted_answer", "")) for q in qs)
        print(f"{label:<12} {len(qs):>6} {am/len(qs):>8.1%} {ab/len(qs):>9.1%} "
              f"{_mean([_ans(M[q]) for q in qs]):>8.4f} "
              f"{_mean([_ans(B[q]) for q in qs]):>9.4f} "
              f"{_mean(paired(qs)):>+9.4f}")

    if nulls and answerable:
        d_null, d_ans = _mean(paired(nulls)), _mean(paired(answerable))
        share = (d_null * len(nulls)) / (
            d_null * len(nulls) + d_ans * len(answerable)) if (
            d_null * len(nulls) + d_ans * len(answerable)) else None
        print()
        if share is not None and share > 0.7:
            print(f"  -> D DOMINATES: ~{share:.0%} of the total advantage "
                  "comes from the NULL queries, where abstaining is the "
                  "correct answer. C is incidental.")
        elif d_ans > 0 and d_null > 0:
            print("  -> BOTH LIVE: the advantage appears on answerable "
                  "queries too, so it is not only abstention accounting. "
                  "Report C and D separately.")
        else:
            print("  -> MIXED SIGNS: the advantage on one group offsets a "
                  "deficit on the other. Report both; the pooled mean hides "
                  "it.")

    print()
    if abs(gap) >= 0.2:
        print("LEANS B (summaries help): M4's answer advantage is "
              "CONCENTRATED on queries where it actually retrieved summary "
              "nodes. That is RAPTOR's own claim and the reading the "
              "hierarchy story needs.")
    else:
        print("LEANS C/D (context length or abstention): the advantage is "
              "roughly UNIFORM in whether M4 used summaries, so it is not "
              "coming from the hierarchy. Compare the evidence-token and "
              "abstention lines above - and note that a context-length "
              "effect would be a finding about PROMPT SIZE, not about "
              "RAPTOR.")
    print("\nNEITHER verdict is significance. Run "
          "scripts/significance_diagnostic.py on the paired answer deltas "
          "before reporting any of this as an effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
