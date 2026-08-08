"""Is M9's action-mix drift CORPUS DIFFICULTY or MISCALIBRATION? Read-only.

THE PROBLEM. M9's thresholds were derived once on QASPER validation
(tau_high 0.6395 = non-gold p90, tau_low 0.5001 = gold p5) and are
applied unchanged everywhere, with the realised action mix as the
transfer check. QASPER-val derived 49.1/50.9/0.0 C/A/I; MultiHop came
back 64.4/35.6/0.0.

**THE MIX ALONE CANNOT TELL YOU WHY.** Two hypotheses predict the
identical signature:

  DIFFICULTY    MultiHop retrieval is genuinely better, so top-1
                reranker confidence is genuinely higher, so more queries
                legitimately clear tau_high. The threshold transferred
                fine; the corpus is easier.

  MISCALIBRATION The reranker's score distribution is shifted on this
                corpus for reasons unrelated to relevance, so tau_high
                now sits at an arbitrary point. The mix moved for no
                meaningful reason.

WHAT SEPARATES THEM. Under DIFFICULTY, the action must still
DISCRIMINATE: queries labelled CORRECT should have measurably better
retrieval than queries labelled AMBIGUOUS, because that is what the
confidence is supposed to track. Under MISCALIBRATION the two groups
have similar retrieval and the label is noise.

So this compares retrieval quality BETWEEN action groups. It is a
property of data already banked -- no re-run, no GPU.

    python -m scripts.m9_action_diagnostic <jsonl> [<jsonl> ...]

REFUSES TO REPORT rather than reporting vacuously (standing rule 2): if
there are no m9_* rows, or only one action is present, or a group is too
small to compare, it exits non-zero saying so. A diagnostic that cannot
observe its subject has not passed.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MIN_GROUP = 20


def _mean(xs):
    return statistics.mean(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--metric", default="f1",
                    choices=["f1", "recall", "hit10"],
                    help="Retrieval metric to compare across action groups.")
    args = ap.parse_args()

    files: list[Path] = []
    for spec in args.inputs:
        p = Path(spec)
        files.extend(sorted(p.rglob("*.jsonl")) if p.is_dir()
                     else [Path(x) for x in glob.glob(spec)])
    files = [f for f in files if f.is_file()]

    by_action: dict[str, list[float]] = defaultdict(list)
    conf_by_action: dict[str, list[float]] = defaultdict(list)
    n_rows = n_m9 = 0
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            md = r.get("metadata") or {}
            action = md.get("m9_action")
            if not action:
                continue
            retr = r.get("retrieval") or {}
            if retr.get("skipped"):
                continue
            if args.metric == "hit10":
                hits = retr.get("hit_at_k") or {}
                val = hits.get("10", hits.get(10))
                if val is None:
                    continue
                val = float(val)
            else:
                val = float(retr.get(args.metric, 0.0))
            n_m9 += 1
            by_action[action].append(val)
            conf = md.get("m9_max_conf")
            if conf is not None:
                conf_by_action[action].append(float(conf))

    print(f"scanned {len(files)} files, {n_rows} rows, {n_m9} scoreable M9 rows")

    if not by_action:
        print("\nINCONCLUSIVE: no rows carry m9_action. Point this at an M9 "
              "cell; the diagnostic did not run.")
        return 2

    total = sum(len(v) for v in by_action.values())
    print(f"\n{'action':<12} {'n':>6} {'share':>7} {'retr_'+args.metric:>12} "
          f"{'max_conf':>9}")
    print("-" * 50)
    for action in sorted(by_action, key=lambda a: -len(by_action[a])):
        vals = by_action[action]
        m = _mean(vals)
        c = _mean(conf_by_action.get(action, []))
        print(f"{action:<12} {len(vals):>6} {len(vals)/total:>6.1%} "
              f"{m if m is None else round(m, 4):>12} "
              f"{c if c is None else round(c, 4):>9}")

    present = {a for a, v in by_action.items() if len(v) >= MIN_GROUP}
    if len(present) < 2:
        print(f"\nINCONCLUSIVE: fewer than two action groups with >= "
              f"{MIN_GROUP} rows, so there is nothing to compare. The "
              "prediction was never tested.")
        return 2

    correct = by_action.get("correct", [])
    ambiguous = by_action.get("ambiguous", [])
    if len(correct) < MIN_GROUP or len(ambiguous) < MIN_GROUP:
        print("\nINCONCLUSIVE: need both CORRECT and AMBIGUOUS groups.")
        return 2

    mc, ma = _mean(correct), _mean(ambiguous)
    delta = mc - ma
    pooled_sd = statistics.pstdev(correct + ambiguous) or 1e-9
    d = delta / pooled_sd

    print(f"\nCORRECT - AMBIGUOUS on retr_{args.metric}: {delta:+.4f} "
          f"(Cohen's d = {d:+.2f})")
    print()
    if d >= 0.2:
        print("VERDICT: CORPUS DIFFICULTY, not miscalibration.")
        print("  The action still DISCRIMINATES — queries the evaluator "
              "called CORRECT do have better retrieval than the ones it "
              "called AMBIGUOUS. The threshold transferred; this corpus is "
              "simply easier, so more queries clear it legitimately.")
        print("  The cell is trustworthy. Report the realised mix as a "
              "corpus property, not as a calibration warning.")
        rc = 0
    elif d <= -0.2:
        print("VERDICT: INVERTED — the evaluator is ANTI-correlated with "
              "retrieval quality on this corpus. Do not trust this cell.")
        rc = 1
    else:
        print("VERDICT: MISCALIBRATION. The action does NOT discriminate: "
              "CORRECT and AMBIGUOUS queries have indistinguishable "
              "retrieval quality, so the threshold is sitting at an "
              "arbitrary point on this corpus.")
        print("  The mix shift is not meaningful and the corrective layer "
              "is firing on noise. Flag before trusting the cell.")
        rc = 1

    incorrect = by_action.get("incorrect", [])
    print()
    if not incorrect:
        print("*** INCORRECT BRANCH DEAD (0 rows). ***")
        print("  STRUCTURAL, not per-benchmark: tau_low = 0.5001 and the "
              "evaluator only ever sees chunks the M3 retriever already "
              "returned, i.e. topically plausible ones with logits >= ~0, "
              "hence sigmoid >= 0.5. A corpus-internal CRAG cannot reach "
              "its own INCORRECT branch by construction.")
        print("  Consequence: M9's corrective layer reduces to the "
              "AMBIGUOUS branch alone, and the paper's INCORRECT path "
              "(web search, here replaced by corpus-internal "
              "re-retrieval) is not merely weaker but UNREACHABLE.")
    else:
        print(f"INCORRECT branch fired on {len(incorrect)} rows — the "
              "dead-branch finding does NOT hold here. Report it.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
