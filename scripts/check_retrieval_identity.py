"""Are two cells' retrieval scores IDENTICAL PER QUERY, or just on average?

WHY THIS EXISTS. Two MultiHop cells reproduced their gpt-4o-mini-era
retrieval F1 to four decimals (M2 0.4202, M3 0.4450). The natural
inference is that the paired significance result carries across columns
unchanged. That inference needs per-query identity, and a matching MEAN
does not establish it: two different per-query vectors can share a mean
to any number of decimals, and a paired sign-flip permutation test reads
the DIFFERENCES, not the mean.

This is the same shape as every check that has misled this project: an
account and an agreeing check, where the check is weaker than the claim.
So compare the vectors, not the summaries.

    python -m scripts.check_retrieval_identity <cell_a.jsonl> <cell_b.jsonl>

WHAT IT ESTABLISHES

  IDENTICAL   every shared query has bit-equal retrieval F1. The paired
              differences are unchanged, so any paired test on retrieval
              gives an identical p-value. The significance result
              genuinely carries.

  NOT         the means agreed by coincidence or near-coincidence. The
              paired test must be re-run; do not carry the old verdict.

Refuses to report on an empty or non-overlapping intersection rather
than declaring identity over nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = r.get("query_id")
        if qid is not None:
            out[str(qid)] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--field", default="f1",
                    choices=["f1", "recall", "precision"])
    args = ap.parse_args()

    A, B = _load(args.a), _load(args.b)
    shared = sorted(set(A) & set(B))
    print(f"{args.a.name}: {len(A)} rows")
    print(f"{args.b.name}: {len(B)} rows")
    print(f"shared query_ids: {len(shared)}")

    if not shared:
        print("\nINCONCLUSIVE: no shared query_ids. Nothing was compared.")
        return 2
    if len(shared) < min(len(A), len(B)):
        print(f"  NOTE: {min(len(A), len(B)) - len(shared)} rows in the "
              "smaller file have no counterpart and are excluded.")

    diffs = []
    n_skip_mismatch = 0
    for q in shared:
        ra = (A[q].get("retrieval") or {})
        rb = (B[q].get("retrieval") or {})
        if bool(ra.get("skipped")) != bool(rb.get("skipped")):
            n_skip_mismatch += 1
            continue
        if ra.get("skipped"):
            continue
        diffs.append((q, float(ra.get(args.field, 0.0)),
                      float(rb.get(args.field, 0.0))))

    if not diffs:
        print("\nINCONCLUSIVE: every shared query was skipped or mismatched; "
              "no scores were compared.")
        return 2

    exact = [d for d in diffs if d[1] == d[2]]
    worst = max(diffs, key=lambda d: abs(d[1] - d[2]))
    mean_a = sum(d[1] for d in diffs) / len(diffs)
    mean_b = sum(d[2] for d in diffs) / len(diffs)

    print(f"\ncompared {len(diffs)} scoreable queries on retr_{args.field}")
    print(f"  mean A = {mean_a:.6f}   mean B = {mean_b:.6f}   "
          f"delta = {mean_b - mean_a:+.2e}")
    print(f"  bit-equal per query: {len(exact)}/{len(diffs)} "
          f"({len(exact)/len(diffs):.2%})")
    print(f"  largest per-query gap: {abs(worst[1]-worst[2]):.6f} "
          f"on {worst[0]}")
    if n_skip_mismatch:
        print(f"  skip-flag mismatches: {n_skip_mismatch} "
              "(one cell scored a query the other skipped)")

    print()
    if len(exact) == len(diffs) and not n_skip_mismatch:
        print("IDENTICAL. Every shared query has bit-equal retrieval, so the "
              "paired differences are unchanged and any paired test on "
              "retrieval returns an identical p-value. A significance "
              "verdict from the earlier column CARRIES.")
        return 0

    print("NOT IDENTICAL. The means may agree while the per-query vectors "
          "differ, and a paired test reads the differences. RE-RUN the "
          "significance test on the new column; do not carry the old "
          "verdict across.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
