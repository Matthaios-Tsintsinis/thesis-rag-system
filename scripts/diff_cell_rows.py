"""Per-row comparison of two banked cells on the SAME benchmark.

Built for the F-X4 verification: on HotpotQA-distractor, M2 and M3 report
retrieval F1 identical to four decimals, and the claimed mechanism is "set
unchanged, order degraded". That claim is per-row checkable — count the
queries where the packed sets actually differ — and this script does it at
the strength the data supports.

WHAT THE BANK CAN AND CANNOT ANSWER, stated up front. Rows banked before
2026-08-24 carry counts, token totals and per-row retrieval scores, but NOT
the packed chunk ids — the identity existed at run time and nothing
recorded it (the project's recurring lesson; rows carry `packed_ids` from
`3f516d1`'s successor commit onward, so this question is never
unrecoverable again). For id-less rows the comparison uses sound
implications instead:

  evidence_tokens or n_packed differ  =>  the packed SET differs
      (same set => same token total, whatever the order)
  per-row set R/P/F1 differ           =>  the scored set differs
  predicted_answer differs            =>  the PROMPT differs, i.e. the
      set OR its order differs (temp-0 decoding, same host: an identical
      prompt reproduces its answer)

So over id-less rows:
  * "set differs" count      = a LOWER BOUND (rows the implications catch)
  * "order-only candidates"  = answers differ while every set signal
      matches — exactly the F-X4 mechanism, counted
  * rows where everything matches are CONSISTENT WITH identical set and
      order, not proven identical

Where BOTH files carry `packed_ids`, the comparison is exact and says so:
set equality by id multiset, order equality by sequence.

    python -m scripts.diff_cell_rows hotpotqa_M2.jsonl hotpotqa_M3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[str(r["query_id"])] = r
    return rows


def compare_rows(a: dict, b: dict) -> dict:
    """One row pair -> which signals differ. Pure; tested."""
    ra, rb = a.get("retrieval") or {}, b.get("retrieval") or {}
    set_signals_differ = (
        a.get("evidence_tokens") != b.get("evidence_tokens")
        or a.get("n_packed") != b.get("n_packed")
        or any(ra.get(k) != rb.get(k) for k in ("f1", "recall", "precision"))
    )
    answer_differs = a.get("predicted_answer") != b.get("predicted_answer")

    ids_a = (a.get("metadata") or {}).get("packed_ids")
    ids_b = (b.get("metadata") or {}).get("packed_ids")
    exact = None
    if ids_a is not None and ids_b is not None:
        exact = {
            "set_differs": sorted(ids_a) != sorted(ids_b),
            "order_differs": (sorted(ids_a) == sorted(ids_b)
                              and list(ids_a) != list(ids_b)),
        }
    return {
        "set_signals_differ": set_signals_differ,
        "answer_differs": answer_differs,
        "order_only_candidate": answer_differs and not set_signals_differ,
        "exact": exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file_a", type=Path)
    ap.add_argument("file_b", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows_a, rows_b = _load(args.file_a), _load(args.file_b)
    bench_a = {r.get("benchmark") for r in rows_a.values()}
    bench_b = {r.get("benchmark") for r in rows_b.values()}
    if bench_a != bench_b:
        raise SystemExit(f"[rowdiff] REFUSING: different benchmarks "
                         f"{bench_a} vs {bench_b}")
    if set(rows_a) != set(rows_b):
        only_a = sorted(set(rows_a) - set(rows_b))[:5]
        only_b = sorted(set(rows_b) - set(rows_a))[:5]
        raise SystemExit(f"[rowdiff] REFUSING: query sets differ "
                         f"(only-A e.g. {only_a}, only-B e.g. {only_b})")

    n = len(rows_a)
    c = {"set_signals_differ": 0, "answer_differs": 0,
         "order_only_candidate": 0, "exact_rows": 0,
         "exact_set_differs": 0, "exact_order_differs": 0}
    for qid in rows_a:
        r = compare_rows(rows_a[qid], rows_b[qid])
        for k in ("set_signals_differ", "answer_differs",
                  "order_only_candidate"):
            c[k] += bool(r[k])
        if r["exact"] is not None:
            c["exact_rows"] += 1
            c["exact_set_differs"] += bool(r["exact"]["set_differs"])
            c["exact_order_differs"] += bool(r["exact"]["order_differs"])

    result = {"n_rows": n, **c,
              "systems": [next(iter(rows_a.values())).get("system_id"),
                          next(iter(rows_b.values())).get("system_id")]}
    if args.json:
        print(json.dumps(result, indent=2))
        return

    sa, sb = result["systems"]
    print(f"\n{sa} vs {sb}, {n} shared rows")
    print(f"  set differs (PROVEN via tokens/counts/PRF)   "
          f"{c['set_signals_differ']}   <- lower bound on set differences")
    print(f"  answer text differs (prompt differs)          "
          f"{c['answer_differs']}")
    print(f"  order-only candidates (answer moved, every    "
          f"{c['order_only_candidate']}")
    print("      set signal identical) <- the F-X4 mechanism, counted")
    if c["exact_rows"]:
        print(f"  EXACT (rows carrying packed_ids): {c['exact_rows']} rows, "
              f"set differs {c['exact_set_differs']}, "
              f"order-only differs {c['exact_order_differs']}")
    else:
        print("  no packed_ids in these rows (banked before the field): "
              "identical-looking rows are CONSISTENT WITH identical "
              "set+order, not proven identical")


if __name__ == "__main__":
    main()
