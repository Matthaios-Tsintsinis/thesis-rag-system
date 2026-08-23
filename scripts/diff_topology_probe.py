"""Diff two probe_python_topology JSON captures — the interpreter-drift check.

    python -m scripts.diff_topology_probe topo_312.json topo_313.json

WHAT "IDENTICAL" MEANS HERE, exactly: for every probed leaf count, the
cluster COUNT, the sorted cluster SIZES, and the full MEMBERSHIP (which
leaves share which cluster — the probe already canonicalises this as sorted
index tuples, themselves sorted) are equal, and so are the guard-trip
counters. Counts alone would miss a reshuffle that preserves sizes, which is
precisely the kind of change a last-digit float move produces.

WHAT IS CHECKED BEFORE THE DIFF, because a comparison that could not fail
for the right reason has not passed:
  * the two captures must carry IDENTICAL package versions for the whole
    probed stack — otherwise the diff conflates the interpreter with a
    package drift and answers a different question;
  * the two captures must carry DIFFERENT python versions — diffing a file
    against a copy of itself passes vacuously, and this refuses to;
  * both must probe the SAME leaf counts and the SAME clustering params.

VERDICTS
  IDENTICAL  -> topology stack cleared. The risk NARROWS to the embedder
                (mpnet under torch, whose cp312/cp313 wheels also differ) —
                it is not cleared. The single-unit GPU rebuild remains the
                authority.
  DIFFERENT  -> the probe alone convicts: the interpreter moves topology on
                this host, cell 6 re-runs under the locked interpreter, and
                no GPU rebuild is needed to establish that.

Exit code 0 on IDENTICAL, 1 on DIFFERENT, 2 on a refused (confounded)
comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"[topo-diff] cannot read {path}: {e}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = _load(a_path), _load(b_path)

    refused = []
    if a.get("packages") != b.get("packages"):
        diffs = {
            k: (a.get("packages", {}).get(k), b.get("packages", {}).get(k))
            for k in set(a.get("packages", {})) | set(b.get("packages", {}))
            if a.get("packages", {}).get(k) != b.get("packages", {}).get(k)
        }
        refused.append(f"package versions differ: {diffs} -- this would "
                       "conflate interpreter drift with package drift")
    if a.get("python") == b.get("python"):
        refused.append(f"both captures ran python {a.get('python')} -- "
                       "a self-diff passes vacuously and proves nothing")
    if a.get("params") != b.get("params"):
        refused.append("clustering params differ between captures")
    ns_a = [r["n_leaves"] for r in a.get("results", [])]
    ns_b = [r["n_leaves"] for r in b.get("results", [])]
    if ns_a != ns_b:
        refused.append(f"probed leaf counts differ: {ns_a} vs {ns_b}")
    if refused:
        print("[topo-diff] REFUSED — comparison is confounded:")
        for r in refused:
            print(f"  - {r}")
        sys.exit(2)

    print(f"[topo-diff] {a_path}  python {a['python']}")
    print(f"[topo-diff] {b_path}  python {b['python']}")
    print(f"[topo-diff] shared stack: "
          + "  ".join(f"{k}={v}" for k, v in sorted(a["packages"].items())))
    print()

    different = False
    for ra, rb in zip(a["results"], b["results"]):
        n = ra["n_leaves"]
        checks = [
            ("n_clusters", ra["n_clusters"], rb["n_clusters"]),
            ("cluster_sizes", ra["cluster_sizes"], rb["cluster_sizes"]),
            ("membership", ra["membership"], rb["membership"]),
            ("guard_trips", ra.get("guard_trips"), rb.get("guard_trips")),
        ]
        bad = [(name, va, vb) for name, va, vb in checks if va != vb]
        if not bad:
            print(f"  n={n:>3}  MATCH  ({ra['n_clusters']} clusters, "
                  f"sizes {ra['cluster_sizes']}, membership identical)")
            continue
        different = True
        print(f"  n={n:>3}  *** DIFFERS ***")
        for name, va, vb in bad:
            if name == "membership":
                moved = sum(1 for x, y in zip(va, vb) if x != y)
                moved += abs(len(va) - len(vb))
                print(f"        membership: {moved} cluster(s) differ")
                for i, (x, y) in enumerate(zip(va, vb)):
                    if x != y:
                        print(f"          cluster {i}: {x}  vs  {y}")
            else:
                print(f"        {name}: {va}  vs  {vb}")

    print()
    if different:
        print("[topo-diff] VERDICT: DIFFERENT — the interpreter moves tree "
              "topology on this host. The probe alone convicts: cell 6 "
              "re-runs under the locked interpreter; no GPU rebuild is "
              "needed to establish the drift.")
        sys.exit(1)
    print("[topo-diff] VERDICT: IDENTICAL at every probed n, memberships "
          "included. The topology stack is cleared; the risk NARROWS to "
          "the embedder (torch cp312 vs cp313 wheels) and is not yet "
          "cleared — the single-unit rebuild under the locked interpreter "
          "remains the authority.")


if __name__ == "__main__":
    main()
