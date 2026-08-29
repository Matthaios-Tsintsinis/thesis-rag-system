"""Diff banked M4 tree TOPOLOGIES for one benchmark across two build eras.

WHY THIS SCRIPT EXISTS. The P11 slot-6 cell (M4/hotpotqa under Llama)
failed the exact-reproduction check: children/parent micro 4.897 against
the required 4.901, total parents 3,426 against 3,423, while macro
(5.059) and the degenerate count (83) reproduced exactly. Two hypotheses
explain a +3-parent delta, and they demand opposite actions:

  A. GENERATOR-DEPENDENT UPPER LAYERS. Some units carry a third layer in
     BOTH eras; layer-2 clustering runs on SUMMARY embeddings, which
     differ by summariser, so layer-2 parent counts may legitimately
     move. Signature: every differing unit has layer 0 and layer 1
     IDENTICAL across eras and differs only at layers >= 2. Action:
     refine the check ("2-layer subset exact; >=3-layer units may
     differ, count them") and land the cell.

  B. CROSS-ERA LEAF-CLUSTERING DRIFT. The Qwen cell-6 trees were built
     under CPython 3.13.15 (the declared exception); the Llama trees are
     the first full-scale re-derivation under the pinned 3.12.13. Layer-1
     clustering sees only leaf embeddings and is summariser-independent,
     so a layer-1 count differing ANYWHERE is not explainable by the
     generator swap. Signature: any differing unit whose layer-1 (or
     layer-0) size moved. Action: the cell does NOT land until explained,
     and the cell-6 interpreter declaration has its first counter-
     evidence at full scale (note which era is the pinned one).

THE DISCRIMINATOR IS PER-UNIT LAYER SIZES, which every manifest records
(`extra.index_stats.tree_depth_counts`). This script matches units across
the eras by `corpus_hash` (content-only; survives every cache-key move),
selects one manifest per unit per era, refuses ambiguity rather than
resolving it silently, and classifies every differing unit as L0_DIFFERS
/ L1_DIFFERS / UPPER_ONLY.

THE PROBE ASSERTS IT MEASURED WHAT IT CLAIMS. Before any verdict it
reproduces each era's banked aggregates (total parents, degenerate
count) from the manifests it selected; if they do not match the
expectations passed on the command line, the manifest population is
wrong (bad selector, wrong window) and the diff is REFUSED -- a diff
over the wrong trees would be a vacuous pass wearing a verdict.

Era selection, combinable per era (same machinery as
report_children_per_parent):
  --summary-a/--summary-b PATH   keep manifests created inside that
                                 cell's run window (start stamp +
                                 elapsed, 30 min slack)
  --pick-env-a/--pick-env-b S    substring of the manifest's build_env
                                 (e.g. "python=3.13" uniquely names the
                                 cp313-era Qwen cell-6 trees)

Run-host only for narrativeqa/hotpotqa (the loaders read the HF cache
there). Typical slot-6 invocation:

    python -m scripts.diff_tree_topologies --benchmark hotpotqa \\
      --pick-env-a "python=3.13" \\
      --summary-b /content/drive/MyDrive/thesis_rag/outputs/p11/hotpotqa_M4_validation.summary.json \\
      --expect-parents-a 3423 --expect-parents-b 3426 --expect-degenerate 83
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths
from src.retrievers.m4_raptor import RaptorSystem
from src.cache import corpus_content_hash
from scripts.report_children_per_parent import (
    BENCHMARKS,
    _benchmark,
    _created_in,
    _index_manifests,
    _run_window,
    _stats_of,
)


def _select(cands: list[dict], window, pick_env: str | None) -> list[dict]:
    if pick_env:
        cands = [m for m in cands
                 if pick_env in str((m.get("extra") or {}).get("build_env", ""))]
    if window is not None:
        cands = [m for m in cands if _created_in(m, window)]
    return cands


def _depths(st: dict) -> dict[int, int]:
    """tree_depth_counts with int keys (JSON round-trips them to strings)."""
    raw = st.get("tree_depth_counts") or {}
    return {int(k): int(v) for k, v in raw.items()}


def _classify(da: dict[int, int], db: dict[int, int]) -> str:
    if da.get(0) != db.get(0):
        return "L0_DIFFERS"
    if da.get(1) != db.get(1):
        return "L1_DIFFERS"
    return "UPPER_ONLY"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--summary-a", default=None,
                    help="era A cell .summary.json (run-window selector)")
    ap.add_argument("--summary-b", default=None,
                    help="era B cell .summary.json (run-window selector)")
    ap.add_argument("--pick-env-a", default=None,
                    help="build_env substring selecting era A manifests")
    ap.add_argument("--pick-env-b", default=None,
                    help="build_env substring selecting era B manifests")
    ap.add_argument("--expect-parents-a", type=int, default=None,
                    help="banked total parents for era A; the probe refuses "
                         "a verdict unless it reproduces this from manifests")
    ap.add_argument("--expect-parents-b", type=int, default=None)
    ap.add_argument("--expect-degenerate", type=int, default=None,
                    help="banked degenerate count; asserted for BOTH eras")
    args = ap.parse_args()

    if not (args.summary_a or args.pick_env_a) or not (args.summary_b or args.pick_env_b):
        raise SystemExit("[diff] each era needs at least one selector "
                         "(--summary-X and/or --pick-env-X)")

    cache_root = Path(args.cache_root) if args.cache_root else paths.cache_dir()
    by_hash = _index_manifests(cache_root)
    win_a = _run_window(Path(args.summary_a)) if args.summary_a else None
    win_b = _run_window(Path(args.summary_b)) if args.summary_b else None
    for tag, w in (("A", win_a), ("B", win_b)):
        if w is not None:
            print(f"[diff] era {tag} window: {w[0].isoformat()} -> {w[1].isoformat()}")

    system = RaptorSystem()  # config only; no model, no GPU
    rows: list[dict] = []
    missing: list[tuple[str, str]] = []      # (era, corpus_id)
    ambiguous: list[tuple[str, str]] = []

    for u in _benchmark(args.benchmark).iter_eval_units(split=args.split):
        with tempfile.TemporaryDirectory(prefix="diff_") as td:
            system._write_corpus_layout(list(u.corpus), Path(td))
            chash = corpus_content_hash(Path(td))
        cands = by_hash.get(chash, [])
        per_era: dict[str, dict] = {}
        for tag, window, env in (("A", win_a, args.pick_env_a),
                                 ("B", win_b, args.pick_env_b)):
            sel = _select(cands, window, env)
            if not sel:
                missing.append((tag, str(u.corpus_id)))
                continue
            if len(sel) > 1:
                ambiguous.append((tag, str(u.corpus_id)))
                print(f"[diff] AMBIGUOUS era {tag} {u.corpus_id!r}: "
                      f"{len(sel)} manifests share corpus_hash {chash[:12]}...")
                for m in sel:
                    print(f"        dir={m['_dir'][:16]}... "
                          f"created={m.get('created_at')} "
                          f"env={(m.get('extra') or {}).get('build_env')}")
                continue
            st = _stats_of(sel[0])
            if st is None:
                missing.append((tag, str(u.corpus_id)))
                continue
            per_era[tag] = st
        if len(per_era) == 2:
            rows.append({"corpus_id": str(u.corpus_id),
                         "a": per_era["A"], "b": per_era["B"]})

    if ambiguous:
        raise SystemExit(f"[diff] REFUSING: {len(ambiguous)} (era, unit) pairs "
                         "match several manifests -- sharpen the selectors.")
    if missing:
        preview = ", ".join(f"{t}:{c}" for t, c in missing[:8])
        raise SystemExit(f"[diff] REFUSING: {len(missing)} (era, unit) pairs "
                         f"have no manifest ({preview}...) -- a diff over "
                         "partial coverage cannot support a verdict.")
    if not rows:
        raise SystemExit("[diff] no units matched -- nothing to compare")

    # --- reproduce the banked aggregates BEFORE any verdict -------------
    tot = {"A": 0, "B": 0}
    deg = {"A": 0, "B": 0}
    for r in rows:
        for tag, st in (("A", r["a"]), ("B", r["b"])):
            tot[tag] += int(st.get("n_summary_nodes_at_index") or 0)
            deg[tag] += 1 if st.get("degenerate_no_tree") else 0
    print(f"[diff] units compared: {len(rows)}")
    print(f"[diff] era A: parents {tot['A']}  degenerate {deg['A']}")
    print(f"[diff] era B: parents {tot['B']}  degenerate {deg['B']}")
    bad = []
    if args.expect_parents_a is not None and tot["A"] != args.expect_parents_a:
        bad.append(f"era A parents {tot['A']} != banked {args.expect_parents_a}")
    if args.expect_parents_b is not None and tot["B"] != args.expect_parents_b:
        bad.append(f"era B parents {tot['B']} != banked {args.expect_parents_b}")
    if args.expect_degenerate is not None:
        for tag in ("A", "B"):
            if deg[tag] != args.expect_degenerate:
                bad.append(f"era {tag} degenerate {deg[tag]} != banked "
                           f"{args.expect_degenerate}")
    if bad:
        raise SystemExit("[diff] PROBE INVALID -- selected manifests do not "
                         "reproduce the banked aggregates; the diff below "
                         "would compare the wrong trees:\n  " + "\n  ".join(bad))

    # --- the diff -------------------------------------------------------
    diffs = []
    for r in rows:
        da, db = _depths(r["a"]), _depths(r["b"])
        pa = int(r["a"].get("n_summary_nodes_at_index") or 0)
        pb = int(r["b"].get("n_summary_nodes_at_index") or 0)
        if da != db or pa != pb:
            diffs.append({"corpus_id": r["corpus_id"],
                          "class": _classify(da, db),
                          "depths_a": da, "depths_b": db,
                          "parents_a": pa, "parents_b": pb})

    print(f"\n[diff] identical units: {len(rows) - len(diffs)} / {len(rows)}")
    print(f"[diff] differing units: {len(diffs)}")
    for d in diffs:
        print(f"  {d['corpus_id']}  [{d['class']}]")
        print(f"    era A layers {d['depths_a']}  parents {d['parents_a']}")
        print(f"    era B layers {d['depths_b']}  parents {d['parents_b']}")

    if not diffs:
        print("\nVERDICT: topologies IDENTICAL across the eras.")
        return
    classes = {d["class"] for d in diffs}
    if classes <= {"UPPER_ONLY"}:
        n3 = sum(1 for d in diffs
                 if max(d["depths_a"]) >= 2 or max(d["depths_b"]) >= 2)
        print(f"\nVERDICT: HYPOTHESIS A -- every differing unit matches on "
              f"layers 0 and 1 and differs only at layers >= 2 ({n3} of "
              f"{len(diffs)} carry a third layer). Layer >= 2 clustering runs "
              "on SUMMARY embeddings and is generator-dependent by design. "
              "Refine the exactness check to the 2-layer subset, correct any "
              "'all units are 2-layer' claim, count the deep units, and LAND "
              "the cell.")
    else:
        print("\nVERDICT: HYPOTHESIS B -- at least one unit differs at layer "
              "0 or layer 1, which the summariser cannot touch. This is "
              "cross-era topology drift. The cell does NOT land until "
              "explained. Note which era ran under the pinned interpreter "
              "(era B / python=3.12 for slot 6; the Qwen cell-6 trees are "
              "the declared 3.13.15 exception) -- the drift indicts the "
              "unpinned era's trees, and the cell-6 interpreter declaration "
              "now has counter-evidence at full scale. Escalate to the "
              "operator before ANY further landing.")


if __name__ == "__main__":
    main()
