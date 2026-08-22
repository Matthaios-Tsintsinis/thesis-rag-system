"""Aggregate the App. C children-per-parent gate per cell, from banked trees.

WHY THIS SCRIPT EXISTS. RAPTOR's paper reports an average of 6.7 children per
parent (App. C; reference range 5.7-6.8). Our builds compute the per-unit
figure at index time -- `tree_stats()['mean_children_per_parent']` lands in
each unit's `manifest.json` under `extra.index_stats.gate_children_per_parent`
-- but nothing aggregated it per CELL, so the gate was measured everywhere and
reported nowhere. Same shape as this project's recurring defect: a correct
value nothing consumed.

HOW UNITS ARE MATCHED TO MANIFESTS, and why not by cache path. The M4
substrate key moved at e907d68 (the interpreter joined `_topology_env_id`), so
`substrate_warm_path` computed at current HEAD names keys the BANKED trees do
not sit under -- by design. Manifests, however, record `corpus_hash`, which is
content-only and does not move with the key. So this script recomputes each
unit's corpus hash through the pipeline's own layout (`_write_corpus_layout`
-> `corpus_content_hash`) and matches manifests on that.

AMBIGUITY IS SURFACED, NEVER RESOLVED SILENTLY. The old bank is never deleted,
so one corpus_hash can match manifests from several build eras. When that
happens the script lists every candidate with its `build_env` and creation
time and refuses to aggregate unless `--pick-env SUBSTRING` selects exactly
one per unit. A silently-picked wrong-era tree would be a wrong number wearing
a measured one's provenance.

AGGREGATION. Per-unit `gate_children_per_parent` is the mean over that unit's
summary nodes, and `n_summary_nodes_at_index` is how many there are, so:

    micro = sum(mean_i * parents_i) / sum(parents_i)   # = total child links
                                                       #   / total parents
    macro = mean over units with >= 1 parent

Micro is the paper's quantity (an average over parents, not over corpora).
Degenerate units have zero parents and are EXCLUDED from both by arithmetic,
not by filtering -- they contribute nothing to either sum; their count is
reported beside the figures.

COVERAGE IS PART OF THE RESULT. A unit with no matching manifest is counted
and named; a headline over partial coverage prints INCOMPLETE beside it.
Run-host only for narrativeqa/hotpotqa (the loaders read the HF cache there).

    python -m scripts.report_children_per_parent --benchmark narrativeqa
    python -m scripts.report_children_per_parent --benchmark hotpotqa --json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths
from src.cache import corpus_content_hash
from src.retrievers.m4_raptor import M4_SUBSTRATE_NAMESPACE, RaptorSystem

BENCHMARKS = ("multihop_rag", "narrativeqa", "hotpotqa", "hotpotqa_pooled")
PAPER_MEAN = 6.7
PAPER_RANGE = (5.7, 6.8)


def _benchmark(name: str):
    if name == "multihop_rag":
        from src.eval.multihop import MultiHopBenchmark
        return MultiHopBenchmark()
    if name == "narrativeqa":
        from src.eval.narrativeqa import NarrativeQABenchmark
        return NarrativeQABenchmark()
    if name == "hotpotqa":
        from src.eval.hotpotqa import HotpotQABenchmark
        return HotpotQABenchmark()
    if name == "hotpotqa_pooled":
        from src.eval.hotpotqa import HotpotQAPooledBenchmark
        return HotpotQAPooledBenchmark()
    raise SystemExit(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")


def _index_manifests(cache_root: Path) -> dict[str, list[dict]]:
    """corpus_hash -> [manifest dict + _dir], across the M4 namespace."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    ns = cache_root / M4_SUBSTRATE_NAMESPACE
    if not ns.is_dir():
        raise SystemExit(
            f"no {M4_SUBSTRATE_NAMESPACE} namespace under {cache_root} -- "
            "wrong --cache-root, or this host holds no banked trees"
        )
    n = 0
    for mp in ns.glob("*/manifest.json"):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[cpp] WARN: unreadable manifest {mp}: {e}")
            continue
        m["_dir"] = str(mp.parent.name)
        by_hash[str(m.get("corpus_hash"))].append(m)
        n += 1
    print(f"[cpp] indexed {n} manifests under {ns}")
    return by_hash


def _stats_of(m: dict) -> dict | None:
    st = (m.get("extra") or {}).get("index_stats")
    return st if isinstance(st, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cache-root", default=None,
                    help="cache root (default: the pipeline's own paths.cache_dir())")
    ap.add_argument("--pick-env", default=None,
                    help="substring of build_env selecting ONE manifest when a "
                         "corpus_hash matches several build eras")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else paths.cache_dir()
    by_hash = _index_manifests(cache_root)

    system = RaptorSystem()  # config only; no model, no GPU
    per_unit: list[dict] = []
    missing: list[str] = []
    ambiguous: list[str] = []

    for u in _benchmark(args.benchmark).iter_eval_units(split=args.split):
        with tempfile.TemporaryDirectory(prefix="cpp_") as td:
            system._write_corpus_layout(list(u.corpus), Path(td))
            chash = corpus_content_hash(Path(td))
        cands = by_hash.get(chash, [])
        if args.pick_env:
            cands = [m for m in cands
                     if args.pick_env in str((m.get("extra") or {}).get("build_env", ""))]
        if not cands:
            missing.append(str(u.corpus_id))
            continue
        if len(cands) > 1:
            ambiguous.append(str(u.corpus_id))
            print(f"[cpp] AMBIGUOUS {u.corpus_id!r}: {len(cands)} manifests share "
                  f"corpus_hash {chash[:12]}...")
            for m in cands:
                print(f"        dir={m['_dir'][:16]}... created={m.get('created_at')} "
                      f"env={(m.get('extra') or {}).get('build_env')}")
            continue
        st = _stats_of(cands[0])
        if st is None:
            missing.append(str(u.corpus_id))
            continue
        per_unit.append({
            "corpus_id": str(u.corpus_id),
            "parents": int(st.get("n_summary_nodes_at_index") or 0),
            "mean_cpp": float(st.get("gate_children_per_parent") or 0.0),
            "mean_summary_tokens": float(st.get("gate_mean_summary_tokens") or 0.0),
            "degenerate": bool(st.get("degenerate_no_tree")),
        })

    if ambiguous:
        raise SystemExit(
            f"[cpp] REFUSING to aggregate: {len(ambiguous)} unit(s) match "
            "several build eras. Re-run with --pick-env to select one "
            "(the cell summary's environment block names the right stack)."
        )
    if not per_unit:
        raise SystemExit("[cpp] no units matched any manifest -- nothing to report")

    tree_units = [r for r in per_unit if r["parents"] > 0]
    degenerate = [r for r in per_unit if r["degenerate"]]
    total_parents = sum(r["parents"] for r in tree_units)
    micro = (sum(r["mean_cpp"] * r["parents"] for r in tree_units) / total_parents
             if total_parents else None)
    macro = (sum(r["mean_cpp"] for r in tree_units) / len(tree_units)
             if tree_units else None)
    sum_tok = (sum(r["mean_summary_tokens"] * r["parents"] for r in tree_units)
               / total_parents if total_parents else None)

    n_expected = len(per_unit) + len(missing)
    complete = not missing
    result = {
        "benchmark": args.benchmark,
        "split": args.split,
        "n_units_expected": n_expected,
        "n_units_matched": len(per_unit),
        "n_units_missing": len(missing),
        "missing_corpus_ids": missing[:20],
        "n_tree_building": len(tree_units),
        "n_degenerate": len(degenerate),
        "total_parents": total_parents,
        "children_per_parent_micro": micro,
        "children_per_parent_macro": macro,
        "paper_mean": PAPER_MEAN,
        "paper_range": list(PAPER_RANGE),
        "in_range_micro": (PAPER_RANGE[0] <= micro <= PAPER_RANGE[1])
                          if micro is not None else None,
        "mean_summary_tokens_micro": sum_tok,
        "coverage_complete": complete,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    tag = "" if complete else "  *** INCOMPLETE COVERAGE ***"
    print(f"\n{args.benchmark} ({args.split}) -- children per parent, from "
          f"banked manifests{tag}")
    print(f"  units matched          {len(per_unit)}/{n_expected}"
          + (f"  (missing: {missing[:5]}{' ...' if len(missing) > 5 else ''})"
             if missing else ""))
    print(f"  tree-building units    {len(tree_units)}   degenerate {len(degenerate)}")
    print(f"  total parents          {total_parents}")
    if micro is not None:
        verdict = ("IN RANGE" if result["in_range_micro"] else "OUT OF RANGE")
        print(f"  children/parent micro  {micro:.3f}   (paper mean {PAPER_MEAN}, "
              f"range {PAPER_RANGE[0]}-{PAPER_RANGE[1]})  {verdict}")
        print(f"  children/parent macro  {macro:.3f}")
        print(f"  mean summary tokens    {sum_tok:.1f}   (paper's 131 was measured "
              "under ITS summariser; ours caps completions at 100 -- "
              "informational, not a gate)")
    else:
        print("  no tree-building units matched -- nothing to aggregate")
    if not complete:
        print("  A headline over partial coverage is not the cell's figure; "
              "resolve the missing units first.")


if __name__ == "__main__":
    main()
