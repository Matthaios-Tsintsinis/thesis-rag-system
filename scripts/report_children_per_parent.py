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
time and refuses to aggregate unless a selector reduces it to one per unit. A
silently-picked wrong-era tree would be a wrong number wearing a measured
one's provenance. Two selectors, combinable:

  --pick-env SUBSTRING     matches on the manifest's build_env string.
                           Sufficient where the eras differ by stack (the
                           old HotpotQA era records env=None).
  --from-summary PATH      reads the banked cell's `.summary.json` and keeps
                           only manifests whose `created_at` falls inside
                           that cell's RUN WINDOW - [timestamp - 30 min,
                           timestamp + elapsed_s + 30 min]; the summary
                           timestamp marks the run's START. This is the
                           discriminator where probe builds share the cell's
                           env string (NarrativeQA: five probe builds plus
                           the banked one under identical stacks). The
                           window is READ from the summary, never
                           hand-typed. Timestamps: the summary's stamp is
                           the runner host's local time and manifests are
                           UTC; on the Colab run host local time IS UTC,
                           and this script is run-host-only for exactly
                           that class of reason.

ALSO REPORTED: POOL AVAILABILITY (the AF-10 ceiling diagnostic). Each
manifest carries the collapsed pool's composition (`flat_n_chunks`,
`flat_n_summaries`), so the cell-level non-leaf AVAILABILITY - the fraction
of the pool that is non-leaf, i.e. the ceiling on any retrieved non-leaf
share - aggregates here for cells whose ROWS predate the per-row
availability field. Compare it against `analyse`'s retrieved share: if
retrieved >= available, the App. I band presupposes tree depth and the
summary preference is intact conditional on availability.

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


def _run_window(summary_path: Path, slack_s: int = 1800):
    """[start - slack, start + elapsed + slack] of the banked cell run.

    THE TIMESTAMP MARKS THE RUN'S START, NOT ITS END. `runner.py` captures
    `stamp` at entry (line ~604, for output-file naming) -- roughly 150
    lines before the run loop's own clock starts -- and only WRITES it
    into the summary at the end. The first version of this function read
    "written at the end" as "marks the end" and subtracted `elapsed_s`,
    shifting the window one run-length too early. Caught on real data:
    cell 2's manifests (created 23:13-23:37 UTC) matched a window whose
    parsed timestamp was 23:09 -- four minutes BEFORE the first manifest,
    impossible for an end-stamp on a three-hour run. So:

        start  = parse(timestamp)
        end    = start + elapsed_s
        window = [start - slack, end + slack]

    Timestamps: the stamp is the runner host's local time and manifests
    are UTC; on the Colab run host local time IS UTC, and this script is
    run-host-only for exactly that class of reason."""
    from datetime import datetime, timedelta, timezone

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stamp = summary.get("timestamp")
    elapsed = summary.get("elapsed_s")
    if not stamp or elapsed is None:
        raise SystemExit(
            f"[cpp] {summary_path} lacks timestamp/elapsed_s -- cannot "
            "derive a run window; fall back to --pick-env"
        )
    start = datetime.strptime(str(stamp), "%Y%m%d-%H%M%S").replace(
        tzinfo=timezone.utc
    )
    end = start + timedelta(seconds=float(elapsed))
    pad = timedelta(seconds=slack_s)
    return (start - pad, end + pad)


def _created_in(m: dict, window) -> bool:
    from datetime import datetime, timezone

    raw = m.get("created_at")
    if not raw:
        return False
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return window[0] <= ts <= window[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cache-root", default=None,
                    help="cache root (default: the pipeline's own paths.cache_dir())")
    ap.add_argument("--pick-env", default=None,
                    help="substring of build_env selecting ONE manifest when a "
                         "corpus_hash matches several build eras")
    ap.add_argument("--from-summary", default=None,
                    help="path to the banked cell's .summary.json; keeps only "
                         "manifests created inside that cell's run window")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else paths.cache_dir()
    by_hash = _index_manifests(cache_root)

    window = None
    if args.from_summary:
        window = _run_window(Path(args.from_summary))
        print(f"[cpp] run window from summary: {window[0].isoformat()} -> "
              f"{window[1].isoformat()}")

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
        if window is not None:
            cands = [m for m in cands if _created_in(m, window)]
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
        n_leaf_pool = int(st.get("flat_n_chunks") or 0)
        n_sum_pool = int(st.get("flat_n_summaries") or 0)
        per_unit.append({
            "corpus_id": str(u.corpus_id),
            "parents": int(st.get("n_summary_nodes_at_index") or 0),
            "mean_cpp": float(st.get("gate_children_per_parent") or 0.0),
            "mean_summary_tokens": float(st.get("gate_mean_summary_tokens") or 0.0),
            "degenerate": bool(st.get("degenerate_no_tree")),
            "pool_leaves": n_leaf_pool,
            "pool_summaries": n_sum_pool,
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

    # Pool availability over TREE-BUILDING units (AF-10 ceiling): micro =
    # all summary nodes over all pool nodes; macro = mean per-unit fraction.
    tb_pool_total = sum(r["pool_leaves"] + r["pool_summaries"] for r in tree_units)
    tb_pool_sum = sum(r["pool_summaries"] for r in tree_units)
    avail_micro = (tb_pool_sum / tb_pool_total) if tb_pool_total else None
    per_unit_avail = [
        r["pool_summaries"] / (r["pool_leaves"] + r["pool_summaries"])
        for r in tree_units
        if (r["pool_leaves"] + r["pool_summaries"]) > 0
    ]
    avail_macro = (sum(per_unit_avail) / len(per_unit_avail)
                   if per_unit_avail else None)
    mean_pool = (tb_pool_total / len(tree_units)) if tree_units else None

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
        "pool_non_leaf_available_micro": avail_micro,
        "pool_non_leaf_available_macro": avail_macro,
        "pool_mean_size_treebuilding": mean_pool,
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
        if avail_micro is not None:
            print(f"  pool non-leaf avail.   micro {avail_micro:.1%} / "
                  f"macro {avail_macro:.1%}   mean pool "
                  f"{mean_pool:.1f} nodes   <- CEILING on any retrieved "
                  "non-leaf share (AF-10): compare analyse output for the "
                  "tree-building retrieved share against this, not only "
                  "against the paper band")
    else:
        print("  no tree-building units matched -- nothing to aggregate")
    if not complete:
        print("  A headline over partial coverage is not the cell's figure; "
              "resolve the missing units first.")


if __name__ == "__main__":
    main()
