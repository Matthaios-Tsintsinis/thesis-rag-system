"""Derive the M9 CorrectiveRAG evaluator thresholds empirically.

The CRAG paper's published thresholds (0.59 / -0.99) live on its
fine-tuned-T5 score scale and do not transfer to our off-the-shelf
bge-reranker evaluator. This script derives (tau_high, tau_low) from
the reranker's actual sigmoid-confidence distribution on a labeled
VALIDATION sample, with gold labels coming from the existing CK-2
alignment (chunk.gold_provenance vs the union of per-annotator gold
passage sets):

  tau_high  — the LOWEST confidence such that
              P(gold | conf >= tau_high) >= --precision-target (0.8),
              subject to a minimum support of --min-support chunks
              above the cut. "At least one confidently-relevant doc"
              is the CORRECT trigger, so this is precision-oriented.
  tau_low   — the --gold-low-percentile (5th) percentile of the GOLD
              confidence distribution: discarding everything below it
              loses at most ~5% of gold chunks. "ALL docs below" is
              the INCORRECT trigger, so this is recall-protective.
  tau_strip — fixed to tau_low by design (one fewer free parameter).

Derived ONCE on QASPER validation (rule: develop on validation,
reserve test; no per-benchmark threshold tuning — MultiHop transfer
is checked by comparing realized action mixes in analyse.py, not by
re-deriving). The realized Correct/Ambiguous/Incorrect mix of the
first real M9 run must roughly match the derivation-time mix printed
here; large drift = miscalibration, flag before trusting results.

Cost: NO OpenAI calls (M3 retrieval + reranker only). The reranker
wants a GPU (T4 fine); CPU works for a small sample but crawls.
The M3 substrate cache is reused when present.

Usage (from the repo root, Colab or local):

    python -m scripts.derive_corrective_thresholds \
        [--split validation] [--max-units 20] [--max-queries N] \
        [--precision-target 0.8] [--gold-low-percentile 5.0] \
        [--min-support 20] [--output PATH]

Output: a JSON artifact (distributions, chosen taus, action mix,
parameters) under <OUTPUT_DIR>/m9_thresholds/ — disk-only, NOT
committed. The chosen taus get baked into CorrectiveConfig defaults
with a provenance comment in a follow-up commit.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from src import paths
from src.config import DEFAULT_CONFIG
from src.eval.qasper import QasperBenchmark
from src.models import rerank_scores
from src.retrievers.m3_hybrid import HybridRRFSystem
from src.retrievers.m9_corrective import (
    ACTION_AMBIGUOUS,
    ACTION_CORRECT,
    ACTION_INCORRECT,
    decide_action,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def derive_tau_high(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    precision_target: float,
    min_support: int,
) -> tuple[float | None, dict]:
    """Lowest threshold whose above-cut precision meets the target.

    Scans every unique score as a candidate cut (descending). Precision
    above a cut is not monotonic in the threshold, so we take the
    MINIMUM candidate (most inclusive CORRECT trigger) among all cuts
    that satisfy precision >= target with support >= min_support.
    Returns (tau, diagnostics); tau is None when no cut qualifies
    (degenerate — caller must flag, not silently bake).
    """
    order = np.argsort(-scores)
    sorted_labels = labels[order].astype(np.float64)
    sorted_scores = scores[order]
    cum_gold = np.cumsum(sorted_labels)
    counts = np.arange(1, len(sorted_scores) + 1, dtype=np.float64)
    precision_at = cum_gold / counts

    qualifying: list[tuple[float, float, int]] = []
    for i in range(len(sorted_scores)):
        # Cut at this score: everything with conf >= sorted_scores[i].
        # Use the LAST index of a tied score so support/precision
        # reflect the full tie group.
        if i + 1 < len(sorted_scores) and sorted_scores[i + 1] == sorted_scores[i]:
            continue
        support = int(counts[i])
        if support < min_support:
            continue
        if precision_at[i] >= precision_target:
            qualifying.append((float(sorted_scores[i]), float(precision_at[i]), support))

    diag = {
        "n_candidates_qualifying": len(qualifying),
        "best_precision_any_cut": float(precision_at.max()) if len(precision_at) else 0.0,
    }
    if not qualifying:
        return None, diag
    tau, prec, support = min(qualifying, key=lambda t: t[0])
    diag["precision_at_tau_high"] = prec
    diag["support_at_tau_high"] = support
    return tau, diag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive M9 evaluator thresholds from the bge-reranker "
        "confidence distribution on a labeled QASPER validation sample."
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-units", type=int, default=20,
                        help="Papers in the derivation sample (QASPER small-sample gate size).")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Optional cap on total queries (debug runs).")
    parser.add_argument("--precision-target", type=float, default=0.8)
    parser.add_argument("--gold-low-percentile", type=float, default=5.0)
    parser.add_argument("--min-support", type=int, default=20,
                        help="Minimum chunks above a tau_high candidate cut.")
    parser.add_argument("--output", default=None,
                        help="JSON output path. Defaults to "
                        "<OUTPUT_DIR>/m9_thresholds/derivation_{split}_{stamp}.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    benchmark = QasperBenchmark()
    system = HybridRRFSystem(DEFAULT_CONFIG)

    gold_scores: list[float] = []
    nongold_scores: list[float] = []
    per_query_confs: list[list[float]] = []
    n_queries = 0
    n_queries_no_gold = 0

    for unit in benchmark.iter_eval_units(split=args.split, max_units=args.max_units):
        system.index_items(unit.corpus)
        for q in unit.queries:
            if args.max_queries is not None and n_queries >= args.max_queries:
                break
            retrieved = system.retrieve(q.question_text)  # natural top-15
            if not retrieved:
                continue
            logits = rerank_scores(q.question_text, [r.chunk.text for r in retrieved])
            confs = _sigmoid(np.asarray(logits, dtype=np.float32))
            per_query_confs.append([float(c) for c in confs])
            n_queries += 1

            gold_union: set[tuple[str, str]] = set()
            for atom_set in q.gold_passage_sets:
                gold_union |= set(atom_set)
            if not gold_union:
                # Unanswerable / table-only gold: no positive class for
                # the distributions, but the query still counts toward
                # the action-mix simulation (runtime sees it too).
                n_queries_no_gold += 1
                continue
            for r, c in zip(retrieved, confs):
                provenance = set(getattr(r.chunk, "gold_provenance", ()) or ())
                is_gold = bool(provenance & gold_union)
                (gold_scores if is_gold else nongold_scores).append(float(c))
        if args.max_queries is not None and n_queries >= args.max_queries:
            break

    if not gold_scores:
        raise SystemExit(
            "[derive] FATAL: no gold-labeled chunks in the sample — check "
            "gold_provenance stamping (index_items) and the split/sample size."
        )

    scores = np.asarray(gold_scores + nongold_scores, dtype=np.float64)
    labels = np.asarray([1] * len(gold_scores) + [0] * len(nongold_scores), dtype=np.int64)

    tau_high, tau_high_diag = derive_tau_high(
        scores, labels,
        precision_target=args.precision_target,
        min_support=args.min_support,
    )
    tau_low = float(np.percentile(np.asarray(gold_scores), args.gold_low_percentile))

    flags: list[str] = []
    if tau_high is None:
        flags.append(
            f"NO tau_high cut reaches precision >= {args.precision_target} "
            f"with support >= {args.min_support} "
            f"(best precision {tau_high_diag['best_precision_any_cut']:.3f}). "
            "Lower the target or grow the sample; do NOT bake a guess."
        )
    elif tau_low >= tau_high:
        flags.append(
            f"DEGENERATE: tau_low ({tau_low:.3f}) >= tau_high ({tau_high:.3f}) "
            "— the AMBIGUOUS band is empty or inverted. Adjust targets."
        )

    def _mix(th: float, tl: float) -> dict[str, float]:
        counts = Counter(decide_action(cs, th, tl) for cs in per_query_confs)
        total = max(1, len(per_query_confs))
        return {
            a: round(counts.get(a, 0) / total, 4)
            for a in (ACTION_CORRECT, ACTION_AMBIGUOUS, ACTION_INCORRECT)
        }

    derived_mix = _mix(tau_high, tau_low) if tau_high is not None else None
    placeholder_mix = _mix(
        DEFAULT_CONFIG.corrective.tau_high, DEFAULT_CONFIG.corrective.tau_low
    )
    if derived_mix:
        for action, frac in derived_mix.items():
            if frac > 0.9:
                flags.append(
                    f"DEGENERATE action mix: {action} takes {frac:.0%} of "
                    "queries — the thresholds barely discriminate. Adjust "
                    "targets and re-derive."
                )

    def _dist(xs: list[float]) -> dict:
        a = np.asarray(xs, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "percentiles": {
                str(p): float(np.percentile(a, p))
                for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            },
            "values": [round(float(v), 5) for v in xs],
        }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = {
        "params": {
            "split": args.split,
            "max_units": args.max_units,
            "max_queries": args.max_queries,
            "precision_target": args.precision_target,
            "gold_low_percentile": args.gold_low_percentile,
            "min_support": args.min_support,
            "reranker": "BAAI/bge-reranker-v2-m3 (sigmoid of raw logits)",
            "retriever": "M3 HybridRRFSystem natural top-15",
        },
        "sample": {
            "n_queries": n_queries,
            "n_queries_no_gold": n_queries_no_gold,
            "n_gold_chunks": len(gold_scores),
            "n_nongold_chunks": len(nongold_scores),
        },
        "tau_high": tau_high,
        "tau_high_diagnostics": tau_high_diag,
        "tau_low": tau_low,
        "action_mix_derived": derived_mix,
        "action_mix_placeholder_0p7_0p3": placeholder_mix,
        "flags": flags,
        "gold_conf_distribution": _dist(gold_scores),
        "nongold_conf_distribution": _dist(nongold_scores),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "timestamp": stamp,
    }

    if args.output is None:
        out_dir = paths.output_dir() / "m9_thresholds"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"derivation_{args.split}_{stamp}.json"
    else:
        from pathlib import Path

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[derive] sample: {n_queries} queries "
          f"({n_queries_no_gold} without gold), "
          f"{len(gold_scores)} gold / {len(nongold_scores)} non-gold chunks")
    print(f"[derive] tau_high = {tau_high}  (target precision "
          f">= {args.precision_target}, diagnostics {tau_high_diag})")
    print(f"[derive] tau_low  = {tau_low:.4f}  "
          f"({args.gold_low_percentile}th pct of gold confidences)")
    print(f"[derive] action mix (derived):     {derived_mix}")
    print(f"[derive] action mix (placeholder): {placeholder_mix}")
    for f in flags:
        print(f"[derive] WARNING: {f}")
    print(f"[derive] artifact -> {out_path}")
    if tau_high is not None and not flags:
        print("\n[derive] NEXT: bake these values into CorrectiveConfig "
              "(tau_high / tau_low defaults) with a provenance comment "
              "citing this artifact; remove the PROVISIONAL marker.")


if __name__ == "__main__":
    main()
