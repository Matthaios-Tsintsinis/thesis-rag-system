"""Derive the M9 CorrectiveRAG evaluator thresholds empirically.

The CRAG paper's published thresholds (0.59 / -0.99) live on its
fine-tuned-T5 score scale and do not transfer to our off-the-shelf
bge-reranker evaluator. This script derives (tau_high, tau_low) from
the reranker's actual sigmoid-confidence distribution on a labeled
VALIDATION sample, with gold labels coming from the existing CK-2
alignment (chunk.gold_provenance vs the union of per-annotator gold
passage sets).

CRITERION (v2 — non-gold percentile / FPR control):

  tau_high  — the --nongold-high-percentile (90th) percentile of the
              NON-GOLD confidence distribution: a chunk above tau_high
              scores higher than ~all known-irrelevant chunks, i.e.
              P(conf >= tau_high | non-gold) <= 10% by construction.
              "At least one confidently-relevant doc" is the CORRECT
              trigger, so this is false-positive-rate-oriented.
  tau_low   — the --gold-low-percentile (5th) percentile of the GOLD
              confidence distribution: discarding everything below it
              loses at most ~5% of gold chunks. "ALL docs below" is
              the INCORRECT trigger, so this is recall-protective.
  tau_strip — fixed to tau_low by design (one fewer free parameter).

WHY NOT AN ABSOLUTE PRECISION TARGET (v1, retired): QASPER's
gold-paragraph base rate in the retrieved pools is ~9%, and
topically-relevant-but-unannotated chunks count as false positives,
so precision-against-gold is structurally capped well below an
uncalibrated absolute target (the v1 run measured a 0.50 ceiling at
every cut — itself a reportable finding: the off-the-shelf
reranker-as-evaluator cannot reach the precision an absolute target
implies on QASPER gold, which is part of why the paper fine-tuned its
evaluator; see the deviations/limitations note in
src/retrievers/m9_corrective.py). The percentile criterion is robust
to label sparsity because it conditions on the fat, stable class
(non-gold, ~91% of chunks). Precision and LIFT over the base rate at
the chosen cut are still computed and reported as cross-check
diagnostics in the artifact, not as selection criteria.

The script also reports, per the calibration review:
  * decile tables for the gold and non-gold distributions
    (entanglement visibility);
  * the derived action mix, with an explicit DEAD-BRANCH warning when
    the INCORRECT branch fires on <1% of queries (document, don't
    hide);
  * a strip-refinement simulation at tau_strip(=tau_low): the
    fraction of 2-sentence strips that would survive refinement, with
    an INERT-REFINEMENT warning when survival is >95%.

Derived ONCE on QASPER validation (rule: develop on validation,
reserve test; no per-benchmark threshold tuning — MultiHop transfer
is checked by comparing realized action mixes in analyse.py, not by
re-deriving). The realized Correct/Ambiguous/Incorrect mix of the
first real M9 run must roughly match the derivation-time mix printed
here; large drift = miscalibration, flag before trusting results.

Cost: NO OpenAI calls (M3 retrieval + reranker only). The reranker
wants a GPU (T4 fine); the strip simulation roughly triples the
reranker pairs — disable with --no-strip-sim if pressed. The M3
substrate cache is reused when present.

Usage (from the repo root, Colab or local):

    python -m scripts.derive_corrective_thresholds \
        [--split validation] [--max-units 20] [--max-queries N] \
        [--nongold-high-percentile 90.0] [--gold-low-percentile 5.0] \
        [--lift-k 4.0] [--no-strip-sim] [--output PATH]

Output: a JSON artifact (distributions, chosen taus, action mix,
strip survival, parameters) under <OUTPUT_DIR>/m9_thresholds/ —
disk-only, NOT committed. The chosen taus get baked into
CorrectiveConfig defaults with a provenance comment in a follow-up
commit.
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
    split_strips,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def derive_tau_high_percentile(
    nongold_scores: np.ndarray,
    *,
    percentile: float,
) -> float:
    """tau_high = the given percentile of the non-gold distribution.

    FPR control: by construction at most (100 - percentile)% of
    known-irrelevant chunks score above the cut. Conditions on the
    statistically fat class, so it is robust to gold-label sparsity
    (the trap that killed the v1 absolute-precision criterion).
    """
    return float(np.percentile(nongold_scores, percentile))


def precision_lift_at(
    tau: float,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Cross-check diagnostics at a cut: precision, base rate, lift, FPR."""
    above = scores >= tau
    n_above = int(above.sum())
    base_rate = float(labels.mean()) if labels.size else 0.0
    precision = float(labels[above].mean()) if n_above else 0.0
    nongold = labels == 0
    fpr = (
        float((scores[nongold] >= tau).mean()) if int(nongold.sum()) else 0.0
    )
    return {
        "support_above": n_above,
        "precision": precision,
        "base_rate": base_rate,
        "lift": (precision / base_rate) if base_rate > 0 else None,
        "fpr_nongold": fpr,
    }


def _decile_table(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    table = {f"p{p}": float(np.percentile(a, p)) for p in range(10, 100, 10)}
    table["min"] = float(a.min())
    table["max"] = float(a.max())
    return table


def _print_deciles(name: str, xs: list[float]) -> None:
    a = np.asarray(xs, dtype=np.float64)
    print(f"[derive] {name}: n={a.size}  mean={a.mean():.3f}  std={a.std():.3f}")
    deciles = "  ".join(
        f"p{p}={np.percentile(a, p):.3f}" for p in range(10, 100, 10)
    )
    print(f"[derive]   {deciles}")
    hist, edges = np.histogram(a, bins=10, range=(0.0, 1.0))
    peak = max(1, int(hist.max()))
    for c, lo in zip(hist.tolist(), edges[:-1].tolist()):
        print(f"[derive]   {lo:.1f}-{lo + 0.1:.1f}  {c:6d}  {'#' * int(40 * c / peak)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive M9 evaluator thresholds from the bge-reranker "
        "confidence distribution on a labeled QASPER validation sample "
        "(non-gold-percentile criterion)."
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-units", type=int, default=20,
                        help="Papers in the derivation sample (QASPER small-sample gate size).")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Optional cap on total queries (debug runs).")
    parser.add_argument("--nongold-high-percentile", type=float, default=90.0,
                        help="tau_high = this percentile of NON-GOLD confidences "
                        "(FPR control; 90 -> at most 10%% of known-irrelevant "
                        "chunks score above the cut).")
    parser.add_argument("--gold-low-percentile", type=float, default=5.0,
                        help="tau_low = this percentile of GOLD confidences "
                        "(recall-protective discard line).")
    parser.add_argument("--lift-k", type=float, default=4.0,
                        help="Reported-only cross-check: flag whether the lift "
                        "at tau_high reaches k x base rate.")
    parser.add_argument("--no-strip-sim", action="store_true",
                        help="Skip the strip-refinement survival simulation "
                        "(saves ~2/3 of the reranker pairs).")
    parser.add_argument("--output", default=None,
                        help="JSON output path. Defaults to "
                        "<OUTPUT_DIR>/m9_thresholds/derivation_{split}_{stamp}.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    benchmark = QasperBenchmark()
    system = HybridRRFSystem(DEFAULT_CONFIG)
    strip_cfg = DEFAULT_CONFIG.corrective.strip_sentences

    gold_scores: list[float] = []
    nongold_scores: list[float] = []
    per_query_confs: list[list[float]] = []
    per_query_strip_survival: list[tuple[int, int]] = []  # (n_strips, query idx into strip confs)
    strip_conf_store: list[list[float]] = []  # per-query strip confidences
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

            if not args.no_strip_sim:
                strips = [
                    s
                    for r in retrieved
                    for s in split_strips(r.chunk.text, strip_cfg)
                ]
                if strips:
                    s_logits = rerank_scores(q.question_text, strips)
                    s_confs = _sigmoid(np.asarray(s_logits, dtype=np.float32))
                    strip_conf_store.append([float(c) for c in s_confs])
                else:
                    strip_conf_store.append([])

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

    if not gold_scores or not nongold_scores:
        raise SystemExit(
            "[derive] FATAL: empty gold or non-gold class in the sample — "
            "check gold_provenance stamping (index_items) and the "
            "split/sample size."
        )

    scores = np.asarray(gold_scores + nongold_scores, dtype=np.float64)
    labels = np.asarray([1] * len(gold_scores) + [0] * len(nongold_scores), dtype=np.int64)

    # --- distributions (entanglement visibility) ---
    print()
    _print_deciles("GOLD confidences", gold_scores)
    print()
    _print_deciles("NON-GOLD confidences", nongold_scores)

    # --- thresholds ---
    tau_high = derive_tau_high_percentile(
        np.asarray(nongold_scores), percentile=args.nongold_high_percentile
    )
    tau_low = float(np.percentile(np.asarray(gold_scores), args.gold_low_percentile))
    diag = precision_lift_at(tau_high, scores, labels)

    flags: list[str] = []
    if tau_low >= tau_high:
        flags.append(
            f"DEGENERATE: tau_low ({tau_low:.3f}) >= tau_high ({tau_high:.3f}) "
            "— the AMBIGUOUS band is empty or inverted. Adjust percentiles."
        )
    if diag["lift"] is not None and diag["lift"] < args.lift_k:
        flags.append(
            f"CROSS-CHECK: lift at tau_high = {diag['lift']:.2f}x is below "
            f"the k={args.lift_k}x reference. The cut enriches weakly over "
            "the base rate — evaluator signal on this benchmark is limited; "
            "report alongside the precision-ceiling finding."
        )

    # --- action-mix simulation ---
    def _mix(th: float, tl: float) -> dict[str, float]:
        counts = Counter(decide_action(cs, th, tl) for cs in per_query_confs)
        total = max(1, len(per_query_confs))
        return {
            a: round(counts.get(a, 0) / total, 4)
            for a in (ACTION_CORRECT, ACTION_AMBIGUOUS, ACTION_INCORRECT)
        }

    derived_mix = _mix(tau_high, tau_low)
    placeholder_mix = _mix(
        DEFAULT_CONFIG.corrective.tau_high, DEFAULT_CONFIG.corrective.tau_low
    )
    for action, frac in derived_mix.items():
        if frac > 0.9:
            flags.append(
                f"DEGENERATE action mix: {action} takes {frac:.0%} of "
                "queries — the thresholds barely discriminate. Adjust "
                "percentiles and re-derive."
            )
    if derived_mix[ACTION_INCORRECT] < 0.01:
        flags.append(
            f"DEAD BRANCH: the INCORRECT branch fires on "
            f"{derived_mix[ACTION_INCORRECT]:.1%} of dev queries (needs ALL "
            f"15 chunks < tau_low={tau_low:.3f}). On QASPER the corrective "
            "re-retrieval path is effectively limited to the AMBIGUOUS "
            "branch — a documented finding, not a bug."
        )

    # --- strip-refinement survival at tau_strip(=tau_low) ---
    strip_survival: dict | None = None
    if not args.no_strip_sim and strip_conf_store:
        per_query_frac: list[float] = []
        total_strips = 0
        surviving = 0
        for s_confs in strip_conf_store:
            if not s_confs:
                continue
            n = len(s_confs)
            kept = sum(1 for c in s_confs if c >= tau_low)
            total_strips += n
            surviving += kept
            per_query_frac.append(kept / n)
        overall = surviving / max(1, total_strips)
        strip_survival = {
            "tau_strip": tau_low,
            "n_strips_total": total_strips,
            "n_strips_surviving": surviving,
            "overall_survival": round(overall, 4),
            "per_query_survival_mean": (
                round(float(np.mean(per_query_frac)), 4) if per_query_frac else None
            ),
            "per_query_survival_p10": (
                round(float(np.percentile(per_query_frac, 10)), 4)
                if per_query_frac
                else None
            ),
        }
        if overall > 0.95:
            flags.append(
                f"INERT REFINEMENT: {overall:.1%} of strips survive "
                f"tau_strip={tau_low:.3f} — strip refinement is a near-no-op "
                "on QASPER at this threshold. Document it (config flag stays "
                "for the ablation); do not claim refinement lift here."
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
            "deciles": _decile_table(xs),
            "values": [round(float(v), 5) for v in xs],
        }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = {
        "criterion": {
            "version": "v2_nongold_percentile",
            "tau_high_rule": f"p{args.nongold_high_percentile} of NON-GOLD "
            "confidences (FPR control)",
            "tau_low_rule": f"p{args.gold_low_percentile} of GOLD confidences "
            "(recall-protective)",
            "tau_strip_rule": "tau_low (fixed by design)",
            "v1_retired": "absolute precision target (0.8) uncalibratable on "
            "QASPER gold: ~9% base rate + unannotated-but-relevant chunks "
            "cap precision at ~0.50 at every cut (v1 run, 2026-06-12)",
        },
        "params": {
            "split": args.split,
            "max_units": args.max_units,
            "max_queries": args.max_queries,
            "nongold_high_percentile": args.nongold_high_percentile,
            "gold_low_percentile": args.gold_low_percentile,
            "lift_k_reference": args.lift_k,
            "strip_sim": not args.no_strip_sim,
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
        "tau_high_diagnostics": diag,
        "tau_low": tau_low,
        "action_mix_derived": derived_mix,
        "action_mix_placeholder_0p7_0p3": placeholder_mix,
        "strip_survival": strip_survival,
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
          f"{len(gold_scores)} gold / {len(nongold_scores)} non-gold chunks "
          f"(base rate {diag['base_rate']:.3f})")
    print(f"[derive] tau_high = {tau_high:.4f}  "
          f"(p{args.nongold_high_percentile} of non-gold; precision "
          f"{diag['precision']:.3f}, lift "
          f"{diag['lift']:.2f}x, FPR {diag['fpr_nongold']:.3f}, "
          f"support {diag['support_above']})")
    print(f"[derive] tau_low  = {tau_low:.4f}  "
          f"(p{args.gold_low_percentile} of gold confidences)")
    print(f"[derive] action mix (derived):     {derived_mix}")
    print(f"[derive] action mix (placeholder): {placeholder_mix}")
    if strip_survival:
        print(f"[derive] strip survival at tau_strip={tau_low:.3f}: "
              f"{strip_survival['overall_survival']:.1%} overall "
              f"({strip_survival['n_strips_surviving']}/"
              f"{strip_survival['n_strips_total']} strips)")
    for f in flags:
        print(f"[derive] WARNING: {f}")
    print(f"[derive] artifact -> {out_path}")
    hard_degenerate = any(f.startswith("DEGENERATE") for f in flags)
    if not hard_degenerate:
        print("\n[derive] NEXT: bake tau_high / tau_low into CorrectiveConfig "
              "defaults with a provenance comment citing this artifact and "
              "the v2 criterion; remove the PROVISIONAL marker. DEAD-BRANCH "
              "and INERT-REFINEMENT warnings (if any) are documented "
              "findings, not blockers.")


if __name__ == "__main__":
    main()
