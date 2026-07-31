"""Paired significance diagnostic over banked eval JSONLs. No API, no GPU.

WHY THIS EXISTS
---------------
The rollup tables report bare means, and the spreads between systems are
small: MultiHop M9 vs M3 differs by ~0.011 retrieval-F1 and ~0.007
answer-F1; QASPER spanned 0.338-0.343 across four systems; QuALITY
0.723-0.735. Differences that size are only claims if they clear the
detection threshold at the n each benchmark actually provides. This
script decides that, per pair, per metric, from the banked per-query
scores — so an ordering in the table is either supported or explicitly
labelled as not.

It answers three questions:
  1. Which pairwise orderings are statistically supported at full n?
  2. What effect size must a new system (M7) clear on each benchmark to
     be a defensible result rather than a rounding difference?
  3. What should the matrix report — bare means, or means with CIs?

STATISTICS, AND WHY THESE TESTS
-------------------------------
Comparisons are PAIRED: every system answers the same queries, so the
unit of analysis is the per-query difference, and the relevant spread is
SD of that difference — NOT the SD of either system's scores. Paired
analysis is far more powerful here because most of the variance is
"this query is hard", which is common to both systems and cancels.

The difference distribution is badly non-normal: for M9 vs M3 it is
zero-inflated by construction (M9 returns M3's exact set whenever the
corrective action is CORRECT, ~49% of queries) with large swings on the
remainder. So:

  * P-VALUE: paired sign-flip PERMUTATION test. Under the null "the two
    systems are exchangeable on each query", flipping the sign of a
    per-query difference is an equally likely outcome, so the sign-flip
    distribution IS the exact null. Zero deltas flip to zero and
    correctly contribute nothing. This makes no normality assumption and
    handles the spike at zero exactly right — which a t-test does not.
  * INTERVAL: paired BOOTSTRAP over queries (resample query indices with
    replacement, recompute the mean delta). Reports a percentile CI on
    the observed difference, which is what belongs in a results table.
  * MDE: the normal-approximation minimum detectable effect,
    2.8 * SD_diff / sqrt(n), at 80% power and alpha=.05 two-sided. Kept
    as a PLANNING number — it is what tells Phase 2 what M7 must beat —
    while the permutation p-value is what adjudicates an observed
    result. Where the two disagree, trust the permutation test.

MULTIPLICITY: 5 systems give 10 pairs per (benchmark, metric). At
alpha=.05 across ~40 tests you expect ~2 spurious "significant" results,
so Holm-Bonferroni is applied WITHIN each (benchmark, metric) family and
both raw and adjusted p-values are printed. A single pre-registered
comparison (e.g. M7 vs the best baseline) carries no such penalty, which
is a reason to name it in advance rather than sweep all pairs.

USAGE
-----
    python -m scripts.significance_diagnostic --dir outputs/matrix_baseline

Filenames are the banked convention `{benchmark}_{system}_{split}.jsonl`
(the runner's default adds a timestamp; the matrix cells were written
with an explicit --output and carry none). Both are accepted.

CAVEATS THE OUTPUT CANNOT SHOW YOU
----------------------------------
  * NarrativeQA and QuALITY have no gold evidence spans, so every
    retrieval score is skipped; retrieval rows there are empty by
    construction, not by failure.
  * M1 is closed-book and has no retrieval at all.
  * narrativeqa_M4 was never completed, so its pairs are absent.
  * These are gpt-4o-mini-era answer scores. Retrieval deltas are
    generator-independent and survive a generator change; ANSWER deltas
    do not, and must be re-measured under the local generators.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


SYSTEMS = ("M1", "M2", "M3", "M4", "M9")
# QASPER and QuALITY are being dropped from the matrix, but their banked
# cells are included deliberately: they are where the tightest orderings
# were read off (QASPER 0.338-0.343, QuALITY 0.723-0.735 across four
# systems), so they are the sharpest test of whether differences in that
# range were ever supported. Answering that is worth more than the
# benchmarks themselves.
#
# NOTE on QuALITY: its answer metric is multiple-choice accuracy, so
# per-query deltas are in {-1, 0, +1}. The paired sign-flip permutation
# test handles that correctly and degenerate-free — on binary paired
# data it reduces to an exact McNemar test, which is the textbook-right
# choice. A t-test on {-1,0,+1} would not be.
BENCHMARKS = ("multihop_rag", "narrativeqa", "qasper", "quality")
METRICS = ("retrieval", "answer")

N_BOOT = 10_000
N_PERM = 10_000
# 80% power, alpha=.05 two-sided: (z_{1-a/2} + z_{power}) = 1.96 + 0.84.
MDE_Z = 2.80
SEED = 12345


def _find(root: Path, benchmark: str, system: str, split: str) -> Path | None:
    exact = root / f"{benchmark}_{system}_{split}.jsonl"
    if exact.exists():
        return exact
    stamped = sorted(root.glob(f"{benchmark}_{system}_{split}_*.jsonl"))
    return stamped[-1] if stamped else None


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    """query_id -> {'retrieval': f1 or None, 'answer': value}."""
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            retr = r.get("retrieval") or {}
            ans = r.get("answer") or {}
            out[r["query_id"]] = {
                "retrieval": (
                    None if retr.get("skipped", True) else float(retr.get("f1", 0.0))
                ),
                "answer": float(ans.get("value", 0.0)),
            }
    return out


def paired_deltas(
    a: dict[str, dict[str, float]],
    b: dict[str, dict[str, float]],
    metric: str,
) -> np.ndarray:
    """b - a over queries both scored on `metric` (None => not scored)."""
    keys = sorted(a.keys() & b.keys())
    vals = [
        b[k][metric] - a[k][metric]
        for k in keys
        if a[k][metric] is not None and b[k][metric] is not None
    ]
    return np.asarray(vals, dtype=float)


def permutation_p(d: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """Two-sided paired sign-flip permutation test on the mean difference.

    Exact null for paired exchangeable data; immune to the spike at zero
    and to heavy tails. +1 in numerator and denominator is the standard
    unbiased small-sample correction (never reports p=0).
    """
    if len(d) == 0:
        return float("nan")
    obs = abs(float(np.mean(d)))
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def bootstrap_ci(
    d: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI on the mean paired difference, resampling queries."""
    if len(d) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="dir holding the banked JSONLs")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.dir)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    rng = np.random.default_rng(SEED)

    for benchmark in BENCHMARKS:
        loaded: dict[str, dict] = {}
        for s in systems:
            p = _find(root, benchmark, s, args.split)
            if p is None:
                print(f"[{benchmark}] MISSING cell for {s} — pairs skipped")
                continue
            loaded[s] = load_scores(p)
        if len(loaded) < 2:
            print(f"[{benchmark}] fewer than two cells present; skipping\n")
            continue

        for metric in METRICS:
            rows = []
            for a, b in itertools.combinations(sorted(loaded), 2):
                d = paired_deltas(loaded[a], loaded[b], metric)
                if len(d) == 0:
                    continue
                sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
                mean = float(np.mean(d))
                mde = MDE_Z * sd / np.sqrt(len(d)) if len(d) else float("nan")
                lo, hi = bootstrap_ci(d, rng, alpha=args.alpha)
                p = permutation_p(d, rng)
                rows.append({
                    "pair": f"{a} vs {b}", "n": len(d), "mean": mean, "sd": sd,
                    "mde": mde, "lo": lo, "hi": hi, "p": p,
                    "nonzero": int(np.sum(np.abs(d) > 1e-12)),
                })
            if not rows:
                print(f"[{benchmark}/{metric}] no scored pairs "
                      "(no gold evidence for this benchmark?)\n")
                continue

            adj = holm([r["p"] for r in rows])
            for r, a_ in zip(rows, adj):
                r["p_holm"] = a_

            print(f"\n=== {benchmark} / {metric} "
                  f"({len(rows)} pairs, Holm-corrected within this family) ===")
            print(f"{'pair':14s} {'n':>5s} {'delta':>9s} {'SD_diff':>8s} "
                  f"{'MDE':>8s} {'95% CI':>19s} {'p':>8s} {'p_holm':>8s} "
                  f"{'nonzero':>8s}  verdict")
            for r in sorted(rows, key=lambda x: x["p"]):
                clears = abs(r["mean"]) >= r["mde"]
                ci_excl = (r["lo"] > 0) or (r["hi"] < 0)
                sig = r["p_holm"] < args.alpha
                verdict = (
                    "SUPPORTED" if (sig and ci_excl)
                    else "marginal" if (ci_excl or clears)
                    else "NOT SUPPORTED"
                )
                print(f"{r['pair']:14s} {r['n']:5d} {r['mean']:+9.4f} "
                      f"{r['sd']:8.4f} {r['mde']:8.4f} "
                      f"[{r['lo']:+.4f},{r['hi']:+.4f}] {r['p']:8.4f} "
                      f"{r['p_holm']:8.4f} {r['nonzero']:8d}  {verdict}")

            sds = [r["sd"] for r in rows]
            ns = [r["n"] for r in rows]
            pooled_sd = float(np.median(sds))
            n_med = int(np.median(ns))
            print(
                f"\n  PHASE-2 TARGET for {benchmark}/{metric}: a new system must "
                f"beat its comparator by >= {MDE_Z * pooled_sd / np.sqrt(n_med):.4f} "
                f"(median SD_diff {pooled_sd:.4f}, n {n_med}) to be detectable "
                "at 80% power as a single pre-registered comparison."
            )

    print(
        "\nNOTE: 'SUPPORTED' = Holm-adjusted permutation p < alpha AND the "
        "bootstrap CI excludes zero. 'marginal' = one of the two only. "
        "Retrieval deltas are generator-independent; answer deltas are "
        "gpt-4o-mini-era and must be re-measured under the local generators."
    )


if __name__ == "__main__":
    main()
