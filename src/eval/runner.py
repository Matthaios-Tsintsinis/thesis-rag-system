"""CLI runner: one system x one benchmark x one split, JSONL output.

Usage:
    python -m src.eval.runner --system M2 --benchmark qasper --split validation \
        --output local_runs/eval/qasper_M2_validation.jsonl --max-units 20

Sharding across systems / benchmarks is done by running this script
once per combination from a wrapper (shell script or Colab notebook
cell). One process, one system, one benchmark — simple to bisect, simple
to retry, no shared state. Each invocation writes a single JSONL file
that downstream analysis aggregates.

`--max-units` runs the small-sample validation gate (per the Pass-1
plan: ~20 papers x all systems before the full validation run, to
catch architecture bugs cheaply).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import paths
from ..config import DEFAULT_CONFIG, HarnessConfig
from ..retrievers.base import BaseSystem
from ..retrievers.m1_closedbook import ClosedBookSystem
from ..retrievers.m2_flat_dense import FlatDenseSystem
from ..retrievers.m3_hybrid import HybridRRFSystem
from ..retrievers.m4_raptor import RaptorSystem
from ..retrievers.m6_hipporag import HippoRAGSystem
from ..retrievers.m7_three_axis import ThreeAxisSystem
from ..retrievers.m9_corrective import CorrectiveRAGSystem
from .base import BenchmarkRunner
from .multihop import MultiHopBenchmark
from .qasper import QasperBenchmark
from .quality import QualityBenchmark


SYSTEM_REGISTRY: dict[str, type[BaseSystem]] = {
    "M1": ClosedBookSystem,
    "M2": FlatDenseSystem,
    "M3": HybridRRFSystem,
    "M4": RaptorSystem,
    "M6": HippoRAGSystem,
    "M7": ThreeAxisSystem,
    "M9": CorrectiveRAGSystem,
}

BENCHMARK_REGISTRY: dict[str, type] = {
    "qasper": QasperBenchmark,
    "multihop_rag": MultiHopBenchmark,
    "quality": QualityBenchmark,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one system x one benchmark x one split to JSONL."
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=sorted(SYSTEM_REGISTRY),
        help="Retrieval system id (M1/M2/M3/M4/M6/M7/M9).",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARK_REGISTRY),
        help="Benchmark id (qasper, multihop_rag, quality).",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Benchmark-specific split. QASPER: train/validation/test. "
        "MultiHop-RAG: validation/test/all (single underlying split).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path. Defaults to "
        "<OUTPUT_DIR>/eval/{benchmark}_{system}_{split}_{stamp}.jsonl",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Cap the number of EvalUnits processed. For QASPER this is "
        "max papers (~20 is the recommended small-sample gate before "
        "the full validation run). For MultiHop-RAG only 0 or 1 is "
        "meaningful (the dataset is one EvalUnit).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap TOTAL queries across units. Useful for MultiHop-RAG "
        "where the single EvalUnit holds 2556 queries — pass "
        "`--max-queries 50` for a small-sample shared-corpus "
        "validation run before the full 2556.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-unit progress logs (still writes JSONL).",
    )
    parser.add_argument(
        "--evidence-budget",
        type=int,
        default=None,
        help="OPT-IN CK-4 context-volume ablation: cap the evidence "
        "block fed to the generator at this many tokens (measured via "
        "tiktoken gpt-4o-mini). Default OFF — baselines feed their "
        "natural full retrieval per professor's directive. Pass e.g. "
        "--evidence-budget 3000 to run an ablation that equalises "
        "the chunks-only context size across systems for diagnostic "
        "comparison. Monkey-patches src.config.EVIDENCE_TOKEN_BUDGET "
        "for the process lifetime.",
    )
    args = parser.parse_args()

    # Opt-in CK-4 budget. Runs before any system instantiation so the
    # config constant is in place before pack_context resolves it.
    if args.evidence_budget is not None and args.evidence_budget > 0:
        from .. import config as _cfg
        _cfg.EVIDENCE_TOKEN_BUDGET = int(args.evidence_budget)
        print(
            f"[eval] CK-4 ABLATION ENABLED: evidence_budget="
            f"{args.evidence_budget} tokens (monkey-patched). "
            f"Default is no-budget; this is an opt-in comparison."
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output is None:
        out_root = paths.output_dir() / "eval"
        out_root.mkdir(parents=True, exist_ok=True)
        suffix = ""
        if args.evidence_budget:
            suffix = f"_budget{args.evidence_budget}"
        args.output = out_root / (
            f"{args.benchmark}_{args.system}_{args.split}{suffix}_{stamp}.jsonl"
        )

    print(
        f"[eval] {args.system} x {args.benchmark} x {args.split} -> {args.output}"
    )
    if args.max_units is not None:
        print(f"[eval] max_units={args.max_units} (small-sample mode)")

    system_cls = SYSTEM_REGISTRY[args.system]
    system: BaseSystem = system_cls(config=HarnessConfig())

    benchmark_cls = BENCHMARK_REGISTRY[args.benchmark]
    benchmark = benchmark_cls()

    runner = BenchmarkRunner(output_path=args.output, verbose=not args.quiet)
    n_scored = 0
    sum_retr_f1 = 0.0
    sum_retr_skipped = 0
    sum_ans = 0.0
    for scored in runner.run(
        system,
        benchmark,
        split=args.split,
        max_units=args.max_units,
        max_queries=args.max_queries,
    ):
        n_scored += 1
        if scored.retrieval.skipped:
            sum_retr_skipped += 1
        else:
            sum_retr_f1 += scored.retrieval.f1
        sum_ans += scored.answer.value

    # Aggregate summary alongside the JSONL.
    summary_path = args.output.with_suffix(".summary.json")
    n_retr_scored = max(1, n_scored - sum_retr_skipped)
    summary = {
        "system": args.system,
        "benchmark": args.benchmark,
        "split": args.split,
        "max_units": args.max_units,
        "n_queries_scored": n_scored,
        "n_retrieval_skipped": sum_retr_skipped,
        "mean_retrieval_f1": sum_retr_f1 / n_retr_scored,
        "mean_answer_score": sum_ans / max(1, n_scored),
        "benchmark_stats": getattr(benchmark, "stats", {}),
        "output_path": str(args.output),
        "timestamp": stamp,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[eval] summary -> {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
