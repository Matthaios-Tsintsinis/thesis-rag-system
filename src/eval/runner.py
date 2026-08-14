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
from dataclasses import replace
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
from .hotpotqa import HotpotQABenchmark, HotpotQAPooledBenchmark
from .multihop import MultiHopBenchmark
from .narrativeqa import NarrativeQABenchmark
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
    "narrativeqa": NarrativeQABenchmark,
    # HotpotQA ships as TWO benchmarks, not one flag: variant A is the
    # comparable headline, variant B is where a hierarchy exists. They
    # produce different corpora and different unit counts, so a shared
    # entry with a mode switch would make every downstream table
    # ambiguous about which was run.
    "hotpotqa": HotpotQABenchmark,
    "hotpotqa_pooled": HotpotQAPooledBenchmark,
}


def _environment_provenance() -> dict:
    """Lockfile hash, GPU model, python and the pinned versions.

    The GPU string is here because the reproducibility target is
    same-lockfile-SAME-GPU-CLASS: a tree that reproduces on an L4 is not
    thereby claimed to reproduce on another accelerator, and a row that
    does not say which GPU produced it cannot support either claim.
    """
    try:
        from pathlib import Path as _Path

        from scripts.pin_environment import environment_provenance

        return environment_provenance(_Path("requirements.lock"))
    except Exception as e:  # never fatal: provenance must not kill a run
        return {"error": f"{type(e).__name__}: {e}"}


def _model_revisions(system) -> dict:
    """Resolved model ids for the three roles, with HF revision hashes.

    An id alone does not pin a model: a repo can move. The revision is
    what makes "the same embedder" checkable rather than assumed. Absent
    revisions degrade to None rather than failing the run — this is
    provenance, not a gate.
    """
    out: dict = {}
    try:
        resolved = getattr(system, "resolved_components", None)
        if resolved is not None:
            out["embedder"] = getattr(resolved, "embedder_id", None)
            out["reranker"] = getattr(resolved, "reranker_id", None)
            out["index_llm"] = getattr(resolved, "index_llm_id", None)
        out["generator"] = system.config.generation.model
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    revisions: dict = {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for role, model_id in list(out.items()):
            if not model_id or "/" not in str(model_id):
                continue
            try:
                revisions[role] = api.model_info(str(model_id)).sha
            except Exception:
                revisions[role] = None
    except Exception:
        pass
    out["revisions"] = revisions
    return out


def _tree_build_env() -> str | None:
    """The resolved topology stack, recorded per cell. None if raptor_paper
    is not importable (no tree-building system in this run)."""
    try:
        from ..raptor_paper import PAPER_TREE_BUILD_ENV

        return PAPER_TREE_BUILD_ENV
    except Exception:
        return None


def _git_commit_short() -> str:
    """Short HEAD hash for run provenance in summary.json. Never fatal —
    a clone without git metadata (or no git binary) degrades to
    "unknown" rather than killing an eval run."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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
        help="Benchmark id. hotpotqa = standard distractor (one corpus "
        "per question; M4 has NO TREE there and its rows are not a RAPTOR "
        "result). hotpotqa_pooled = shards of 100 questions (a real tree, "
        "but NOT comparable to published HotpotQA).",
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
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Enable TWO-PHASE answering: retrieve every query in a unit "
            "first, then generate the unit in batches of this size. "
            "Omit for the historic sequential path. Only affects systems "
            "with supports_batched_answer=True (M1 and M7 stay "
            "sequential). Needed for local generation, where sequential "
            "answering wastes most of the GPU; measured feasible batch "
            "on an L4 with Qwen2.5-7B fp16 is 8 at 4k context and 32 at "
            "800 tokens. NOTE: batch composition can change generated "
            "text even at temperature 0, so keep this FIXED across the "
            "cells you intend to compare."
        ),
    )
    parser.add_argument(
        "--max-padded-tokens",
        type=int,
        default=None,
        help=(
            "Cap n * longest-prompt within a generation batch, instead of "
            "using a fixed batch count. Real prompts are ragged and a batch "
            "pads to its longest member, so a count tuned on uniform "
            "synthetic prompts OOMs on real ones: batch 8 survived uniform "
            "4k prompts at 21.7GB and OOM'd on real MultiHop prompts. With "
            "this set, --batch-size becomes an upper bound on COUNT only. "
            "One knob covers both context regimes (M4 ~2k, M2/M3/M9 ~4k). "
            "Suggested starting point on a 24GB L4 with Qwen2.5-7B fp16: "
            "20000, then raise until it stops fitting."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override the generation cap for this run. THIS IS THE LEVER "
            "THAT ACTUALLY REACHES GENERATION. Rebinding "
            "src.config.GEN_MAX_NEW_TOKENS in-process does NOT work and "
            "fails SILENTLY: GenerationConfig.max_new_tokens takes that "
            "constant as a dataclass field DEFAULT, which Python evaluates "
            "once at class-definition time, so a later rebind changes "
            "nothing the generator reads. This flag instead constructs the "
            "GenerationConfig explicitly. When set, the runner VERIFIES "
            "every emitted answer against the cap and aborts if the cap "
            "did not apply -- a measurement that silently did not measure "
            "what it claimed is the failure mode this exists to prevent."
        ),
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help=(
            "Run this cell under a DIFFERENT model. Sets the READER "
            "(HarnessConfig.generation.model) AND the INDEX-TIME "
            "SUMMARISER (M4/M7 summary_model) together, because the "
            "matrix design is FULL INDEPENDENT REPLICATION: each column "
            "builds its own trees with its own summariser and reads them "
            "with the same model. "
            "Rebinding src.config.GENERATOR_MODEL or JUDGE_MODEL "
            "in-process does NOT work -- both are dataclass field "
            "defaults evaluated once at class-definition time. "
            "VERIFIED: changing this moves M4's and M7's substrate cache "
            "keys, so a Llama cell cannot silently hit a Qwen tree. "
            "M2/M3/M9 keys do NOT move, which is CORRECT -- their "
            "substrate contains no LLM output, so it is a "
            "model-independent artifact and rebuilding it would produce "
            "byte-identical files."
        ),
    )
    parser.add_argument(
        "--expand-summary-nodes",
        action="store_true",
        help=(
            "M4 DIAGNOSTIC TWIN. Replace each retrieved summary node with "
            "its top-N descendant LEAVES, which DO carry gold_provenance, "
            "so CK-2 can score them. Exists to quantify how much of M4's "
            "retrieval deficit is a MEASUREMENT ARTIFACT: summary nodes "
            "are unscoreable by construction, so a share of M4's returned "
            "units cannot contribute to recall no matter how good they "
            "are. Query-time only -- the substrate cache key does NOT "
            "move, so the tree is reused rather than rebuilt. "
            "NEVER A REPORTABLE M4 CELL: the evidence text becomes leaves, "
            "so the answers are a different system's. Write it to a "
            "separate directory; every row carries "
            "metadata.m4_summary_expansion and analyse prints a banner. "
            "Pair with --max-new-tokens 1: the twin exists for RETRIEVAL "
            "scores, and retrieval is generator-independent, so paying for "
            "full answers buys nothing."
        ),
    )
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help=(
            "Load the generator BEFORE the timed run and report the load "
            "time separately. Without this the first system measured pays "
            "a ~15GB download/load that later ones do not, which makes "
            "cross-system timings meaningless -- observed: a probe where "
            "M1 appeared slower than M2 purely because M1 ran first."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to an existing output JSONL and SKIP query_ids already "
            "in it, instead of truncating and starting over. Use after a "
            "session dies mid-pass: index caches survive on their own, but "
            "without this the answers do not. A torn final line from a "
            "killed write is tolerated and that query is re-answered."
        ),
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
    elif not args.output.is_absolute():
        # A RELATIVE --output resolves against the CURRENT DIRECTORY, not
        # against OUTPUT_DIR. On Colab that is the cloned repo under
        # /content, which a runtime restart DELETES.
        #
        # Worse than losing the file: `--resume` reads the JSONL at this
        # path to decide what to skip, and when the file is gone it finds
        # nothing and opens in "w" mode. A resume aimed at a vanished
        # path therefore TRUNCATES and silently restarts from zero, which
        # is exactly the situation resume exists to prevent.
        resolved = args.output.resolve()
        print(
            "[eval] *** WARNING: --output is a RELATIVE path. It resolves "
            f"to {resolved}\n"
            "    which is NOT under OUTPUT_DIR and is EPHEMERAL on Colab "
            "(a runtime restart deletes /content).\n"
            f"    Drive-backed location would be: "
            f"{paths.output_dir() / args.output}\n"
            "    Pass an ABSOLUTE path for anything you intend to keep, "
            "especially matrix cells you may need to --resume. ***"
        )

    print(
        f"[eval] {args.system} x {args.benchmark} x {args.split} -> {args.output}"
    )
    if args.max_units is not None:
        print(f"[eval] max_units={args.max_units} (small-sample mode)")

    # Build the config EXPLICITLY rather than rebinding module constants.
    # Only src.config.EVIDENCE_TOKEN_BUDGET is patchable that way (it is
    # read through the module object at call time, deliberately); every
    # other constant in src/config.py is baked into a dataclass field
    # default at import and a rebind is silently ignored.
    harness_cfg = HarnessConfig()
    if args.generator is not None:
        # READER AND SUMMARISER TOGETHER. The matrix is a full
        # independent replication: each column builds its own trees with
        # its own summariser and reads them with the same model. Setting
        # only the reader would produce a column whose trees came from
        # the other model, which is the confound this design exists to
        # avoid on M1/M2/M3/M9 and cannot avoid on M4.
        harness_cfg = replace(
            harness_cfg,
            generation=replace(harness_cfg.generation, model=args.generator),
            m4=replace(harness_cfg.m4, summary_model=args.generator),
            m7=replace(harness_cfg.m7, summary_model=args.generator),
        )
        print(
            f"[eval] GENERATOR OVERRIDE: {args.generator}\n"
            f"    reader           = {harness_cfg.generation.model}\n"
            f"    index summariser = {harness_cfg.m4.summary_model}\n"
            "    M4/M7 substrate cache keys MOVE with this, so this cell "
            "builds its own trees and cannot hit the other column's.\n"
            "    M2/M3/M9 keys do NOT move — their substrate has no LLM "
            "in it, so a cache hit there reuses a model-INDEPENDENT "
            "artifact, not the other column's work."
        )
    if args.expand_summary_nodes:
        harness_cfg = replace(
            harness_cfg,
            m4=replace(harness_cfg.m4, expand_summary_nodes=True),
        )
        print("[eval] M4 LEAF-EXPANDED DIAGNOSTIC TWIN enabled.")
        print("    Retrieval becomes CK-2-comparable to a leaf-only system.")
        print("    Substrate key is UNMOVED, so the tree is reused.")
        print("    *** NOT a reportable M4 cell - answers are a different "
              "system's. Keep this out of the matrix directory. ***")
    if args.max_new_tokens is not None:
        if args.max_new_tokens < 1:
            parser.error("--max-new-tokens must be >= 1")
        harness_cfg = replace(
            harness_cfg,
            generation=replace(
                harness_cfg.generation, max_new_tokens=args.max_new_tokens
            ),
        )
        print(
            f"[eval] generation cap OVERRIDDEN to "
            f"{harness_cfg.generation.max_new_tokens} tokens (verified after "
            "each answer)"
        )

    system_cls = SYSTEM_REGISTRY[args.system]
    system: BaseSystem = system_cls(config=harness_cfg)

    # BENCHMARK FIRST, AND PREFLIGHT BEFORE PREWARM. A HotpotQA run once
    # died on an unresolvable dataset id at the first iter_eval_units
    # call -- after --prewarm had already pulled 15 GB of Qwen into VRAM.
    # Cheap preconditions get checked before expensive ones are paid.
    # `preflight` is optional: a benchmark without one is simply not
    # checked, and any benchmark can add the same two-second guard.
    benchmark_cls = BENCHMARK_REGISTRY[args.benchmark]
    benchmark = benchmark_cls()
    preflight = getattr(benchmark, "preflight", None)
    if callable(preflight):
        preflight()

    load_s = None
    if args.prewarm:
        t_load = time.perf_counter()
        from ..models import load_generator

        load_generator(
            harness_cfg.generation.model, harness_cfg.generation.load_in_4bit
        )
        load_s = time.perf_counter() - t_load
        print(f"[eval] prewarm: generator resident in {load_s:.1f}s "
              "(EXCLUDED from the timings below)")

    runner = BenchmarkRunner(
        output_path=args.output,
        verbose=not args.quiet,
        batch_size=args.batch_size,
        resume=args.resume,
        max_padded_tokens=args.max_padded_tokens,
        verify_max_new_tokens=args.max_new_tokens,
    )
    n_scored = 0
    sum_retr_f1 = 0.0
    sum_retr_skipped = 0
    sum_ans = 0.0
    # COMPOSITION DISCLOSURE. The answer column is ONE number, but on
    # MultiHop it averages two different measurement scales: token-F1 on
    # answerable queries and a binary rule on null ones. A micro-mean
    # over two scales is not legible unless the split is published
    # beside it, so the parts are counted here and reported in the
    # summary. The table still carries one column; its caption states
    # the composition.
    n_null = 0
    sum_ans_null = 0.0
    # Timed HERE, around runner.run only, so the prewarm load and every
    # import sit outside it. Recorded in the summary because otherwise
    # the only source of a timing is stdout, and reading a wall clock off
    # a Colab cell is how the first 1-token probe ended up quoting model
    # downloads as compute.
    t_run = time.perf_counter()
    for scored in runner.run(
        system,
        benchmark,
        split=args.split,
        max_units=args.max_units,
        max_queries=args.max_queries,
    ):
        n_scored += 1
        if scored.answer.method == "unanswerable_rule":
            n_null += 1
            sum_ans_null += scored.answer.value
        if scored.retrieval.skipped:
            sum_retr_skipped += 1
        else:
            sum_retr_f1 += scored.retrieval.f1
        sum_ans += scored.answer.value

    elapsed_s = time.perf_counter() - t_run

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
        # The two parts of the one answer column, so the composition is
        # inspectable without re-reading the JSONL.
        "n_answerable": n_scored - n_null,
        "mean_answer_score_answerable": (
            (sum_ans - sum_ans_null) / (n_scored - n_null)
            if n_scored - n_null else None
        ),
        "n_null_queries": n_null,
        "mean_answer_score_null": (
            sum_ans_null / n_null if n_null else None
        ),
        "benchmark_stats": getattr(benchmark, "stats", {}),
        # LOADER-DERIVED, never a literal. P8 asserts the post-dedup row
        # count against this, and a hardcoded constant would abort every
        # NarrativeQA cell the moment P7 re-drew the sample.
        "expected_n_queries": (getattr(benchmark, "stats", {}) or {}).get(
            "n_queries"
        ),
        # Run-condition provenance (the aggregator's conditions columns;
        # every matrix row must be self-describing from birth). The
        # generator field is what keeps Qwen-era and gpt-4o-mini-era
        # numbers from ever conflating in one table.
        "generator": system.config.generation.model,
        # The INDEX-TIME LLM, recorded separately from the reader. Under
        # full independent replication they are the same model, but they
        # are distinct roles and a row must say which model built the
        # trees it read -- that is precisely the M4 confound.
        "index_llm": system.config.m4.summary_model,
        "chunking_strategy": system.config.chunking.strategy,
        "evidence_budget": args.evidence_budget,
        # Recorded so a probe artifact is self-describing. The 1-token
        # probe that silently ran uncapped would have been caught here.
        "max_new_tokens": harness_cfg.generation.max_new_tokens,
        "prewarm_load_s": load_s,
        # Wall clock of runner.run() ALONE — model load excluded when
        # --prewarm is used. s_per_query is the number every cost
        # forecast in this project is built from, so it is recorded
        # rather than re-derived by hand each time.
        "elapsed_s": round(elapsed_s, 2),
        "s_per_query": (
            round(elapsed_s / n_scored, 4) if n_scored else None
        ),
        "batch_size": args.batch_size,
        "max_padded_tokens": args.max_padded_tokens,
        # Cold-tree provenance. None for systems with no tree; False on
        # the P10 pass means the lever took and the tree was rebuilt.
        # P9: which environment produced this row. Recorded per CELL, not
        # per session — a matrix assembled over several sessions must be
        # able to say which stack produced which row.
        "environment": _environment_provenance(),
        "model_revisions": _model_revisions(system),
        "tree_cache_hit": getattr(system, "tree_cache_hit", None),
        "tree_build_env": _tree_build_env(),
        "git_commit": _git_commit_short(),
        "output_path": str(args.output),
        "timestamp": stamp,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[eval] summary -> {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
