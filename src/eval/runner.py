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
from ..config import DEFAULT_CONFIG, MATRIX_BATCH_SIZE, HarnessConfig
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


def assert_environment_pinned(
    lockfile: Path,
    *,
    allow_unpinned: bool,
) -> None:
    """Abort before any model loads unless this environment is the locked one.

    THE FAILURE THIS GATES IS SILENT AND UNRECOVERABLE AFTER THE FACT.
    The M4 substrate key folds `build_env` — the umap-learn / scikit-learn
    / numpy triple. Colab updates its base image without notice, so a
    session on a drifted image computes a DIFFERENT substrate key. The
    cache then MISSES rather than colliding: the tree rebuilds cleanly,
    the cell succeeds, and the matrix quietly holds two tree populations.
    Nothing in any output says which image built which tree, so no
    post-hoc check can separate them. The only place to catch it is
    before it happens.

    `scripts.pin_environment.check_lockfile` did this correctly and had
    NO CALLERS. That is the sixth instance in this project of a check
    that exists, works, and is inert in the pipeline.

    `allow_unpinned` means "I have no lockfile" — for probes and dev
    runs. It does NOT mean "ignore the lockfile I have": a MISMATCH still
    aborts, because a present-but-violated pin is an operator asserting
    something untrue, which is worse than asserting nothing.
    """
    from scripts.pin_environment import check_lockfile

    if not Path(lockfile).exists():
        if allow_unpinned:
            print(
                f"[eval] WARNING: no lockfile at {lockfile}; running "
                "UNPINNED because --allow-unpinned was passed. M4 tree "
                "topology is not claimed to reproduce from this run."
            )
            return
        raise SystemExit(
            f"PREFLIGHT FAILED: no lockfile at {lockfile}.\n"
            "  The M4 substrate key folds umap-learn/scikit-learn/numpy, "
            "so an unpinned session can build trees under a different key "
            "than the rest of the matrix. That rebuild SUCCEEDS and is "
            "invisible afterwards.\n"
            "  Generate one:  python -m scripts.pin_environment write\n"
            "  Then verify:   python -m scripts.pin_environment check\n"
            "  Probes and dev runs may pass --allow-unpinned."
        )

    if check_lockfile(Path(lockfile)) != 0:
        raise SystemExit(
            "PREFLIGHT FAILED: this environment does not match "
            f"{lockfile} (mismatches printed above).\n"
            "  Reinstall from the lock:  pip install -r "
            f"{lockfile}\n"
            "  --allow-unpinned does NOT bypass this: a lockfile that is "
            "present and violated is a pin the operator asserted and the "
            "environment broke."
        )
    print(f"[eval] PREFLIGHT: environment matches {lockfile}")


def resolve_expected_n_queries(benchmark) -> int | None:  # noqa: ANN001
    """The loader-derived query count P8 asserts a cell against.

    One key, read from one place. HotpotQA used to record only
    `n_questions` while the other loaders recorded `n_queries`, so this
    field was null on ten of twenty cells and P8's short-cell guard had
    nothing to compare against. The loaders now agree; this reads the
    agreed key and returns None rather than guessing at a synonym.
    """
    stats = getattr(benchmark, "stats", {}) or {}
    value = stats.get("n_queries")
    return int(value) if value else None


def assert_expected_n_queries_usable(
    expected: int | None,
    *,
    max_units: int | None,
    max_queries: int | None,
    only_unit: str | None = None,
) -> None:
    """A full cell must carry a count to be checked against.

    P8's guard exists so a TRUNCATED cell aborts instead of reporting a
    partial mean. A null `expected_n_queries` removes that guard without
    removing the appearance of it, which is worse than a short cell —
    nothing downstream can tell the difference between "complete" and
    "unchecked".

    A capped run legitimately has no full-cell expectation, so the check
    applies only when neither cap is set.
    """
    if expected is not None:
        return
    if max_units is not None or max_queries is not None or only_unit:
        return
    raise SystemExit(
        "PROVENANCE FAILED: expected_n_queries is null on an UNCAPPED "
        "run. P8 asserts each cell's row count against this number, so a "
        "null silently disables the short-cell guard — nothing downstream "
        "can then tell a complete cell from an unchecked one. The loader "
        "must record `n_queries` in its stats."
    )


def resolve_chunking_strategy(system) -> str | None:  # noqa: ANN001
    """The chunker the system RESOLVED, not the harness default.

    `system.config.chunking.strategy` is the harness-wide default, which
    M4 does not use: it resolves `raptor_100tok` through
    `resolved_components.chunker_config`. Recording the default meant
    every M4 row in the final table would have named the wrong chunker,
    while the run's own components line printed the right one.
    """
    resolved = getattr(system, "resolved_components", None)
    chunker = getattr(resolved, "chunker_config", None) if resolved else None
    strategy = getattr(chunker, "strategy", None)
    if strategy:
        return strategy
    return getattr(
        getattr(getattr(system, "config", None), "chunking", None),
        "strategy", None,
    )


def assert_population_as_declared(
    benchmark,  # noqa: ANN001
    *,
    n_units_processed: int,
    explicit_reason: str | None,
) -> None:
    """Abort if the cell resolved to a different population than declared.

    THE BACKSTOP to the loader default. NarrativeQA's seeded 40-story
    draw used to materialise only when `--max-units` was typed, so a
    forgotten flag produced a 115-story, 3,461-question cell that ran to
    completion and looked entirely normal. The loader now carries that
    property itself; this checks the OUTCOME, so a future loader change,
    a dataset that grew, or a benchmark whose draw stops matching its
    declaration cannot pass silently.

    An explicit `--max-units` is an operator decision and is honoured:
    explicit 115 stays possible, silent 115 does not.

    A benchmark with no declared `cell_units` is skipped WITH A PRINTED
    NOTE rather than in silence — an undeclared population is exactly
    the condition that hid this defect.
    """
    declared = getattr(benchmark, "cell_units", None)
    if explicit_reason:
        # NAME THE FLAG THAT WAS ACTUALLY GIVEN. This used to take an int
        # and print "--max-units N given explicitly" even when the caller
        # had passed --only-unit and no --max-units at all, so the gate's
        # own message described a run that did not happen.
        print(
            f"[eval] population: {n_units_processed} units "
            f"({explicit_reason}; declared {declared})"
        )
        return
    if declared is None:
        print(
            f"[eval] population: {n_units_processed} units; "
            f"{benchmark.name} declares no cell_units, so nothing was "
            "checked. An undeclared population is how the NarrativeQA "
            "115-vs-40 defect hid."
        )
        return
    if n_units_processed != declared:
        raise SystemExit(
            f"POPULATION MISMATCH: {benchmark.name} processed "
            f"{n_units_processed} units but declares cell_units="
            f"{declared}.\n"
            "  A cell built on the wrong population runs to completion "
            "and looks normal — its rows are simply about different "
            "data than the rest of the matrix.\n"
            "  Pass --max-units explicitly if you intend this."
        )
    print(
        f"[eval] population OK: {n_units_processed} units "
        f"= declared cell_units"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one system x one benchmark x one split to JSONL."
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help=(
            "Run without a requirements.lock. FOR PROBES AND DEV RUNS "
            "ONLY. Without it the run ABORTS when the lockfile is absent, "
            "because the M4 substrate key folds the umap/sklearn/numpy "
            "versions: a session on a drifted Colab image computes a "
            "different key, MISSES the cache rather than colliding, "
            "rebuilds the tree cleanly, and leaves the matrix holding two "
            "tree populations with no error anywhere. This flag does NOT "
            "bypass a MISMATCH against a lockfile that is present."
        ),
    )
    parser.add_argument(
        "--allow-warm-trees",
        action="store_true",
        help=(
            "Permit an M4 cell to serve a WARM substrate. Off by default: "
            "a warm tree may have been built under a different topology "
            "stack and nothing in the output says which, so P10 requires "
            "every M4 cell to build cold. Recorded in the summary, so a "
            "cell can never claim a cold build it did not do."
        ),
    )
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=Path("requirements.lock"),
        help="Environment lock checked before anything loads.",
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
        help="Benchmark id. hotpotqa = standard distractor, one corpus per "
        "question. M4 there is a REAL RAPTOR result with a small flat "
        "tail: most units build a 2-layer hierarchy, while a minority "
        "falls at or below RAPTOR's own stop condition "
        "(<= reduction_dimension + 1 = 11 leaves) and is scored on flat "
        "dense retrieval. THE SIZE OF THAT TAIL IS PENDING MEASUREMENT: "
        "the earlier 36/1000 (3.6%) figure predates the single-item-rule "
        "corpus layout and a 2026-08-22 re-derivation estimates 83/1000 "
        "(8.3%); BOTH ARE SUPERSEDED and analyse over the banked cell is "
        "authoritative (see m4_raptor.py deviation 4). analyse reports "
        "the degenerate count per cell. "
        "hotpotqa_pooled = shards of 100 questions (a real tree, but NOT "
        "comparable to published HotpotQA).",
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
        "--only-unit",
        type=str,
        default=None,
        help=(
            "Run ONLY units whose corpus_id starts with this prefix, "
            "drawn from within the benchmark's normal population. Use "
            "this instead of --max-units to target a specific unit: on "
            "NarrativeQA the seeded draw depends on N, so --max-units 1 "
            "selects a DIFFERENT story than the first of the 40-story "
            "cell, and would build a tree the cell never touches."
        ),
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
        default=MATRIX_BATCH_SIZE,
        help=(
            f"Generation batch size. DEFAULTS TO config.MATRIX_BATCH_SIZE "
            f"({MATRIX_BATCH_SIZE}) so every matrix cell batches "
            "identically WITHOUT the operator remembering a flag: batch "
            "composition can move generated text at temperature 0, so a "
            "cell that silently fell back to sequential answering would "
            "not be strictly comparable to the other nineteen. Pass a "
            "smaller value to override for a cost probe; pass 0 for the "
            "historic sequential path. "
            "A NONZERO value enables TWO-PHASE answering: retrieve every "
            "query in a unit first, then generate the unit in batches. "
            "MEASURED SLOWER THAN SEQUENTIAL ON THE ANSWER PATH "
            "(M2 x MultiHop, 64 queries, L4): sequential 4.2558 s/query "
            "against 5.1654 at the best batched cap, degrading to 6.2492 "
            "at padded 20000. A batch runs until its LONGEST member "
            "stops, so the 512-token answer cap makes one slow "
            "generation charge the whole batch; sequential lets each "
            "query stop at its own EOS. The batching win is real only "
            "where the cap is tight, i.e. M4's 100-token tree summaries, "
            "and that path uses M4Config.summary_batch_size instead. "
            "NOTE: batch composition can change generated text even at "
            "temperature 0, so keep this FIXED across the cells you "
            "intend to compare."
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

    # FIRST, before any model loads or any dataset is touched. A gate that
    # fires after a 15 GB load has already cost the thing it protects.
    assert_environment_pinned(
        args.lockfile, allow_unpinned=args.allow_unpinned
    )

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

    # 0 is the explicit opt-out: argparse cannot express "None by
    # request" once the default is an int, and a sequential path that
    # can only be reached by editing config is not an escape hatch.
    batch_size = args.batch_size if args.batch_size else None
    if batch_size is None:
        print("[eval] batch_size=0 -> SEQUENTIAL answering (explicitly "
              "requested; matrix cells must use the default)")
    runner = BenchmarkRunner(
        output_path=args.output,
        verbose=not args.quiet,
        batch_size=batch_size,
        resume=args.resume,
        max_padded_tokens=args.max_padded_tokens,
        verify_max_new_tokens=args.max_new_tokens,
        # Tree systems only: M1/M2/M3/M9 build no tree, so the rule has
        # nothing to say about them and a blanket gate would fire on a
        # legitimate embedding-substrate cache hit.
        require_cold_tree=(args.system == "M4" and not args.allow_warm_trees),
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
        only_unit=args.only_unit,
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

    # POPULATION CHECK — after the pass, because the resolved unit count
    # is only known once the units have been drawn, and BEFORE the
    # summary is written, so a cell built on the wrong population does
    # not acquire a provenance block that makes it look finished.
    assert_population_as_declared(
        benchmark,
        n_units_processed=getattr(runner, "n_units_processed", 0),
        explicit_reason=(
            f"--max-units {args.max_units} given explicitly"
            if args.max_units is not None
            else f"--only-unit {args.only_unit!r} given explicitly"
            if args.only_unit
            else None
        ),
    )

    # Resolved AFTER the pass, because the loader fills its stats as it
    # yields. Checked before the summary is written so an uncapped cell
    # cannot be banked with P8's short-cell guard silently disarmed.
    # A PARTIAL RUN'S LOADER STATS ARE NOT A CELL COUNT. `iter_eval_units`
    # fills stats as it YIELDS, and a capped or filtered pass stops
    # consuming the generator early — so an --only-unit run on story 12
    # of 40 left n_stories=12 and n_queries=372. Recording that as
    # expected_n_queries would hand P8's guard a number a third the size
    # of the cell, and the guard would then certify a third of the data
    # as complete.
    _partial_run = bool(
        args.max_units is not None
        or args.max_queries is not None
        or args.only_unit
    )
    _expected_n_queries = (
        None if _partial_run else resolve_expected_n_queries(benchmark)
    )
    assert_expected_n_queries_usable(
        _expected_n_queries,
        max_units=args.max_units,
        max_queries=args.max_queries,
        only_unit=args.only_unit,
    )

    # Aggregate summary alongside the JSONL.
    summary_path = args.output.with_suffix(".summary.json")
    n_retr_scored = max(1, n_scored - sum_retr_skipped)
    summary = {
        "system": args.system,
        "benchmark": args.benchmark,
        "split": args.split,
        "max_units": args.max_units,
        "only_unit": args.only_unit,
        "allow_warm_trees": args.allow_warm_trees,
        "n_queries_scored": n_scored,
        "n_retrieval_skipped": sum_retr_skipped,
        # THE DENOMINATOR, NAMED. mean_retrieval_f1 is over the rows that
        # HAD retrieval ground truth, not over every row: MultiHop's 301
        # null queries are skipped, so the mean is over 2,255 of 2,556. A
        # reader given only the mean assumes the cell's row count, and a
        # mean without its n is the same defect class as a guard without
        # its comparison. NarrativeQA makes this stark - every row there
        # is skipped, so the denominator is zero and the column is n/a.
        "n_retrieval_scored": n_scored - sum_retr_skipped,
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
        "expected_n_queries": _expected_n_queries,
        # Says WHY the field is null, so a reader never has to guess
        # whether a partial run is a short cell.
        "expected_n_queries_scope": (
            "PARTIAL RUN - loader stats describe only the units consumed, "
            "not the cell" if _partial_run else "full cell"
        ),
        "partial_run": _partial_run,
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
        "chunking_strategy": resolve_chunking_strategy(system),
        # The CK-4 ABLATION FLAG, not the effective budget. M4 carries a
        # 2,000-token paper budget through its own config and shows null
        # here; both are recorded so neither is read as the other.
        "evidence_budget": args.evidence_budget,
        "evidence_budget_effective": getattr(
            system.config.m4, "retrieval_budget_tokens", None
        ) if args.system == "M4" else None,
        # Recorded so a probe artifact is self-describing. The 1-token
        # probe that silently ran uncapped would have been caught here.
        # ANSWER-path cap. The index-time summariser uses its own,
        # recorded beside it so a reader is not left inferring which.
        "max_new_tokens": harness_cfg.generation.max_new_tokens,
        "summary_max_new_tokens": getattr(
            system.config.m4, "summary_max_tokens", None
        ) if args.system == "M4" else None,
        "prewarm_load_s": load_s,
        # Wall clock of runner.run() ALONE — model load excluded when
        # --prewarm is used. s_per_query is the number every cost
        # forecast in this project is built from, so it is recorded
        # rather than re-derived by hand each time.
        "elapsed_s": round(elapsed_s, 2),
        "s_per_query": (
            round(elapsed_s / n_scored, 4) if n_scored else None
        ),
        "batch_size": batch_size,
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
