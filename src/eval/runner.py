"""CLI runner: one system x one benchmark x one split, JSONL output.

One process per cell. Writes the JSONL plus a .summary.json beside it.
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
from .base import BenchmarkRunner
from .hotpotqa import HotpotQABenchmark, HotpotQAPooledBenchmark
from .multihop import MultiHopBenchmark
from .narrativeqa import NarrativeQABenchmark


SYSTEM_REGISTRY: dict[str, type[BaseSystem]] = {
    "M1": ClosedBookSystem,
    "M2": FlatDenseSystem,
    "M3": HybridRRFSystem,
    "M4": RaptorSystem,
}

BENCHMARK_REGISTRY: dict[str, type] = {
    "multihop_rag": MultiHopBenchmark,
    "narrativeqa": NarrativeQABenchmark,
    # HotpotQA is two benchmarks with different corpora and unit counts:
    # distractor is the comparable headline, pooled is where a tree exists.
    # harness choice: our construction, not comparable to published HotpotQA (METHODS §B.4)
    "hotpotqa": HotpotQABenchmark,
    "hotpotqa_pooled": HotpotQAPooledBenchmark,
}


def _environment_provenance(lockfile: Path) -> dict:
    """Lockfile hash, GPU model, python and pinned versions for the summary."""
    try:
        from scripts.pin_environment import environment_provenance

        # Read the lockfile the gate checked, so the recorded hash matches.
        return environment_provenance(Path(lockfile))
    except Exception as e:  # provenance never kills a run
        return {"error": f"{type(e).__name__}: {e}"}


def _model_revisions(system) -> dict:
    """Resolved model ids per role plus their HF revision hashes."""
    out: dict = {}
    # Collect the ids the system resolved; the generator comes from config.
    try:
        resolved = getattr(system, "resolved_components", None)
        if resolved is not None:
            out["embedder"] = getattr(resolved, "embedder_id", None)
            out["reranker"] = getattr(resolved, "reranker_id", None)
            out["index_llm"] = getattr(resolved, "index_llm_id", None)
        out["generator"] = system.config.generation.model
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    # Look up each hub id's revision; a missing one is None, never fatal.
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
    """The topology stack string, or None when raptor_paper cannot import."""
    try:
        from ..raptor_paper import PAPER_TREE_BUILD_ENV

        return PAPER_TREE_BUILD_ENV
    except Exception:
        return None


def _git_commit_short() -> str:
    """Short HEAD hash for the summary; "unknown" without git."""
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


def assert_bank_generator_consistent(output_dir: Path, generator: str) -> None:
    """Refuse a cell whose generator differs from the bank's summaries."""
    out = Path(output_dir)
    if not out.is_dir():
        return
    # Compare the resolved generator with every readable summary in the
    # bank; a summary that names no generator cannot vouch for it.
    mismatched: list[str] = []
    for sp in sorted(out.glob("*.summary.json")):
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[eval] WARN: unreadable summary {sp.name} during the "
                  f"bank-generator check: {e}")
            continue
        g = (summary.get("model_revisions") or {}).get("generator")
        if g is None:
            print(f"[eval] WARN: {sp.name} records no generator; it cannot "
                  "vouch for this bank")
            continue
        if str(g) != generator:
            mismatched.append(f"{sp.name}: {g}")
    # Any mismatch aborts: each generator column is its own bank directory.
    if mismatched:
        listing = "\n  ".join(mismatched[:8])
        more = "" if len(mismatched) <= 8 else f"\n  ... {len(mismatched) - 8} more"
        raise SystemExit(
            f"PREFLIGHT FAILED: the output directory {out} already holds "
            f"cells from a DIFFERENT generator than the resolved "
            f"{generator!r}:\n  {listing}{more}\n"
            "  Per ADDENDUM 2 the columns are separate banks — write the "
            "Llama column into its own directory (outputs/p11), never "
            "into p10, and vice versa."
        )


def assert_bank_gpu_consistent(
    output_dir: Path,
    *,
    current_gpu: str | None = None,
) -> None:
    """Refuse a cell whose GPU differs from the bank's recorded hardware."""
    out = Path(output_dir)
    if not out.is_dir():
        return
    # Resolve this host's GPU. No visible CUDA device is an absence of
    # measurement, not a measured change, so it warns instead of failing.
    if current_gpu is None:
        try:
            from scripts.pin_environment import gpu_model

            current_gpu = gpu_model()
        except Exception as e:
            print(f"[eval] WARN: could not resolve the GPU for the bank "
                  f"check ({type(e).__name__}); hardware NOT verified")
            return
    if not current_gpu or current_gpu == "unknown":
        print("[eval] WARN: no CUDA device visible; the bank's hardware "
              "consistency was NOT verified")
        return

    # Collect the GPU strings the bank's summaries record.
    bank_gpus: set[str] = set()
    for sp in sorted(out.glob("*.summary.json")):
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[eval] WARN: unreadable summary {sp.name} during the "
                  f"bank-GPU check: {e}")
            continue
        g = (summary.get("environment") or {}).get("gpu")
        if g:
            bank_gpus.add(str(g))
    if not bank_gpus:
        return  # empty bank: this cell sets the hardware
    # fp16 numerics differ across GPU architectures, so the first cell into
    # a bank sets its hardware and every later cell must match it.
    if len(bank_gpus) > 1:
        print(f"[eval] WARN: the bank already holds MULTIPLE GPU strings "
              f"{sorted(bank_gpus)} — inspect before trusting any "
              "cross-cell comparison")
    if current_gpu in bank_gpus:
        print(f"[eval] PREFLIGHT: GPU {current_gpu!r} matches the bank")
        return
    raise SystemExit(
        f"PREFLIGHT FAILED: this host's GPU is {current_gpu!r} but the "
        f"bank at {out} was built on {sorted(bank_gpus)}.\n"
        "  fp16 numerics differ across GPU architectures — a cell run on "
        "different hardware is a confound inside its own column (the P11 "
        "canary drew a Tesla T4 this way and OOM'd; had it fit, it would "
        "have banked silently).\n"
        "  Fix: set the runtime accelerator to the bank's GPU and re-run. "
        "A deliberate hardware change belongs in a DIFFERENT bank "
        "directory, never beside cells built on other hardware."
    )


def assert_generator_accessible(model_id: str) -> None:
    """Fail before GPU time if the hub will not serve the model's files."""
    # Local paths and API-routed models have nothing to gate.
    if "/" not in str(model_id):
        return
    low = str(model_id).lower()
    if low.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except Exception as e:  # hub not importable: nothing to check with
        print(f"[eval] WARN: huggingface_hub unavailable for the access "
              f"preflight ({type(e).__name__}); model access NOT verified")
        return
    # Download config.json, a file the license gates; repo metadata is
    # public even on gated repos. A cached copy counts as access. No-access
    # errors abort with the steps to gain access; an unreachable hub only
    # warns, because a locally cached model still runs.
    try:
        hf_hub_download(str(model_id), "config.json")
        print(f"[eval] PREFLIGHT: hub FILE access to {model_id} verified "
              "(config.json served or cached)")
    except EntryNotFoundError:
        print(f"[eval] WARN: {model_id} has no config.json to probe; "
              "FILE access NOT verified")
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise SystemExit(
            f"PREFLIGHT FAILED: no access to {model_id!r} "
            f"({type(e).__name__}).\n"
            "  For a gated repo (meta-llama/*):\n"
            "  1. open the model page with the SAME account as the token "
            "and accept the license;\n"
            "  2. create a READ token at huggingface.co/settings/tokens;\n"
            "  3. export HF_TOKEN=<token> in the session BEFORE this "
            "runner starts (Colab: store it in Secrets and export in "
            "Block F2).\n"
            "  Verified access prints a PREFLIGHT line; nothing loads "
            "until it does."
        )
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            raise SystemExit(
                f"PREFLIGHT FAILED: HTTP {code} for {model_id!r} — the "
                "token is missing, expired, or the license is not "
                "accepted. See the gated-repo steps above (export "
                "HF_TOKEN before the runner starts)."
            )
        print(f"[eval] WARN: hub returned HTTP {code} for {model_id}; "
              "access not verified (a locally cached model still works)")
    except Exception as e:  # network down, DNS, proxy
        print(f"[eval] WARN: hub unreachable ({type(e).__name__}: {e}); "
              f"access to {model_id} not verified — a locally cached "
              "model still works")


def assert_environment_pinned(lockfile: Path) -> None:
    """Abort before any model loads unless the environment matches the lock."""
    from scripts.pin_environment import check_lockfile

    # The M4 substrate key folds umap-learn/scikit-learn/numpy, so an
    # unpinned session rebuilds trees under a different key and nothing
    # afterwards can tell them apart. A missing lockfile aborts and a
    # violated one aborts; no flag bypasses either.
    if not Path(lockfile).exists():
        raise SystemExit(
            f"PREFLIGHT FAILED: no lockfile at {lockfile}.\n"
            "  The M4 substrate key folds umap-learn/scikit-learn/numpy, "
            "so an unpinned session can build trees under a different key "
            "than the rest of the matrix. That rebuild SUCCEEDS and is "
            "invisible afterwards.\n"
            "  Copy the banked requirements.lock (Drive root) beside the "
            "checkout, or pass --lockfile <path>; then verify with\n"
            "  python -m scripts.pin_environment check --lockfile <path>"
        )

    if check_lockfile(Path(lockfile)) != 0:
        raise SystemExit(
            "PREFLIGHT FAILED: this environment does not match "
            f"{lockfile} (mismatches printed above).\n"
            "  Reinstall from the lock:  pip install -r "
            f"{lockfile}\n"
            "  A lockfile that is present and violated is a pin the "
            "operator asserted and the environment broke; no flag "
            "bypasses it."
        )
    print(f"[eval] PREFLIGHT: environment matches {lockfile}")


def resolve_expected_n_queries(benchmark) -> int | None:  # noqa: ANN001
    """The loader-derived query count the row-count check uses."""
    # One agreed key across loaders; None rather than a guessed synonym.
    stats = getattr(benchmark, "stats", {}) or {}
    value = stats.get("n_queries")
    return int(value) if value else None


def assert_expected_n_queries_usable(expected: int | None) -> None:
    """Abort when a cell carries no expected query count to check against."""
    # A null count would disarm the short-cell guard while looking checked.
    if expected is not None:
        return
    raise SystemExit(
        "PROVENANCE FAILED: expected_n_queries is null on a full "
        "run. P8 asserts each cell's row count against this number, so a "
        "null silently disables the short-cell guard — nothing downstream "
        "can then tell a complete cell from an unchecked one. The loader "
        "must record `n_queries` in its stats."
    )


def resolve_chunking_strategy(system) -> str | None:  # noqa: ANN001
    """The chunker the system resolved, else the harness default."""
    # M4 resolves its own chunker through resolved_components; the other
    # systems use the harness-wide default.
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
) -> None:
    """Abort if the processed unit count differs from declared cell_units."""
    # An undeclared population is skipped with a printed note, not silently.
    # NarrativeQA declares 40 stories.
    # harness choice: preregistered seeded draw of 40 (METHODS §B.2)
    declared = getattr(benchmark, "cell_units", None)
    if declared is None:
        print(
            f"[eval] population: {n_units_processed} units; "
            f"{benchmark.name} declares no cell_units, so nothing was "
            "checked. An undeclared population is how the NarrativeQA "
            "115-vs-40 defect hid."
        )
        return
    # A cell on the wrong population runs to completion and looks normal.
    if n_units_processed != declared:
        raise SystemExit(
            f"POPULATION MISMATCH: {benchmark.name} processed "
            f"{n_units_processed} units but declares cell_units="
            f"{declared}.\n"
            "  A cell built on the wrong population runs to completion "
            "and looks normal — its rows are simply about different "
            "data than the rest of the matrix."
        )
    print(
        f"[eval] population OK: {n_units_processed} units "
        f"= declared cell_units"
    )


def main() -> None:
    """Parse the CLI, run every gate, score one cell, write its summary."""
    parser = argparse.ArgumentParser(
        description="Run one system x one benchmark x one split to JSONL."
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
        help="Retrieval system id (M1/M2/M3/M4).",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARK_REGISTRY),
        help="Benchmark id; hotpotqa is the distractor setting, "
        "hotpotqa_pooled the shard-pooled construction.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Dataset split; every matrix cell runs 'validation'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (a bank cell is "
        "<bank>/<benchmark>_<system>_validation.jsonl).",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help="Reader model id; also M4's summariser (moves M4's cache key, "
        "not M2/M3's).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output JSONL, skipping questions "
        "already in it.",
    )
    args = parser.parse_args()

    # Gate the environment first, before any model or dataset loads.
    assert_environment_pinned(args.lockfile)

    # Resolve the output path. A relative path lands in the current
    # directory, which a runtime restart deletes; --resume would then find
    # nothing there and truncate, so warn loudly.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output is None:
        out_root = paths.output_dir() / "eval"
        out_root.mkdir(parents=True, exist_ok=True)
        args.output = out_root / (
            f"{args.benchmark}_{args.system}_{args.split}_{stamp}.jsonl"
        )
    elif not args.output.is_absolute():
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
    # Build the config explicitly: module constants are baked into
    # dataclass defaults at import, so rebinding them does nothing.
    harness_cfg = HarnessConfig()
    if args.generator is not None:
        # --generator sets reader and index summariser together: each
        # column builds its own trees and reads them with the same model.
        harness_cfg = replace(
            harness_cfg,
            generation=replace(harness_cfg.generation, model=args.generator),
            m4=replace(harness_cfg.m4, summary_model=args.generator),
        )
        print(
            f"[eval] GENERATOR OVERRIDE: {args.generator}\n"
            f"    reader           = {harness_cfg.generation.model}\n"
            f"    index summariser = {harness_cfg.m4.summary_model}\n"
            "    M4's substrate cache key MOVES with this, so this cell "
            "builds its own trees and cannot hit the other column's.\n"
            "    M2/M3 keys do NOT move — their substrate has no LLM "
            "in it, so a cache hit there reuses a model-INDEPENDENT "
            "artifact, not the other column's work."
        )
    # Bank gates before the hub check: disk first, then the network.
    assert_bank_generator_consistent(
        Path(args.output).parent, harness_cfg.generation.model
    )
    assert_bank_gpu_consistent(Path(args.output).parent)
    assert_generator_accessible(harness_cfg.generation.model)

    system_cls = SYSTEM_REGISTRY[args.system]
    system: BaseSystem = system_cls(config=harness_cfg)

    # Build the benchmark and run its cheap preflight, when it has one,
    # before any model loads.
    benchmark_cls = BENCHMARK_REGISTRY[args.benchmark]
    benchmark = benchmark_cls()
    preflight = getattr(benchmark, "preflight", None)
    if callable(preflight):
        preflight()

    # Only M4 builds a tree, so only M4 must build it cold.
    runner = BenchmarkRunner(
        output_path=args.output,
        resume=args.resume,
        require_cold_tree=(args.system == "M4"),
    )
    # Running sums for the summary. The answer column mixes token-F1 on
    # answerable queries with the null rule on null ones, so the null part
    # is counted separately and reported beside the mean.
    n_scored = 0
    sum_retr_f1 = 0.0
    sum_retr_skipped = 0
    sum_ans = 0.0
    n_null = 0
    sum_ans_null = 0.0
    # Time the pass alone; the generator loads inside the first answer and
    # is part of it.
    t_run = time.perf_counter()
    for scored in runner.run(system, benchmark, split=args.split):
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

    # Check the population after the pass, when the unit count is known,
    # and before the summary is written.
    assert_population_as_declared(
        benchmark,
        n_units_processed=getattr(runner, "n_units_processed", 0),
    )

    # The loader fills its stats while yielding, so read them after the
    # pass and check them before the summary is written.
    _expected_n_queries = resolve_expected_n_queries(benchmark)
    assert_expected_n_queries_usable(_expected_n_queries)

    # Write the summary beside the JSONL.
    summary_path = args.output.with_suffix(".summary.json")
    n_retr_scored = max(1, n_scored - sum_retr_skipped)
    summary = {
        "system": args.system,
        "benchmark": args.benchmark,
        "split": args.split,
        "n_queries_scored": n_scored,
        "n_retrieval_skipped": sum_retr_skipped,
        # mean_retrieval_f1 averages only rows with retrieval gold: MultiHop
        # skips its 301 null queries and NarrativeQA skips every row.
        # dataset: yixuantt/MultiHopRAG (609 articles, 2,556 queries, 301 null)
        "n_retrieval_scored": n_scored - sum_retr_skipped,
        "mean_retrieval_f1": sum_retr_f1 / n_retr_scored,
        "mean_answer_score": sum_ans / max(1, n_scored),
        # The two parts of the one answer column.
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
        # Loader-derived, never a literal.
        "expected_n_queries": _expected_n_queries,
        # Always False: a run is the full cell. The bank gates refuse a
        # truthy value.
        "partial_run": False,
        # Run conditions, so every row is self-describing.
        "generator": system.config.generation.model,
        # The index-time model, recorded apart from the reader.
        "index_llm": system.config.m4.summary_model,
        "chunking_strategy": resolve_chunking_strategy(system),
        # M4 alone carries an evidence budget.
        # harness choice: no shared evidence budget (METHODS §D)
        # RAPTOR paper §3: "2000 maximum tokens ... top-20 nodes" (paper over repo): see METHODS §A.4.3
        "evidence_budget_effective": getattr(
            system.config.m4, "retrieval_budget_tokens", None
        ) if args.system == "M4" else None,
        # Answer-path cap and the index summariser's own cap, side by side.
        # harness choice: one reader across all systems (METHODS §D)
        # ref: raptor/tree_builder.py::TreeBuilderConfig @ 7da1d48a (summarization_length=100); the paper's 131 is a measured mean (App. C)
        "max_new_tokens": harness_cfg.generation.max_new_tokens,
        "summary_max_new_tokens": getattr(
            system.config.m4, "summary_max_tokens", None
        ) if args.system == "M4" else None,
        # Wall clock of runner.run alone, and the per-query rate.
        "elapsed_s": round(elapsed_s, 2),
        "s_per_query": (
            round(elapsed_s / n_scored, 4) if n_scored else None
        ),
        # Provenance per cell: environment, model revisions, tree cache
        # state (None without a tree), topology stack and code revision.
        "environment": _environment_provenance(args.lockfile),
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
