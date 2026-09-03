"""CLI runner: one system x one benchmark x one split, JSONL output.

Usage:
    python -m src.eval.runner --system M2 --benchmark multihop_rag --split validation \
        --output /content/drive/MyDrive/thesis_rag/outputs/p10/multihop_rag_M2_validation.jsonl --resume

Sharding across systems / benchmarks is done by running this script
once per combination from a wrapper (shell script or Colab notebook
cell). One process, one system, one benchmark — simple to bisect, simple
to retry, no shared state. Each invocation writes a single JSONL file
plus a `.summary.json` beside it; scripts/export_comparison.py reads
both.

There is no small-sample mode: every cell is the full declared
population. The caps and escapes that once existed were pruned in the
repo reduction (the full tree lives at tag thesis-full-2026-09-03), so
the only way to run a cell is to run all of it under every gate.
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


def assert_bank_generator_consistent(output_dir: Path, generator: str) -> None:
    """Refuse to write a cell into a bank another generator owns.

    ADDENDUM 2's Llama column is a SEPARATE bank (p11), and the plan
    documents noted that output-dir separation was enforced by
    convention only — a Llama cell written into p10 would sit beside
    Qwen cells with nothing but a filename convention keeping analysis
    from pooling them. This gate is DATA-DRIVEN rather than a name
    registry: it reads the generator every existing summary in the
    output directory records, and refuses on any mismatch with the
    resolved generator — which also guards the reverse mistake (a Qwen
    cell into p11) and any future third column, with no code change.

    Summaries predating model_revisions, or unreadable ones, are skipped
    with a warning rather than trusted: an unreadable file cannot vouch
    for the bank, but it cannot convict it either — the P10 bank is
    complete and every one of its summaries records the generator.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return
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
    """Refuse to add a cell to a bank on DIFFERENT hardware.

    Instance 16 of the recurring lesson, caught by an incident rather
    than a review: Colab silently assigned a Tesla T4 (14.56 GiB) to a
    P11 canary, Llama-8B fp16 OOM'd at weight load (266/291 shards) —
    and the pin gate had printed `gpu=Tesla T4` and said OK, because the
    GPU string has been RECORDED per-cell since P9 and compared by
    NOTHING. Had the model fit, the cell would have banked a hardware
    confound: fp16 numerics differ across GPU architectures, and every
    P10 cell ran on NVIDIA L4.

    Data-driven like the bank-generator gate: the first cell into an
    empty bank SETS the hardware; every later cell must match the GPU
    strings the bank's summaries record. There is no escape: a cell on
    different hardware belongs in a different bank directory. A current
    GPU of "unknown" (no
    CUDA visible) WARNS rather than fails: that is an absence of
    measurement, not a measured change, and cells cannot run without a
    GPU anyway.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return
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
        return  # empty bank, or summaries predate the field: this cell sets it
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
    """Fail LOUDLY before GPU time if the hub will not serve the model's
    FILES — probed by downloading a gated file, never by repo metadata.

    Earned twice. First: without a granted license and a valid HF_TOKEN
    the failure arrives mid-session as a 401 inside from_pretrained,
    after the setup blocks are paid for. Second, the sharper one: the
    original probe called `model_info`, which SUCCEEDS on gated repos —
    the metadata is public, only the files are gated — so it printed
    "verified" over a 403-bound run (the operator fetched the revision
    sha days BEFORE the license was granted, through this very gap). The
    probe is now `hf_hub_download` of `config.json`: a file the license
    actually gates. A cache-satisfied download counts as verified, which
    is correct — cached files are how an offline run works.

    Local-path and API-routed models are skipped (nothing to gate).
    Errors that mean "no access" — gated/401/403/not-found — FAIL with
    the fix steps; errors that mean "hub unreachable" WARN and continue,
    because a locally cached model is a legitimate way to run and a
    preflight must not add a network single-point-of-failure to an
    offline-capable path.
    """
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
    try:
        # A GATED FILE, never repo metadata: model_info succeeds on gated
        # repos (metadata is public, files are not) and once printed
        # "verified" over a 403-bound run. config.json is tiny and sits
        # behind the same license as the weights.
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

    There is no escape: a missing lockfile aborts and a present-but-
    violated one aborts. Copy the banked `requirements.lock` (Drive root)
    beside the checkout, or pass --lockfile, before running anything.
    """
    from scripts.pin_environment import check_lockfile

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


def assert_expected_n_queries_usable(expected: int | None) -> None:
    """A cell must carry a count to be checked against.

    P8's guard exists so a TRUNCATED cell aborts instead of reporting a
    partial mean. A null `expected_n_queries` removes that guard without
    removing the appearance of it, which is worse than a short cell —
    nothing downstream can tell the difference between "complete" and
    "unchecked". Every run is a full cell (there are no caps), so the
    check always applies.
    """
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
) -> None:
    """Abort if the cell resolved to a different population than declared.

    THE BACKSTOP to the loader default. NarrativeQA's seeded 40-story
    draw used to materialise only when `--max-units` was typed, so a
    forgotten flag produced a 115-story, 3,461-question cell that ran to
    completion and looked entirely normal. The loader now carries that
    property itself; this checks the OUTCOME, so a future loader change,
    a dataset that grew, or a benchmark whose draw stops matching its
    declaration cannot pass silently.

    A benchmark with no declared `cell_units` is skipped WITH A PRINTED
    NOTE rather than in silence — an undeclared population is exactly
    the condition that hid this defect.
    """
    declared = getattr(benchmark, "cell_units", None)
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
            "data than the rest of the matrix."
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
        help="Benchmark id. hotpotqa = standard distractor, one corpus per "
        "question. M4 there is a REAL RAPTOR result with a small flat "
        "tail, MEASURED from the banked cell (2026-08-22): "
        "917/1000 (91.7%%) build a 2-layer hierarchy, 83/1000 (8.3%%) fall "
        "at or below RAPTOR's own stop condition "
        "(<= reduction_dimension + 1 = 11 leaves) and are scored on flat "
        "dense retrieval. Leaves: 17,443 total, median 17, max 37. The "
        "old 36/1000 (3.6%%) figure is DEAD - it predates the "
        "single-item-rule corpus layout. "
        "hotpotqa_pooled = shards of 100 questions (a real tree, but NOT "
        "comparable to published HotpotQA).",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Benchmark-specific split; every matrix cell runs 'validation'. "
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
        "--generator",
        type=str,
        default=None,
        help=(
            "Run this cell under a DIFFERENT model. Sets the READER "
            "(HarnessConfig.generation.model) AND the INDEX-TIME "
            "SUMMARISER (M4 summary_model) together, because the "
            "matrix design is FULL INDEPENDENT REPLICATION: each column "
            "builds its own trees with its own summariser and reads them "
            "with the same model. "
            "Rebinding src.config.GENERATOR_MODEL or JUDGE_MODEL "
            "in-process does NOT work -- both are dataclass field "
            "defaults evaluated once at class-definition time. "
            "VERIFIED: changing this moves M4's substrate cache "
            "key, so a Llama cell cannot silently hit a Qwen tree. "
            "M2/M3 keys do NOT move, which is CORRECT -- their "
            "substrate contains no LLM output, so it is a "
            "model-independent artifact and rebuilding it would produce "
            "byte-identical files."
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
    args = parser.parse_args()

    # FIRST, before any model loads or any dataset is touched. A gate that
    # fires after a 15 GB load has already cost the thing it protects.
    assert_environment_pinned(args.lockfile)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if args.output is None:
        out_root = paths.output_dir() / "eval"
        out_root.mkdir(parents=True, exist_ok=True)
        args.output = out_root / (
            f"{args.benchmark}_{args.system}_{args.split}_{stamp}.jsonl"
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
    # Build the config EXPLICITLY rather than rebinding module constants:
    # every constant in src/config.py is baked into a dataclass field
    # default at import, so a rebind is silently ignored.
    harness_cfg = HarnessConfig()
    if args.generator is not None:
        # READER AND SUMMARISER TOGETHER. The matrix is a full
        # independent replication: each column builds its own trees with
        # its own summariser and reads them with the same model. Setting
        # only the reader would produce a column whose trees came from
        # the other model, which is the confound this design exists to
        # avoid on M1/M2/M3 and cannot avoid on M4.
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
    # TWO REPLICATION GATES, both before any model loads (ADDENDUM 2
    # activation, 2026-08-24). Order matters: the bank check is pure disk
    # and runs first; the hub check may touch the network.
    assert_bank_generator_consistent(
        Path(args.output).parent, harness_cfg.generation.model
    )
    assert_bank_gpu_consistent(Path(args.output).parent)
    assert_generator_accessible(harness_cfg.generation.model)

    system_cls = SYSTEM_REGISTRY[args.system]
    system: BaseSystem = system_cls(config=harness_cfg)

    # BENCHMARK FIRST, AND PREFLIGHT BEFORE ANY MODEL LOADS. A HotpotQA
    # run once died on an unresolvable dataset id at the first
    # iter_eval_units call -- after a prewarm had already pulled 15 GB of
    # Qwen into VRAM. Cheap preconditions get checked before expensive
    # ones are paid.
    # `preflight` is optional: a benchmark without one is simply not
    # checked, and any benchmark can add the same two-second guard.
    benchmark_cls = BENCHMARK_REGISTRY[args.benchmark]
    benchmark = benchmark_cls()
    preflight = getattr(benchmark, "preflight", None)
    if callable(preflight):
        preflight()

    runner = BenchmarkRunner(
        output_path=args.output,
        resume=args.resume,
        # Tree systems only: M1/M2/M3 build no tree, so the rule has
        # nothing to say about them and a blanket gate would fire on a
        # legitimate embedding-substrate cache hit. M4 has no escape: a
        # warm tree refuses the cell.
        require_cold_tree=(args.system == "M4"),
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
    # Timed HERE, around runner.run only, so every import sits outside
    # it (the generator's own load happens inside the first answer and
    # is part of the run, as on every banked cell). Recorded in the
    # summary because otherwise
    # the only source of a timing is stdout, and reading a wall clock off
    # a Colab cell is how the first 1-token probe ended up quoting model
    # downloads as compute.
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

    # POPULATION CHECK — after the pass, because the resolved unit count
    # is only known once the units have been drawn, and BEFORE the
    # summary is written, so a cell built on the wrong population does
    # not acquire a provenance block that makes it look finished.
    assert_population_as_declared(
        benchmark,
        n_units_processed=getattr(runner, "n_units_processed", 0),
    )

    # Resolved AFTER the pass, because the loader fills its stats as it
    # yields. Checked before the summary is written so a cell cannot be
    # banked with P8's short-cell guard silently disarmed.
    _expected_n_queries = resolve_expected_n_queries(benchmark)
    assert_expected_n_queries_usable(_expected_n_queries)

    # Aggregate summary alongside the JSONL.
    summary_path = args.output.with_suffix(".summary.json")
    n_retr_scored = max(1, n_scored - sum_retr_skipped)
    summary = {
        "system": args.system,
        "benchmark": args.benchmark,
        "split": args.split,
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
        # Constant since the reduction removed every cap: a run is always
        # the full cell. Still written because the bank gates (the
        # exporter's read_cell, the replay) refuse a truthy value.
        "partial_run": False,
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
        # The harness imposes no evidence budget (locked decision 4);
        # M4 carries its own 2,000-token paper budget, recorded here.
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
        # Wall clock of runner.run() ALONE (the generator load happens
        # inside the first answer and is part of it). s_per_query is the
        # number every cost forecast in this project is built from, so
        # it is recorded rather than re-derived by hand each time.
        "elapsed_s": round(elapsed_s, 2),
        "s_per_query": (
            round(elapsed_s / n_scored, 4) if n_scored else None
        ),
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
