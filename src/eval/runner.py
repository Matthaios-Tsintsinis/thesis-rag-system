"""CLI runner: one system x one benchmark x one split, JSONL output.

Usage:
    python -m src.eval.runner --system M2 --benchmark multihop_rag --split validation \
        --output local_runs/eval/multihop_rag_M2_validation.jsonl --max-units 20

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
    allow_mismatch: bool = False,
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
    strings the bank's summaries record. `--allow-gpu-mismatch` is the
    deliberate escape — loud, and recorded in the cell summary — for a
    ruled exception, never a default. A current GPU of "unknown" (no
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
    if allow_mismatch:
        print(
            f"[eval] *** GPU MISMATCH ALLOWED BY FLAG: running on "
            f"{current_gpu!r} against a bank built on {sorted(bank_gpus)}. "
            "This cell is a HARDWARE CONFOUND relative to its bank and the "
            "summary records the override. ***"
        )
        return
    raise SystemExit(
        f"PREFLIGHT FAILED: this host's GPU is {current_gpu!r} but the "
        f"bank at {out} was built on {sorted(bank_gpus)}.\n"
        "  fp16 numerics differ across GPU architectures — a cell run on "
        "different hardware is a confound inside its own column (the P11 "
        "canary drew a Tesla T4 this way and OOM'd; had it fit, it would "
        "have banked silently).\n"
        "  Fix: set the runtime accelerator to the bank's GPU and re-run. "
        "A DELIBERATE hardware change is --allow-gpu-mismatch, which is "
        "loud and recorded in the summary."
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
        "--allow-gpu-mismatch",
        action="store_true",
        help="run on a GPU other than the one this bank's summaries "
             "record. LOUD and recorded in the summary - fp16 numerics "
             "differ across architectures, so this cell becomes a "
             "declared hardware confound within its bank. Exists because "
             "Colab silently swapped an L4 for a T4 once and only the "
             "OOM caught it.",
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
        "917/1000 (91.7%) build a 2-layer hierarchy, 83/1000 (8.3%) fall "
        "at or below RAPTOR's own stop condition "
        "(<= reduction_dimension + 1 = 11 leaves) and are scored on flat "
        "dense retrieval. Leaves: 17,443 total, median 17, max 37. The "
        "old 36/1000 (3.6%) figure is DEAD - it predates the "
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

    # TWO REPLICATION GATES, both before any model loads (ADDENDUM 2
    # activation, 2026-08-24). Order matters: the bank check is pure disk
    # and runs first; the hub check may touch the network.
    assert_bank_generator_consistent(
        Path(args.output).parent, harness_cfg.generation.model
    )
    assert_bank_gpu_consistent(
        Path(args.output).parent, allow_mismatch=args.allow_gpu_mismatch
    )
    assert_generator_accessible(harness_cfg.generation.model)

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
        # Tree systems only: M1/M2/M3 build no tree, so the rule has
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
        **({"gpu_mismatch_allowed": True} if args.allow_gpu_mismatch else {}),
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
