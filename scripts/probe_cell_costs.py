"""Measure the two costs the matrix has never measured, on the pinned stack.

WHAT IS MISSING FROM EVERY FORECAST SO FAR

  1. M4 COLD TREE-BUILD TIME, isolated from query cost. Paid ONCE PER
     CELL and invisible to `s_per_query`, which is wall-clock over
     queries only. It has never been measured under the local summariser,
     and it has never been paid at all on MultiHop or NarrativeQA,
     because those cells ran against trees built in earlier sessions. The
     cold-tree lever (`cf8a7b8`) now forces a rebuild, so this cost is
     new to the matrix rather than merely unmeasured.

     NarrativeQA is the concern: FORTY separate story-trees, each built
     cold. If one story's build is a meaningful fraction of the session
     guard, that cell needs intra-cell tree checkpointing or it is a
     re-run risk — `--resume` protects the remainder of a cell's QUERIES,
     not a tree that was half-built when the session died.

  2. FIVE-SYSTEM `s_per_query` ON MULTIHOP, the largest benchmark at
     2,556 queries. M9 has never run a timed slice under the local
     generator; M4's per-query cost was derived, not measured.

THE PROBE RULES THIS FILE OBEYS (earned five times over in this project)

  * It ASSERTS that it measured what it claims. A tree "build" that was
    served from cache is a cache read, and reporting its 0.4 s as a build
    time would be worse than not measuring at all — so a cache hit
    ABORTS.
  * A VACUOUS PASS IS A FAILURE. A degenerate tree (no summary layer)
    exits non-zero rather than reporting a suspiciously small number.
  * It PREWARMS. The first system measured otherwise pays a ~15 GB model
    load the second does not, which has already inverted the sign of one
    measurement in this project.

USAGE (Colab, pinned stack, GPU attached)

    python -m scripts.probe_cell_costs --mode tree
    python -m scripts.probe_cell_costs --mode queries --n 50
    python -m scripts.probe_cell_costs --mode both --out /content/cell_costs.json

`--mode tree` builds ONE unit per benchmark. Expect it to spend real
money on summaries; that is the point, and it is a rounding error against
the run it de-risks.
"""

from __future__ import annotations

import os

# BEFORE ANY TORCH IMPORT. The allocator config is read once, when torch
# initialises CUDA; setting it later is a no-op that looks like it worked.
# Baked in here rather than left to the shell so it is part of the
# MEASUREMENT — an ambient export that differs between the probe session
# and the real run would make the probe's headroom a fiction.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any


def _fail(msg: str) -> None:
    print(f"[probe] FAILED: {msg}")
    sys.exit(1)


def _prewarm() -> float:
    """Load the generator BEFORE any timing. Returns load seconds."""
    from src.config import DEFAULT_CONFIG
    from src.models import load_generator

    t0 = time.perf_counter()
    load_generator(DEFAULT_CONFIG.generation.model)
    dt = time.perf_counter() - t0
    print(f"[probe] prewarm: generator resident in {dt:.1f}s (EXCLUDED)")
    return dt


# The NarrativeQA cell draws 40 stories; this is how many the probe must
# draw too. Using --max-units 1 runs subsample_indices(115, 1), which is a
# DIFFERENT seeded draw from subsample_indices(115, 40) — it selected story
# index 65 where the matrix's first story is index 1. A probe that measures
# a story the run will not build is measuring the wrong thing.
NARRATIVEQA_CELL_UNITS = 40


def _units_for(benchmark_id: str, max_units: int | None):
    from src.eval.runner import BENCHMARK_REGISTRY

    bench = BENCHMARK_REGISTRY[benchmark_id]()
    units = list(bench.iter_eval_units(split="validation", max_units=max_units))
    if not units:
        _fail(f"{benchmark_id}: loader yielded no unit")
    return bench, units


def _one_unit(benchmark_id: str, story_id: str | None = None):
    """One unit, drawn from THE SAME sample the matrix cell will use.

    `story_id` accepts a prefix. "largest" picks the biggest corpus in the
    cell's own draw, which is the memory worst case and therefore the only
    unit whose success bounds the whole cell.
    """
    max_units = (NARRATIVEQA_CELL_UNITS
                 if benchmark_id == "narrativeqa" else 1)
    bench, units = _units_for(benchmark_id, max_units)

    if story_id in (None, ""):
        return bench, units[0]
    if story_id == "largest":
        unit = max(units, key=lambda u: sum(len(i.text) for i in u.corpus))
        print(f"[probe] largest unit in the {len(units)}-unit draw: "
              f"{unit.corpus_id} "
              f"({sum(len(i.text) for i in unit.corpus):,} chars)")
        return bench, unit
    matches = [u for u in units if str(u.corpus_id).startswith(story_id)]
    if not matches:
        _fail(f"{benchmark_id}: no unit in the {len(units)}-unit draw whose "
              f"corpus_id starts with {story_id!r}. The probe must target a "
              "unit the CELL will actually build.")
    if len(matches) > 1:
        _fail(f"{benchmark_id}: {story_id!r} matches {len(matches)} units")
    return bench, matches[0]


def probe_tree_builds(benchmarks: list[str]) -> list[dict]:
    """Wall-time for M4 tree construction alone, one unit per benchmark."""
    from src.config import DEFAULT_CONFIG
    from src.retrievers.m4_raptor import RaptorSystem

    rows: list[dict] = []
    for benchmark_id in benchmarks:
        bench, unit = _one_unit(benchmark_id)
        system = RaptorSystem(DEFAULT_CONFIG)

        t0 = time.perf_counter()
        system.index_items(unit.corpus)
        build_s = time.perf_counter() - t0

        # THE ASSERTIONS. A cache read is not a build.
        if system.tree_cache_hit is None:
            _fail(f"{benchmark_id}: index() never recorded a cache verdict")
        if system.tree_cache_hit:
            _fail(
                f"{benchmark_id}: substrate was served WARM, so {build_s:.1f}s "
                "is a cache read, not a build. Clear the cache dir for this "
                "corpus, or the cold-tree lever did not take."
            )
        stats = dict(system.index_stats)
        n_leaves = int(stats.get("n_leaves", 0) or 0)
        if n_leaves <= 0:
            _fail(f"{benchmark_id}: build produced no leaves; nothing was measured")
        if int(stats.get("n_summary_calls_at_index", 0) or 0) <= 0:
            _fail(
                f"{benchmark_id}: zero summary calls, so no summariser work "
                "was timed. That is not a tree build."
            )
        if stats.get("degenerate_no_tree"):
            _fail(
                f"{benchmark_id}: FLAT INDEX — the corpus fell below the layer "
                "stop condition, so this is not a tree-build time and must not "
                "be extrapolated as one."
            )

        row = {
            "benchmark": benchmark_id,
            "corpus_id": unit.corpus_id,
            "n_corpus_items": len(unit.corpus),
            "n_queries_in_unit": len(unit.queries),
            "build_s": round(build_s, 2),
            "n_leaves": n_leaves,
            "n_summary_nodes": int(stats.get("flat_n_summaries", 0) or 0),
            "n_summary_calls": int(stats.get("n_summary_calls_at_index", 0) or 0),
            "layer_sizes": stats.get("layer_sizes"),
            "tree_cache_hit": system.tree_cache_hit,
        }
        rows.append(row)
        print(f"[probe] tree {benchmark_id:<16} build={build_s:8.2f}s  "
              f"items={row['n_corpus_items']:<6} leaves={n_leaves:<6} "
              f"summary_nodes={row['n_summary_nodes']:<5} "
              f"summary_calls={row['n_summary_calls']}")
    return rows


def _cap_moves_key() -> bool:
    """Does summary_max_padded_tokens actually change the substrate key?

    Pure: compares two keys over a FIXED stub corpus hash, so it asks
    about the LEVER and not about any particular story. That is what lets
    it separate the two reasons a build can come back warm.
    """
    import dataclasses

    from src.cache import compute_cache_key
    from src.config import DEFAULT_CONFIG
    from src.raptor_paper import paper_substrate_extra

    def key(cap: int) -> str:
        m4 = dataclasses.replace(DEFAULT_CONFIG.m4,
                                 summary_max_padded_tokens=cap)
        extra = paper_substrate_extra(
            params=m4.paper, summary_model=m4.summary_model,
            summary_prompt_version=m4.summary_prompt_version,
            summary_max_tokens=m4.summary_max_tokens,
            summary_batch_size=m4.summary_batch_size,
            summary_max_padded_tokens=m4.summary_max_padded_tokens,
            rrf_k=m4.rrf_k, include_root=m4.include_root_in_flat_index)
        return compute_cache_key(chunking_config=m4.chunker,
                                 embedder_model="stub", corpus_hash="stub",
                                 extra=extra, parsing_identity={})

    return key(8000) != key(4000)


def _warm_hit_message(cap: int, story: str, build_s: float) -> str:
    """Two different situations wore the same message. Separate them.

    A warm substrate means EITHER this exact cap was already built for
    this story - a repeat request, not a defect - OR the cap failed to
    move the key, which is a lever bug that would silently serve one
    cap tree for another. Reporting both as "the cap is supposed to move
    the key" made a routine repeat look like a defect, and would have
    made a real defect look routine.
    """
    if _cap_moves_key():
        return (
            f"cap={cap} was ALREADY BUILT for {story}: the substrate came "
            f"back warm in {build_s:.1f}s, which is a cache read and not a "
            "build. REPEAT REQUEST, not a defect - the cap does move the "
            "key, verified just now against two other values. Pass "
            "--force-cold to rebuild, or sweep a cap this story has not "
            "seen."
        )
    return (
        f"cap={cap}: LEVER BUG. The substrate came back warm AND two "
        "different caps produce the SAME cache key, so a tree built at one "
        "cap can be served for another. Every sweep result is suspect until "
        "paper_substrate_extra folds the cap again."
    )


def _vram_peak_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:
        pass
    return None


def _vram_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == (
        "OutOfMemoryError")


def _force_cold():
    """Make the next index() take the BUILD path, whatever is cached.

    DELIBERATELY NOT `rm -rf` ON THE SUBSTRATE DIRECTORY, and the
    deviation is worth stating. A probe that deletes from a Drive path is
    one typo away from removing a banked tree that cost hours, and
    working out WHICH directory to delete means reproducing the key
    derivation - the thing under test. Overriding the cache CHECK instead
    makes index() take the miss branch, rebuild every artifact, and
    overwrite in place. The manifest is still written last, so an
    interrupted rebuild leaves an incomplete directory the next run
    treats as a miss.

    Same cold compute; no destructive filesystem operation. Returns the
    original method so the caller restores it.
    """
    from src import cache as cache_mod

    original = cache_mod.CacheDir.is_complete

    def always_cold(self, required):  # noqa: ANN001
        return False

    cache_mod.CacheDir.is_complete = always_cold
    return original


def _restore_cache_check(original) -> None:
    from src import cache as cache_mod

    if original is not None:
        cache_mod.CacheDir.is_complete = original


def probe_padding_sweep(
    benchmark_id: str, story_id: str, values: list[int], sweep_all: bool,
    force_cold: bool = False,
) -> list[dict]:
    """Largest `summary_max_padded_tokens` that builds the worst-case unit.

    WHY THIS KNOB AND NOT summary_batch_size. `generate_batch` packs until
    `n * longest_prompt` reaches the cap, so on ~800-token summary prompts
    the effective batch is already clipped well below the nominal 32.
    Halving the CAP halves peak activation directly; halving the batch
    size may not move the effective batch at all.

    WHY THE LARGEST THAT FITS, NOT THE SMALLEST THAT RUNS. The cap is not
    purely operational: a smaller cap means fewer summary contexts per
    batch, which can change what gets summarised, and therefore tree
    topology and M4's retrieval numbers. Staying near the intended RAPTOR
    construction argues for the high end of what survives.

    EACH VALUE IS ITS OWN COLD BUILD, free of cross-contamination: the cap
    sits in `paper_substrate_extra`, so every sweep step lands on a
    different cache key. The cache-hit assertion still runs at each step,
    because "different key" is a claim about the code and the probe checks
    it against the artifact.

    Default is DESCENDING, STOPPING AT THE FIRST SUCCESS: that success IS
    the answer, and every lower value is a strictly worse choice by the
    fidelity argument above. --sweep-all measures the rest for the
    tradeoff curve, at the price of three more full builds of a ~4,900-leaf
    story.
    """
    from src.config import DEFAULT_CONFIG
    from src.retrievers.m4_raptor import RaptorSystem

    bench, unit = _one_unit(benchmark_id, story_id)
    n_chars = sum(len(i.text) for i in unit.corpus)
    print(f"[probe] sweep target: {unit.corpus_id} ({n_chars:,} chars, "
          f"{len(unit.queries)} queries)")

    rows: list[dict] = []
    for cap in sorted(values, reverse=True):
        cfg = dataclasses.replace(
            DEFAULT_CONFIG,
            m4=dataclasses.replace(DEFAULT_CONFIG.m4,
                                   summary_max_padded_tokens=cap),
        )
        system = RaptorSystem(cfg)
        restore = _force_cold() if force_cold else None
        if force_cold:
            print(f"[probe] --force-cold: cache check overridden for cap={cap}")
        _vram_reset()
        print(f"[probe] --- summary_max_padded_tokens={cap} ---")
        t0 = time.perf_counter()
        try:
            system.index_items(unit.corpus)
        except Exception as exc:  # noqa: BLE001 - the OOM IS the result
            _restore_cache_check(restore)
            dt = time.perf_counter() - t0
            oom = _is_oom(exc)
            rows.append({
                "summary_max_padded_tokens": cap, "ok": False,
                "oom": oom, "error": f"{type(exc).__name__}: {exc}"[:300],
                "elapsed_s": round(dt, 2), "peak_vram_gb": _vram_peak_gb(),
            })
            print(f"[probe] cap={cap:<6} {'OOM' if oom else 'ERROR'} after "
                  f"{dt:.1f}s  peak={_vram_peak_gb()}GB")
            if not oom:
                _fail(f"cap={cap} failed for a reason that is NOT OOM; the "
                      "sweep cannot interpret it. Fix that first.")
            _vram_reset()
            continue

        dt = time.perf_counter() - t0
        _restore_cache_check(restore)
        if system.tree_cache_hit:
            _fail(_warm_hit_message(cap, str(unit.corpus_id), dt))
        stats = dict(system.index_stats)
        if int(stats.get("n_summary_calls_at_index", 0) or 0) <= 0:
            _fail(f"cap={cap}: zero summary calls; no summariser work timed.")
        row = {
            "summary_max_padded_tokens": cap, "ok": True, "oom": False,
            "build_s": round(dt, 2), "peak_vram_gb": _vram_peak_gb(),
            "n_leaves": int(stats.get("n_leaves", 0) or 0),
            "n_summary_calls": int(stats.get("n_summary_calls_at_index", 0) or 0),
            "n_summary_nodes": int(stats.get("flat_n_summaries", 0) or 0),
            "layer_sizes": stats.get("layer_sizes"),
        }
        rows.append(row)
        print(f"[probe] cap={cap:<6} OK build={dt:8.1f}s  "
              f"peak={row['peak_vram_gb']}GB  leaves={row['n_leaves']} "
              f"summary_calls={row['n_summary_calls']}")
        _vram_reset()
        if not sweep_all:
            print(f"[probe] LARGEST SURVIVING CAP = {cap} (stopping; every "
                  "lower value is strictly worse for tree fidelity)")
            break

    survivors = [r for r in rows if r.get("ok")]
    if not survivors:
        _fail("no value in the sweep completed. Lower the range, or fall "
              "back to summary_batch_size as the second knob.")
    return rows


def probe_query_slice(system_ids: list[str], benchmark_id: str, n: int) -> list[dict]:
    """Timed slice: s_per_query net of indexing and of model load."""
    from src.config import DEFAULT_CONFIG
    from src.eval.runner import SYSTEM_REGISTRY

    rows: list[dict] = []
    bench, unit = _one_unit(benchmark_id)
    queries = list(unit.queries)[:n]
    if len(queries) < n:
        _fail(f"{benchmark_id}: only {len(queries)} queries available, wanted {n}")

    for system_id in system_ids:
        system = SYSTEM_REGISTRY[system_id](DEFAULT_CONFIG)

        t_index = time.perf_counter()
        system.index_items(unit.corpus)
        index_s = time.perf_counter() - t_index

        t0 = time.perf_counter()
        n_done = 0
        for q in queries:
            system.answer(q.question_text)
            n_done += 1
        answer_s = time.perf_counter() - t0

        if n_done != n:
            _fail(f"{system_id}: answered {n_done} of {n} queries")

        row = {
            "system": system_id,
            "benchmark": benchmark_id,
            "n": n_done,
            "index_s": round(index_s, 2),
            "answer_s": round(answer_s, 2),
            "s_per_query": round(answer_s / n_done, 4),
            "tree_cache_hit": getattr(system, "tree_cache_hit", None),
        }
        rows.append(row)
        print(f"[probe] queries {system_id:<4} {benchmark_id:<14} "
              f"index={index_s:7.1f}s  {row['s_per_query']:.3f} s/query "
              f"over n={n_done}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",
                    choices=("tree", "queries", "both", "sweep"),
                    default="both")
    ap.add_argument("--story-id", default="largest",
                    help="corpus_id prefix to target, or 'largest' (default) "
                         "for the biggest unit in the CELL'S OWN draw")
    ap.add_argument("--sweep-benchmark", default="narrativeqa")
    ap.add_argument("--sweep-values", default="16000,8000,4000,2000")
    ap.add_argument("--force-cold", action="store_true",
                    help="rebuild even when this story and cap are already "
                         "cached (overrides the cache check; deletes nothing)")
    ap.add_argument("--sweep-all", action="store_true",
                    help="measure every value instead of stopping at the "
                         "largest that survives")
    ap.add_argument("--n", type=int, default=50, help="queries in the timed slice")
    ap.add_argument("--tree-benchmarks", default="narrativeqa,multihop_rag,hotpotqa")
    ap.add_argument("--query-systems", default="M4,M9")
    ap.add_argument("--query-benchmark", default="multihop_rag")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from scripts.pin_environment import environment_provenance

    report: dict[str, Any] = {
        "environment": environment_provenance(Path("requirements.lock")),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "prewarm_s": None,
        "tree_builds": [],
        "query_slices": [],
        "padding_sweep": [],
    }
    print(f"[probe] PYTORCH_CUDA_ALLOC_CONF="
          f"{report['pytorch_cuda_alloc_conf']}")
    print(f"[probe] gpu={report['environment'].get('gpu')}")

    report["prewarm_s"] = round(_prewarm(), 1)

    if args.mode == "sweep":
        report["padding_sweep"] = probe_padding_sweep(
            args.sweep_benchmark, args.story_id,
            [int(v) for v in args.sweep_values.split(",") if v.strip()],
            args.sweep_all, force_cold=args.force_cold)
    if args.mode in ("tree", "both"):
        report["tree_builds"] = probe_tree_builds(
            [b.strip() for b in args.tree_benchmarks.split(",") if b.strip()])
    if args.mode in ("queries", "both"):
        report["query_slices"] = probe_query_slice(
            [s.strip() for s in args.query_systems.split(",") if s.strip()],
            args.query_benchmark, args.n)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[probe] wrote {args.out}")
    print()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
