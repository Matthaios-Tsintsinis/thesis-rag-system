"""Does running UMAP poison generation in the same process?

THE HYPOTHESIS. Everything GPU-shaped is dead. The generation path is
exonerated — the build's own call-3 shape runs in 6.87 s isolated against
228 s inside a build, a 33x gap on identical work, with placement clean
and the KV cache on. Weight residency is dead by code read: nothing in
the indexing path releases or reloads the generator. If the snapshot
build also returns healthy `free_gb`, memory pressure dies too and what
is left is CPU-side contention.

Mechanism: numba spawns a worker pool for UMAP. Seeding forces
`n_jobs=1`, which constrains parallelism but need not tear down pools
already created. Contended CPU threads starve the Python-side loop that
sits between CUDA kernel launches, which produces a cost that is flat per
call, independent of batch shape, invisible to every GPU metric, and
absent from a process that never ran UMAP. That is the entire observed
signature.

WHAT MAKES THIS A MEASUREMENT RATHER THAN A DEMONSTRATION

  * TWO baseline calls before any UMAP, not one. A first-call warmup
    effect would otherwise read as a UMAP effect, and telling those apart
    is the whole result. If the baselines disagree by more than the
    tolerance, the probe reports INCONCLUSIVE_UNSTABLE_BASELINE instead
    of a ratio computed off a floor that moves.
  * The SLOWER baseline is the floor. Comparing against the faster of the
    two would inflate the slowdown.
  * The UMAP call is shaped from REAL BUILD DATA, not invented: story
    d431326b recorded layer sizes 142 -> 30 -> 4 with mpnet's 768
    dimensions, and the paper params fix n_components=10, metric=cosine,
    random_state=42, local n_neighbors=10.
  * It tests the CANDIDATE FIX in the same run. A verdict that names a
    cause but cannot say whether the obvious remedy works costs another
    round trip, and this investigation has spent several.
  * A post-UMAP call that failed is carried as None and reported
    INCONCLUSIVE. It must never read as fast.

USAGE (Colab, pinned stack, GPU attached). No build, no cache, no
substrate. About two minutes, most of it the model load.

    python -m scripts.probe_umap_contention
    python -m scripts.probe_umap_contention --width 2 --prompt-tokens 1092 \
        --out /content/umap_contention.json

RUN IT ONLY IF the snapshot build returns healthy free_gb. If free_gb is
small, memory pressure is the live hypothesis and this probe is measuring
the wrong thing.
"""

from __future__ import annotations

import os

# BEFORE ANY TORCH IMPORT, matching every other probe in this
# investigation so the comparison is like for like.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# A post-UMAP call this many times the baseline counts as contention.
# The observed gap is ~33x, so this is not a fine judgement.
CONTENTION_FACTOR = 5.0

# Two baseline calls may differ by this fraction before the floor is
# treated as unstable.
BASELINE_DRIFT_TOLERANCE = 0.25

# A post-teardown call within this multiple of baseline counts as fixed.
FIXED_FACTOR = 2.0

# Layer sizes recorded from the real build of story d431326b
# (142 leaves -> 30 -> 4), embedded by mpnet at 768 dimensions.
BUILD_LAYER_SIZES = (142, 30)
EMBED_DIM = 768


def classify_umap_contention(
    *,
    pre_umap: list[float | None],
    post_umap: float | None,
    post_teardown: float | None,
) -> dict:
    """Map the three readings onto a named outcome.

    All values are seconds per decode step. `pre_umap` holds the baseline
    calls issued before UMAP ran; the SLOWER of them is the floor.
    """
    usable = [v for v in pre_umap if v is not None]
    if not usable:
        raise ValueError(
            "no usable baseline call: without a floor measured in this "
            "same process there is nothing to compare against, and a "
            "ratio against another run's number would not be a control"
        )

    baseline = max(usable)          # conservative: the slower call
    fastest = min(usable)
    drift = (baseline - fastest) / fastest if fastest else 0.0

    out: dict[str, Any] = {
        "baseline_s_per_step": baseline,
        "baseline_drift": round(drift, 4),
        "pre_umap": pre_umap,
        "post_umap": post_umap,
        "post_teardown": post_teardown,
        "slowdown_factor": (
            round(post_umap / baseline, 2)
            if post_umap is not None and baseline
            else None
        ),
        "teardown_factor": (
            round(post_teardown / baseline, 2)
            if post_teardown is not None and baseline
            else None
        ),
    }

    if drift > BASELINE_DRIFT_TOLERANCE:
        out["verdict"] = "INCONCLUSIVE_UNSTABLE_BASELINE"
        out["explanation"] = (
            f"The two pre-UMAP calls differ by {drift:.0%}, above the "
            f"{BASELINE_DRIFT_TOLERANCE:.0%} tolerance. The floor is not "
            "stable enough to support a ratio — a warmup effect and a "
            "UMAP effect would be indistinguishable. Re-run with an extra "
            "untimed warmup."
        )
        return out

    if post_umap is None:
        out["verdict"] = "INCONCLUSIVE"
        out["explanation"] = (
            "The post-UMAP call did not complete, so nothing is shown "
            "either way. A call that failed is not a fast call."
        )
        return out

    if out["slowdown_factor"] < CONTENTION_FACTOR:
        out["verdict"] = "NOT_UMAP"
        out["explanation"] = (
            f"Generation after UMAP runs at {out['slowdown_factor']}x the "
            "same-process baseline, so running UMAP does not poison "
            "generation. CPU contention from the clustering stack is out. "
            "What remains is something else the build does that this "
            "probe does not reproduce — the embedder held resident, the "
            "tree structures themselves, or the interleaving of phases "
            "rather than any one phase."
        )
        return out

    if post_teardown is None:
        out["verdict"] = "UMAP_CONTENTION"
        out["explanation"] = (
            f"Generation after UMAP runs at {out['slowdown_factor']}x the "
            "same-process baseline, in a process that has done nothing "
            "else. Running UMAP poisons generation. No teardown variant "
            "was measured, so the remedy is untested."
        )
        return out

    if out["teardown_factor"] <= FIXED_FACTOR:
        out["verdict"] = "UMAP_CONTENTION_FIXED_BY_TEARDOWN"
        out["explanation"] = (
            f"UMAP slows generation by {out['slowdown_factor']}x, and "
            f"restricting the numba thread pool restores it to "
            f"{out['teardown_factor']}x baseline. Cause and remedy both "
            "identified, and the remedy is a thread-pool setting rather "
            "than a fidelity tradeoff — it changes no tree content."
        )
        return out

    out["verdict"] = "UMAP_CONTENTION_TEARDOWN_INEFFECTIVE"
    out["explanation"] = (
        f"UMAP slows generation by {out['slowdown_factor']}x and "
        f"restricting the thread pool afterwards leaves it at "
        f"{out['teardown_factor']}x, so the pool cannot be reclaimed once "
        "created. The lever is then NUMBA_NUM_THREADS set BEFORE numba "
        "imports, or running clustering in a subprocess that exits."
    )
    return out


def _thread_state() -> dict:
    """Everything about threading this process can see, cheaply."""
    import threading

    out: dict[str, Any] = {
        "python_active_threads": threading.active_count(),
        "os_cpu_count": os.cpu_count(),
        "env_NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS"),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        import numba

        out["numba_num_threads"] = numba.get_num_threads()
        out["numba_config_NUMBA_NUM_THREADS"] = numba.config.NUMBA_NUM_THREADS
        out["numba_threading_layer"] = getattr(
            numba.config, "THREADING_LAYER", None
        )
    except Exception as exc:  # noqa: BLE001
        out["numba_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        import torch

        out["torch_num_threads"] = torch.get_num_threads()
        out["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except Exception:  # noqa: BLE001
        pass
    return out


def _timed_call(model, tokenizer, enc, *, max_new_tokens: int) -> dict | None:  # noqa: ANN001
    """One generate() call, synchronised, per decode step. None on failure."""
    import torch

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 - a failed call is data
        print(f"[probe] call FAILED: {type(exc).__name__}: {exc}")
        return None

    new_tokens = int(out.shape[1] - enc["input_ids"].shape[1])
    return {
        "generate_s": round(dt, 3),
        "new_tokens": new_tokens,
        "s_per_decode_step": (
            round(dt / new_tokens, 6) if new_tokens > 0 else None
        ),
    }


def _encode(tokenizer, model, *, width: int, prompt_tokens: int):  # noqa: ANN001
    filler = "the tree grows upward from its leaves. " * 4000
    ids = tokenizer(filler, add_special_tokens=False)["input_ids"]
    if len(ids) < prompt_tokens:
        raise RuntimeError(
            f"filler yielded {len(ids)} tokens, wanted {prompt_tokens}"
        )
    text = tokenizer.decode(ids[:prompt_tokens])

    prev_side, prev_pad = tokenizer.padding_side, tokenizer.pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        enc = tokenizer(
            [text] * width, return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(model.device)
    finally:
        tokenizer.padding_side, tokenizer.pad_token = prev_side, prev_pad

    actual = int(enc["input_ids"].shape[1])
    if abs(actual - prompt_tokens) > max(8, prompt_tokens * 0.02):
        raise RuntimeError(
            f"encoded {actual} tokens, wanted ~{prompt_tokens}; the probe "
            "must measure the shape it claims"
        )
    return enc


def _run_umap_like_the_build() -> dict:
    """One UMAP fit per recorded layer, at the build's own parameters.

    Shapes come from story d431326b's recorded layers rather than being
    chosen: a fixture standing in for real data must have its parameters
    measured from that data, or it is testing something else.
    """
    import numpy as np

    from src.config import DEFAULT_CONFIG
    from src.raptor_paper import _global_cluster_embeddings

    params = DEFAULT_CONFIG.m4.paper
    rng = np.random.default_rng(0)
    fits = []
    for n in BUILD_LAYER_SIZES:
        emb = rng.normal(size=(n, EMBED_DIM)).astype("float32")
        t0 = time.perf_counter()
        reduced = _global_cluster_embeddings(
            emb, params.reduction_dimension, params
        )
        fits.append({
            "n_points": n,
            "umap_s": round(time.perf_counter() - t0, 3),
            "out_shape": list(reduced.shape),
        })
        print(f"[probe] umap fit n={n}: {fits[-1]['umap_s']}s")
    return {"fits": fits}


def _restrict_numba_threads() -> dict:
    """The candidate remedy: shrink the pool that UMAP created."""
    out: dict[str, Any] = {"attempted": True}
    try:
        import numba

        numba.set_num_threads(1)
        out["numba_num_threads_after"] = numba.get_num_threads()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=2,
                    help="batch width; 2 reproduces the build's call 3")
    ap.add_argument("--prompt-tokens", type=int, default=1092)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch

    from src.config import DEFAULT_CONFIG
    from src.models import load_generator

    report: dict[str, Any] = {
        "shape": {"width": args.width, "prompt_tokens": args.prompt_tokens,
                  "max_new_tokens": args.max_new_tokens},
        "umap_layer_sizes": list(BUILD_LAYER_SIZES),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "torch_version": torch.__version__,
    }

    tokenizer, model = load_generator(DEFAULT_CONFIG.generation.model)
    enc = _encode(tokenizer, model,
                  width=args.width, prompt_tokens=args.prompt_tokens)

    # Untimed warmup. The first call on a fresh model pays autotuning no
    # later call pays, and letting that land on baseline call 1 would
    # manufacture the very drift the control exists to detect.
    _timed_call(model, tokenizer, enc, max_new_tokens=4)

    report["threads_before_umap"] = _thread_state()
    print(f"[probe] threads before umap: {report['threads_before_umap']}")

    # TWO baselines, both before any UMAP.
    report["pre_umap_calls"] = [
        _timed_call(model, tokenizer, enc,
                    max_new_tokens=args.max_new_tokens)
        for _ in range(2)
    ]
    for i, c in enumerate(report["pre_umap_calls"], 1):
        print(f"[probe] pre-umap call {i}: {c}")

    report["umap"] = _run_umap_like_the_build()
    report["threads_after_umap"] = _thread_state()
    print(f"[probe] threads after umap: {report['threads_after_umap']}")

    report["post_umap_call"] = _timed_call(
        model, tokenizer, enc, max_new_tokens=args.max_new_tokens
    )
    print(f"[probe] post-umap call: {report['post_umap_call']}")

    report["teardown"] = _restrict_numba_threads()
    report["post_teardown_call"] = _timed_call(
        model, tokenizer, enc, max_new_tokens=args.max_new_tokens
    )
    report["threads_after_teardown"] = _thread_state()
    print(f"[probe] post-teardown call: {report['post_teardown_call']}")

    def step(c: dict | None) -> float | None:
        return c.get("s_per_decode_step") if c else None

    report["classification"] = classify_umap_contention(
        pre_umap=[step(c) for c in report["pre_umap_calls"]],
        post_umap=step(report["post_umap_call"]),
        post_teardown=step(report["post_teardown_call"]),
    )

    print()
    print(f"[probe] VERDICT: {report['classification']['verdict']}")
    print(f"[probe] {report['classification']['explanation']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[probe] wrote {args.out}")
    print()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
