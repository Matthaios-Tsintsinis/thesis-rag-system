"""Reproduce the 230 s generate() call WITHOUT a tree build, then name it.

THE MEASUREMENT THIS ANSWERS. A tree build spends 96% of its time in
summarisation at ~230 s per `model.generate()` call. The per-call
instrumentation put all of it inside `generate()` itself — tokenise
~0.02 s, decode ~0.001 s — and showed it FLAT across an 8x range of batch
width and a 2x range of prompt length:

    call  width  prompt_tokens  generate_s  s_per_decode_step
    1     16     498            238.26      2.383
    2     12     648            236.20      2.362
    3     2      1092           228.14      2.281
    4     4      957            228.80      2.288

The healthy baseline is 6.6 s for 100 steps on one ~2,000-token prompt
with fp16 weights resident on an L4 — about 66 ms/step. So the build runs
~35x slower per step, on SHORTER prompts.

WHAT THE FLATNESS ALREADY RULES OUT, before this probe runs. Call 1
carries 16 x 498 = 7,968 padded input cells against call 3's
2 x 1,092 = 2,184 — 3.65x the tokens for 1.04x the time. Any explanation
whose cost scales with tokens processed is dead on that arithmetic, and
that INCLUDES `use_cache=False`, which re-prefills every step and would
therefore spread those two calls by a factor of ~3.6. It is measured here
as a CONTROL, not as a candidate.

What survives is a cost that ignores how much data is processed: roughly
15 GB of weights moved per decode step at ~6.5 GB/s, which is PCIe
bandwidth rather than HBM. That is the signature of parameters not being
resident on the GPU.

WHY THIS IS NOT ALREADY ANSWERED by `diagnose_generator_placement.py`.
That script loads the generator into a FRESH process with empty VRAM and
reports 339/339 tensors on cuda:0. But `device_map="auto"` decides
placement AT LOAD TIME from whatever VRAM is free at that moment, and
`describe_generator_runtime` defaults to `release=True`, which clears the
`lru_cache` that keeps the model resident. Nobody has yet measured
placement AT THE MOMENT OF A SLOW CALL. Every placement reading so far
was taken under conditions that cannot reproduce the fault.

THE PROBE RULES THIS FILE OBEYS

  * It ASSERTS that it measured what it claims: a variant that failed to
    run reports None and is carried as None, never as fast.
  * A VACUOUS PASS IS A FAILURE. If the default variant is already
    healthy, that is reported as NOT_REPRODUCED rather than dressed up —
    it would mean the cost lives in the build context, not in the
    generation path, which is a different search.
  * It states a VERDICT, so the mapping from four numbers to a cause is
    written down and can be argued with instead of guessed.

USAGE (Colab, pinned stack, GPU attached). About a minute; builds
nothing, caches nothing, and touches no substrate.

    python -m scripts.probe_generate_call_isolation
    python -m scripts.probe_generate_call_isolation --width 2 \
        --prompt-tokens 1092 --out /content/call_isolation.json
"""

from __future__ import annotations

import os

# BEFORE ANY TORCH IMPORT — the allocator config is read once, when CUDA
# initialises. Matched to the probe that produced the numbers above, so
# this is a comparison and not a new configuration.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# 6.6 s for 100 decode steps, one ~2,000-token prompt, Qwen2.5-7B fp16
# resident on an L4, KV cache on. Recorded in
# docs/EVAL_CORRECTION_PLAN.md under COST REFERENCE.
HEALTHY_S_PER_STEP = 0.066

# A variant within this multiple of the baseline counts as healthy. Wide
# enough that ordinary batch-width overhead does not read as a fault,
# far below the ~35x actually observed.
FAST_RATIO = 3.0

# Below this, nothing here is the explanation for a 35x gap.
SLOW_RATIO = 10.0


def classify_per_call_cost(
    measurements: dict[str, float | None],
    *,
    healthy_s_per_step: float = HEALTHY_S_PER_STEP,
) -> dict:
    """Map measured per-step costs onto a named cause.

    `measurements` maps variant name to seconds per decode step, or None
    where the variant did not run. `default` is required: without the
    production path there is nothing to explain.
    """
    if "default" not in measurements or measurements["default"] is None:
        raise ValueError(
            "no `default` measurement: the production path is the thing "
            "being explained, so its absence is an error, not a verdict"
        )

    ratios = {
        k: (round(v / healthy_s_per_step, 2) if v is not None else None)
        for k, v in measurements.items()
    }

    def is_fast(name: str) -> bool:
        r = ratios.get(name)
        return r is not None and r <= FAST_RATIO

    def ran(name: str) -> bool:
        return ratios.get(name) is not None

    default_ratio = ratios["default"]

    if default_ratio < SLOW_RATIO:
        return {
            "verdict": "NOT_REPRODUCED",
            "explanation": (
                f"An isolated call runs at {default_ratio}x the healthy "
                "baseline, so the fault does not live in the generation "
                "path on its own. Something the BUILD does to the process "
                "is the trigger — resident embedder, UMAP worker threads, "
                "allocator state after clustering, or a reload of the "
                "generator into VRAM that is no longer empty. Search "
                "there, and re-measure placement inside a build."
            ),
            "ratios": ratios,
            "healthy_s_per_step": healthy_s_per_step,
        }

    placement_fixed = is_fast("all_on_gpu")
    cache_fixed = is_fast("use_cache_true")

    if placement_fixed:
        explanation = (
            "Forcing every layer onto cuda:0 restores healthy speed, so "
            "the production load is NOT keeping the weights resident and "
            "each decode step is moving parameters across PCIe. Fix the "
            "load, not the batch shape."
        )
        if cache_fixed:
            explanation += (
                " AMBIGUOUS in one respect, stated rather than resolved: "
                "the all-on-GPU variant also reloads the model, so it "
                "could mask a cache effect. Placement is reported as "
                "primary because it is the variant that changes weight "
                "residency; confirm by reading the placement block."
            )
        return {
            "verdict": "PLACEMENT",
            "explanation": explanation,
            "ratios": ratios,
            "healthy_s_per_step": healthy_s_per_step,
        }

    if cache_fixed:
        return {
            "verdict": "KV_CACHE_DEFAULT",
            "explanation": (
                "Passing use_cache=True explicitly restores healthy "
                "speed, so the default resolved at call time is not the "
                "one assumed. NOTE the tension with the build's own call "
                "table: a missing cache re-prefills every step and its "
                "cost scales with padded tokens, which the flat build "
                "measurements contradict. Reconcile before acting."
            ),
            "ratios": ratios,
            "healthy_s_per_step": healthy_s_per_step,
        }

    if not (ran("all_on_gpu") and ran("use_cache_true")):
        return {
            "verdict": "INCONCLUSIVE",
            "explanation": (
                "The production path is slow and at least one control "
                "variant did not run, so nothing is excluded. A variant "
                "that failed is carried as None and must not be read as "
                "fast."
            ),
            "ratios": ratios,
            "healthy_s_per_step": healthy_s_per_step,
        }

    return {
        "verdict": "INTRINSIC",
        "explanation": (
            f"Every variant runs at ~{default_ratio}x the healthy "
            "baseline. Not placement, not the cache default, not batch "
            "shape, not prompt length. The cost is intrinsic to "
            "generate() on this stack at this shape — attention "
            "implementation, a per-step host synchronisation inside the "
            "decode loop, or a transformers 5.x regression. Compare the "
            "resolved config and attn_implementation blocks against the "
            "host that produced the 6.6 s baseline."
        ),
        "ratios": ratios,
        "healthy_s_per_step": healthy_s_per_step,
    }


def _placement(model) -> dict:  # noqa: ANN001
    """Where the parameters ACTUALLY are, at this moment."""
    from collections import Counter

    n_by_device: Counter[str] = Counter()
    tensors_by_device: Counter[str] = Counter()
    for _, p in model.named_parameters():
        dev = str(p.device)
        n_by_device[dev] += p.numel()
        tensors_by_device[dev] += 1
    total = sum(n_by_device.values())
    off = sum(n for d, n in n_by_device.items() if not d.startswith("cuda"))
    return {
        "param_tensors_by_device": dict(tensors_by_device),
        "params_by_device_billions": {
            d: round(n / 1e9, 3) for d, n in n_by_device.items()
        },
        "fraction_params_off_gpu": round(off / total, 4) if total else None,
        "hf_device_map": getattr(model, "hf_device_map", None),
        "dtype": str(getattr(model, "dtype", "?")),
        "attn_implementation": getattr(
            model.config, "_attn_implementation", "unknown"
        ),
    }


def _resolved_config(model) -> dict:
    """What generate() will actually use for the fields that matter.

    Read from both the model config and the generation config, because
    the call sites pass neither — they pass plain kwargs, so every field
    not named in the kwargs resolves from these objects.
    """
    gc = getattr(model, "generation_config", None)
    return {
        "config_use_cache": getattr(model.config, "use_cache", None),
        "generation_config_use_cache": getattr(gc, "use_cache", None),
        "generation_config_cache_implementation": getattr(
            gc, "cache_implementation", None
        ),
        "generation_config_max_new_tokens": getattr(gc, "max_new_tokens", None),
        "generation_config_do_sample": getattr(gc, "do_sample", None),
    }


def _timed_call(model, tokenizer, enc, *, max_new_tokens: int,
                use_cache=None) -> dict:  # noqa: ANN001
    """One generate() call, synchronised, reported per decode step."""
    import torch

    kwargs: dict[str, Any] = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if use_cache is not None:
        kwargs["use_cache"] = use_cache

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    new_tokens = int(out.shape[1] - enc["input_ids"].shape[1])
    return {
        "generate_s": round(dt, 3),
        "new_tokens": new_tokens,
        "s_per_decode_step": (
            round(dt / new_tokens, 6) if new_tokens > 0 else None
        ),
        "peak_vram_gb": (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available() else None
        ),
    }


def _encode(tokenizer, model, *, width: int, prompt_tokens: int):  # noqa: ANN001
    """A batch of `width` prompts, each ~`prompt_tokens` long.

    Built from a repeated token rather than real prose on purpose: the
    question is about SHAPE, and using a fixture that varies in content
    would add a variable the build's own numbers do not have.
    """
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=2,
                    help="batch width; 2 reproduces the build's call 3")
    ap.add_argument("--prompt-tokens", type=int, default=1092,
                    help="prompt length; 1092 reproduces the build's call 3")
    ap.add_argument("--max-new-tokens", type=int, default=100,
                    help="the summary path's cap")
    ap.add_argument("--skip-no-cache", action="store_true",
                    help="skip the use_cache=False control (it is slow by "
                         "construction and is only a reference point)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch

    from src.config import DEFAULT_CONFIG
    from src.models import load_generator, release_generator

    report: dict[str, Any] = {
        "shape": {"width": args.width, "prompt_tokens": args.prompt_tokens,
                  "max_new_tokens": args.max_new_tokens},
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "variants": {},
    }
    model_id = DEFAULT_CONFIG.generation.model

    # ---- VARIANT SET 1: the production load path, exactly as the build
    # reaches it. `load_generator` is what m4_raptor -> generate_batch
    # calls, so this is the object whose behaviour is in question.
    t0 = time.perf_counter()
    tokenizer, model = load_generator(model_id)
    report["load_s"] = round(time.perf_counter() - t0, 1)
    report["placement_production"] = _placement(model)
    report["resolved_config_production"] = _resolved_config(model)
    print(f"[probe] placement: {report['placement_production']}")
    print(f"[probe] resolved:  {report['resolved_config_production']}")

    enc = _encode(tokenizer, model,
                  width=args.width, prompt_tokens=args.prompt_tokens)

    # Warm the kernels once, untimed. The first call on a fresh model
    # pays autotuning that no later call pays, and reporting it as the
    # measurement would inflate every variant equally but honestly
    # mislead about the absolute number.
    _timed_call(model, tokenizer, enc, max_new_tokens=4)

    for name, use_cache in (("default", None), ("use_cache_true", True)):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        report["variants"][name] = _timed_call(
            model, tokenizer, enc, max_new_tokens=args.max_new_tokens,
            use_cache=use_cache,
        )
        print(f"[probe] {name:<16} {report['variants'][name]}")

    if not args.skip_no_cache:
        # CONTROL, not a candidate: already ruled out by the flatness of
        # the build's own call table. Measured so the reference point
        # exists at THESE shapes rather than being scaled from a
        # width-1 reading.
        report["variants"]["use_cache_false"] = _timed_call(
            model, tokenizer, enc, max_new_tokens=args.max_new_tokens,
            use_cache=False,
        )
        print(f"[probe] use_cache_false  "
              f"{report['variants']['use_cache_false']}")

    # ---- VARIANT SET 2: every layer pinned to cuda:0, bypassing
    # device_map="auto". If this is fast and the production load is not,
    # the production load is not keeping the weights resident.
    # DROP THIS PROBE'S OWN REFERENCES FIRST. `release_generator` clears
    # the lru_cache and calls empty_cache, but the cache is not the only
    # thing holding the model: these locals are. The first run of this
    # probe kept them and the variant died with "Tried to allocate
    # 14.16 GiB ... 14.45 GiB already in use" — the first copy, still
    # referenced, still resident. Reported at the time as a failed
    # control, which it was; it was also a bug in the control.
    del enc, model, tokenizer
    release_generator()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok2 = AutoTokenizer.from_pretrained(model_id)
        model2 = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map={"": 0},
        )
        model2.eval()
        report["placement_all_on_gpu"] = _placement(model2)
        report["resolved_config_all_on_gpu"] = _resolved_config(model2)
        enc2 = _encode(tok2, model2,
                       width=args.width, prompt_tokens=args.prompt_tokens)
        _timed_call(model2, tok2, enc2, max_new_tokens=4)
        report["variants"]["all_on_gpu"] = _timed_call(
            model2, tok2, enc2, max_new_tokens=args.max_new_tokens,
        )
        print(f"[probe] all_on_gpu       {report['variants']['all_on_gpu']}")
    except Exception as exc:  # noqa: BLE001 - a failed control is data
        report["variants"]["all_on_gpu"] = None
        report["all_on_gpu_error"] = f"{type(exc).__name__}: {exc}"[:300]
        print(f"[probe] all_on_gpu FAILED: {report['all_on_gpu_error']}")

    report["classification"] = classify_per_call_cost(
        {k: (v or {}).get("s_per_decode_step") if v else None
         for k, v in report["variants"].items()}
    )

    print()
    print(f"[probe] VERDICT: {report['classification']['verdict']}")
    print(f"[probe] {report['classification']['explanation']}")
    print(f"[probe] ratios vs healthy baseline: "
          f"{report['classification']['ratios']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[probe] wrote {args.out}")
    print()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
