"""Is the generator actually ON the GPU, and what does a summary call cost?

WHY THIS EXISTS. A sweep step built a 4,953-leaf story in 20,691 s — 766
summary calls, ~27 s each. Decode on an L4 is ~50 ms/token, so a
100-token summary should cost ~5 s at batch 1 and far less batched. Two
orders of magnitude of disagreement is not a tuning problem; it is the
model not running where it is supposed to run.

THE LEADING HYPOTHESIS, to be MEASURED here rather than argued:
`load_generator` passes `device_map="auto"` with no `max_memory`, and the
run host has **transformers 5.15.0**, where the `torch_dtype` kwarg the
loader passes was deprecated in 4.x and removed in v5 in favour of
`dtype`. If it is being ignored, the model loads in **fp32** — 7.6 B
params x 4 bytes = ~30 GB against a 22 GB card — so accelerate places
what fits on the GPU and spills the rest to CPU. That single fact would
explain the 21.6 GB resident, the meta-device parameters in the
traceback, and the 27 s/call, all at once.

It is falsifiable in one line: if `model.dtype` reports float16 and no
parameter sits on CPU or meta, the hypothesis is dead and the cost is
somewhere else.

WHAT THIS DELIBERATELY DOES NOT DO: complete a build. It runs a bounded
number of summary calls and stops. The question is s/call and placement,
and paying five hours to learn them again would be its own kind of error.

    python -m scripts.diagnose_generator_placement --cap 8000 --n 50
    python -m scripts.diagnose_generator_placement --cap 4000 --n 50

Run each in a FRESH process. The 8000 sweep step ran immediately after a
16000 OOM, so it started against a fragmented allocator and its numbers
carry that contamination.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def describe_placement() -> dict:
    """Where every parameter actually lives, and in what dtype."""
    import torch

    from src.config import DEFAULT_CONFIG
    from src.models import load_generator

    t0 = time.perf_counter()
    tokenizer, model = load_generator(DEFAULT_CONFIG.generation.model)
    load_s = time.perf_counter() - t0

    by_device: Counter[str] = Counter()
    by_dtype: Counter[str] = Counter()
    n_params_by_device: Counter[str] = Counter()
    for _, p in model.named_parameters():
        dev = str(p.device)
        by_device[dev] += 1
        n_params_by_device[dev] += p.numel()
        by_dtype[str(p.dtype)] += 1

    total_params = sum(n_params_by_device.values())
    off_gpu = sum(n for d, n in n_params_by_device.items()
                  if not d.startswith("cuda"))
    info = {
        "load_s": round(load_s, 1),
        "model_dtype": str(getattr(model, "dtype", "?")),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "params_total_billions": round(total_params / 1e9, 3),
        "param_tensors_by_device": dict(by_device),
        "params_by_device_billions": {
            d: round(n / 1e9, 3) for d, n in n_params_by_device.items()},
        "param_tensors_by_dtype": dict(by_dtype),
        "fraction_params_off_gpu": (
            round(off_gpu / total_params, 4) if total_params else None),
        "hf_device_map": getattr(model, "hf_device_map", None),
        # KV CACHE. generate() is called without use_cache in
        # models.py, so it inherits these. Decoding WITHOUT a cache
        # re-runs the full prefill at every step, which turns a ~2.4 s
        # prefill plus 100 decode steps into ~240 s — and ~236 s is what
        # a summary generate() call actually measured. Reported because
        # that arithmetic fits far too well to leave unchecked, and
        # because transformers 5.x is a major version where a default is
        # worth reading rather than assuming.
        "config_use_cache": getattr(model.config, "use_cache", None),
        "generation_config_use_cache": getattr(
            getattr(model, "generation_config", None), "use_cache", None),
    }
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        info["vram_total_gb"] = round(total_b / 2**30, 2)
        info["vram_free_after_load_gb"] = round(free_b / 2**30, 2)
        info["vram_allocated_gb"] = round(
            torch.cuda.memory_allocated() / 2**30, 2)

    # THE VERDICT, stated rather than left to the reader.
    fp32 = "float32" in info["model_dtype"]
    spilled = (info["fraction_params_off_gpu"] or 0) > 0
    info["verdict"] = (
        "OFFLOADED — parameters are not all on the GPU; every forward pass "
        "crosses PCIe. This is the cost, not the token cap."
        if spilled else
        "FULLY RESIDENT — all parameters on GPU; the cost is elsewhere."
    )
    if info.get("config_use_cache") is False or info.get(
            "generation_config_use_cache") is False:
        info["verdict"] += (
            " ⚠ KV CACHE IS OFF. Every decode step re-runs the full "
            "prefill, which multiplies a 100-token summary by ~100x and "
            "matches the measured 236 s/call almost exactly. This is a "
            "bug, not a memory limit, and fixing it costs nothing "
            "methodologically."
        )
    if fp32:
        info["verdict"] += (
            " AND the model is FP32: the dtype kwarg did not take, which on "
            "transformers 5.x is exactly what a removed `torch_dtype` looks "
            "like. ~30 GB of weights cannot fit 22 GB, so the spill follows."
        )
    return info, tokenizer, model


def time_summary_calls(cap: int, n: int, story_id: str) -> dict:
    """Cost of the FIRST `n` summary calls, then stop. No full build."""
    import torch

    from src.config import DEFAULT_CONFIG
    from src.raptor_paper import split_text_raptor
    from src.summarization import chat_messages  # noqa: F401  (import check)

    from scripts.probe_cell_costs import _one_unit

    _, unit = _one_unit("narrativeqa", story_id)
    text = unit.corpus[0].text
    spans = split_text_raptor(text, max_tokens=DEFAULT_CONFIG.m4.chunker.chunk_words)
    print(f"[diag] story {unit.corpus_id} -> {len(spans)} leaves")

    # Layer-1 clusters average ~7.8 leaves under this build (4953 -> 638),
    # so a realistic summary context is a handful of leaves joined. Build
    # `n` such contexts from the real text rather than synthetic filler:
    # a fixture whose parameters are invented measures the assumption.
    per_context = max(1, len(spans) // 638) if len(spans) > 638 else 1
    contexts = []
    for i in range(n):
        lo = (i * per_context) % max(1, len(spans) - per_context)
        contexts.append("\n\n".join(s.text for s in spans[lo:lo + per_context]))
    approx_tokens = sum(len(c) // 4 for c in contexts) / max(1, len(contexts))
    print(f"[diag] {n} contexts, ~{approx_tokens:.0f} tokens each "
          f"({per_context} leaves per context)")

    from src.raptor_paper import summarize_paper_style_batch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    summaries = summarize_paper_style_batch(
        contexts,
        model=DEFAULT_CONFIG.m4.summary_model,
        max_tokens=DEFAULT_CONFIG.m4.summary_max_tokens,
        batch_size=DEFAULT_CONFIG.m4.summary_batch_size,
        max_padded_tokens=cap,
    )
    dt = time.perf_counter() - t0

    if len(summaries) != n:
        print(f"[diag] FAILED: asked for {n} summaries, got {len(summaries)}")
        sys.exit(1)
    if not any(s.strip() for s in summaries):
        print("[diag] FAILED: every summary is empty; nothing was generated")
        sys.exit(1)

    effective_batch = max(1, int(cap // max(1, approx_tokens)))
    out = {
        "summary_max_padded_tokens": cap,
        "n_calls": n,
        "total_s": round(dt, 1),
        "s_per_call": round(dt / n, 2),
        "approx_context_tokens": int(approx_tokens),
        "effective_batch_est": effective_batch,
        "s_per_batch_est": round(dt / max(1, n / effective_batch), 1),
        "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                         if torch.cuda.is_available() else None),
        "sample_summary": summaries[0][:160],
    }
    print(f"[diag] cap={cap}  {out['s_per_call']} s/call  "
          f"(~{out['s_per_batch_est']} s/batch at est. batch "
          f"{effective_batch})  peak={out['peak_vram_gb']}GB")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--story-id", default="57523a48")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--placement-only", action="store_true")
    args = ap.parse_args(argv)

    report: dict[str, Any] = {"cap": args.cap}
    placement, _tok, _model = describe_placement()
    report["placement"] = placement
    print(json.dumps(placement, indent=2))
    print(f"\n[diag] VERDICT: {placement['verdict']}\n")

    if not args.placement_only:
        report["summary_timing"] = time_summary_calls(
            args.cap, args.n, args.story_id)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[diag] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
