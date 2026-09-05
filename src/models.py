"""Model loaders and generation for the harness: the shared embedder, the
one local reader, and the placement and timing diagnostics the run summary
records.
"""

from __future__ import annotations

import os

# Set the CUDA allocator config before anything imports torch. The
# embedder loads first and the variable is read once at allocator setup,
# so it has to be in place here. setdefault keeps an explicit env value.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import functools
import time
from typing import Any

import numpy as np

from .config import (
    EMBEDDER_MODEL,
    GENERATOR_MAX_MEMORY,
    GENERATOR_MODEL,
    GenerationConfig,
)


# --- Embedder -------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def load_embedder(model_name: str = EMBEDDER_MODEL) -> Any:
    """Load a SentenceTransformer, one instance per model name."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = EMBEDDER_MODEL) -> np.ndarray:
    """L2-normalised float32 embeddings, so inner product equals cosine."""
    model = load_embedder(model_name)
    # Empty input returns shape (0, dim) with dim read from the model.
    if not texts:
        dim = int(model.get_sentence_embedding_dimension())
        return np.zeros((0, dim), dtype=np.float32)
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype(np.float32, copy=False)


# --- Generator ------------------------------------------------------------
#
# One local HuggingFace causal LM (Qwen2.5-7B-Instruct fp16 by default,
# Llama-3.1-8B-Instruct for the second reader column) is every system's
# reader and M4's tree summariser. OpenAI ids are recognised only to be
# refused, here and in M4's index-LLM guard.
# harness choice: one reader across all systems (METHODS §D)


_OPENAI_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "chatgpt-",
    "o1",
    "o3",
    "o4",
)


def _is_openai_model(model_name: str) -> bool:
    """True if `model_name` is an OpenAI API id (prefix match)."""
    name = model_name.strip().lower()
    return any(name.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


_API_PATH_REMOVED = (
    "generator {model!r} names an OpenAI API model, and the API answer "
    "path was removed in the repo reduction: every matrix cell runs a "
    "LOCAL HuggingFace model (tag thesis-full-2026-09-03 keeps the old "
    "path)."
)


def assert_loaded_generator_matches(
    *,
    requested_name: str,
    loaded_name: str | None,
    is_quantized: bool,
    dtype_str: str,
) -> None:
    """Raise unless the loaded checkpoint, quantisation and dtype match."""
    # Three ways the loaded model can differ from the name it reports
    # under: another checkpoint, a quantised weight set, or a dtype other
    # than the fp16/bf16 the harness loads.
    if loaded_name and loaded_name != requested_name:
        raise RuntimeError(
            f"generator mismatch: requested {requested_name!r} but loaded "
            f"{loaded_name!r}. Refusing to run — this is the b6e35c6 "
            "failure mode (reporting one model while running another)."
        )
    if is_quantized:
        raise RuntimeError(
            f"quantization mismatch for {requested_name!r}: the harness "
            "loads fp16 only but the loaded model IS quantized. A silently "
            "quantized model is not the model the thesis names."
        )
    if dtype_str not in ("torch.float16", "torch.bfloat16"):
        raise RuntimeError(
            f"unexpected dtype {dtype_str} for {requested_name!r}. "
            "Unquantized local generators must load in fp16/bf16; fp32 "
            "doubles VRAM and changes numerics vs the reported run."
        )


def generator_identity(model_name: str) -> dict:
    """Model name, torch version and GPU class, for the run summary."""
    import torch

    # The GPU class is recorded because local decoding is not
    # bit-identical across GPU generations.
    gpu = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu"
    )
    return {
        "generator_model": model_name,
        "torch_version": torch.__version__,
        "gpu": gpu,
    }


def assert_generator_fully_resident(
    placement: dict,
    *,
    model_name: str,
    cuda_available: bool,
    allow_offload: bool = False,
) -> None:
    """Raise if any parameters sit off the GPU on a CUDA host."""
    # Without CUDA everything lives on the CPU and there is nothing to
    # check; a caller may also opt in to offload.
    if not cuda_available or allow_offload:
        return
    frac = placement.get("fraction_params_off_gpu")
    if frac is None or frac <= 0:
        return
    devices = placement.get("param_tensors_by_device", {})
    raise RuntimeError(
        f"GENERATOR NOT FULLY RESIDENT: {frac:.2%} of {model_name}'s "
        f"parameters are off the GPU (tensors by device: {devices}). "
        "Every decode step would stream those weights across PCIe, which "
        "measured a 33x slowdown and a flat ~230 s per generate() call. "
        "CAUSE, almost always: the generator was ALREADY loaded once and "
        "a second load landed in VRAM that was no longer empty, so "
        "device_map='auto' spilled the remainder. Check "
        "len(GENERATOR_LOADS) — it should be 1. Pass allow_offload=True "
        "only if you intend to pay this."
    )


def _load_generator_impl(model_name: str) -> Any:
    """Load tokenizer and model, fp16 on the GPU, and check what arrived."""
    # Every local load passes through here, and this is the last moment
    # before CUDA allocates anything.
    configure_cuda_allocator()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Explicit fp16, never "auto", which can resolve to fp32 or bf16 from
    # the checkpoint. With "cpu": "0GiB" a load that does not fit raises
    # instead of spilling weights to the CPU.
    # harness choice: one reader across all systems (METHODS §D)
    kwargs: dict = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = dict(GENERATOR_MAX_MEMORY)
    else:
        kwargs["torch_dtype"] = torch.float32  # CPU-only hosts

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    # Refuse a checkpoint, quantisation or dtype other than requested.
    assert_loaded_generator_matches(
        requested_name=model_name,
        loaded_name=getattr(model.config, "_name_or_path", None),
        is_quantized=getattr(model.config, "quantization_config", None) is not None,
        dtype_str=str(model.dtype),
    )
    return tokenizer, model


@functools.lru_cache(maxsize=2)
def _load_generator_cached(model_name: str) -> Any:
    """Load on a cache miss, record the load, then check its placement."""
    tokenizer, model = _load_generator_impl(model_name)

    # Record the load; this body runs once per cache miss, so a second
    # entry means a real reload.
    placement = model_placement_snapshot(model)
    GENERATOR_LOADS.append({
        "model": model_name,
        "memory_after_load": cuda_memory_snapshot(),
        "placement_at_load": placement,
    })
    # The identity block is best-effort: a log line must not take a load
    # down.
    try:
        identity: Any = generator_identity(model_name)
    except Exception:  # noqa: BLE001
        identity = {"generator_model": model_name}
    print(
        f"[models] loaded local generator: {identity} "
        f"(load #{len(GENERATOR_LOADS)} this process)"
    )

    # Read the placement snapshot and abort on any off-GPU weights.
    cuda = False
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        pass
    assert_generator_fully_resident(
        placement, model_name=model_name, cuda_available=cuda,
    )
    return tokenizer, model


def load_generator(model_name: str = GENERATOR_MODEL) -> Any:
    """Return (tokenizer, model), fp16, cached once per model name."""
    # One canonical key (the name as str) so every call site shares one
    # cache entry and the model loads once per process.
    return _load_generator_cached(str(model_name))


# Callers use cache_clear() / cache_info() on the public name; the cache
# lives on the inner function, so forward the attributes.
load_generator.cache_clear = _load_generator_cached.cache_clear  # type: ignore[attr-defined]
load_generator.cache_info = _load_generator_cached.cache_info  # type: ignore[attr-defined]


def _generate_local(
    system_prompt: str,
    user_prompt: str,
    cfg: GenerationConfig,
) -> str:
    """Answer one prompt with the local model."""
    import torch

    tokenizer, model = load_generator(cfg.model)

    # Build the chat prompt and generate the continuation.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated span, after the prompt.
    generated = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def deterministic_batch_order(lengths: list[int]) -> tuple[list[int], list[int]]:
    """Stable length-sorted permutation and its inverse."""
    # Key (length, index): the order depends on the lengths alone, and
    # `[out[i] for i in inverse]` restores input order.
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    inverse = [0] * len(order)
    for position, original in enumerate(order):
        inverse[original] = position
    return order, inverse


def token_budget_batches(
    order: list[int],
    lengths: list[int],
    *,
    max_padded_tokens: int,
    max_batch_size: int | None = None,
    reserve_tokens_per_seq: int = 0,
) -> list[list[int]]:
    """Group sorted indices so that n * longest <= max_padded_tokens."""
    # A batch pads every member to its longest one, so the padded width
    # bounds memory, not the count. Grouping is a pure function of the
    # order, the lengths and the budget, because batch composition can
    # move generated text even at temperature 0.
    if max_padded_tokens < 1:
        raise ValueError("max_padded_tokens must be >= 1")
    if reserve_tokens_per_seq < 0:
        raise ValueError("reserve_tokens_per_seq must be >= 0")
    # The reserve is headroom for the tokens generation appends; the KV
    # cache grows with every decoded token. Default 0; callers opt in.
    lengths = [n + reserve_tokens_per_seq for n in lengths]
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    # Close the batch when the padded width or the count would overflow.
    # A lone item wider than the budget still gets its own batch.
    for idx in order:
        candidate_max = max(current_max, lengths[idx])
        n_after = len(current) + 1
        over_tokens = n_after * candidate_max > max_padded_tokens
        over_count = (
            max_batch_size is not None and n_after > max_batch_size
        )
        if current and (over_tokens or over_count):
            batches.append(current)
            current, current_max = [], 0
            candidate_max = lengths[idx]
        current.append(idx)
        current_max = candidate_max
    if current:
        batches.append(current)
    return batches


def configure_cuda_allocator() -> str | None:
    """Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True if unset."""
    import os

    # Only effective before torch initialises CUDA; the allocator reads
    # the variable once. Expandable segments let ragged padded widths
    # reuse freed blocks. Returns the value in force.
    key = "PYTORCH_CUDA_ALLOC_CONF"
    if not os.environ.get(key):
        os.environ[key] = "expandable_segments:True"
    return os.environ[key]


# Per-process record of every model.generate() call: count, widths and the
# per-call records. Module-level so the tree builder can read it without
# threading a counter through the signatures.
GENERATE_CALLS: dict = {}


def reset_generate_calls() -> None:
    """Clear the per-process generate() call record."""
    GENERATE_CALLS.clear()


def cuda_memory_snapshot() -> dict:
    """Allocator and device memory right now; None fields without CUDA."""
    out: dict = {
        "cuda_available": False,
        "allocated_gb": None,
        "reserved_gb": None,
        "reserved_minus_allocated_gb": None,
        "free_gb": None,
        "total_gb": None,
    }
    # reserved - allocated is memory torch holds but cannot hand out;
    # free_gb is what the device itself has left.
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        allocated = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        free_b, total_b = torch.cuda.mem_get_info()
        out.update(
            cuda_available=True,
            allocated_gb=round(allocated, 3),
            reserved_gb=round(reserved, 3),
            reserved_minus_allocated_gb=round(reserved - allocated, 3),
            free_gb=round(free_b / 2**30, 3),
            total_gb=round(total_b / 2**30, 3),
        )
    except Exception:  # noqa: BLE001 - a diagnostic must never be fatal
        pass
    return out


def model_placement_snapshot(model) -> dict:  # noqa: ANN001
    """Parameter tensors and counts by device, as placed right now."""
    from collections import Counter

    # Walk the parameters and read their devices; no data is touched.
    tensors: Counter[str] = Counter()
    params: Counter[str] = Counter()
    try:
        for _, p in model.named_parameters():
            dev = str(p.device)
            tensors[dev] += 1
            params[dev] += p.numel()
    except Exception:  # noqa: BLE001
        pass
    total = sum(params.values())
    off = sum(n for d, n in params.items() if not d.startswith("cuda"))
    return {
        "param_tensors_by_device": dict(tensors),
        "params_by_device_billions": {
            d: round(n / 1e9, 3) for d, n in params.items()
        },
        "fraction_params_off_gpu": (
            round(off / total, 4) if total else None
        ),
        "hf_device_map": getattr(model, "hf_device_map", None),
        "attn_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
    }


# One entry per real load in this process. A second entry means a
# reload into whatever VRAM is free at that later moment.
GENERATOR_LOADS: list = []


def record_generate_call(
    *,
    width: int,
    prompt_tokens_padded: int,
    new_tokens: int,
    max_new_tokens: int,
    tokenise_s: float,
    generate_s: float,
    decode_s: float,
    placement: dict | None = None,
    memory: dict | None = None,
) -> dict:
    """Record one model.generate() call: shape, phase timings, placement."""
    # Three timings and the batch shape separate prefill, decode steps
    # and CPU-side work; new_tokens is the steps the loop ran, which is
    # the cap only when some member never emits EOS.
    call = {
        "call_no": int(GENERATE_CALLS.get("n_calls", 0)) + 1,
        "width": int(width),
        "prompt_tokens_padded": int(prompt_tokens_padded),
        # What the padded-token budget bounds.
        "padded_input_cells": int(width) * int(prompt_tokens_padded),
        "new_tokens": int(new_tokens),
        "max_new_tokens": int(max_new_tokens),
        "tokenise_s": round(float(tokenise_s), 4),
        "generate_s": round(float(generate_s), 4),
        "decode_s": round(float(decode_s), 4),
        "total_s": round(
            float(tokenise_s) + float(generate_s) + float(decode_s), 4
        ),
        # None when nothing was emitted, not 0.0 and not an exception.
        "s_per_decode_step": (
            round(float(generate_s) / int(new_tokens), 6)
            if int(new_tokens) > 0
            else None
        ),
        "s_per_seq": (
            round(
                (float(tokenise_s) + float(generate_s) + float(decode_s))
                / int(width),
                4,
            )
            if int(width) > 0
            else None
        ),
        # Whole snapshots; None means none was taken.
        "placement": dict(placement) if placement is not None else None,
        "memory": dict(memory) if memory is not None else None,
    }
    # Counters beside the records, for consumers that read them.
    GENERATE_CALLS["n_calls"] = int(GENERATE_CALLS.get("n_calls", 0)) + 1
    GENERATE_CALLS.setdefault("widths", []).append(int(width))
    GENERATE_CALLS.setdefault("calls", []).append(call)
    return call


def generate_calls_summary() -> dict:
    """Copy of GENERATE_CALLS plus width aggregates and the load history."""
    # Copy the whole dict so a field added later reaches the consumer
    # without this function changing.
    widths = [int(w) for w in GENERATE_CALLS.get("widths", [])]
    out = dict(GENERATE_CALLS)
    out["n_calls"] = int(GENERATE_CALLS.get("n_calls", 0))
    out["widths"] = widths
    out["calls"] = list(GENERATE_CALLS.get("calls", []))
    out["mean_width"] = (
        round(sum(widths) / len(widths), 2) if widths else None
    )
    out["max_width"] = max(widths) if widths else None
    out["min_width"] = min(widths) if widths else None
    # The load history sits next to the calls so the two are read together.
    out["generator_loads"] = list(GENERATOR_LOADS)
    out["n_generator_loads"] = len(GENERATOR_LOADS)
    return out


def generate_batch(
    system_prompts: list[str],
    user_prompts: list[str],
    cfg: GenerationConfig | None = None,
    *,
    batch_size: int = 8,
    sort_by_length: bool = True,
    progress_every: int = 0,
    max_padded_tokens: int | None = None,
) -> list[str]:
    """Batched generation; answers come back aligned to the input order."""
    cfg = cfg or GenerationConfig()
    if len(system_prompts) != len(user_prompts):
        raise ValueError(
            f"prompt list length mismatch: {len(system_prompts)} system vs "
            f"{len(user_prompts)} user"
        )
    if not user_prompts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    # Only the local model serves; an OpenAI id is refused.
    if _is_openai_model(cfg.model):
        raise ValueError(_API_PATH_REMOVED.format(model=cfg.model))

    configure_cuda_allocator()
    import torch

    tokenizer, model = load_generator(cfg.model)

    # Render every prompt through the chat template.
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s, u in zip(system_prompts, user_prompts)
    ]

    # Choose the processing order: a stable length sort, or input order.
    # Only the forward permutation is needed; results are written back by
    # original index.
    need_lengths = sort_by_length or max_padded_tokens is not None
    lengths = (
        [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
        if need_lengths
        else [0] * len(texts)
    )
    if sort_by_length:
        order, _ = deterministic_batch_order(lengths)
    else:
        order = list(range(len(texts)))

    # Group by padded-token budget when one is given, else by count.
    if max_padded_tokens is not None:
        groups = token_budget_batches(
            order, lengths,
            max_padded_tokens=max_padded_tokens,
            max_batch_size=batch_size,
        )
    else:
        groups = [
            order[s : s + batch_size] for s in range(0, len(order), batch_size)
        ]

    # Decoder-only batches need left padding (so every row continues from
    # its last real token and shares one slice offset), a real pad token,
    # and an attention mask. The tokenizer state is restored afterwards.
    prev_side = tokenizer.padding_side
    prev_pad = tokenizer.pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[str] = [""] * len(texts)
    try:
        n_done = 0
        # CUDA work is queued asynchronously, so each phase is bracketed
        # by a sync or the timings would attribute one phase's work to
        # the next.
        def _sync() -> None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Tokenise, generate and decode each group, timing every phase and
        # recording the call.
        for group_no, idxs in enumerate(groups):
            batch_texts = [texts[i] for i in idxs]

            _sync()
            t_tok = time.perf_counter()
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            _sync()
            tokenise_s = time.perf_counter() - t_tok

            # Placement and memory read right before the call, with the
            # batch tensors already resident.
            placement = model_placement_snapshot(model)
            memory = cuda_memory_snapshot()

            t_gen = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=cfg.temperature > 0,
                    temperature=max(cfg.temperature, 1e-5),
                    top_p=cfg.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            _sync()
            generate_s = time.perf_counter() - t_gen

            t_dec = time.perf_counter()
            # Left padding puts every row's continuation at the same
            # offset, so one slice serves the whole batch.
            gen = out[:, enc["input_ids"].shape[1] :]
            for i, row in zip(idxs, gen):
                results[i] = tokenizer.decode(
                    row, skip_special_tokens=True
                ).strip()
            decode_s = time.perf_counter() - t_dec

            call = record_generate_call(
                width=len(idxs),
                prompt_tokens_padded=int(enc["input_ids"].shape[1]),
                # Actual decode steps, not the cap: a batch runs until
                # every member has emitted EOS.
                new_tokens=int(gen.shape[1]),
                max_new_tokens=int(cfg.max_new_tokens),
                tokenise_s=tokenise_s,
                generate_s=generate_s,
                decode_s=decode_s,
                placement=placement,
                memory=memory,
            )
            if progress_every:
                print(
                    f"[generate_batch] call {call['call_no']} "
                    f"w={call['width']} "
                    f"prompt={call['prompt_tokens_padded']} "
                    f"new={call['new_tokens']}/{call['max_new_tokens']}  "
                    f"tok={call['tokenise_s']:.2f}s "
                    f"gen={call['generate_s']:.2f}s "
                    f"dec={call['decode_s']:.2f}s  "
                    f"s/step={call['s_per_decode_step']}"
                )
                print(
                    f"[generate_batch]   off_gpu="
                    f"{placement['fraction_params_off_gpu']} "
                    f"alloc={memory['allocated_gb']}GB "
                    f"reserved={memory['reserved_gb']}GB "
                    f"(reserved-alloc={memory['reserved_minus_allocated_gb']}) "
                    f"free={memory['free_gb']}GB  "
                    f"generator_loads={len(GENERATOR_LOADS)}"
                )

            n_done += len(idxs)
            if progress_every and group_no % progress_every == 0:
                print(f"[generate_batch] {n_done}/{len(order)}")
    finally:
        tokenizer.padding_side = prev_side
        tokenizer.pad_token = prev_pad

    return results


def generate(
    system_prompt: str,
    user_prompt: str,
    cfg: GenerationConfig | None = None,
) -> str:
    """Answer one prompt with the shared local reader; OpenAI ids refused."""
    cfg = cfg or GenerationConfig()
    if _is_openai_model(cfg.model):
        raise ValueError(_API_PATH_REMOVED.format(model=cfg.model))
    return _generate_local(system_prompt, user_prompt, cfg)
