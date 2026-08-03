"""Lazy model loaders.

Heavy deps (transformers, sentence-transformers) are imported inside
the loaders so `import src.*` stays cheap. Smoke tests on Colab/GPU
pay the cost; harness skeleton on CPU does not.

Loaders are memoised by argument tuple so repeated calls return the
same instance.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from .config import (
    EMBEDDER_MODEL,
    GENERATOR_MODEL,
    GenerationConfig,
    RERANKER_MODEL,
)


# --- Embedder -------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def load_embedder(model_name: str = EMBEDDER_MODEL) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = EMBEDDER_MODEL) -> np.ndarray:
    """L2-normalised embeddings; inner product == cosine.

    The empty-input branch probes the loaded model for its sentence-
    embedding dimension rather than reading a hardcoded constant, so the
    function works correctly across embedders with different native dims
    (e.g. bge-m3 at 1024, multi-qa-mpnet-base-cos-v1 at 768, Contriever
    at 768). load_embedder is lru-cached, so the probe is free after the
    first real call.
    """
    model = load_embedder(model_name)
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


# --- Reranker -------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def load_reranker(model_name: str = RERANKER_MODEL) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank_scores(
    query: str,
    passages: list[str],
    model_name: str = RERANKER_MODEL,
) -> np.ndarray:
    """Raw cross-encoder logits. Apply sigmoid externally for probabilities."""
    if not passages:
        return np.zeros((0,), dtype=np.float32)
    reranker = load_reranker(model_name)
    scores = reranker.predict([(query, p) for p in passages], show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)


# --- Generator ------------------------------------------------------------
#
# Two backends share one `generate()` entrypoint, routed by model name:
#
#   * OpenAI Chat Completions API (gpt-*, o*, chatgpt-*) -- the
#     methodology-locked default. The shared final-answer generator
#     across ALL systems is gpt-4o-mini (professor-confirmed); the
#     index-time summariser / OpenIE LLM is also gpt-4o-mini and goes
#     through `summarization.chat_messages` -- the same client + retry
#     policy this path reuses, so prompt-cache behaviour and rate-limit
#     handling stay consistent across answer-time and index-time calls.
#
#   * Local HuggingFace causal LM (anything else) -- preserved
#     behind an explicit `cfg.model="Qwen/..."` opt-in so a future
#     "local-vs-API answer generator" ablation has a working code path
#     at zero cost. Not used by the default eval grid.


_OPENAI_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "chatgpt-",
    "o1",
    "o3",
    "o4",
)


def _is_openai_model(model_name: str) -> bool:
    """True if `model_name` should route to the OpenAI API path.

    Heuristic on prefix -- avoids hard-coding the exact model id list and
    lets new OpenAI families (o3-*, o4-*, chatgpt-*) work without code
    change. Any name not matching a known OpenAI prefix is treated as a
    local HuggingFace causal-LM id (the opt-in Qwen path).
    """
    name = model_name.strip().lower()
    return any(name.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


def _generate_openai(
    system_prompt: str,
    user_prompt: str,
    cfg: GenerationConfig,
) -> str:
    """OpenAI Chat Completions answer. Reuses summarization.chat_messages
    so the answer-time client / retry / transient-error policy is
    identical to the index-time path.
    """
    # Late import: keeps this module importable in environments where
    # `openai` is not installed (CPU-only smoke that exercises only the
    # local path).
    from .summarization import chat_messages

    return chat_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=cfg.model,
        max_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        _label="generate",
    )


def assert_loaded_generator_matches(
    *,
    requested_name: str,
    loaded_name: str | None,
    want_4bit: bool,
    is_quantized: bool,
    dtype_str: str,
) -> None:
    """Refuse a generator that is not the one the caller asked for.

    THE b6e35c6 LESSON, ENFORCED. That bug ran Qwen2.5-3B locally while
    every report said gpt-4o-mini, and it survived because nothing
    compared the configured generator against the loaded one — the only
    tell was a 6.17GB download in a Colab log. Recording provenance is
    not enough; a mismatch has to ABORT, because the failure is silent
    by construction and every downstream number inherits it.

    Three ways the artifact can differ from the name:
      * a different checkpoint than requested;
      * quantization applied (or skipped) against the config, which
        makes it a materially different model from the one the thesis
        names;
      * an unintended dtype — `torch_dtype="auto"` can yield fp32 or
        bf16 depending on the checkpoint, which changes both numerics
        and VRAM.

    Pure function so the policy is testable without a GPU or a download.
    """
    if loaded_name and loaded_name != requested_name:
        raise RuntimeError(
            f"generator mismatch: requested {requested_name!r} but loaded "
            f"{loaded_name!r}. Refusing to run — this is the b6e35c6 "
            "failure mode (reporting one model while running another)."
        )
    if bool(is_quantized) != bool(want_4bit):
        raise RuntimeError(
            f"quantization mismatch for {requested_name!r}: config asked for "
            f"load_in_4bit={want_4bit} but the loaded model "
            f"{'IS' if is_quantized else 'is NOT'} quantized. A silently "
            "quantized model is not the model the thesis names."
        )
    if not want_4bit and dtype_str not in ("torch.float16", "torch.bfloat16"):
        raise RuntimeError(
            f"unexpected dtype {dtype_str} for {requested_name!r}. "
            "Unquantized local generators must load in fp16/bf16; fp32 "
            "doubles VRAM and changes numerics vs the reported run."
        )


def generator_identity(model_name: str, *, load_in_4bit: bool) -> dict:
    """Runtime identity of the generator, for the manifest / run summary.

    Recorded so a cell can be traced to the exact artifact that produced
    it. GPU class is included because local decoding is not bit-identical
    across GPU generations — see the determinism note in the M4 fidelity
    plan; a tree or an answer set is reproducible against a pinned
    runtime, not absolutely.
    """
    import torch

    gpu = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu"
    )
    return {
        "generator_model": model_name,
        "load_in_4bit": bool(load_in_4bit),
        "torch_version": torch.__version__,
        "gpu": gpu,
    }


@functools.lru_cache(maxsize=2)
def load_generator(
    model_name: str = GENERATOR_MODEL,
    load_in_4bit: bool = False,
) -> Any:
    """Return (tokenizer, model). fp16 by default; 4-bit only on request.

    Used by the local-HF answer path (`_generate_local`). The default is
    now fp16 — see config.LOAD_GENERATOR_IN_4BIT for why the previous
    True default was a latent repeat of b6e35c6. The load is gated by
    `assert_loaded_generator_matches`, which aborts rather than warns.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {}
    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        # EXPLICIT fp16, never torch_dtype="auto" — "auto" resolves from
        # the checkpoint and can hand back fp32 or bf16 without saying so.
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32  # CPU smoke only

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    assert_loaded_generator_matches(
        requested_name=model_name,
        loaded_name=getattr(model.config, "_name_or_path", None),
        want_4bit=bool(load_in_4bit and torch.cuda.is_available()),
        is_quantized=getattr(model.config, "quantization_config", None) is not None,
        dtype_str=str(model.dtype),
    )
    print(
        "[models] loaded local generator: "
        f"{generator_identity(model_name, load_in_4bit=load_in_4bit)}"
    )
    return tokenizer, model


def _generate_local(
    system_prompt: str,
    user_prompt: str,
    cfg: GenerationConfig,
) -> str:
    """Local HuggingFace causal-LM answer (Qwen path). Opt-in only via
    `cfg.model` set to a non-OpenAI model id; kept for a future
    local-vs-API generator ablation.
    """
    import torch

    tokenizer, model = load_generator(cfg.model, cfg.load_in_4bit)

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
    generated = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def deterministic_batch_order(lengths: list[int]) -> tuple[list[int], list[int]]:
    """Length-sorted permutation + its inverse. Pure, so it is testable.

    Batched generation pads every sequence to the batch maximum, so a
    batch mixing a 300-token prompt with a 4,000-token one pays 4,000 for
    both. Sorting by length before batching removes most of that waste.

    The sort key is (length, original_index) and the sort is stable, so
    the permutation is a pure function of the input lengths — never of
    dict iteration order, scores, or arrival timing. That matters beyond
    tidiness: batch COMPOSITION can change generated text (padding and
    batched-matmul reduction order can flip argmax on near-ties), so a
    non-deterministic order would make the run non-reproducible. See
    `generate_batch` for the rest of that argument.

    Returns (order, inverse) with `[out[i] for i in inverse]` restoring
    input order from results computed in `order`.
    """
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
) -> list[list[int]]:
    """Group pre-sorted indices so that n * longest <= max_padded_tokens.

    WHY THIS EXISTS, measured rather than theorised. A fixed batch COUNT
    has to be sized for the worst-case batch, because a batch pads every
    member to its longest one — so the memory cost is `n * max_len`, not
    `sum(len)`. Real prompts are ragged: a synthetic benchmark of uniform
    4,000-token prompts survived batch 8 at 21.7 GB, and real MultiHop
    prompts at the same batch size OOM'd, because their padded width
    exceeded 4,000.

    Worse, length sorting — which is on, and which does reduce total
    padding waste — CONCENTRATES the longest prompts into a single
    batch. That batch sets peak memory. Sorting is a throughput win and
    a peak-memory non-win.

    Bounding padded tokens instead adapts to raggedness: many short
    prompts give a wide batch, a few long ones give a narrow batch, and
    peak memory is bounded by construction rather than by guessing a
    count that happens to fit the longest case. It also means one knob
    covers both context regimes — M4's ~2k prompts and M2/M3/M9's ~4k —
    instead of a separate batch size per system.

    `order` is the length-sorted index order; `lengths[i]` is item i's
    token count. Pure and deterministic, so batch composition is a
    function of the inputs and the budget alone. Note that composition
    therefore depends on the budget: if the batch-invariance probe shows
    composition changes output, the budget is part of the artifact's
    identity wherever the artifact is cached.

    A single item wider than the budget gets its own batch rather than
    being dropped — the same "include it anyway" policy the context
    packer uses, for the same reason.
    """
    if max_padded_tokens < 1:
        raise ValueError("max_padded_tokens must be >= 1")
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
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
    """Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True if unset.

    Must run BEFORE torch initialises CUDA — the variable is read at
    allocator setup and changing it afterwards does nothing.

    Ragged batches are the allocator's worst case: every distinct padded
    width asks for a differently-sized block, and freed blocks of the
    wrong size cannot be reused. The observed OOM had 1.34 GiB
    reserved-but-unallocated, i.e. memory torch was holding but could not
    hand out — expandable segments is the mechanism that targets exactly
    that. Returns the value in force, or None if torch is absent.
    """
    import os

    key = "PYTORCH_CUDA_ALLOC_CONF"
    if not os.environ.get(key):
        os.environ[key] = "expandable_segments:True"
    return os.environ[key]


def release_generator() -> None:
    """Drop the cached generator and return its VRAM.

    `load_generator` is lru_cached, so the model outlives any single
    inspection and keeps ~15 GB resident. That is correct for a run and
    wrong for a probe: it OOM'd a subsequent `python -m src.eval.runner`
    subprocess launched from the same notebook.
    """
    import gc

    load_generator.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def describe_generator_config(model_name: str = GENERATOR_MODEL) -> dict:
    """Architecture facts WITHOUT loading weights. Safe to call anywhere.

    Enough to compute KV-cache arithmetic (layers x kv_heads x head_dim)
    when planning a batch budget. What it cannot tell you is the
    attention implementation, which is resolved at model construction —
    for that you need `describe_generator_runtime`.
    """
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(model_name)
    n_layers = getattr(c, "num_hidden_layers", None)
    n_kv = getattr(c, "num_key_value_heads", None)
    head_dim = getattr(c, "head_dim", None) or (
        (getattr(c, "hidden_size", 0) // getattr(c, "num_attention_heads", 1))
        or None
    )
    kv_bytes_per_token = (
        2 * n_layers * n_kv * head_dim * 2
        if None not in (n_layers, n_kv, head_dim)
        else None
    )
    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_attention_heads": getattr(c, "num_attention_heads", None),
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "max_position_embeddings": getattr(c, "max_position_embeddings", None),
        "vocab_size": getattr(c, "vocab_size", None),
        # fp16 KV per token per sequence; multiply by padded tokens to
        # size a batch budget.
        "kv_bytes_per_token_fp16": kv_bytes_per_token,
    }


def describe_generator_runtime(
    model_name: str = GENERATOR_MODEL,
    *,
    load_in_4bit: bool = False,
    release: bool = True,
) -> dict:
    """Load the generator, report what actually got built, then RELEASE it.

    Exists for the measurement round: attention implementation is the
    single biggest free lever on batch size at 4k context, and it is not
    knowable from config — recent transformers picks SDPA when torch
    supports it, older ones fall back to eager, and eager materialises a
    batch x heads x seq x seq attention matrix that dominates VRAM.

    `release=True` by DEFAULT, and that default is a bug fix: because
    `load_generator` is lru_cached, an earlier version left ~15 GB
    resident in the calling process and OOM'd the next
    `python -m src.eval.runner` subprocess launched from the same
    notebook. A probe must not cost the run that follows it. Pass
    release=False only when the caller intends to generate immediately
    afterwards in the same process.
    """
    import torch

    tokenizer, model = load_generator(model_name, load_in_4bit)
    out = {
        **generator_identity(model_name, load_in_4bit=load_in_4bit),
        "attn_implementation": getattr(
            model.config, "_attn_implementation", "unknown"
        ),
        "dtype": str(model.dtype),
        "n_layers": getattr(model.config, "num_hidden_layers", None),
        "n_kv_heads": getattr(model.config, "num_key_value_heads", None),
        "max_position_embeddings": getattr(
            model.config, "max_position_embeddings", None
        ),
        "tokenizer_padding_side": tokenizer.padding_side,
        "vram_allocated_gb": (
            round(torch.cuda.memory_allocated() / 1e9, 2)
            if torch.cuda.is_available() else None
        ),
        "cuda_alloc_conf": __import__("os").environ.get(
            "PYTORCH_CUDA_ALLOC_CONF"
        ),
    }
    del tokenizer, model
    if release:
        release_generator()
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
    """Batched generation. Returns answers aligned to the INPUT order.

    Phase B of the two-phase runner. Against an API the harness could
    stay sequential; against a local model that leaves ~90% of the
    throughput unused, and the measured cost of the matrix is dominated
    by this call.

    FOUR REQUIREMENTS THAT SILENTLY CORRUPT OUTPUT IF MISSED. Each
    produces plausible-looking text rather than an error, which is why
    they are enforced here rather than left to call sites:

      1. LEFT padding. Decoder-only generation continues from the last
         position, so right-padded rows continue from PAD and emit
         garbage. The tokenizer's padding_side is forced to "left" for
         the duration and restored afterwards.
      2. A real pad token. Many chat checkpoints ship pad_token=None;
         falling back to eos_token is the standard fix.
      3. An explicit attention_mask, so padded positions are not attended.
      4. Correct slicing of the generated span. With LEFT padding every
         row's continuation begins at the same index (the padded input
         width), which is precisely why left padding makes this safe —
         with right padding the offset differs per row and a shared
         slice silently truncates or leaks prompt text.

    DETERMINISM. Batch composition can change generated text even at
    temperature 0. We control batching completely — no continuous
    batching, no arrival-timing dependence — so the run is reproducible
    provided batch_size and the ordering are fixed. `sort_by_length` uses
    the pure, stable key in `deterministic_batch_order`. If the
    batch-invariance probe shows composition matters, batch_size becomes
    part of the artifact's identity and belongs in the cache key of
    anything cached (see raptor_paper.paper_substrate_extra).

    OpenAI-model ids fall back to the sequential API path — batching is a
    local-model concern, and the API already parallelises server-side.
    """
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

    if _is_openai_model(cfg.model):
        return [
            _generate_openai(s, u, cfg)
            for s, u in zip(system_prompts, user_prompts)
        ]

    configure_cuda_allocator()
    import torch

    tokenizer, model = load_generator(cfg.model, cfg.load_in_4bit)

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": s}, {"role": "user", "content": u}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s, u in zip(system_prompts, user_prompts)
    ]

    need_lengths = sort_by_length or max_padded_tokens is not None
    lengths = (
        [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
        if need_lengths
        else [0] * len(texts)
    )
    if sort_by_length:
        # Only the forward permutation is needed: results are written
        # back by original index, so no inverse pass is required.
        order, _ = deterministic_batch_order(lengths)
    else:
        order = list(range(len(texts)))

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

    prev_side = tokenizer.padding_side
    prev_pad = tokenizer.pad_token
    tokenizer.padding_side = "left"          # requirement 1
    if tokenizer.pad_token is None:          # requirement 2
        tokenizer.pad_token = tokenizer.eos_token

    results: list[str] = [""] * len(texts)
    try:
        n_done = 0
        for group_no, idxs in enumerate(groups):
            batch_texts = [texts[i] for i in idxs]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],   # requirement 3
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=cfg.temperature > 0,
                    temperature=max(cfg.temperature, 1e-5),
                    top_p=cfg.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Requirement 4: left padding aligns every row's continuation
            # to the same offset.
            gen = out[:, enc["input_ids"].shape[1] :]
            for i, row in zip(idxs, gen):
                results[i] = tokenizer.decode(
                    row, skip_special_tokens=True
                ).strip()

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
    """Final-answer generator. Routes by `cfg.model` to OpenAI or local.

    Methodology: the shared final-answer generator is gpt-4o-mini (held
    constant across ALL systems per professor's directive). `cfg.model`
    defaults to `GENERATOR_MODEL = "gpt-4o-mini"`, so unless a caller
    explicitly overrides with a non-OpenAI id, every system's answer
    goes through the API path.
    """
    cfg = cfg or GenerationConfig()
    if _is_openai_model(cfg.model):
        return _generate_openai(system_prompt, user_prompt, cfg)
    return _generate_local(system_prompt, user_prompt, cfg)
