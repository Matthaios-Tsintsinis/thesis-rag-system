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


@functools.lru_cache(maxsize=2)
def load_generator(
    model_name: str = GENERATOR_MODEL,
    load_in_4bit: bool = True,
) -> Any:
    """Return (tokenizer, model). 4-bit NF4 quant on CUDA via bitsandbytes.

    Only used by the local-HF answer path (`_generate_local`). The
    default eval grid routes to OpenAI via `generate()` and never calls
    this function. Kept for the opt-in local-generator ablation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {"torch_dtype": "auto"}
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
        kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
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
