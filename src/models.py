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
    EMBEDDING_DIM,
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
    """L2-normalised embeddings; inner product == cosine."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = load_embedder(model_name)
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


@functools.lru_cache(maxsize=2)
def load_generator(
    model_name: str = GENERATOR_MODEL,
    load_in_4bit: bool = True,
) -> Any:
    """Return (tokenizer, model). 4-bit NF4 quant on CUDA via bitsandbytes."""
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


def generate(
    system_prompt: str,
    user_prompt: str,
    cfg: GenerationConfig | None = None,
) -> str:
    import torch

    cfg = cfg or GenerationConfig()
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
