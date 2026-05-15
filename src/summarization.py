"""Abstractive summarization for RAPTOR-family tree nodes.

M4 (and later M7) needs an LLM-generated summary on every internal node
of the RAPTOR cluster tree per PIPELINE_DESIGN.md §3.4. The summarizer
is fixed at gpt-4o-mini (project decision), with credentials resolved
at call time from Colab Secrets or env var — never baked into the cache,
never logged, never committed.

`SUMMARY_PROMPT_VERSION` is folded into the M4/M7 cache key via
`summarization_identity()`. Bumping the version (e.g. v1 → v2) is the
explicit lever for invalidating every cached tree summary when the
prompt wording changes.

One API call per node — no batching. Simple, debuggable, traceable in
the manifest. Batching is a follow-up optimisation iff real-corpus
indexing turns out to be the wall-clock bottleneck.
"""

from __future__ import annotations

import os
import time
from typing import Any

# Public knobs --------------------------------------------------------------

DEFAULT_SUMMARY_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PASSAGE_SEPARATOR = "\n\n---\n\n"

SUMMARY_PROMPT_VERSION = "v1"
SUMMARY_PROMPT_TEMPLATE = (
    "Summarize the following passages in the same language they are written in. "
    "Output 3-5 sentences of factual content. Do not invent.\n\n"
    "Passages:\n{passages}"
)


# Identity for cache invalidation ------------------------------------------


def summarization_identity(
    model: str = DEFAULT_SUMMARY_MODEL,
    prompt_version: str = SUMMARY_PROMPT_VERSION,
) -> dict[str, str]:
    """Fold into cache key extras alongside chunking/embedder/parser.

    Bumping `SUMMARY_PROMPT_VERSION` or swapping the model invalidates
    every cached tree-summary artifact across systems that share the
    summarization substrate (M4, M7).
    """
    return {
        "summary_model": model,
        "summary_prompt_version": prompt_version,
    }


# OpenAI key resolution ----------------------------------------------------


def _resolve_openai_key() -> str:
    """Colab Secrets first, env var fallback. Raise with actionable message."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        key = userdata.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        # Not on Colab, or secret not set; fall through to env var.
        pass

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    raise RuntimeError(
        "OPENAI_API_KEY not found. Set Colab secret 'OPENAI_API_KEY' "
        "(Colab → key icon → add secret, then enable notebook access), "
        "or `export OPENAI_API_KEY=...` locally."
    )


_CLIENT: Any | None = None


def _get_client() -> Any:
    """Lazily construct an OpenAI client. Cached for the process lifetime."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. `pip install openai` "
            "(>=1.0) is required for RAPTOR summarization."
        ) from e
    _CLIENT = OpenAI(api_key=_resolve_openai_key())
    return _CLIENT


# Core call ----------------------------------------------------------------


def summarize_passages(
    passages: list[str],
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> str:
    """One API call. Returns the summary text, stripped.

    Retries on transient errors (rate limits, timeouts). Non-transient
    errors (auth, bad request) propagate immediately so misconfiguration
    surfaces loudly during smoke instead of hiding in retry loops.
    """
    if not passages:
        raise ValueError("summarize_passages: passages must be non-empty")

    joined = DEFAULT_PASSAGE_SEPARATOR.join(p.strip() for p in passages if p.strip())
    if not joined:
        raise ValueError("summarize_passages: all passages were empty after strip")

    prompt = SUMMARY_PROMPT_TEMPLATE.format(passages=joined)
    client = _get_client()

    # Local imports so the module imports cleanly without openai installed.
    try:
        from openai import APIError, RateLimitError, APITimeoutError
        transient_excs: tuple[type[BaseException], ...] = (
            RateLimitError,
            APITimeoutError,
            APIError,
        )
    except ImportError:
        transient_excs = ()  # client construction already raised; unreachable

    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except transient_excs as e:
            last_exc = e
            if attempt == max_retries - 1:
                break
            time.sleep(retry_backoff_s * (2**attempt))

    # All retries exhausted on a transient error.
    raise RuntimeError(
        f"summarize_passages: exhausted {max_retries} retries against {model}"
    ) from last_exc


__all__ = [
    "DEFAULT_SUMMARY_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "SUMMARY_PROMPT_VERSION",
    "SUMMARY_PROMPT_TEMPLATE",
    "summarization_identity",
    "summarize_passages",
]
