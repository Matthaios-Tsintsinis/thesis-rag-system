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

import json
import os
import time
from dataclasses import dataclass
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

# --- M7 query-time prompts (PIPELINE_DESIGN.md §4.1) -----------------------
# Each template has its own _VERSION constant folded into
# summarization_identity() so a wording change invalidates only the
# M7-namespace cache, never the shared RAPTOR substrate (the tree
# summary prompt is the only substrate-affecting prompt).

ASPECT_EXTRACTION_PROMPT_VERSION = "v1"
ASPECT_EXTRACTION_PROMPT_TEMPLATE = (
    "You analyze a search question for a retrieval system. "
    "Classify it as \"simple\" (one information need) or \"multi_aspect\" "
    "(several distinct sub-questions that need separate evidence).\n"
    "For \"simple\": return exactly one aspect named \"main\" with "
    "importance 1.0.\n"
    "For \"multi_aspect\": return up to 3 aspects. Each aspect is a short "
    "self-contained noun phrase or sub-question covering one part of the "
    "answer. Give each an importance in [0,1] reflecting how central it is "
    "to fully answering the question; importances should sum to roughly 1.\n"
    "Answer in the same language as the question. Output ONLY a JSON object, "
    "no prose, no code fences:\n"
    "{{\"type\": \"simple\" | \"multi_aspect\", "
    "\"aspects\": [{{\"name\": string, \"importance\": number}}]}}\n\n"
    "Question: {query}"
)

PARAPHRASE_PROMPT_VERSION = "v1"
PARAPHRASE_PROMPT_TEMPLATE = (
    "Rephrase the following search query so the wording and surface form "
    "change but the information need is preserved exactly. Keep it in the "
    "same language. Output only the rephrased query on a single line, with "
    "no preamble or quotation marks.\n\n"
    "Query: {aspect}"
)

HYDE_PROMPT_VERSION = "v1"
HYDE_PROMPT_TEMPLATE = (
    "Write a short hypothetical passage of 2-3 sentences that would "
    "directly answer the question below, written in the style of an "
    "extract from a relevant source document. Be specific and factual in "
    "tone. Do not hedge and do not state that it is hypothetical. Write "
    "in the same language as the question. Output only the passage.\n\n"
    "Question: {aspect}"
)


# Identity for cache invalidation ------------------------------------------


def summarization_identity(
    model: str = DEFAULT_SUMMARY_MODEL,
    prompt_version: str = SUMMARY_PROMPT_VERSION,
    *,
    aspect_prompt_version: str = ASPECT_EXTRACTION_PROMPT_VERSION,
    paraphrase_prompt_version: str = PARAPHRASE_PROMPT_VERSION,
    hyde_prompt_version: str = HYDE_PROMPT_VERSION,
) -> dict[str, str]:
    """Summariser + M7-prompt identity for M7-namespace cache keys.

    The shared RAPTOR substrate key does NOT use this helper — it folds
    only the tree-summary model + `SUMMARY_PROMPT_VERSION` via
    `raptor.raptor_substrate_extra()`, because the aspect/paraphrase/HyDE
    prompts are query-time only and never alter the cached tree. M7's
    own cache key uses this full identity so bumping any M7 prompt
    version invalidates M7-only artifacts without rebuilding the
    substrate that M4 also depends on.
    """
    return {
        "summary_model": model,
        "summary_prompt_version": prompt_version,
        "aspect_prompt_version": aspect_prompt_version,
        "paraphrase_prompt_version": paraphrase_prompt_version,
        "hyde_prompt_version": hyde_prompt_version,
    }


# OpenAI key resolution ----------------------------------------------------


def _resolve_openai_key() -> str:
    """Env var first, Colab Secrets fallback. Raise with actionable message.

    Env var is checked first because Colab `userdata` does NOT propagate
    secrets to subprocesses spawned via `!python -m ...` — only to
    in-kernel cells. The smoke runner and full-eval driver run as
    subprocesses, so env-first is the order that works in every context
    (notebook-direct and subprocess). The notebook sets
    `os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")`
    once at the top; the userdata branch below remains as a fallback
    for in-kernel use where the env var was not exported.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        key = userdata.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        # Not on Colab, or secret not set; fall through to the error.
        pass

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


def _chat(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
    _label: str = "chat",
) -> str:
    """One single-user-message completion. Returns stripped content.

    Retries on transient errors (rate limits, timeouts). Non-transient
    errors (auth, bad request) propagate immediately so misconfiguration
    surfaces loudly during smoke instead of hiding in retry loops. Shared
    by the RAPTOR tree summariser and every M7 query-time prompt.
    """
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

    raise RuntimeError(
        f"{_label}: exhausted {max_retries} retries against {model}"
    ) from last_exc


def summarize_passages(
    passages: list[str],
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> str:
    """One API call. Returns the RAPTOR node summary text, stripped."""
    if not passages:
        raise ValueError("summarize_passages: passages must be non-empty")

    joined = DEFAULT_PASSAGE_SEPARATOR.join(p.strip() for p in passages if p.strip())
    if not joined:
        raise ValueError("summarize_passages: all passages were empty after strip")

    return _chat(
        SUMMARY_PROMPT_TEMPLATE.format(passages=joined),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        _label="summarize_passages",
    )


# --- M7 query-time helpers (PIPELINE_DESIGN.md §4.1) -----------------------

ASPECT_MAX_TOKENS = 220
PARAPHRASE_MAX_TOKENS = 80
HYDE_MAX_TOKENS = 160


@dataclass(frozen=True)
class Aspect:
    """One answer aspect extracted from the user query (§4.1)."""
    name: str
    importance: float


@dataclass(frozen=True)
class AspectExtraction:
    """Result of query decomposition. `kind` is "simple" or "multi_aspect"."""
    kind: str
    aspects: list[Aspect]


def _strip_json_fences(text: str) -> str:
    """Drop ```json ... ``` / ``` ... ``` fences an LLM may wrap JSON in."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def extract_aspects(
    query: str,
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    max_tokens: int = ASPECT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> AspectExtraction:
    """LLM simple/multi-aspect classification + aspect list (§4.1).

    Raises `ValueError` if the model output is not parseable as the
    expected JSON schema. Callers (the M7 orchestrator) catch this and
    fall back to a single protected `main` aspect — aspect-extractor
    failure must degrade, never crash the pipeline.
    """
    if not query or not query.strip():
        raise ValueError("extract_aspects: query must be non-empty")

    raw = _chat(
        ASPECT_EXTRACTION_PROMPT_TEMPLATE.format(query=query.strip()),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        _label="extract_aspects",
    )

    try:
        obj = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"extract_aspects: unparseable JSON: {raw!r}") from e

    kind = obj.get("type")
    if kind not in ("simple", "multi_aspect"):
        raise ValueError(f"extract_aspects: bad type field: {obj!r}")

    raw_aspects = obj.get("aspects")
    if not isinstance(raw_aspects, list) or not raw_aspects:
        raise ValueError(f"extract_aspects: missing/empty aspects: {obj!r}")

    aspects: list[Aspect] = []
    for a in raw_aspects:
        if not isinstance(a, dict) or "name" not in a or "importance" not in a:
            raise ValueError(f"extract_aspects: malformed aspect: {a!r}")
        name = str(a["name"]).strip()
        if not name:
            raise ValueError(f"extract_aspects: empty aspect name: {a!r}")
        try:
            importance = float(a["importance"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"extract_aspects: non-numeric importance: {a!r}"
            ) from e
        aspects.append(Aspect(name=name, importance=max(0.0, min(1.0, importance))))

    return AspectExtraction(kind=kind, aspects=aspects)


def paraphrase_view(
    aspect_text: str,
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    max_tokens: int = PARAPHRASE_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Intent-preserving reword of an aspect (§4.1 paraphrase view)."""
    if not aspect_text or not aspect_text.strip():
        raise ValueError("paraphrase_view: aspect_text must be non-empty")
    return _chat(
        PARAPHRASE_PROMPT_TEMPLATE.format(aspect=aspect_text.strip()),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        _label="paraphrase_view",
    )


def hyde_view(
    aspect_text: str,
    *,
    model: str = DEFAULT_SUMMARY_MODEL,
    max_tokens: int = HYDE_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Hypothetical 2-3 sentence answer to an aspect (§4.1 HyDE view)."""
    if not aspect_text or not aspect_text.strip():
        raise ValueError("hyde_view: aspect_text must be non-empty")
    return _chat(
        HYDE_PROMPT_TEMPLATE.format(aspect=aspect_text.strip()),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        _label="hyde_view",
    )


__all__ = [
    "DEFAULT_SUMMARY_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "SUMMARY_PROMPT_VERSION",
    "SUMMARY_PROMPT_TEMPLATE",
    "ASPECT_EXTRACTION_PROMPT_VERSION",
    "ASPECT_EXTRACTION_PROMPT_TEMPLATE",
    "PARAPHRASE_PROMPT_VERSION",
    "PARAPHRASE_PROMPT_TEMPLATE",
    "HYDE_PROMPT_VERSION",
    "HYDE_PROMPT_TEMPLATE",
    "Aspect",
    "AspectExtraction",
    "summarization_identity",
    "summarize_passages",
    "extract_aspects",
    "paraphrase_view",
    "hyde_view",
]
