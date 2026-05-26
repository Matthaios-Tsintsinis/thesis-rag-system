"""CK-4 context packer (token budget OPT-IN; default OFF).

`pack_context` formats retrieved chunks into a "[N] {text}" evidence
block. When `token_budget` is None (the default sourced from
`EVIDENCE_TOKEN_BUDGET`), it packs the FULL retrieved list with no
truncation — baselines feed their natural strength to the generator,
per professor's directive on not constraining the baselines.

When `token_budget` is a positive int (set via the CLI ablation flag
`python -m src.eval.runner --evidence-budget 3000`), the packer
becomes a greedy top-by-rank cutoff: take chunks in order until the
next would push the cumulative evidence-block token count over
budget. The cutoff measures FORMATTED CHUNK STRINGS only — system
prompt, question, and per-system structural overhead (M7's aspect
headers + orientation lines) live outside the packer's budget and
surface only in `AnswerResult.n_input_tokens`.

Tokenizer: cached `tiktoken.encoding_for_model(EVIDENCE_TOKEN_BUDGET_
TOKENIZER)`. Fixed across runs; changing it would invalidate
budget-equality assertions across ablation runs.

CK-2 independence: the packer reads `retrieved` but does NOT modify
it. The caller is responsible for passing the FULL retrieval to the
CK-2 scorer (`score_retrieval_ck2`) and the PACKED subset to the
generator. Under the no-budget default `packed == retrieved`.
AnswerResult carries both fields so both invariants hold either way.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable

from .config import EVIDENCE_TOKEN_BUDGET, EVIDENCE_TOKEN_BUDGET_TOKENIZER

if TYPE_CHECKING:
    from .retrievers.base import RetrievedChunk


@functools.lru_cache(maxsize=4)
def _get_tokenizer(name: str) -> Any:
    """Cached tiktoken encoding loader.

    `tiktoken.encoding_for_model` falls back to a default if the model
    name is unknown to tiktoken; we explicitly catch that and prefer
    cl100k_base (gpt-4o-family) to keep budget calc stable.
    """
    import tiktoken

    try:
        return tiktoken.encoding_for_model(name)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, *, tokenizer_name: str = EVIDENCE_TOKEN_BUDGET_TOKENIZER) -> int:
    """One-shot token count for arbitrary text (whole prompts, single
    chunks, anything). Used by the BaseSystem.answer() default to
    populate AnswerResult.n_input_tokens.
    """
    enc = _get_tokenizer(tokenizer_name)
    return len(enc.encode(text))


_DEFAULT_FORMAT = "[{rank}] {text}"
_DEFAULT_SEPARATOR = "\n\n"


def _format_chunk(r: "RetrievedChunk", *, format_per_chunk: str) -> str:
    return format_per_chunk.format(rank=r.rank + 1, text=r.chunk.text)


def _resolve_budget(token_budget: int | None) -> int | None:
    """Pull token_budget from the (possibly monkey-patched) module
    constant when caller passed nothing. Late import so the ablation
    CLI can mutate `src.config.EVIDENCE_TOKEN_BUDGET` before the first
    pack_context call.
    """
    if token_budget is not None:
        return token_budget
    from . import config as _cfg
    return _cfg.EVIDENCE_TOKEN_BUDGET  # may be None — no-budget default


def pack_context(
    retrieved: list,  # list[RetrievedChunk]; loosely typed to avoid cycle
    *,
    token_budget: int | None = None,
    tokenizer_name: str = EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    format_per_chunk: str = _DEFAULT_FORMAT,
    separator: str = _DEFAULT_SEPARATOR,
) -> tuple[list, int, str]:
    """Format retrieved chunks into an evidence block (token budget OPT-IN).

    `token_budget=None` (default) -> no budget enforcement: pack the
    FULL retrieved list, return the full evidence block + its token
    count. This is the professor-aligned default — baselines feed
    their natural strength to the generator unconstrained.

    `token_budget > 0` -> greedy top-by-rank cutoff: take chunks in
    order until the cumulative evidence block would exceed
    `token_budget` tokens when tokenised with
    `tiktoken.encoding_for_model(tokenizer_name)`. Used for opt-in
    context-volume ablation runs (`python -m src.eval.runner
    --evidence-budget 3000`).

    Returns:
        packed         — list[RetrievedChunk]. With no budget,
                         packed == retrieved.
        n_tokens       — int, total tokens in `evidence_block`.
        evidence_block — str, the assembled evidence text. Empty
                         string if no chunks were retrieved.

    Degenerate cases:
      * Empty `retrieved` -> ([], 0, "").
      * (budget mode) first chunk alone exceeds budget -> include it
        anyway with stderr warning; otherwise system feeds zero
        evidence which is strictly worse than over-budget.
    """
    enc = _get_tokenizer(tokenizer_name)
    budget = _resolve_budget(token_budget)

    if not retrieved:
        return [], 0, ""

    packed: list = []
    cumulative_text = ""
    cumulative_tokens = 0

    for r in retrieved:
        chunk_str = _format_chunk(r, format_per_chunk=format_per_chunk)
        candidate = (cumulative_text + separator + chunk_str) if cumulative_text else chunk_str
        candidate_tokens = len(enc.encode(candidate))

        if budget is not None:
            if candidate_tokens > budget and packed:
                # Out of budget; stop without including this chunk.
                break

            # First-chunk-exceeds-budget special case: include and
            # continue (caller sees the warning + n_tokens > budget).
            if candidate_tokens > budget and not packed:
                import sys as _sys
                _sys.stderr.write(
                    f"[pack_context] WARN: top-1 chunk alone exceeds budget "
                    f"({candidate_tokens} tok > {budget}); including anyway.\n"
                )

        packed.append(r)
        cumulative_text = candidate
        cumulative_tokens = candidate_tokens

    return packed, cumulative_tokens, cumulative_text


__all__ = [
    "count_tokens",
    "pack_context",
]
