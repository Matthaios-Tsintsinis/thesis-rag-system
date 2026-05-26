"""CK-4 shared context packer.

`pack_context` enforces a uniform generator-context token budget across
all benchmarked systems. Each system retrieves a deep ranking (up to
RETRIEVAL_RANKING_DEPTH chunks) via `retrieve()`, then `pack_context`
greedily takes top-by-rank chunks until the next would exceed
EVIDENCE_TOKEN_BUDGET when formatted into the chunk-evidence block.

Scope guarantee: this module measures tokens of the FORMATTED CHUNK
STRINGS only (e.g. "[1] {text}\n\n[2] {text}\n\n..."). System-prompt,
question prompt, and per-system structural overhead (M7's aspect
headers + orientation lines) live OUTSIDE the packer's budget. The
analyser's `n_input_tokens` field captures the FULL assembled prompt
tokens, so M7's structural overhead is visible in analysis even
though the packer doesn't enforce it.

Tokenizer: cached `tiktoken.encoding_for_model(EVIDENCE_TOKEN_BUDGET_
TOKENIZER)`. The encoding is fixed across the eval grid; changing it
would invalidate CK-4 budget equality across runs.

CK-2 independence: the packer reads `retrieved` but does NOT modify it.
The caller is responsible for passing the FULL retrieval to the
CK-2 scorer (`score_retrieval_ck2`) and the PACKED subset to the
generator. AnswerResult carries both fields so both invariants hold.
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


def pack_context(
    retrieved: list,  # list[RetrievedChunk]; loosely typed to avoid cycle
    *,
    token_budget: int = EVIDENCE_TOKEN_BUDGET,
    tokenizer_name: str = EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    format_per_chunk: str = _DEFAULT_FORMAT,
    separator: str = _DEFAULT_SEPARATOR,
) -> tuple[list, int, str]:
    """Greedy top-by-rank packer; returns (packed, n_tokens, evidence_block).

    Iterates `retrieved` in rank order. For each chunk, formats with
    `format_per_chunk` (template uses {rank} and {text}) and prepends
    `separator` after the first chunk. Tokenises the cumulative
    evidence block; stops when adding the next chunk would exceed
    `token_budget`.

    Returns:
        packed         — list[RetrievedChunk], top-by-rank subset.
        n_tokens       — int, total tokens in `evidence_block`.
        evidence_block — str, the assembled evidence block ready to
                         drop into a system's user-prompt template.
                         Empty string if no chunks were retrieved.

    Degenerate cases:
      * Empty `retrieved` -> ([], 0, "").
      * First chunk alone exceeds budget -> include it anyway (system
        feeds at least one chunk; otherwise generator sees zero
        evidence — strictly worse than over-budget). Log via stderr.
    """
    enc = _get_tokenizer(tokenizer_name)

    if not retrieved:
        return [], 0, ""

    packed: list = []
    cumulative_text = ""
    cumulative_tokens = 0

    for r in retrieved:
        chunk_str = _format_chunk(r, format_per_chunk=format_per_chunk)
        candidate = (cumulative_text + separator + chunk_str) if cumulative_text else chunk_str
        candidate_tokens = len(enc.encode(candidate))

        if candidate_tokens > token_budget and packed:
            # Out of budget; stop without including this chunk.
            break

        # First-chunk-exceeds-budget special case: include and continue
        # (caller sees the warning in the returned n_tokens > budget).
        if candidate_tokens > token_budget and not packed:
            import sys as _sys
            _sys.stderr.write(
                f"[pack_context] WARN: top-1 chunk alone exceeds budget "
                f"({candidate_tokens} tok > {token_budget}); including anyway.\n"
            )

        packed.append(r)
        cumulative_text = candidate
        cumulative_tokens = candidate_tokens

    return packed, cumulative_tokens, cumulative_text


__all__ = [
    "count_tokens",
    "pack_context",
]
