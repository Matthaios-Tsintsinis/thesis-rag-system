"""Format retrieved chunks into the evidence block the reader sees, and
count tokens with one tokenizer so the counts compare across cells.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .config import EVIDENCE_TOKEN_BUDGET_TOKENIZER

if TYPE_CHECKING:
    from .retrievers.base import RetrievedChunk


@functools.lru_cache(maxsize=4)
def _get_tokenizer(name: str) -> Any:
    """Load the tiktoken encoding for a model name, cl100k_base if unknown."""
    import tiktoken

    try:
        return tiktoken.encoding_for_model(name)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, *, tokenizer_name: str = EVIDENCE_TOKEN_BUDGET_TOKENIZER) -> int:
    """Count the tokens in a text with the evidence tokenizer."""
    enc = _get_tokenizer(tokenizer_name)
    return len(enc.encode(text))


# harness choice: no shared evidence budget (METHODS §D)
_DEFAULT_FORMAT = "[{rank}] {text}"
_DEFAULT_SEPARATOR = "\n\n"


def _format_chunk(r: "RetrievedChunk", *, format_per_chunk: str) -> str:
    """Render one chunk as its 1-based rank plus text."""
    return format_per_chunk.format(rank=r.rank + 1, text=r.chunk.text)


def pack_context(
    retrieved: list,  # list[RetrievedChunk]; untyped to avoid an import cycle
    *,
    tokenizer_name: str = EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    format_per_chunk: str = _DEFAULT_FORMAT,
    separator: str = _DEFAULT_SEPARATOR,
) -> tuple[list, int, str]:
    """Return (packed, n_tokens, evidence_block); packed is the whole list."""
    enc = _get_tokenizer(tokenizer_name)

    if not retrieved:
        return [], 0, ""

    packed: list = []
    cumulative_text = ""
    cumulative_tokens = 0

    # Append every chunk in rank order; nothing is dropped, the token count
    # is the size of the finished block.
    for r in retrieved:
        chunk_str = _format_chunk(r, format_per_chunk=format_per_chunk)
        candidate = (cumulative_text + separator + chunk_str) if cumulative_text else chunk_str
        candidate_tokens = len(enc.encode(candidate))
        packed.append(r)
        cumulative_text = candidate
        cumulative_tokens = candidate_tokens

    return packed, cumulative_tokens, cumulative_text


__all__ = [
    "count_tokens",
    "pack_context",
]
