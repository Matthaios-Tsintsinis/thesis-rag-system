"""Evidence-block packer.

`pack_context` formats retrieved chunks into a "[N] {text}" evidence
block and returns the FULL retrieved list with no truncation — baselines
feed their natural strength to the generator (locked decision 4; the
opt-in token budget the packer once implemented left in the repo
reduction). M4's 2,000-token paper budget is applied by M4 itself,
inside its retrieve(), before anything reaches this packer.

Tokenizer: cached `tiktoken.encoding_for_model(EVIDENCE_TOKEN_BUDGET_
TOKENIZER)`, fixed across runs, so `evidence_tokens` and
`n_input_tokens` are comparable across cells.

CK-2 independence: the packer reads `retrieved` but does NOT modify
it. The caller passes the FULL retrieval to the CK-2 scorer
(`score_retrieval_ck2`) and the packed list to the generator; here
`packed == retrieved` always, and AnswerResult carries both fields.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .config import EVIDENCE_TOKEN_BUDGET_TOKENIZER

if TYPE_CHECKING:
    from .retrievers.base import RetrievedChunk


@functools.lru_cache(maxsize=4)
def _get_tokenizer(name: str) -> Any:
    """Cached tiktoken encoding loader.

    `tiktoken.encoding_for_model` falls back to a default if the model
    name is unknown to tiktoken; we explicitly catch that and prefer
    cl100k_base (gpt-4o-family) to keep the count stable.
    """
    import tiktoken

    try:
        return tiktoken.encoding_for_model(name)
    except (KeyError, ValueError):
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, *, tokenizer_name: str = EVIDENCE_TOKEN_BUDGET_TOKENIZER) -> int:
    """One-shot token count for arbitrary text (whole prompts, single
    chunks, anything). Used by the BaseSystem.answer() default to
    populate AnswerResult.n_input_tokens and by the runner's fixed-cap
    output check.
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
    tokenizer_name: str = EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    format_per_chunk: str = _DEFAULT_FORMAT,
    separator: str = _DEFAULT_SEPARATOR,
) -> tuple[list, int, str]:
    """Format retrieved chunks into an evidence block, no budget.

    Returns:
        packed         — list[RetrievedChunk], == retrieved.
        n_tokens       — int, total tokens in `evidence_block`.
        evidence_block — str, the assembled evidence text. Empty
                         string if no chunks were retrieved.

    Empty `retrieved` -> ([], 0, "").
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
        packed.append(r)
        cumulative_text = candidate
        cumulative_tokens = candidate_tokens

    return packed, cumulative_tokens, cumulative_text


__all__ = [
    "count_tokens",
    "pack_context",
]
