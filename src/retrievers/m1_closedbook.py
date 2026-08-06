"""M1 — Closed-book LLM baseline.

No retrieval. The generator answers from parametric knowledge alone.
Establishes the floor for retrieval-based systems (per evaluation plan).
"""

from __future__ import annotations

from pathlib import Path

from ..config import (
    CLOSED_BOOK_SYSTEM_PROMPT,
    DEFAULT_CONFIG,
    EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    HarnessConfig,
)
from ..prompt_packing import count_tokens
from .base import BaseSystem, PreparedQuery, RetrievedChunk


class ClosedBookSystem(BaseSystem):
    system_id = "M1"
    # BATCHABLE as of the 1-token probe (2026-08-05). The previous value
    # was False, justified by "M1's prompts are ~100 tokens so batching
    # buys almost nothing" — which is PREFILL reasoning, and prefill is
    # not what M1 pays.
    #
    # MEASURED, generating exactly one token so the fixed cost is
    # isolated: M1 0.106 s/query against M2's 1.496 s at 4k. So M1's
    # ~4.1 s/query is ~97% DECODE, making it the purest decode-bound
    # system in the matrix — the same regime as the index summaries that
    # batch 13x, and the regime where a batched decode step serves the
    # whole batch for the price of one.
    #
    # CAVEAT, and it is why the forecast for this is labelled a
    # projection: batching measurably LOST at 4k (25 s for a batch of 5)
    # and that failure is still unexplained. The prefill split does not
    # account for it. M1 is a different regime and should not inherit
    # that result, but it has not yet been measured either — treat the
    # win as unverified until a batched M1 pass is timed.
    supports_batched_answer = True

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)

    def index(self, corpus_path: Path) -> None:
        # No retrieval index needed. Mark indexed so the harness contract
        # holds; the corpus is intentionally unused.
        del corpus_path
        self._indexed = True

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        del query, k
        return []

    def prepare(self, query: str, k: int | None = None) -> PreparedQuery:
        """PHASE A for M1: assemble the prompt. There is nothing to retrieve.

        Overriding prepare/finish rather than answer() is what makes M1
        batchable — the two-phase runner calls prepare() for every query,
        then hands all the prompts to `generate_batch` at once. The
        BaseSystem default cannot be reused because it retrieves and
        packs evidence, and M1 has neither: its prompt is the bare
        closed-book system prompt plus the question.

        Per CK-4 this still populates n_input_tokens so the analyser's
        --check-budget-equality keeps a meaningful M1 column. M1's
        evidence-block count is 0 by definition; n_input_tokens is
        system + query only.
        """
        del k
        self._require_indexed()
        t0 = self._now()
        n_input_tokens = count_tokens(
            CLOSED_BOOK_SYSTEM_PROMPT + "\n" + query,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        return PreparedQuery(
            query=query,
            retrieved=[],
            packed=[],
            system_prompt=CLOSED_BOOK_SYSTEM_PROMPT,
            user_prompt=query,
            evidence_tokens=0,  # no retrieval, no evidence block
            n_input_tokens=n_input_tokens,
            n_retrieval_calls=0,
            prepare_s=self._now() - t0,
        )
