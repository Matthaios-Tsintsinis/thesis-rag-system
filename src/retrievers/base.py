"""BaseSystem ABC.

The active benchmarked roster is M1, M2, M3, M4, M6, M7; M5 (GraphRAG)
and M8 (hierarchical cluster-tree port) are archived under
src/retrievers/deprecated/. The harness treats each system as a black
box: `index(corpus_path)` once, then `answer(query)` per question.

`retrieve` is exposed separately so the harness can score retrieval
quality (Recall@k, RAGAS context_precision) independently from answer
quality.

M1 (closed-book) returns an empty list from `retrieve` — same interface,
no chunks. This keeps the harness uniform.

`index_items(items)` is an alternative entry point used by the
benchmark eval layer to feed an in-memory list of CorpusItems rather
than a filesystem path. The default implementation writes each item to
a temp directory as a .txt file and calls self.index(temp_dir), then
stamps each produced Chunk's gold_provenance from the per-item
(parent_id, span_id) for CK-2 retrieval-recall scoring. Per-system
overrides skip the disk roundtrip when it proves to be the bottleneck;
do not pre-optimise all six.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

from ..chunking import Chunk
from ..config import (
    BASE_ANSWER_SYSTEM_PROMPT,
    DEFAULT_CONFIG,
    EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    HarnessConfig,
)

if TYPE_CHECKING:
    from ..eval.types import CorpusItem


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    # CK-4: classification of the retrieval unit this RetrievedChunk
    # came from. "chunk" for raw chunk hits (M2/M3/M6 always; M4/M7
    # for direct chunk-position flat-index hits). "summary_low" /
    # "summary_mid" / "summary_high" for M4/M7 flat-index summary-node
    # hits per `FlatCollapsedIndex.node_types`. "multibranch" for M7
    # tree-traversal hits whose origin lies outside the flat index.
    # Default "chunk" preserves back-compat with cached chunks that
    # pre-date CK-4.
    source_unit_type: str = "chunk"


@dataclass
class AnswerResult:
    query: str
    answer: str
    # FULL ranking returned by system.retrieve(). CK-2 retrieval-recall
    # scores against this — the CK-4 packer does not narrow it.
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    # CK-4: the subset of `retrieved` that actually fed the generator
    # after `src.prompt_packing.pack_context` enforced the shared
    # EVIDENCE_TOKEN_BUDGET. Equal to `retrieved` for M1 (no retrieval)
    # and for any system whose retrieved set already fits in budget.
    packed: list[RetrievedChunk] = field(default_factory=list)
    latency_s: float = 0.0
    n_retrieval_calls: int = 0
    # CK-4: full prompt tokens fed to the generator (system + question
    # + assembled evidence + any per-system structural overhead like
    # M7's aspect headers / orientation lines). Captured via tiktoken
    # on the actual prompt text. ANALYSIS visibility — see how much
    # each system's prompt actually weighs.
    n_input_tokens: int = 0
    # CK-4: chunks-only token count of the evidence block AFTER the
    # shared packer enforced EVIDENCE_TOKEN_BUDGET. THIS is the
    # quantity --check-budget-equality measures — the experimental
    # control. For M7 (or any future system) with heavier structural
    # overhead, n_input_tokens will exceed evidence_tokens; the
    # difference IS the per-system structural overhead.
    evidence_tokens: int = 0
    n_output_tokens: int = 0
    extra: dict = field(default_factory=dict)


def _safe_item_filename(item_id: str) -> str:
    """Deterministic 16-hex-char tag for use as a temp filename. The real
    item_id stays on CorpusItem; this only needs to be filesystem-safe
    and unique within one tempdir, NOT human-readable or invertible.
    Bounds length so URL-keyed ids (MultiHop) don't blow Linux's 255-byte
    filename limit.
    """
    return hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:16]


@dataclass
class PreparedQuery:
    """Everything phase A produces for one query, minus the generation.

    The two-phase runner exists because the harness now generates with a
    LOCAL model. Against an API, sequential answering was fine; against
    HF transformers it leaves ~90% of throughput unused, and generation
    is ~90% of the matrix cost. So retrieval (cheap, sequential, GPU-light)
    is separated from generation (expensive, batched).

    `prepare()` does retrieval, any query-time LLM work a system needs,
    context packing and prompt assembly. `finish()` wraps a generated
    string back into an AnswerResult. `answer()` is prepare + generate +
    finish, so the single-query API is unchanged for smoke and for any
    system that has not been converted.
    """

    query: str
    retrieved: list[RetrievedChunk]
    packed: list[RetrievedChunk]
    system_prompt: str
    user_prompt: str
    evidence_tokens: int
    n_input_tokens: int
    n_retrieval_calls: int = 1
    prepare_s: float = 0.0
    extra: dict = field(default_factory=dict)


class BaseSystem(ABC):
    """Abstract benchmarked system."""

    system_id: str = "base"

    # False for systems that still override answer() wholesale and have
    # not been split into prepare/finish (M1 closed-book, M7). The runner
    # falls back to sequential answering for those rather than guessing.
    # M1 is cheap enough not to matter (~100-token prompts); M7 is not in
    # the baseline matrix.
    supports_batched_answer: bool = True

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._indexed: bool = False
        # All chunking systems (M2/M3/M4/M6/M7) populate this list during
        # index(); M1 leaves it empty. Declared on the base so the eval
        # layer's index_items provenance-stamping can find it uniformly.
        self.chunks: list[Chunk] = []

    @abstractmethod
    def index(self, corpus_path: Path) -> None:
        """Parse, chunk, embed, and build whatever structures the system needs."""

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return up to k retrieved chunks, ordered by descending score."""

    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        """Default retrieve -> pack -> generate -> AnswerResult.

        Each system inherits this default UNLESS its generation
        pipeline is materially different (M1 closed-book; M7
        per-aspect structured prompt). Per-system overrides MUST also
        populate AnswerResult.packed and AnswerResult.n_input_tokens
        so the analyser's --check-budget-equality assertion (opt-in)
        is meaningful.

        Flow:
          1. retrieve(query, k=k) — passes through the caller's k if
             given, OTHERWISE lets each system use its NATURAL top_k
             default (M2/M3 self.config.retrieval.top_k, M4
             m4.top_k_final, M6 m6.top_k_final — all
             FINAL_CONTEXT_CHUNKS=15 by default; M7 retrieves up to
             its un-capped quota ~50). This preserves baseline
             natural strength under the no-budget default per
             professor's directive — baselines feed exactly what
             their papers fed.
          2. pack_context(retrieved) — no enforcement when
             src.config.EVIDENCE_TOKEN_BUDGET is None (the default);
             with --evidence-budget opt-in it truncates by token
             count. Either way returns (packed, evidence_tokens,
             evidence_block).
          3. Assemble `Evidence:\\n{evidence}\\n\\nQuestion: {query}`,
             call generate() with BASE_ANSWER_SYSTEM_PROMPT.
          4. Return AnswerResult(retrieved=full, packed=packed,
             evidence_tokens, n_input_tokens=full assembled prompt).
        """
        from ..models import generate

        prepared = self.prepare(query, k=k)
        ans = generate(
            system_prompt=prepared.system_prompt,
            user_prompt=prepared.user_prompt,
            cfg=self.config.generation,
        )
        return self.finish(prepared, ans)

    def prepare(self, query: str, k: int | None = None) -> PreparedQuery:
        """PHASE A: retrieve, pack, assemble the prompt. No generation.

        Systems whose generation context differs materially from "the
        retrieved chunks, packed" override THIS rather than answer() —
        that keeps them batchable. M9 overrides it to run its corrective
        loop and strip refinement; M9's rewrite is an LLM call, so its
        phase A is not LLM-free, which is accepted and expected.
        """
        # Late import to break the retrievers/base <-> prompt_packing
        # circular (prompt_packing's tiktoken init touches no retriever
        # state, so this is safe).
        from ..prompt_packing import count_tokens, pack_context

        self._require_indexed()
        t0 = self._now()
        # Note: NO RETRIEVAL_RANKING_DEPTH override here. Calling
        # retrieve(query, k=None) (or with an explicit caller-passed
        # k) lets each system's NATURAL default top_k govern. This is
        # the professor-aligned "don't expand baselines" guarantee —
        # M2/M3/M4/M6 feed their original top-15 unchanged.
        retrieved = self.retrieve(query, k=k)
        # token_budget=None (default) -> packer reads
        # src.config.EVIDENCE_TOKEN_BUDGET at call time. None by
        # default = no enforcement (baselines unconstrained, per
        # professor's directive). Opt-in for ablation via
        # `python -m src.eval.runner --evidence-budget 3000` which
        # monkey-patches the config constant before answer() runs.
        packed, evidence_tokens, evidence_block = pack_context(
            retrieved,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        user_prompt = f"Evidence:\n{evidence_block}\n\nQuestion: {query}"
        n_input_tokens = count_tokens(
            BASE_ANSWER_SYSTEM_PROMPT + "\n" + user_prompt,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        return PreparedQuery(
            query=query,
            retrieved=retrieved,
            packed=packed,
            system_prompt=BASE_ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            evidence_tokens=evidence_tokens,
            n_input_tokens=n_input_tokens,
            n_retrieval_calls=1,
            prepare_s=self._now() - t0,
        )

    def finish(
        self,
        prepared: PreparedQuery,
        answer_text: str,
        *,
        generate_s: float = 0.0,
    ) -> AnswerResult:
        """PHASE B tail: wrap a generated string into an AnswerResult.

        `generate_s` is the generation time attributable to this query.
        Under batching that is an AMORTISED share of the batch's wall
        clock, not a measured per-query duration — per-query latency is
        not observable once requests are batched, and reporting it as if
        it were would be a fabricated number.
        """
        return AnswerResult(
            query=prepared.query,
            answer=answer_text,
            retrieved=prepared.retrieved,
            packed=prepared.packed,
            latency_s=prepared.prepare_s + generate_s,
            n_retrieval_calls=prepared.n_retrieval_calls,
            n_input_tokens=prepared.n_input_tokens,
            evidence_tokens=prepared.evidence_tokens,
            extra=dict(prepared.extra),
        )

    def index_items(self, items: Sequence["CorpusItem"]) -> None:
        """Index an in-memory list of CorpusItems (benchmark-eval entry point).

        Default fallback: write each item to a temp directory as a
        `{safe_id}.txt` file, call self.index(temp_dir), then iterate
        self.chunks and stamp gold_provenance from the per-item
        (parent_id, span_id) pair using the doc_id->item mapping that
        walk_corpus produces (doc_id = relative path under the temp
        root). The temp directory is deleted when index() returns; the
        in-memory chunks (and any system-specific cached artefacts on
        disk) survive.

        Per-system overrides can avoid the disk roundtrip if profiling
        shows it is the bottleneck — but do not pre-optimise all six
        systems; the temp-dir fallback is correct for all of them
        today, and benchmark indexing is dominated by LLM calls (M4
        summarisation, M6 OpenIE) not by chunk-text I/O.

        Provenance stamping is a no-op for M1 (self.chunks stays
        empty); for chunking systems it sets each chunk's
        gold_provenance to a single-element tuple ((parent_id,
        span_id),). A semantic chunker that fuses neighbouring
        CorpusItems into one chunk would need a per-system override to
        track multi-atom provenance — flagged as a Pass-2 concern
        (Pass-1 uses 1:1 item-to-chunk granularity for QASPER paragraphs
        and MultiHop articles).
        """
        item_by_doc_id: dict[str, "CorpusItem"] = {}
        with tempfile.TemporaryDirectory(prefix=f"{self.system_id}_corpus_") as td:
            td_path = Path(td)
            for item in items:
                safe = _safe_item_filename(item.item_id)
                filename = f"{safe}.txt"
                # On a collision (two items normalise to the same safe
                # name), suffix with a counter so all items survive.
                if filename in item_by_doc_id:
                    n = 1
                    while f"{safe}_{n}.txt" in item_by_doc_id:
                        n += 1
                    filename = f"{safe}_{n}.txt"
                (td_path / filename).write_text(item.text, encoding="utf-8")
                item_by_doc_id[filename] = item
            self.index(td_path)
        # walk_corpus produces ParsedDocument.doc_id = relative_path
        # under the corpus root, which is just the filename for our
        # flat temp dir. Stamp provenance for every chunk that came
        # from a known item; unknown doc_ids (shouldn't happen but
        # defensive) leave gold_provenance untouched (empty tuple).
        for chunk in self.chunks:
            item = item_by_doc_id.get(chunk.doc_id)
            if item is not None:
                chunk.gold_provenance = ((item.parent_id, item.span_id),)

    def _require_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError(f"{self.system_id} not indexed — call .index() first")

    @staticmethod
    def _now() -> float:
        return time.perf_counter()
