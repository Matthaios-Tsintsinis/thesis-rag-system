"""BaseSystem: the black-box contract every benchmarked system (M1-M4)
implements, plus the per-parent corpus layout the eval layer feeds it.
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
    SCORING_RANKING_DEPTH,
    HarnessConfig,
)
from ..parsing import clean_text

if TYPE_CHECKING:
    from ..eval.types import CorpusItem


@dataclass
class RetrievedChunk:
    """One retrieved unit with its score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int
    # Kind of unit this hit came from: "chunk" for a leaf chunk (M2/M3
    # always, M4 for leaf hits); "summary_low" / "summary_mid" /
    # "summary_high" for an M4 summary node, by tree layer.
    source_unit_type: str = "chunk"


@dataclass
class AnswerResult:
    """What one system produces for one query: text, evidence, counts."""

    query: str
    answer: str
    # Full ranking from retrieve(); set-level retrieval metrics score it.
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    # The subset that reaches the reader. The packer imposes no budget,
    # so this equals `retrieved`; M4 applies its paper budget inside
    # retrieve().
    packed: list[RetrievedChunk] = field(default_factory=list)
    # Fixed-depth ranking for rank-aware metrics. Empty for M1. Never
    # reader input.
    scoring_ranking: list[RetrievedChunk] = field(default_factory=list)
    latency_s: float = 0.0
    n_retrieval_calls: int = 0
    # Tokens of the whole prompt (system + evidence + question), counted
    # with tiktoken on the assembled text.
    n_input_tokens: int = 0
    # Tokens of the evidence block alone; the gap to n_input_tokens is
    # the prompt's structural overhead.
    evidence_tokens: int = 0
    n_output_tokens: int = 0
    extra: dict = field(default_factory=dict)


def _safe_item_filename(item_id: str) -> str:
    """Filesystem-safe 16-hex tag for an item id (URL ids exceed 255 bytes)."""
    return hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:16]


# Separator between a parent's members in its temp file. A newline run
# is a sentence boundary to the RAPTOR chunker, so members never fuse
# mid-sentence, and clean_text collapses only runs of three or more, so
# the join survives the read-back unchanged.
_PARENT_JOIN = "\n\n"


def group_items_by_parent(
    items: Sequence["CorpusItem"],
) -> dict[str, list["CorpusItem"]]:
    """Group CorpusItems by parent_id, keeping first-appearance order."""
    groups: dict[str, list["CorpusItem"]] = {}
    for item in items:
        groups.setdefault(item.parent_id, []).append(item)
    return groups


def build_parent_payload(
    members: Sequence["CorpusItem"],
) -> tuple[str, list[tuple[int, int, str]]]:
    """Join a multi-item parent into text plus (start, end, span_id) spans."""
    # Chunk offsets live in clean_text coordinates and clean_text does
    # not distribute over the join, so clean each member first and
    # measure spans on the joined result. Empty members own no text and
    # get no span.
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for item in members:
        text = clean_text(item.text or "")
        if not text:
            continue
        if parts:
            cursor += len(_PARENT_JOIN)
        parts.append(text)
        spans.append((cursor, cursor + len(text), item.span_id))
        cursor += len(text)
    return _PARENT_JOIN.join(parts), spans


@dataclass
class PreparedQuery:
    """Everything prepare() produces for one query: rankings and prompt."""

    query: str
    retrieved: list[RetrievedChunk]
    packed: list[RetrievedChunk]
    # Fixed-depth ranking for rank-aware metrics; never reader input.
    scoring_ranking: list[RetrievedChunk]
    system_prompt: str
    user_prompt: str
    evidence_tokens: int
    n_input_tokens: int
    n_retrieval_calls: int = 1
    prepare_s: float = 0.0
    extra: dict = field(default_factory=dict)


class BaseSystem(ABC):
    """Abstract benchmarked system: index once, then answer per query."""

    system_id: str = "base"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._indexed: bool = False
        # Chunking systems (M2/M3/M4) fill this during index(); M1 leaves
        # it empty. Declared here so provenance stamping finds it.
        self.chunks: list[Chunk] = []

    @abstractmethod
    def index(self, corpus_path: Path) -> None:
        """Parse, chunk, embed and build what the system needs."""

    @abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return up to k retrieved chunks, ordered by descending score."""

    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        """Retrieve, pack, generate and wrap the result; M1 overrides this."""
        from ..models import generate

        # harness choice: one reader across all systems (METHODS §D)
        prepared = self.prepare(query, k=k)
        ans = generate(
            system_prompt=prepared.system_prompt,
            user_prompt=prepared.user_prompt,
            cfg=self.config.generation,
        )
        return self.finish(prepared, ans)

    def retrieve_for_scoring(
        self, query: str, depth: int = SCORING_RANKING_DEPTH
    ) -> list[RetrievedChunk]:
        """Return the fixed-depth ranking that rank-aware metrics score."""
        # harness choice: one scoring depth for every system (METHODS §D)
        # A deeper cut of the same retrieval: one extra search, no LLM.
        return self.retrieve(query, k=depth)

    def prepare(self, query: str, k: int | None = None) -> PreparedQuery:
        """Retrieve, pack and assemble the prompt; no generation."""
        # Late import: prompt_packing imports this module.
        from ..prompt_packing import count_tokens, pack_context

        self._require_indexed()
        t0 = self._now()
        # k=None lets each system's own top_k govern the reader context.
        # harness choice: baselines at natural strength, no imposed budget (METHODS §A.2)
        retrieved = self.retrieve(query, k=k)
        scoring_ranking = self.retrieve_for_scoring(query)
        # Pack the whole retrieval as [N] text blocks.
        # harness choice: no shared evidence budget (METHODS §D)
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
            scoring_ranking=scoring_ranking,
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
        """Wrap generated text and the prepared query into an AnswerResult."""
        return AnswerResult(
            query=prepared.query,
            answer=answer_text,
            retrieved=prepared.retrieved,
            packed=prepared.packed,
            scoring_ranking=prepared.scoring_ranking,
            latency_s=prepared.prepare_s + generate_s,
            n_retrieval_calls=prepared.n_retrieval_calls,
            n_input_tokens=prepared.n_input_tokens,
            evidence_tokens=prepared.evidence_tokens,
            extra=dict(prepared.extra),
        )

    def index_items(self, items: Sequence["CorpusItem"]) -> None:
        """Index in-memory items, one file per parent; stamp provenance."""
        # A system indexes the document the items came from, not the
        # annotation fragments; the fine item granularity buys provenance.
        with tempfile.TemporaryDirectory(prefix=f"{self.system_id}_corpus_") as td:
            td_path = Path(td)
            layout = self._write_corpus_layout(items, td_path)
            self.index(td_path)

        self._stamp_provenance(layout)

    def _write_corpus_layout(
        self, items: Sequence["CorpusItem"], td_path: Path
    ) -> dict[str, tuple[str, list[tuple[int, int, str]] | None, str]]:
        """Write the per-parent corpus into `td_path`; return its layout."""
        # The corpus hash covers this directory (path and bytes), so the
        # warm-substrate check and the real index share this one writer.
        # harness choice: content-addressed substrates (METHODS §D)
        groups = group_items_by_parent(items)
        # filename -> (parent_id, spans | None, span_id_if_single)
        layout: dict[str, tuple[str, list[tuple[int, int, str]] | None, str]] = {}
        for parent_id, members in groups.items():
            # A single-item parent keeps its item-derived filename and its
            # raw bytes, so 1:1 benchmarks hash the same either way and
            # need no offsets. A multi-item parent is pre-cleaned and
            # carries spans; the chunker must see exactly that string.
            if len(members) == 1:
                only = members[0]
                seed, payload = only.item_id, only.text
                spans, single_span = None, only.span_id
            else:
                seed = parent_id
                payload, spans = build_parent_payload(members)
                single_span = ""
                if clean_text(payload) != payload:
                    raise RuntimeError(
                        f"parsing.clean_text is no longer idempotent on "
                        f"parent {parent_id!r}: the per-parent offsets "
                        "would not match the text the chunker reads."
                    )

            # Suffix a colliding filename with the first free counter.
            filename = f"{_safe_item_filename(seed)}.txt"
            if filename in layout:
                n = 1
                while f"{_safe_item_filename(seed)}_{n}.txt" in layout:
                    n += 1
                filename = f"{_safe_item_filename(seed)}_{n}.txt"
            (td_path / filename).write_text(payload, encoding="utf-8")
            layout[filename] = (parent_id, spans, single_span)
        return layout

    # Systems whose index() writes a key-addressed substrate set this
    # True and override `substrate_warm_path`. Only M4 does.
    has_cacheable_substrate = False

    def substrate_warm_path(self, items: Sequence["CorpusItem"]) -> str | None:
        """Path of a complete cached substrate for `items`, else None."""
        # Read-only: never embeds, clusters or summarises. None means
        # "no cacheable substrate", not "checked and cold".
        del items
        return None

    def _stamp_provenance(
        self,
        layout: dict[str, tuple[str, list[tuple[int, int, str]] | None, str]],
    ) -> None:
        """Set each chunk's gold_provenance from the corpus layout."""
        # chunk.doc_id is the path relative to the corpus root, which in
        # the flat temp dir is the filename.
        n_unmapped = 0
        for chunk in self.chunks:
            entry = layout.get(chunk.doc_id)
            if entry is None:
                continue
            parent_id, spans, single_span = entry
            if spans is None:
                chunk.gold_provenance = ((parent_id, single_span),)
                continue
            lo = chunk.metadata.get("start_char")
            hi = chunk.metadata.get("end_char")
            if lo is None or hi is None:
                raise RuntimeError(
                    f"{self.system_id}: chunk {chunk.chunk_id!r} carries no "
                    "start_char/end_char, so per-parent provenance cannot be "
                    "derived. The word_window and raptor_100tok chunkers "
                    "both supply them."
                )
            # Half-open overlap; a chunk crossing an item boundary carries
            # both atoms.
            hits = tuple(
                (parent_id, span_id)
                for start, end, span_id in spans
                if start < hi and lo < end
            )
            if not hits:
                n_unmapped += 1
            chunk.gold_provenance = hits

        if n_unmapped:
            print(
                f"[{self.system_id}] WARNING: {n_unmapped} chunks intersected "
                "no source item and carry empty gold_provenance"
            )
        # Report into the system's index diagnostics where it keeps any
        # (M4 declares `_index_stats`; the others gain no attribute).
        stats = getattr(self, "_index_stats", None)
        if isinstance(stats, dict):
            stats["n_chunks_without_gold_provenance"] = n_unmapped

    def _require_indexed(self) -> None:
        """Raise unless index() has run."""
        if not self._indexed:
            raise RuntimeError(f"{self.system_id} not indexed — call .index() first")

    @staticmethod
    def _now() -> float:
        """Monotonic clock for latency accounting."""
        return time.perf_counter()
