"""BaseSystem ABC.

The benchmarked roster is M1, M2, M3, M4 (the withdrawn and archived
systems live at tag thesis-full-2026-09-03). The harness treats each
system as a black box: `index(corpus_path)` once, then `answer(query)`
per question.

`retrieve` is exposed separately so the harness can score retrieval
quality (Recall@k, RAGAS context_precision) independently from answer
quality.

M1 (closed-book) returns an empty list from `retrieve` — same interface,
no chunks. This keeps the harness uniform.

`index_items(items)` is an alternative entry point used by the
benchmark eval layer to feed an in-memory list of CorpusItems rather
than a filesystem path. It writes ONE FILE PER PARENT to a temp
directory, calls self.index(temp_dir), and stamps each produced Chunk's
gold_provenance for CK-2 retrieval-recall scoring — by character-span
intersection where a parent holds several items, and directly from the
single (parent_id, span_id) pair where it holds one.

The per-parent layout was M4-local until it was promoted here (see
`index_items` for why it is safe to share). Per-system overrides can
still skip the disk roundtrip if profiling ever shows it is the
bottleneck; do not pre-optimise all six.
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
    # P6: fixed-depth ranking for rank-aware scoring. Empty for systems
    # that do not retrieve (M1). NOT generator input.
    scoring_ranking: list[RetrievedChunk] = field(default_factory=list)
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


# Separator between a parent's member items in the per-parent temp file.
# Two newlines because `split_text_raptor` treats a newline RUN as a
# sentence boundary that it then drops, so members never fuse into one
# chunk mid-sentence, and `parsing.clean_text` collapses only runs of
# THREE or more — so this survives the read-back unchanged.
_PARENT_JOIN = "\n\n"


def group_items_by_parent(
    items: Sequence["CorpusItem"],
) -> dict[str, list["CorpusItem"]]:
    """Group CorpusItems by parent_id, first-appearance order preserved.

    Order matters twice over: it fixes the temp-dir write order (and so
    the collision-suffix assignment) and the order of the provenance
    pairs on a chunk. Both must be a function of the input alone.
    """
    groups: dict[str, list["CorpusItem"]] = {}
    for item in items:
        groups.setdefault(item.parent_id, []).append(item)
    return groups


def build_parent_payload(
    members: Sequence["CorpusItem"],
) -> tuple[str, list[tuple[int, int, str]]]:
    """Concatenate a multi-item parent, returning its text and member spans.

    Returns (text, [(start_char, end_char, span_id), ...]) with offsets
    into the returned text.

    THE CLEANING TRAP, measured rather than assumed. `walk_corpus` does
    not hand the chunker the file bytes — it hands it
    `parsing.clean_text(bytes)`, which collapses ` \\t` runs and 3+
    newlines and strips the ends. So chunk offsets live in CLEANED
    coordinates, and naive raw-text offsets would be silently wrong by a
    drifting amount. Worse, clean_text does NOT distribute over the
    join: clean_text("abc  " + sep + "def") is "abc \\n\\ndef" while
    joining the cleaned parts gives "abc\\n\\ndef".

    So each member is cleaned FIRST and the spans are measured on the
    joined result, which is then written to disk as-is. That is sound
    only because clean_text is IDEMPOTENT (verified: collapsing runs
    cannot create new runs), so reading the file back is a no-op and the
    chunker sees exactly the string these offsets index. The caller
    asserts that idempotence rather than trusting it.

    Empty members are skipped: they contribute no text, so no chunk can
    overlap them and no provenance is owed.
    """
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
    # P6: the ranking rank-aware metrics are measured over, at a FIXED
    # depth for every system. Never fed to the generator.
    scoring_ranking: list[RetrievedChunk]
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

    def retrieve_for_scoring(
        self, query: str, depth: int = SCORING_RANKING_DEPTH
    ) -> list[RetrievedChunk]:
        """The ranking rank-aware metrics are MEASURED over.

        Separate from the reader context on purpose. With K counted over
        documents surfaced by the reader's 15 chunks, the ranking depth
        varied per query and per system, so Hit@10 could mean "within 4
        candidates" for one system and "within 10" for another. Measuring
        every system at one depth is what makes the numbers comparable to
        each other and to published Hit@K.

        The default asks the system for a deeper cut of the SAME
        retrieval it would otherwise do. That costs one extra vector or
        BM25 search per query and no LLM call.
        """
        return self.retrieve(query, k=depth)

    def prepare(self, query: str, k: int | None = None) -> PreparedQuery:
        """PHASE A: retrieve, pack, assemble the prompt. No generation.

        Systems whose generation context differs materially from "the
        retrieved chunks, packed" override THIS rather than answer().
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
        scoring_ranking = self.retrieve_for_scoring(query)
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
            scoring_ranking=prepared.scoring_ranking,
            latency_s=prepared.prepare_s + generate_s,
            n_retrieval_calls=prepared.n_retrieval_calls,
            n_input_tokens=prepared.n_input_tokens,
            evidence_tokens=prepared.evidence_tokens,
            extra=dict(prepared.extra),
        )

    def index_items(self, items: Sequence["CorpusItem"]) -> None:
        """Write ONE file per parent, not per item, and derive provenance
        from character offsets.

        WHY EVERY SYSTEM NEEDS THIS. One file per CorpusItem makes every
        item boundary a forced chunk boundary, so the retrieval unit
        becomes whatever granularity the benchmark's ANNOTATION happens
        to use. On HotpotQA, whose gold is sentence-level, that meant
        M2/M3 indexing single sentences (~31 tokens) while M4 indexed the
        reassembled paragraph — and `walk_corpus(min_chars=200)` then
        silently dropped most of them, crashing outright on the unit
        where nothing cleared the bar. The corpus a system indexes must
        be the DOCUMENT the items came from; the fine item granularity
        buys provenance, not smaller chunks.

        This layout was M4-local (commit cb5c8ed) and was promoted here
        once its key-invariance was measured on the real corpora rather
        than argued from fixtures — see THE SINGLE-ITEM RULE below.

        THE SINGLE-ITEM RULE (rule B), and why it is not a special case
        bolted on. `corpus_content_hash` is computed over the temp
        directory this writes, so the layout is a cache-key input for
        every system. A parent holding exactly one item is therefore
        written with the SAME filename the per-item layout used (derived
        from item_id) and the SAME raw bytes, so its file is
        byte-identical and the corpus_hash cannot move. Both halves are
        load-bearing and were measured, not reasoned:

          * the FILENAME matters because corpus_content_hash folds the
            relative path, not just the bytes. MultiHop happens to have
            item_id == parent_id so a parent-derived name would collide
            harmlessly, but NarrativeQA and QuALITY use "{id}::<whole>"
            against a bare parent id, so a parent-derived name would move
            their hashes despite being 1:1.
          * the RAW BYTES matter because clean_text is not the identity.
            A multi-item parent is written PRE-CLEANED (see
            build_parent_payload); doing that to a single-item parent
            would change its bytes whenever its text contains a double
            space, and move the hash that way instead.

        A single-item parent also needs no offsets: every chunk of that
        document belongs to that one item, which is exactly what the
        per-item layout stamped. Offsets are only required where a parent
        holds several items — and in that case the hash moves regardless,
        which is why this is safe for the banked MultiHop and NarrativeQA
        columns (both strictly 1:1, measured) and changes HotpotQA (which
        has no valid banked cell).

        SIDE EFFECT, recorded: `walk_corpus` drops documents shorter than
        `min_chars_per_doc`, so concatenating a multi-item parent can
        rescue items that would individually have been dropped. That is a
        fidelity improvement (a retriever indexes documents, not
        annotation fragments) rather than a bug, and it cannot affect a
        1:1 benchmark.

        Provenance stamping is a no-op for M1 (self.chunks stays empty).
        """
        with tempfile.TemporaryDirectory(prefix=f"{self.system_id}_corpus_") as td:
            td_path = Path(td)
            layout = self._write_corpus_layout(items, td_path)
            self.index(td_path)

        self._stamp_provenance(layout)

    def _write_corpus_layout(
        self, items: Sequence["CorpusItem"], td_path: Path
    ) -> dict[str, tuple[str, list[tuple[int, int, str]] | None, str]]:
        """Materialise the per-parent corpus into `td_path`; return the layout.

        EXTRACTED SO THE WARM-SUBSTRATE CHECK CANNOT DIVERGE FROM THE REAL
        INDEX. `corpus_content_hash` is computed over this directory, so
        the layout IS a cache-key input; a preflight that wrote the corpus
        even slightly differently would compute a different key and report
        warm/cold about a substrate no cell would ever use. One function,
        both callers — see `substrate_warm_path`.
        """
        groups = group_items_by_parent(items)
        # filename -> (parent_id, spans | None, span_id_if_single)
        layout: dict[str, tuple[str, list[tuple[int, int, str]] | None, str]] = {}
        for parent_id, members in groups.items():
            if len(members) == 1:
                only = members[0]
                seed, payload = only.item_id, only.text
                spans, single_span = None, only.span_id
            else:
                seed = parent_id
                payload, spans = build_parent_payload(members)
                single_span = ""
                if clean_text(payload) != payload:
                    # The offsets in `spans` index `payload`, but the
                    # chunker will see clean_text(payload). They are
                    # the same string only while clean_text stays
                    # idempotent; if that ever breaks, provenance
                    # would drift silently across the document.
                    raise RuntimeError(
                        f"parsing.clean_text is no longer idempotent on "
                        f"parent {parent_id!r}: the per-parent offsets "
                        "would not match the text the chunker reads."
                    )

            filename = f"{_safe_item_filename(seed)}.txt"
            if filename in layout:
                n = 1
                while f"{_safe_item_filename(seed)}_{n}.txt" in layout:
                    n += 1
                filename = f"{_safe_item_filename(seed)}_{n}.txt"
            (td_path / filename).write_text(payload, encoding="utf-8")
            layout[filename] = (parent_id, spans, single_span)
        return layout

    # Systems whose index() writes a cacheable, key-addressed substrate
    # set this True and override `substrate_warm_path`. M4 is the only
    # one today; M1/M2/M3 either build nothing or rebuild cheaply.
    has_cacheable_substrate = False

    def substrate_warm_path(self, items: Sequence["CorpusItem"]) -> str | None:
        """Path of an EXISTING complete substrate for `items`, else None.

        READ-ONLY and index-free: it materialises the corpus layout,
        computes the cache key, and asks whether that directory is
        already complete. It never embeds, clusters or summarises.

        Default None means "this system has no cacheable substrate", which
        is why `has_cacheable_substrate` is a separate flag — None must
        not be read as "checked and cold".
        """
        del items
        return None

    def _stamp_provenance(
        self,
        layout: dict[str, tuple[str, list[tuple[int, int, str]] | None, str]],
    ) -> None:
        # walk_corpus sets ParsedDocument.doc_id to the path relative to
        # the corpus root, which for this flat temp dir is the filename.
        n_unmapped = 0
        for chunk in self.chunks:
            entry = layout.get(chunk.doc_id)
            if entry is None:
                continue  # defensive; should not happen
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
                    "supply them; the semantic chunker does not and cannot "
                    "be used with a multi-item parent."
                )
            # Half-open overlap. A chunk crossing an item boundary
            # legitimately carries BOTH atoms — that is the whole point
            # of contiguous chunking, and CK-2 scores it correctly.
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
        # Recorded where the system keeps index diagnostics. M4/M6/M7
        # declare `_index_stats`; M1/M2/M3 do not, and gain no attribute
        # here — this is a report, not a new base-class contract.
        stats = getattr(self, "_index_stats", None)
        if isinstance(stats, dict):
            stats["n_chunks_without_gold_provenance"] = n_unmapped

    def _require_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError(f"{self.system_id} not indexed — call .index() first")

    @staticmethod
    def _now() -> float:
        return time.perf_counter()
