"""Shared dataclasses for the benchmark eval layer.

CorpusItem feeds `BaseSystem.index_items` at index time. The rest of
the types here are produced by the benchmark loaders and consumed by
the scorers and the CLI runner. Every type is frozen so they are safe
to pass across the parallel-eval boundary and to use as dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- Indexable corpus item (consumed by BaseSystem.index_items) -----------


@dataclass(frozen=True)
class CorpusItem:
    """One indexable unit fed to a system at benchmark eval time.

    For QASPER each item is one paragraph from a paper (item granularity
    matches the gold-passage atom). For MultiHop-RAG each item is one
    full article body.

    Fields:
      item_id   — globally unique stable id. Convention: f"{parent_id}::{span_id}".
                  Used by the harness to name temp files when the default
                  index_items fallback writes to disk before calling
                  index(corpus_path). MUST be filesystem-safe under the
                  sanitiser in `BaseSystem._safe_item_filename`.
      parent_id — paper_id (QASPER) | article_url (MultiHop). Carried
                  into every produced Chunk's gold_provenance via the
                  default index_items fallback.
      span_id   — "sec{N}.para{M}" (QASPER) | "<whole>" (MultiHop).
                  The within-parent gold-span identifier.
      text      — the indexable text. The chunker may split this further;
                  default index_items writes one .txt file per CorpusItem
                  before calling self.index.
      metadata  — arbitrary loader-provided extras (section_name,
                  source, published_at, category, ...). Not part of the
                  index pipeline; surfaces in eval logs only.
    """

    item_id: str
    parent_id: str
    span_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# --- Gold answers (per-annotator ground truth) -----------------------------


# Answer-type strings. Strings rather than an Enum so loaders and scorers
# can be added without circular-import gymnastics.
ANSWER_TYPE_EXTRACTIVE = "extractive"
ANSWER_TYPE_ABSTRACTIVE = "abstractive"
ANSWER_TYPE_YES_NO = "yes_no"
ANSWER_TYPE_UNANSWERABLE = "unanswerable"
ANSWER_TYPE_FREE_FORM = "free_form"  # MultiHop-RAG single-string answers


@dataclass(frozen=True)
class GoldAnswer:
    """One annotator's ground-truth answer.

    Mutually-exclusive in spirit: a well-formed annotator labels a
    question as exactly one type. The dataclass holds all fields so
    callers can read whichever field the type implies. QASPER produces
    multiple GoldAnswers per query (one per annotator); MultiHop-RAG
    produces a single GoldAnswer.
    """

    answer_type: str
    extractive_spans: tuple[str, ...] = ()
    free_form: str = ""
    yes_no: bool | None = None
    unanswerable: bool = False


# --- Eval query (one (paper, question) for QASPER; one query for MultiHop) -


@dataclass(frozen=True)
class EvalQuery:
    """One eval question with per-annotator ground truth + gold-passage atoms.

    `parent_scope` encodes the corpus-shape fork:
      - When set (QASPER): the runner restricts retrieval to the parent
        (paper) — the system was indexed on that paper's corpus only.
      - When None (MultiHop-RAG): retrieval runs over the full shared
        corpus the system was indexed on.

    `gold_passage_sets` is a tuple of per-annotator atom sets (each set
    is a frozenset of (parent_id, span_id) pairs). MultiHop produces a
    single set; QASPER produces one per annotator. CK-2 retrieval-F1
    is max-over-annotators.

    Per the FLOAT SELECTED ruling (table-grounded QASPER evidence
    out-of-scope for text retrieval), evidence strings that flag as
    table/figure markers ARE NOT included in any gold_passage_set —
    they shrink the denominator naturally without polluting it with
    irreducibly-unrecoverable atoms.

    Per the no-match-evidence ruling, evidence strings that cannot be
    aligned to any paragraph after the exact -> ws-normalised ->
    substring fallback ARE DROPPED + counted in a loader-side counter;
    no fuzzy-wrong atom enters the gold set.

    `question_type` is benchmark-specific for slicing in analysis:
      QASPER:      'extractive' | 'abstractive' | 'yes_no' | 'unanswerable' | 'mixed'
      MultiHop:    'comparison' | 'inference' | 'temporal' | 'null'
    """

    query_id: str
    question_text: str
    parent_scope: str | None
    gold_answers: tuple[GoldAnswer, ...]
    gold_passage_sets: tuple[frozenset[tuple[str, str]], ...]
    question_type: str
    metadata: dict = field(default_factory=dict)


# --- Score dataclasses ----------------------------------------------------


@dataclass(frozen=True)
class RetrievalScore:
    """Retrieval-recall result for one (system, query).

    Two metric families, both optional:

    SET-BASED (CK-2 alignment, `score_retrieval_ck2`):
      Max-over-annotators set-F1 / recall / precision over the gold
      atom set. QASPER scoring; back-compat for any benchmark with
      paragraph-level gold passages. Fields:
        skipped, recall, precision, f1, n_gold, n_covered,
        n_retrieved_atoms, per_annotator

    RANK-AWARE (`score_retrieval_rank_aware`):
      Hit@K / MAP@K / MRR over a single gold atom set. MultiHop-RAG
      scoring; matches the paper's retrieval metrics. Fields:
        hit_at_k, map_at_k, mrr
      Empty dicts / 0.0 mrr when not applicable (e.g. QASPER).

    `skipped` is True when all annotators have empty gold sets (every
    annotator either flagged unanswerable OR every piece of evidence
    was table-grounded). Per ruling 5, retrieval recall is not scored
    in that case — the answer-side abstention scorer handles it.
    """

    skipped: bool
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    n_gold: int = 0
    n_covered: int = 0
    n_retrieved_atoms: int = 0
    # Per-annotator detail for analysis. Each entry is a dict with the
    # same {recall, precision, f1, n_gold} keys, or None for an
    # annotator whose gold set was empty (skipped at the annotator
    # level).
    per_annotator: tuple[dict | None, ...] = ()
    # Rank-aware metrics (MultiHop-RAG; empty for QASPER).
    hit_at_k: dict[int, float] = field(default_factory=dict)
    map_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0


@dataclass(frozen=True)
class AnswerScore:
    """Answer-quality result for one (system, query).

    `value` is the primary scalar (0-1). `method` describes what produced
    it ('token_f1', 'exact_match', 'abstention', 'token_f1_placeholder',
    etc.). `per_annotator` carries the raw per-annotator scores prior
    to the max-over-annotators aggregation.
    """

    value: float
    method: str
    per_annotator: tuple[float, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredQuery:
    """One row of the eval output JSONL.

    Carries everything a downstream analysis script needs: the
    benchmark + system identity, the predicted answer text, both
    score blocks, and per-query metadata for slicing.

    CK-4 fields (n_packed, evidence_tokens, n_input_tokens,
    retrieved_unit_types, packed_unit_types) are populated by the
    runner from the AnswerResult. evidence_tokens IS the quantity
    --check-budget-equality measures (chunks-only, budget-controlled).
    n_input_tokens is the full prompt for analysis visibility. The
    unit-type dicts let the analyser slice by retrieval-unit class —
    "chunk" for raw-chunk systems, "summary_low" / "_mid" / "_high"
    for M4 summary-expanded hits.
    """

    system_id: str
    benchmark: str
    split: str
    query_id: str
    parent_scope: str | None
    question_text: str
    predicted_answer: str
    retrieval: RetrievalScore
    answer: AnswerScore
    question_type: str
    latency_s: float
    n_retrieved: int
    # CK-4 instrumentation
    n_packed: int = 0
    evidence_tokens: int = 0
    n_input_tokens: int = 0
    retrieved_unit_types: dict[str, int] = field(default_factory=dict)
    packed_unit_types: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalUnit:
    """One (corpus, queries) pair to be indexed once and answered.

    QASPER yields 888 EvalUnits, one per paper. MultiHop-RAG yields
    one EvalUnit holding the shared 609-article corpus. Same interface
    for the runner regardless of corpus shape.
    """

    corpus_id: str
    corpus: tuple[CorpusItem, ...]
    queries: tuple[EvalQuery, ...]


__all__ = [
    "CorpusItem",
    "GoldAnswer",
    "EvalQuery",
    "EvalUnit",
    "RetrievalScore",
    "AnswerScore",
    "ScoredQuery",
    "ANSWER_TYPE_EXTRACTIVE",
    "ANSWER_TYPE_ABSTRACTIVE",
    "ANSWER_TYPE_YES_NO",
    "ANSWER_TYPE_UNANSWERABLE",
    "ANSWER_TYPE_FREE_FORM",
]
