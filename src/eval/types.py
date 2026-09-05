"""Frozen dataclasses shared by the benchmark loaders, scorers and runner.
Every type is immutable, so instances can be dict keys and cross processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- Indexable corpus item (consumed by BaseSystem.index_items) -----------


@dataclass(frozen=True)
class CorpusItem:
    """One indexable text unit handed to a system at index time."""

    # item_id is "{parent_id}::{span_id}"; the on-disk name is a hash of it,
    # so it need not be filesystem-safe. (parent_id, span_id) is the gold
    # atom the retrieval scorers match against, e.g. (article url, "<whole>")
    # for MultiHop. metadata holds loader extras that reach the logs only.
    item_id: str
    parent_id: str
    span_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# --- Gold answers (per-annotator ground truth) -----------------------------


# Plain strings, not an Enum, so loaders and scorers need not import
# each other.
ANSWER_TYPE_EXTRACTIVE = "extractive"
ANSWER_TYPE_ABSTRACTIVE = "abstractive"
ANSWER_TYPE_YES_NO = "yes_no"
ANSWER_TYPE_UNANSWERABLE = "unanswerable"
ANSWER_TYPE_FREE_FORM = "free_form"  # MultiHop-RAG single-string answers
ANSWER_TYPE_MULTIPLE_CHOICE = "multiple_choice"  # no current benchmark


@dataclass(frozen=True)
class GoldAnswer:
    """One reference answer; answer_type says which field carries it."""

    answer_type: str
    extractive_spans: tuple[str, ...] = ()
    free_form: str = ""
    yes_no: bool | None = None
    unanswerable: bool = False


# --- Eval query ------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuery:
    """One question with its reference answers and gold retrieval atoms."""

    # parent_scope names the story the query belongs to, or None when the
    # unit's whole corpus is the scope. gold_answers holds one entry per
    # reference; answer scores take the max over them. gold_passage_sets
    # holds one frozenset of (parent_id, span_id) atoms per annotator;
    # retrieval scores take the max over them. question_type is the
    # benchmark's own label (MultiHop: comparison, inference, temporal,
    # null) and is used only for slicing.
    # NarrativeQA paper: two references per question, max over references
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
    """Retrieval metrics for one (system, query); skipped when no gold."""

    # skipped is True when every annotator's gold set is empty; retrieval
    # is then not scored and the answer side handles the query.
    skipped: bool
    # Set-level recall / precision / F1 over gold atoms in the reader
    # context, max over annotators.
    # harness choice: chunker-independent recall (METHODS §C.4)
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    n_gold: int = 0
    n_covered: int = 0
    n_retrieved_atoms: int = 0
    # One {recall, precision, f1, n_gold} dict per annotator, or None for
    # an annotator with no gold atoms.
    per_annotator: tuple[dict | None, ...] = ()
    # Rank-aware metrics over the depth-50 scoring ranking; empty dicts and
    # 0.0 where the benchmark has no retrieval gold.
    # harness choice: one scoring depth for every system (METHODS §D)
    hit_at_k: dict[int, float] = field(default_factory=dict)
    map_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0


@dataclass(frozen=True)
class AnswerScore:
    """Answer score for one (system, query); method names the rule used."""

    # value is in [0, 1]. method names the scoring rule (token_f1,
    # exact_match, unanswerable_rule), never an outcome. per_annotator
    # holds the raw per-reference scores before the max. metadata.abstained
    # records abstention detection only and never changes value.
    # official: hotpot_evaluate_v1.py::f1_score @ 36358534
    # official: hotpot_evaluate_v1.py::exact_match_score @ 36358534
    # harness addition: see METHODS §C.9
    value: float
    method: str
    per_annotator: tuple[float, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredQuery:
    """One output JSONL row: identity, prediction, both scores, metadata."""

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
    # Context accounting copied from the AnswerResult: evidence_tokens
    # counts the packed evidence only, n_input_tokens the whole prompt, and
    # the unit-type dicts count retrieved / packed units by class ("chunk",
    # or "summary_low" / "summary_mid" / "summary_high" for M4 nodes).
    n_packed: int = 0
    evidence_tokens: int = 0
    n_input_tokens: int = 0
    retrieved_unit_types: dict[str, int] = field(default_factory=dict)
    packed_unit_types: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalUnit:
    """One corpus indexed once plus the queries answered over it."""

    # MultiHop is a single unit over the shared corpus; NarrativeQA yields
    # one unit per story, HotpotQA one per question or per pooled shard.
    # dataset: yixuantt/MultiHopRAG (609 articles, 2,556 queries, 301 null)
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
    "ANSWER_TYPE_MULTIPLE_CHOICE",
]
