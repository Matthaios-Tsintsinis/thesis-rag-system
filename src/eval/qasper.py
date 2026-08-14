"""QASPER benchmark loader + answer-type-aware scorer.

Loads from `allenai/qasper` on the `refs/convert/parquet` branch
(the default `refs/main` uses a deprecated dataset script and won't
load on modern HuggingFace `datasets`).

Per-paper EvalUnits: each paper becomes its own corpus + queries
bundle so the system indexes only the relevant paper before answering
that paper's questions. 888 EvalUnits on the train split, 281 on
validation, 416 on test.

Each paper's corpus consists of:
  * one CorpusItem per non-empty paragraph in full_text.paragraphs,
    span_id = f"sec{sec_idx}.para{para_idx}"
  * one CorpusItem for the abstract, span_id = "abstract"
  * the title is NOT a separate item (rarely cited as evidence;
    surfaces in metadata only).

Evidence alignment (per the rulings):
  * "FLOAT SELECTED: ..." evidence strings are TABLE/FIGURE markers.
    They are EXCLUDED from the gold-passage set entirely (out-of-scope
    for text retrieval — denominator shrinks, no irreducible-0 floor).
  * Other evidence strings are aligned against the paper's paragraph
    set via:
        exact -> whitespace-normalised exact -> substring containment
    Unalignable evidence is DROPPED and counted in the manifest
    (`n_unalignable_evidence`). Per ruling 4: fuzzy-wrong corrupts
    recall worse than a drop.
  * 84% of QASPER evidence strings match exactly (audit ran during
    schema design); ~9.2% are FLOAT SELECTED; ~6.9% need the fallback
    chain or get dropped.

Per-annotator scoring (QASPER official convention):
  * Multiple annotators per question; each annotator produces one
    GoldAnswer (one of: extractive, abstractive, yes_no,
    unanswerable). Different annotators can label the same question
    with different types.
  * `score_answer` computes a per-annotator score using the type-
    appropriate scorer, then takes the max across annotators.
  * Retrieval F1 (alignment.score_retrieval_ck2) is max-over-annotators
    too — both metrics aligned.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from ..retrievers.base import RetrievedChunk
from .alignment import score_retrieval_ck2
from .scorers import (
    is_abstention,
    score_abstention,
    score_yes_no,
    token_f1,
)
from .types import (
    ANSWER_TYPE_ABSTRACTIVE,
    ANSWER_TYPE_EXTRACTIVE,
    ANSWER_TYPE_UNANSWERABLE,
    ANSWER_TYPE_YES_NO,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)


HF_REPO = "allenai/qasper"
HF_REVISION = "refs/convert/parquet"
FLOAT_PREFIX = "FLOAT SELECTED:"

_WS_RE = re.compile(r"\s+")


def _ws_normalise(s: str) -> str:
    return _WS_RE.sub(" ", s.strip())


def _classify_annotator(ans: dict) -> str:
    """Determine answer_type from one annotator's raw fields.

    QASPER's mutual-exclusion convention:
      unanswerable=True   -> unanswerable
      yes_no is not None  -> yes_no
      extractive_spans    -> extractive (possibly with free_form too — still extractive)
      free_form           -> abstractive
      else                -> degenerate; treat as unanswerable
    """
    if ans.get("unanswerable"):
        return ANSWER_TYPE_UNANSWERABLE
    if ans.get("yes_no") is not None:
        return ANSWER_TYPE_YES_NO
    if ans.get("extractive_spans"):
        return ANSWER_TYPE_EXTRACTIVE
    if ans.get("free_form_answer"):
        return ANSWER_TYPE_ABSTRACTIVE
    return ANSWER_TYPE_UNANSWERABLE


def _gold_answer_from_raw(ans: dict) -> GoldAnswer:
    return GoldAnswer(
        answer_type=_classify_annotator(ans),
        extractive_spans=tuple(ans.get("extractive_spans") or ()),
        free_form=ans.get("free_form_answer") or "",
        yes_no=ans.get("yes_no"),
        unanswerable=bool(ans.get("unanswerable")),
    )


def _build_paragraph_index(
    paper_id: str,
    abstract: str,
    section_names: list[str],
    paragraphs: list[list[str]],
) -> tuple[list[CorpusItem], dict[str, str], dict[str, str]]:
    """Build the CorpusItem list for one paper plus two lookup maps.

    Returns (corpus_items, exact_map, ws_map):
      exact_map: paragraph_text -> span_id  (for direct evidence alignment)
      ws_map:    ws_normalised_text -> span_id  (fallback when exact fails
                 because of stray whitespace differences)

    Empty paragraphs are dropped (some QASPER section headers have no
    body paragraph and surface as ""); they would never appear in
    evidence anyway.
    """
    items: list[CorpusItem] = []
    exact_map: dict[str, str] = {}
    ws_map: dict[str, str] = {}

    def _add(item_id: str, span_id: str, text: str, metadata: dict) -> None:
        items.append(
            CorpusItem(
                item_id=item_id,
                parent_id=paper_id,
                span_id=span_id,
                text=text,
                metadata=metadata,
            )
        )
        exact_map.setdefault(text, span_id)
        ws_map.setdefault(_ws_normalise(text), span_id)

    if abstract and abstract.strip():
        _add(
            item_id=f"{paper_id}::abstract",
            span_id="abstract",
            text=abstract,
            metadata={"section_name": "Abstract"},
        )

    for sec_idx, (sec_name, paras) in enumerate(zip(section_names, paragraphs)):
        for para_idx, para in enumerate(paras):
            if not para or not para.strip():
                continue
            span_id = f"sec{sec_idx}.para{para_idx}"
            _add(
                item_id=f"{paper_id}::{span_id}",
                span_id=span_id,
                text=para,
                metadata={"section_name": sec_name},
            )

    return items, exact_map, ws_map


def _align_evidence(
    evidence_strings: list[str],
    exact_map: dict[str, str],
    ws_map: dict[str, str],
    paper_id: str,
    paragraphs_in_order: list[tuple[str, str]],
    unalignable_counter: list[int],
) -> set[tuple[str, str]]:
    """Map one annotator's evidence strings to (paper_id, span_id) atoms.

    FLOAT SELECTED: drop entirely.
    Exact match: O(1) dict lookup.
    Whitespace-normalised: O(1) on ws_map.
    Substring containment: O(P) scan over paragraphs_in_order — used
      only when both exact and ws fail; QASPER evidence is normally
      whole paragraphs so this branch covers the rare paragraph-fragment
      annotation. Direction: evidence-in-paragraph OR paragraph-in-evidence
      (cross-paragraph evidence is observed in ~1% of cases).
    Else: drop + bump the unalignable counter (passed in as a single-
      element list so the caller can read the count back).

    `paragraphs_in_order` is the list of (text, span_id) tuples in the
    order they appear in the paper. Substring search walks this list.
    """
    atoms: set[tuple[str, str]] = set()

    for ev in evidence_strings or ():
        if not ev:
            continue
        if ev.startswith(FLOAT_PREFIX):
            continue  # ruling 3: out-of-scope, NOT a denominator entry

        # Exact match.
        span = exact_map.get(ev)
        if span is not None:
            atoms.add((paper_id, span))
            continue

        # Whitespace-normalised.
        span = ws_map.get(_ws_normalise(ev))
        if span is not None:
            atoms.add((paper_id, span))
            continue

        # Substring fallback (direction-symmetric).
        matched = False
        ev_norm = _ws_normalise(ev)
        for para_text, para_span in paragraphs_in_order:
            para_norm = _ws_normalise(para_text)
            if ev_norm in para_norm or para_norm in ev_norm:
                atoms.add((paper_id, para_span))
                matched = True
                break
        if not matched:
            unalignable_counter[0] += 1
            # ruling 4: drop + log; do NOT add a fuzzy-wrong atom.

    return atoms


def _consensus_question_type(answers: tuple[GoldAnswer, ...]) -> str:
    """If all annotators agree on answer_type, return it; else "mixed"."""
    types = {a.answer_type for a in answers}
    if len(types) == 1:
        return next(iter(types))
    return "mixed"


class QasperBenchmark:
    """Iterable over QASPER EvalUnits + per-query answer scorer.

    Lazy-loads the HF parquet split on first iter_eval_units call. A
    second iteration (e.g. for the small-sample sanity pass before a
    full run) re-uses the loaded split.
    """

    name = "qasper"

    def __init__(self) -> None:
        self._split_cache: dict[str, Any] = {}
        # Counters for the manifest. Updated during iter_eval_units.
        self.stats: dict[str, int] = {
            "n_papers": 0,
            "n_queries": 0,
            "n_annotators": 0,
            "n_evidence_total": 0,
            "n_evidence_float_selected": 0,
            "n_evidence_unalignable": 0,
        }

    def _load_split(self, split: str) -> Any:
        if split not in self._split_cache:
            from datasets import load_dataset

            self._split_cache[split] = load_dataset(
                HF_REPO, revision=HF_REVISION, split=split
            )
        return self._split_cache[split]

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        ds = self._load_split(split)
        for paper_idx, rec in enumerate(ds):
            if max_units is not None and paper_idx >= max_units:
                break

            paper_id = rec["id"]
            corpus_items, exact_map, ws_map = _build_paragraph_index(
                paper_id=paper_id,
                abstract=rec.get("abstract") or "",
                section_names=rec["full_text"]["section_name"],
                paragraphs=rec["full_text"]["paragraphs"],
            )
            if not corpus_items:
                # Degenerate paper with no abstract and no text content.
                continue
            paragraphs_in_order = [(c.text, c.span_id) for c in corpus_items]

            queries: list[EvalQuery] = []
            qas = rec["qas"]
            for q_idx in range(len(qas["question"])):
                question_id = qas["question_id"][q_idx]
                question_text = qas["question"][q_idx]
                ann_block = qas["answers"][q_idx]

                gold_answers_list: list[GoldAnswer] = []
                gold_passage_sets_list: list[frozenset[tuple[str, str]]] = []
                for ans in ann_block["answer"]:
                    self.stats["n_annotators"] += 1
                    self.stats["n_evidence_total"] += len(ans.get("evidence") or ())
                    self.stats["n_evidence_float_selected"] += sum(
                        1 for e in (ans.get("evidence") or ())
                        if e and e.startswith(FLOAT_PREFIX)
                    )

                    gold_answers_list.append(_gold_answer_from_raw(ans))
                    unalignable = [0]
                    atoms = _align_evidence(
                        evidence_strings=list(ans.get("evidence") or ()),
                        exact_map=exact_map,
                        ws_map=ws_map,
                        paper_id=paper_id,
                        paragraphs_in_order=paragraphs_in_order,
                        unalignable_counter=unalignable,
                    )
                    self.stats["n_evidence_unalignable"] += unalignable[0]
                    gold_passage_sets_list.append(frozenset(atoms))

                gold_answers = tuple(gold_answers_list)
                queries.append(
                    EvalQuery(
                        query_id=question_id,
                        question_text=question_text,
                        parent_scope=paper_id,
                        gold_answers=gold_answers,
                        gold_passage_sets=tuple(gold_passage_sets_list),
                        question_type=_consensus_question_type(gold_answers),
                        metadata={
                            "paper_id": paper_id,
                            "title": rec.get("title") or "",
                        },
                    )
                )
                self.stats["n_queries"] += 1

            self.stats["n_papers"] += 1
            yield EvalUnit(
                corpus_id=paper_id,
                corpus=tuple(corpus_items),
                queries=tuple(queries),
            )

    # --- per-annotator answer scoring ---------------------------------------

    def _score_one_annotator(
        self,
        predicted: str,
        gold: GoldAnswer,
    ) -> tuple[float, str]:
        """Return (score, method) for one annotator."""
        if gold.answer_type == ANSWER_TYPE_UNANSWERABLE:
            return score_abstention(predicted), "abstention"
        if gold.answer_type == ANSWER_TYPE_YES_NO:
            # If the system abstained on a yes/no question, 0.0.
            if is_abstention(predicted):
                return 0.0, "yes_no_abstained"
            assert gold.yes_no is not None
            return score_yes_no(predicted, gold.yes_no), "yes_no"
        if gold.answer_type == ANSWER_TYPE_EXTRACTIVE:
            if is_abstention(predicted):
                return 0.0, "extractive_abstained"
            # Official QASPER metric: join the annotator's extractive spans
            # into ONE reference string (", ".join), then a single token-F1
            # (allenai/qasper-led-baseline scripts/evaluator.py). Extractive
            # spans are co-required parts of one answer, NOT alternatives, so
            # this is NOT a max-over-spans (which would over-credit matching
            # one span while ignoring the rest). extractive_max_f1 remains for
            # genuine alternative references (e.g. NarrativeQA's two refs).
            joined_gold = ", ".join(gold.extractive_spans)
            return token_f1(predicted, joined_gold), "extractive_f1"
        if gold.answer_type == ANSWER_TYPE_ABSTRACTIVE:
            if is_abstention(predicted):
                return 0.0, "abstractive_abstained"
            return token_f1(predicted, gold.free_form), "abstractive_f1"
        # Unknown type — treat as 0.
        return 0.0, "unknown"

    def score_retrieval(
        self,
        retrieved: list[RetrievedChunk],
        query: EvalQuery,
        scoring_ranking: list[RetrievedChunk] | None = None,
    ) -> RetrievalScore:
        """QASPER: paragraph-level CK-2 set-F1, max over annotators.

        Rank-aware metrics intentionally NOT computed for QASPER:
          - Gold is paragraph-level not document-level.
          - Multi-annotator; rank-aware over union vs per-annotator
            is ambiguous; the MultiHop paper's K-values don't
            translate cleanly.
        Leaves hit_at_k / map_at_k / mrr at their RetrievalScore
        defaults (empty dicts / 0.0).
        """
        return score_retrieval_ck2(retrieved, query.gold_passage_sets)

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """Max-over-annotators answer score per QASPER convention."""
        per_annotator: list[float] = []
        methods: list[str] = []
        for gold in query.gold_answers:
            score, method = self._score_one_annotator(predicted, gold)
            per_annotator.append(score)
            methods.append(method)
        if not per_annotator:
            return AnswerScore(value=0.0, method="no_annotators")
        best_idx = max(range(len(per_annotator)), key=lambda i: per_annotator[i])
        return AnswerScore(
            value=per_annotator[best_idx],
            method=methods[best_idx],
            per_annotator=tuple(per_annotator),
            metadata={"per_annotator_methods": methods, "best_annotator_idx": best_idx},
        )


__all__ = ["QasperBenchmark"]
