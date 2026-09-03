"""MultiHop-RAG benchmark loader + Pass-1 skeleton answer scorer.

Loads from `yixuantt/MultiHopRAG` on HuggingFace (single revision, no
auto-conversion needed — the repo ships JSON files directly).

Single shared corpus of 609 news articles; 2556 queries retrieve over
the whole collection. The loader yields ONE EvalUnit holding the
full corpus + all queries. parent_scope=None on every EvalQuery so
the runner indexes the shared corpus once and reuses across all
queries.

Gold-passage atom = (article_url, "<whole>"). 100% url-exact alignment
verified during schema design — no fallback chain needed.

Question types from the dataset's explicit `question_type` field:
  comparison_query, inference_query, temporal_query, null_query.
Null queries carry empty `evidence_list` (out-of-corpus / unanswerable)
and are scored via abstention.

Answer scoring:
  * null_query annotations: the unanswerable rule, method
    `unanswerable_rule`.
  * Other queries: token_f1(predicted, gold.free_form), method
    `token_f1`, ALWAYS COMPUTED — abstention detection is metadata and
    never reaches the value (see score_answer for what that replaced).

  THE OFFICIAL METRIC, read from the benchmark's own `qa_evaluate.py`
  rather than from the paper's prose (corrected 2026-08-18; this block
  had been wrong under three successive readings of the same metric —
  "LLM judge", then "exact-match accuracy", then this, which is the one
  taken from the source):

      def has_intersection(prediction: str, gold: str) -> bool:
          return bool(set(prediction.split()).intersection(gold.split()))

      correct = sum(has_intersection(p.lower(), g.lower())
                    for p, g in zip(predictions, gold_answers))

  A response is scored CORRECT IF ANY SINGLE WORD OVERLAPS with the gold,
  after lowercasing and whitespace splitting. No punctuation stripping,
  no article removal, no exact match, and NO MODEL IS INVOLVED anywhere
  in their evaluation — GPT-4 appears in that project as a
  dataset-construction tool and as an evaluated system, never as a judge.

  Consequences, all of which belong in the results caption:

  1. TOKEN-F1 IS STRICTLY MORE DEMANDING than the official metric. This
     is a deviation in the STRICT direction, which is the easy direction
     to defend: we cannot be accused of flattering the systems by
     scoring them more leniently than the benchmark's own script does.
  2. THERE IS NO OFFICIAL "F1" TO COMPARE AGAINST. `calculate_metrics`
     returns `score, score, score, score` and labels them precision,
     recall, F1 and accuracy — one binary per-question success rate
     printed four times. Comparing our token-F1 to their "F1" crosses
     metric FAMILIES, not merely thresholds.
  3. NULL QUERIES GET NO SPECIAL HANDLING THERE. Their loop has no
     branch for `null_query`; it groups by `question_type` for reporting
     only and routes every type through the same comparison. So the
     pure-refusal rule this module applies is a HARNESS ADDITION with no
     official counterpart — not a deviation from one.
  4. THEIR NORMALISATION IS LOWERCASE + WHITESPACE SPLIT ONLY, so their
     tokens keep trailing punctuation. Ours strips punctuation and
     articles via the shared SQuAD-style normaliser.
  5. WE DO NOT IMPLEMENT `extract_answer`. Their scorer regexes
     `The answer to the question is "(.*?)"` out of the model output and
     falls back to the raw text, so their prompt asks for that wrapper
     and their scorer removes it before comparing. Our prompt does not
     request it and our scorer does not strip it.

  `pass1_placeholder=True` remains on every AnswerScore. Read it as
  "this column is not the benchmark's own metric", which is still true —
  it is NOT to be read as "an LLM judge is coming later".
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..retrievers.base import RetrievedChunk
from .alignment import score_retrieval_ck2, score_retrieval_rank_aware
from .scorers import (
    assert_gold_not_empty,
    is_abstention,
    score_unanswerable,
    substring_match,
    token_f1,
)
from .types import (
    ANSWER_TYPE_FREE_FORM,
    ANSWER_TYPE_UNANSWERABLE,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)


HF_REPO = "yixuantt/MultiHopRAG"

# K grid. K=4 is here so ONE column is the paper's Hits@4 (Tang & Yang
# 2024 report Hits@4, Hits@10, MAP@10, MRR@10). K=1 and K=5 are extra
# diagnostic columns the paper does not report. Every SYSTEM is measured
# at the same K on this benchmark, which is what comparability requires;
# HotpotQA uses its own grid because its two-gold-title structure makes
# Hit@2 the directly interpretable number there.
RANK_K_VALUES = (1, 4, 5, 10)


def _load_files() -> tuple[list[dict], list[dict]]:
    """Download MultiHopRAG.json + corpus.json from HF. Cached locally."""
    from huggingface_hub import hf_hub_download

    queries_path = hf_hub_download(HF_REPO, "MultiHopRAG.json", repo_type="dataset")
    corpus_path = hf_hub_download(HF_REPO, "corpus.json", repo_type="dataset")
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    return queries, corpus


def _corpus_to_items(corpus: list[dict]) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for c in corpus:
        url = c["url"]
        items.append(
            CorpusItem(
                item_id=url,
                parent_id=url,
                span_id="<whole>",
                text=c.get("body") or "",
                metadata={
                    "title": c.get("title") or "",
                    "author": c.get("author") or "",
                    "source": c.get("source") or "",
                    "published_at": c.get("published_at") or "",
                    "category": c.get("category") or "",
                },
            )
        )
    return items


class MultiHopBenchmark:
    """Iterable over MultiHop-RAG EvalUnits + Pass-1 free-form scorer.

    One EvalUnit per call to iter_eval_units — the shared corpus is
    fully loaded into memory. The MultiHopRAG.json is ~5 MB and the
    corpus.json is ~7 MB; both fit comfortably.

    The `split` argument is accepted for protocol parity with the other
    loaders but has only one value here (the dataset ships a single split).
    Passing anything other than 'validation' or 'test' or 'all' raises.

    METHODOLOGY NOTE (single-split asymmetry): because MultiHop-RAG
    ships ONE split, the validation-matrix run doubles as the FINAL
    MultiHop numbers — there is no held-out portion to reserve. The
    thesis reports NarrativeQA and both HotpotQA variants on their
    validation splits and MultiHop-RAG on its single split; the
    asymmetry is stated in the methods section.
    """

    name = "multihop_rag"
    # ONE shared-corpus EvalUnit: 609 articles indexed once, 2,556
    # queries against it. Declared for the runner's population check.
    cell_units = 1
    VALID_SPLITS = ("validation", "test", "all")

    def __init__(self) -> None:
        self._loaded: tuple[list[dict], list[dict]] | None = None
        self.stats: dict[str, int] = {
            "n_corpus_articles": 0,
            "n_queries": 0,
            "n_evidence_total": 0,
            "n_null_queries": 0,
        }

    def _ensure_loaded(self) -> tuple[list[dict], list[dict]]:
        if self._loaded is None:
            self._loaded = _load_files()
        return self._loaded

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"MultiHop-RAG split must be one of {self.VALID_SPLITS}; got {split!r}. "
                "The dataset ships a single split — pass 'all' for the full set or "
                "'validation' / 'test' as synonyms during Pass-1 development."
            )
        raw_queries, raw_corpus = self._ensure_loaded()

        items = _corpus_to_items(raw_corpus)
        self.stats["n_corpus_articles"] = len(items)

        queries: list[EvalQuery] = []
        for q_idx, q in enumerate(raw_queries):
            evidence_list = q.get("evidence_list") or []
            self.stats["n_evidence_total"] += len(evidence_list)

            qtype_raw = q.get("question_type") or "free_form"
            is_null = qtype_raw == "null_query"
            if is_null:
                self.stats["n_null_queries"] += 1

            gold_str = q.get("answer") or ""
            if not is_null:
                # Null queries are exempt: their gold is empty ON PURPOSE
                # and they score under unanswerable_rule.
                assert_gold_not_empty(
                    query_id=f"multihop_{q_idx:06d}", gold=gold_str,
                    benchmark="multihop_rag")
            gold = GoldAnswer(
                answer_type=ANSWER_TYPE_UNANSWERABLE if is_null else ANSWER_TYPE_FREE_FORM,
                free_form=gold_str,
                unanswerable=is_null,
            )

            atoms: set[tuple[str, str]] = set()
            for ev in evidence_list:
                url = ev.get("url")
                if url:
                    atoms.add((url, "<whole>"))

            # MultiHop has a single gold annotator. Use a derived stable id
            # since the dataset doesn't ship per-query ids; q_idx is stable
            # within a fixed dataset version.
            query_id = f"multihop_{q_idx:06d}"

            queries.append(
                EvalQuery(
                    query_id=query_id,
                    question_text=q.get("query") or "",
                    parent_scope=None,
                    gold_answers=(gold,),
                    gold_passage_sets=(frozenset(atoms),),
                    question_type=qtype_raw,
                    metadata={"raw_index": q_idx},
                )
            )

        self.stats["n_queries"] = len(queries)

        unit = EvalUnit(
            corpus_id="multihop_shared",
            corpus=tuple(items),
            queries=tuple(queries),
        )
        # max_units constrains the unit count; MultiHop only has one
        # unit so this is effectively all-or-nothing.
        if max_units is not None and max_units <= 0:
            return
        yield unit

    def score_retrieval(
        self,
        retrieved: list[RetrievedChunk],
        query: EvalQuery,
        scoring_ranking: list[RetrievedChunk] | None = None,
    ) -> RetrievalScore:
        """MultiHop: CK-2 set-F1 + rank-aware (Hit@K, MAP@K, MRR).

        Merges:
          - score_retrieval_ck2 over (url, "<whole>") atoms — gives F1
            consistent with QASPER's scoring shape.
          - score_retrieval_rank_aware at K=(1, 4, 5, 10) over the
            FIXED-DEPTH SCORING RANKING (SCORING_RANKING_DEPTH=50), not
            over the reader's top-15. K=4 and K=10 are the paper's own
            Hits@4 / Hits@10; K=1 and K=5 are extra diagnostic columns.
            DEVIATION FROM PAPER: MAP and MRR are reported at the same
            grid rather than at K=10 only, and MRR is uncapped. MAP's
            NUMERATOR is standard average precision, which the official
            retrieval_evaluate.py does NOT compute (it sums 1/rank per
            newly-found gold; see score_retrieval_rank_aware's docstring)
            -- a declared deviation, ours >= official. MRR here is
            uncapped (first relevant doc over the full ranking), equal to
            the paper's MRR@10 unless the first relevant doc sits at
            rank >10 (rare under top-15). Matrix tables must NOT be read
            as paper-identical K=4. Single gold annotator means we read
            the first (and only) gold_passage_set.

        null_query (empty gold) skips rank-aware (per the helper
        contract); CK-2 also returns skipped=True. Answer-side
        abstention scorer handles those queries.
        """
        # SET-LEVEL over the READER CONTEXT: it measures what the
        # generator actually saw. RANK-AWARE over the fixed-depth
        # SCORING RANKING: it measures retrieval quality at one depth for
        # every system. Two depths, one table; the caption says so.
        ck2 = score_retrieval_ck2(retrieved, query.gold_passage_sets)
        gold = query.gold_passage_sets[0] if query.gold_passage_sets else frozenset()
        ranked = scoring_ranking if scoring_ranking is not None else retrieved
        rank = score_retrieval_rank_aware(ranked, gold, k_values=RANK_K_VALUES)
        if rank.get("skipped"):
            return ck2
        # Merge — extend CK-2 score with rank-aware fields.
        return RetrievalScore(
            skipped=ck2.skipped,
            recall=ck2.recall,
            precision=ck2.precision,
            f1=ck2.f1,
            n_gold=ck2.n_gold,
            n_covered=ck2.n_covered,
            n_retrieved_atoms=ck2.n_retrieved_atoms,
            per_annotator=ck2.per_annotator,
            hit_at_k=rank["hit_at_k"],
            map_at_k=rank["map_at_k"],
            mrr=rank["mrr"],
        )

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """MultiHop answer scorer (Pass-1).

        THE SCORING CONTRACT (identical in narrativeqa.py and
        hotpotqa.py). The score for an answerable query is token-F1
        against gold and is ALWAYS COMPUTED. Abstention detection is
        recorded in metadata.abstained and never reaches a value.

        The gate this replaced returned 0.0 whenever the prediction
        contained a hedging phrase, which discarded real token-F1: a
        prediction carrying the exact gold string scored 0.0000 where its
        F1 was 0.3333, and the same prediction scored differently on
        HotpotQA, which never had the gate. Measured, not argued -- see
        docs/EVAL_AUDIT.md ISSUE-1. A hedge is a property of PHRASING;
        treating it as a refusal conflates the two.

        Primary value:
          - null_query: score_unanswerable -- the PURE-REFUSAL rule
            (method "unanswerable_rule"). Detection alone used to decide
            this and credited fabrication: "I don't know the year, but
            the answer is Tesla." scored 1.0. The rule now requires the
            utterance MINUS its hedge to assert nothing.
            NOTE, from their qa_evaluate.py: the benchmark itself has NO
            null-query rule -- every type goes through the same one-word
            overlap check. This rule is ours, added, with no official
            counterpart to deviate from.
          - Other queries: token_F1 vs gold.free_form (the
            QASPER-consistent default; less false-positive-prone than
            substring). The benchmark's own scorer credits ANY one-word
            overlap, so token-F1 is STRICTER than it; this column and the
            results table must say so.

        Metadata always records both `token_f1` AND `substring_match`
        for analysis — substring's leniency (e.g. boundary-match of a
        short factual answer inside a longer prediction) catches
        verbose-but-correct outputs that token-F1 underestimates, but
        as ruled it stays out of the primary value to avoid inflating
        Pass-1 accuracy via false positives like "not Apple, it's
        Samsung" matching gold "Apple". The `max(token_f1, substring)`
        value is also recorded in metadata for downstream comparison.
        """
        gold = query.gold_answers[0]
        # Detection is METADATA. It is computed once, recorded, and never
        # consulted by any branch that produces a score.
        abstained = is_abstention(predicted)
        if gold.answer_type == ANSWER_TYPE_UNANSWERABLE:
            value = score_unanswerable(predicted)
            # Even on null_query, record substring + token_f1 against
            # the gold string for analytic visibility (how often does
            # the system fabricate a "factual" answer to a null query).
            tf1 = token_f1(predicted, gold.free_form) if gold.free_form else 0.0
            ssm = substring_match(predicted, gold.free_form) if gold.free_form else 0.0
            return AnswerScore(
                value=value,
                method="unanswerable_rule",
                per_annotator=(value,),
                metadata={
                    "abstained": abstained,
                    "token_f1": tf1,
                    "substring_match": ssm,
                    "max_lenient": max(tf1, ssm),
                    "pass1_placeholder": True,
                },
            )
        tf1 = token_f1(predicted, gold.free_form)
        ssm = substring_match(predicted, gold.free_form)
        return AnswerScore(
            value=tf1,
            method="token_f1",
            per_annotator=(tf1,),
            metadata={
                "abstained": abstained,
                "token_f1": tf1,
                "substring_match": ssm,
                "max_lenient": max(tf1, ssm),
                "pass1_placeholder": True,
            },
        )


__all__ = ["MultiHopBenchmark"]
