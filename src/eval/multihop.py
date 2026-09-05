"""MultiHop-RAG benchmark loader and scorer: one shared corpus of news
articles, document-level retrieval gold, token-F1 answers and a
pure-refusal rule for null queries."""

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


# dataset: yixuantt/MultiHopRAG (609 articles, 2,556 queries, 301 null)
HF_REPO = "yixuantt/MultiHopRAG"

# K grid for Hit@K and MAP@K. Every system is scored at the same K here.
# official: retrieval_evaluate.py @ cde8e844 (Hits@4, Hits@10); K = 1, 5 are ours
RANK_K_VALUES = (1, 4, 5, 10)


def _load_files() -> tuple[list[dict], list[dict]]:
    """Download the query and corpus JSON files from HF, cached locally."""
    from huggingface_hub import hf_hub_download

    queries_path = hf_hub_download(HF_REPO, "MultiHopRAG.json", repo_type="dataset")
    corpus_path = hf_hub_download(HF_REPO, "corpus.json", repo_type="dataset")
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    return queries, corpus


def _corpus_to_items(corpus: list[dict]) -> list[CorpusItem]:
    """Turn each raw article into a whole-document CorpusItem keyed by url."""
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
    """MultiHop-RAG loader: one shared-corpus EvalUnit and its two scorers."""

    name = "multihop_rag"
    # One EvalUnit: the corpus is indexed once and every query runs over it.
    cell_units = 1
    # The dataset ships a single split; the three names are synonyms.
    VALID_SPLITS = ("validation", "test", "all")

    def __init__(self) -> None:
        """Start with nothing loaded and zeroed counters."""
        self._loaded: tuple[list[dict], list[dict]] | None = None
        self.stats: dict[str, int] = {
            "n_corpus_articles": 0,
            "n_queries": 0,
            "n_evidence_total": 0,
            "n_null_queries": 0,
        }

    def _ensure_loaded(self) -> tuple[list[dict], list[dict]]:
        """Load the raw files on first use and reuse them afterwards."""
        if self._loaded is None:
            self._loaded = _load_files()
        return self._loaded

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        """Yield the one EvalUnit holding the whole corpus and all queries."""
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"MultiHop-RAG split must be one of {self.VALID_SPLITS}; got {split!r}. "
                "The dataset ships a single split — pass 'all' for the full set or "
                "'validation' / 'test' as synonyms during Pass-1 development."
            )
        raw_queries, raw_corpus = self._ensure_loaded()

        items = _corpus_to_items(raw_corpus)
        self.stats["n_corpus_articles"] = len(items)

        # Build one EvalQuery per raw query; a null query has empty gold.
        queries: list[EvalQuery] = []
        for q_idx, q in enumerate(raw_queries):
            evidence_list = q.get("evidence_list") or []
            self.stats["n_evidence_total"] += len(evidence_list)

            qtype_raw = q.get("question_type") or "free_form"
            is_null = qtype_raw == "null_query"
            if is_null:
                self.stats["n_null_queries"] += 1

            # Answerable queries must carry non-empty gold; null ones are
            # exempt because they score under the refusal rule.
            gold_str = q.get("answer") or ""
            if not is_null:
                assert_gold_not_empty(
                    query_id=f"multihop_{q_idx:06d}", gold=gold_str,
                    benchmark="multihop_rag")
            gold = GoldAnswer(
                answer_type=ANSWER_TYPE_UNANSWERABLE if is_null else ANSWER_TYPE_FREE_FORM,
                free_form=gold_str,
                unanswerable=is_null,
            )

            # Retrieval gold is the set of evidence documents, whole.
            # deviation from official (retrieval_evaluate.py matches gold-fact substrings): see METHODS §B.1
            atoms: set[tuple[str, str]] = set()
            for ev in evidence_list:
                url = ev.get("url")
                if url:
                    atoms.add((url, "<whole>"))

            # The dataset has no query ids; the row index is the stable id.
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
        # With one unit, a non-positive max_units yields nothing.
        if max_units is not None and max_units <= 0:
            return
        yield unit

    def score_retrieval(
        self,
        retrieved: list[RetrievedChunk],
        query: EvalQuery,
        scoring_ranking: list[RetrievedChunk] | None = None,
    ) -> RetrievalScore:
        """Score retrieval: set-F1 on the reader context, Hit@K, MAP@K, MRR."""
        # Set-level scores use the reader context; rank-aware scores use the
        # depth-50 scoring ranking. The single gold annotator is index 0.
        # harness choice: chunker-independent recall (METHODS §C.4)
        # harness choice: one scoring depth for every system (METHODS §D)
        # deviation from official (retrieval_evaluate.py adds newly-matched/rank): see METHODS §C.8
        # deviation from official (MRR@10): see METHODS §B.1
        ck2 = score_retrieval_ck2(retrieved, query.gold_passage_sets)
        gold = query.gold_passage_sets[0] if query.gold_passage_sets else frozenset()
        ranked = scoring_ranking if scoring_ranking is not None else retrieved
        rank = score_retrieval_rank_aware(ranked, gold, k_values=RANK_K_VALUES)
        # A null query has no gold; return the skipped set-level score alone.
        if rank.get("skipped"):
            return ck2
        # Merge the rank-aware fields into the set-level score.
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
        """Score an answer: refusal rule on null queries, else token-F1."""
        gold = query.gold_answers[0]
        # Abstention is metadata only; no branch below reads it for a value.
        abstained = is_abstention(predicted)
        # Null query: credit only a pure refusal. The lenient figures are
        # still recorded so fabricated answers to null queries stay visible.
        # harness addition (official scorer has no null branch): see METHODS §C.9
        if gold.answer_type == ANSWER_TYPE_UNANSWERABLE:
            value = score_unanswerable(predicted)
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
        # Answerable query: token-F1 is the value; substring match and the
        # max of the two are recorded beside it. pass1_placeholder marks the
        # column as not the benchmark's own metric.
        # deviation from official (qa_evaluate.py::has_intersection is one shared token): see METHODS §B.1
        # official: qa_evaluate.py::has_intersection @ cde8e844 (recorded, not scored)
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
