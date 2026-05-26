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

Answer scoring (Pass-1 skeleton):
  * null_query annotations: score_abstention.
  * Other queries: token_f1(predicted, gold.free_form). This is a
    PLACEHOLDER — the official MultiHop-RAG metric is an LLM-judge on
    free-form answers per the paper. Pass-2 swaps in the judge prompt
    (gpt-4o-mini, same controlled-reader as the systems). The
    placeholder method tag `multihop_token_f1_placeholder` is emitted
    so downstream analysis can identify Pass-1 numbers vs final
    Pass-2 numbers.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..retrievers.base import RetrievedChunk
from .alignment import score_retrieval_ck2, score_retrieval_rank_aware
from .scorers import (
    is_abstention,
    score_abstention,
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

    The `split` argument is accepted for protocol parity with QASPER
    but has only one value here (the dataset ships a single split).
    Passing anything other than 'validation' or 'test' or 'all' raises.
    """

    name = "multihop_rag"
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
    ) -> RetrievalScore:
        """MultiHop: CK-2 set-F1 + rank-aware (Hit@K, MAP@K, MRR).

        Merges:
          - score_retrieval_ck2 over (url, "<whole>") atoms — gives F1
            consistent with QASPER's scoring shape.
          - score_retrieval_rank_aware at K=(1, 5, 10) — paper-aligned
            (MultiHop paper uses MAP@10 + Hit + MRR over supporting
            docs). Single gold annotator means we read the first
            (and only) gold_passage_set.

        null_query (empty gold) skips rank-aware (per the helper
        contract); CK-2 also returns skipped=True. Answer-side
        abstention scorer handles those queries.
        """
        ck2 = score_retrieval_ck2(retrieved, query.gold_passage_sets)
        gold = query.gold_passage_sets[0] if query.gold_passage_sets else frozenset()
        rank = score_retrieval_rank_aware(retrieved, gold, k_values=(1, 5, 10))
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

        Primary value:
          - null_query: score_abstention (1.0 if abstained, else 0.0).
          - Other queries: token_F1 vs gold.free_form (the
            QASPER-consistent default; less false-positive-prone than
            substring). Pass-2 swaps the primary to an LLM judge.

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
        if gold.answer_type == ANSWER_TYPE_UNANSWERABLE:
            absc = score_abstention(predicted)
            # Even on null_query, record substring + token_f1 against
            # the gold string for analytic visibility (how often does
            # the system fabricate a "factual" answer to a null query).
            tf1 = token_f1(predicted, gold.free_form) if gold.free_form else 0.0
            ssm = substring_match(predicted, gold.free_form) if gold.free_form else 0.0
            return AnswerScore(
                value=absc,
                method="abstention",
                per_annotator=(absc,),
                metadata={
                    "token_f1": tf1,
                    "substring_match": ssm,
                    "max_lenient": max(tf1, ssm),
                    "pass1_placeholder": True,
                },
            )
        if is_abstention(predicted):
            # Answerable question; predicted abstained -> 0.0 primary.
            # Record substring too for symmetry (will be 0.0).
            return AnswerScore(
                value=0.0,
                method="free_form_abstained",
                per_annotator=(0.0,),
                metadata={
                    "token_f1": 0.0,
                    "substring_match": 0.0,
                    "max_lenient": 0.0,
                    "pass1_placeholder": True,
                },
            )
        tf1 = token_f1(predicted, gold.free_form)
        ssm = substring_match(predicted, gold.free_form)
        return AnswerScore(
            value=tf1,
            method="multihop_token_f1",
            per_annotator=(tf1,),
            metadata={
                "token_f1": tf1,
                "substring_match": ssm,
                "max_lenient": max(tf1, ssm),
                "pass1_placeholder": True,
            },
        )


__all__ = ["MultiHopBenchmark"]
