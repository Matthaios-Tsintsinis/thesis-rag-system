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

from .scorers import is_abstention, score_abstention, token_f1
from .types import (
    ANSWER_TYPE_FREE_FORM,
    ANSWER_TYPE_UNANSWERABLE,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
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

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """Pass-1 skeleton scorer.

        null_query (unanswerable): abstention detection.
        Other queries: token-F1 vs gold answer as PLACEHOLDER for the
          paper's LLM-judge metric. Pass-2 swaps in the judge.
        """
        gold = query.gold_answers[0]
        if gold.answer_type == ANSWER_TYPE_UNANSWERABLE:
            return AnswerScore(
                value=score_abstention(predicted),
                method="abstention",
                per_annotator=(score_abstention(predicted),),
            )
        if is_abstention(predicted):
            return AnswerScore(
                value=0.0,
                method="free_form_abstained",
                per_annotator=(0.0,),
            )
        score = token_f1(predicted, gold.free_form)
        return AnswerScore(
            value=score,
            method="multihop_token_f1_placeholder",
            per_annotator=(score,),
            metadata={"pass1_placeholder": True},
        )


__all__ = ["MultiHopBenchmark"]
