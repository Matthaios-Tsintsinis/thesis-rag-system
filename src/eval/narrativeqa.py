"""NarrativeQA benchmark loader + free-form max-over-references scoring.

NarrativeQA (Kocisky et al., TACL 2018): free-form QA over FULL
narratives — Project Gutenberg books and movie scripts, ~25k-90k
words each. ~1,572 stories (1,102 train / 115 validation / 355 test),
~30 questions per story, TWO reference answers per question.

FULL-TEXT VARIANT (decided 2026-06-12). This is the only benchmark in
the suite that tests retrieval at book length (QASPER ~ paper,
MultiHop ~ article, QuALITY ~ 5k tokens) — i.e. at the scale the
hierarchical systems (M4/M7) are built for. The summary variant
(retrieval over a ~650-word Wikipedia plot summary) is degenerate as
a retrieval test — top-15 returns essentially the whole corpus — and
would waste the slot. Cost consequence: RAPTOR substrates run ~120
gpt-4o-mini summaries per story (~$0.04/story/substrate); the
small-sample gate should run CHEAP SYSTEMS FIRST (M1/M2/M3, M9) and
only then M4/M7, so loader/scorer bugs surface before the substrate
spend. Full-validation subsampling levers (--max-queries, --max-units)
are decided at matrix time.

QUESTIONS-FROM-SUMMARIES CAVEAT (thesis methods note): NarrativeQA
questions were written by annotators who read the Wikipedia plot
SUMMARIES, not the books. Answers are abstractive and sometimes not
verbatim in the full text, so token-F1 scores run lower than
QASPER's; they remain comparable ACROSS systems, which is what the
matrix needs. Token-F1 against the references is the standard
treatment for generative readers on this dataset.

SPLITS: direct mapping, no remap — harness validation -> NarrativeQA
validation, harness test -> NarrativeQA test; both publicly labeled.
RESERVE DISCIPLINE: all development and the small-sample gate happen
on validation; test is reserved for the single final-numbers pass
after the pipeline locks (same discipline as QASPER test / QuALITY
harness-test).

DATA: loaded from `deepmind/narrativeqa` on HuggingFace. The dataset
ships the FULL story text inside each record (no external URL
fetching — the original loading script's dead-link failure mode is
gone with the parquet conversion). Rows repeat the story per QA pair,
so the loader DEDUPES AT LOAD: stories are grouped by document.id and
each story text is stored once (~35 MB for validation, not the naive
~1 GB). Only the requested split is downloaded.

SHAPE: one EvalUnit per story (QASPER pattern); one CorpusItem per
story (span_id "<whole>", MultiHop convention) — the chunker splits
the book downstream (per-benchmark chunk size = CK-3 ablation,
harness default for Pass-1). question_type = document.kind
("gutenberg" | "movie") so --by-type slices books vs scripts for
free. No gold passages -> RetrievalScore(skipped=True) per query
(QuALITY pattern; the analyser excludes skipped rows from retrieval
means).

SCORING: two references -> two GoldAnswers -> score = MAX token-F1
over references via extractive_max_f1 (the same max-over-annotators
convention as QASPER). Abstention on these always-answerable
questions scores 0 and is flagged in metadata.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..retrievers.base import RetrievedChunk
from .scorers import extractive_max_f1, is_abstention, token_f1
from .types import (
    ANSWER_TYPE_FREE_FORM,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)


HF_REPO = "deepmind/narrativeqa"
# QASPER precedent: prefer the parquet auto-conversion branch (immune
# to dataset-script deprecation); fall back to the default revision if
# the branch is absent (repo migrated to native parquet).
HF_REVISION = "refs/convert/parquet"

VALID_SPLITS = ("train", "validation", "test")


class NarrativeQABenchmark:
    """Iterable over per-story NarrativeQA EvalUnits + free-form scorer."""

    name = "narrativeqa"

    def __init__(self) -> None:
        self._split_cache: dict[str, Any] = {}
        self.stats: dict[str, int] = {
            "n_stories": 0,
            "n_queries": 0,
            "n_refs_total": 0,
            "n_gutenberg": 0,
            "n_movie": 0,
        }

    def _load_split(self, split: str) -> Any:
        if split not in self._split_cache:
            from datasets import load_dataset

            try:
                ds = load_dataset(HF_REPO, revision=HF_REVISION, split=split)
            except Exception:
                # Branch absent (native-parquet repo) — default revision.
                ds = load_dataset(HF_REPO, split=split)
            self._split_cache[split] = ds
        return self._split_cache[split]

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"NarrativeQA split must be one of {VALID_SPLITS}; got "
                f"{split!r}. Mapping is direct (no remap); develop on "
                "validation, test is reserved for final numbers."
            )
        ds = self._load_split(split)

        # Dedupe at load: rows repeat the full story per QA pair. Store
        # each story text ONCE, in first-seen order.
        stories: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in ds:
            doc = row["document"]
            did = doc["id"]
            if did not in stories:
                stories[did] = {
                    "text": doc.get("text") or "",
                    "kind": doc.get("kind") or "?",
                    "url": doc.get("url") or "",
                    "title": (row.get("summary") or {}).get("title") or "",
                    "qa": [],
                }
                order.append(did)
            answers = tuple(
                (a.get("text") or "").strip()
                for a in (row.get("answers") or ())
                if (a.get("text") or "").strip()
            )
            stories[did]["qa"].append(
                ((row.get("question") or {}).get("text") or "", answers)
            )

        n_questions = sum(len(s["qa"]) for s in stories.values())
        print(
            f"[narrativeqa] split {split!r}: {len(order)} stories, "
            f"{n_questions} questions (full-text variant)"
        )

        for n_done, did in enumerate(order):
            if max_units is not None and n_done >= max_units:
                break
            story = stories[did]
            kind = story["kind"]
            if kind == "gutenberg":
                self.stats["n_gutenberg"] += 1
            elif kind == "movie":
                self.stats["n_movie"] += 1

            corpus_items = (
                CorpusItem(
                    item_id=f"{did}::<whole>",
                    parent_id=did,
                    span_id="<whole>",
                    text=story["text"],
                    metadata={
                        "kind": kind,
                        "title": story["title"],
                        "url": story["url"],
                    },
                ),
            )

            queries: list[EvalQuery] = []
            for q_idx, (question, refs) in enumerate(story["qa"]):
                if not question or not refs:
                    continue
                self.stats["n_refs_total"] += len(refs)
                queries.append(
                    EvalQuery(
                        query_id=f"{did}::q{q_idx}",
                        question_text=question,
                        parent_scope=did,
                        gold_answers=tuple(
                            GoldAnswer(
                                answer_type=ANSWER_TYPE_FREE_FORM,
                                free_form=ref,
                            )
                            for ref in refs
                        ),
                        gold_passage_sets=(),  # no gold passages in NarrativeQA
                        question_type=kind,
                        metadata={"kind": kind, "n_references": len(refs)},
                    )
                )
                self.stats["n_queries"] += 1

            self.stats["n_stories"] += 1
            yield EvalUnit(
                corpus_id=did,
                corpus=corpus_items,
                queries=tuple(queries),
            )

    def score_retrieval(
        self,
        retrieved: list[RetrievedChunk],
        query: EvalQuery,
    ) -> RetrievalScore:
        """No gold passages in NarrativeQA — retrieval is never scored.

        skipped=True rows are excluded from the analyser's retrieval
        means and counted in retr_n_skipped (QuALITY pattern).
        """
        del retrieved, query
        return RetrievalScore(skipped=True)

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """Max token-F1 over the (two) reference answers.

        extractive_max_f1 IS max-of-token_f1 over a string tuple —
        reused verbatim; consistent with QASPER's max-over-annotators
        convention. Abstention scores 0 (NarrativeQA questions are
        answerable by construction) but stays flagged in metadata.
        """
        refs = tuple(g.free_form for g in query.gold_answers if g.free_form)
        if not refs:
            return AnswerScore(value=0.0, method="no_references")
        if is_abstention(predicted):
            return AnswerScore(
                value=0.0,
                method="narrativeqa_abstained",
                per_annotator=tuple(0.0 for _ in refs),
                metadata={"abstained": True},
            )
        per_ref = tuple(token_f1(predicted, r) for r in refs)
        return AnswerScore(
            value=extractive_max_f1(predicted, refs),
            method="narrativeqa_token_f1",
            per_annotator=per_ref,
            metadata={"abstained": False, "n_references": len(refs)},
        )


__all__ = ["NarrativeQABenchmark"]
