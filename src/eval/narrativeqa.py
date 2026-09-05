"""NarrativeQA loader: one EvalUnit per full story, scored as the max
token-F1 over the two reference answers. Retrieval is never scored.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..retrievers.base import RetrievedChunk
from .sampling import SUBSAMPLE_SEED, subsample_indices

from .scorers import (
    assert_gold_not_empty,
    extractive_max_f1,
    is_abstention,
    token_f1,
)
from .types import (
    ANSWER_TYPE_FREE_FORM,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)


# dataset: deepmind/narrativeqa, validation (115 stories), full-story setting
HF_REPO = "deepmind/narrativeqa"
# Parquet auto-conversion branch; the loader falls back to the default
# revision when the branch is absent.
HF_REVISION = "refs/convert/parquet"

VALID_SPLITS = ("train", "validation", "test")


# Stories in one matrix cell, drawn with the preregistered seed.
# harness choice: preregistered seeded draw of 40 (METHODS §B.2)
CELL_UNITS = 40


def select_units(order: list, max_units: int | None) -> list:
    """Resolve the seeded story draw; None means the cell, not the split."""
    # Each n has its own seeded set, so a smaller n is a different draw,
    # not a prefix. Asking for the full split (115) drops nothing.
    effective = CELL_UNITS if max_units is None else max_units
    if effective is not None and effective < len(order):
        picked = subsample_indices(len(order), effective)
        return [order[i] for i in picked]
    return list(order)



class NarrativeQABenchmark:
    """NarrativeQA benchmark: per-story EvalUnits and the free-form scorer."""

    name = "narrativeqa"
    # Population one cell resolves to; the runner checks the draw against it.
    cell_units = CELL_UNITS

    def __init__(self) -> None:
        """Start with an empty split cache and zeroed stats."""
        self._split_cache: dict[str, Any] = {}
        self.stats: dict[str, int] = {
            "n_stories": 0,
            "n_queries": 0,
            "n_refs_total": 0,
            "n_gutenberg": 0,
            "n_movie": 0,
            "subsample_seed": None,
            "sampled_story_ids": [],
        }

    def _load_split(self, split: str) -> Any:
        """Load one split from HuggingFace once and cache it."""
        if split not in self._split_cache:
            from datasets import load_dataset

            # Try the parquet branch first, then the default revision.
            try:
                ds = load_dataset(HF_REPO, revision=HF_REVISION, split=split)
            except Exception:
                ds = load_dataset(HF_REPO, split=split)
            self._split_cache[split] = ds
        return self._split_cache[split]

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        """Yield one EvalUnit per drawn story; each question is a query."""
        if split not in VALID_SPLITS:
            raise ValueError(
                f"NarrativeQA split must be one of {VALID_SPLITS}; got "
                f"{split!r}. Mapping is direct (no remap); develop on "
                "validation, test is reserved for final numbers."
            )
        ds = self._load_split(split)

        # Group rows by story: the dataset repeats the full text per QA
        # pair, so each story text is stored once, in first-seen order.
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
                    # The title sits under document.summary in the HF schema.
                    "title": (doc.get("summary") or {}).get("title") or "",
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

        # Draw the cell's stories with the preregistered seed and record
        # the drawn ids, so the run summary names the exact stories.
        # harness choice: preregistered seeded draw of 40 (METHODS §B.2)
        # harness choice: preregistered seed (METHODS §B)
        order = select_units(order, max_units)
        self.stats["subsample_seed"] = SUBSAMPLE_SEED
        self.stats["n_units_requested"] = (
            CELL_UNITS if max_units is None else max_units
        )
        self.stats["sampled_story_ids"] = list(order)

        n_questions = sum(len(s["qa"]) for s in stories.values() if True)
        print(
            f"[narrativeqa] split {split!r}: {len(order)} stories, "
            f"{n_questions} questions (full-text variant)"
        )

        # One unit per story: a single whole-story CorpusItem, and one
        # query per question with every reference as a gold answer.
        for did in order:
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
                for ref in refs:
                    assert_gold_not_empty(
                        query_id=f"{did}::q{q_idx}", gold=ref,
                        benchmark="narrativeqa")
                self.stats["n_refs_total"] += len(refs)
                # dataset: no passage annotation, retrieval never scored
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
                        gold_passage_sets=(),
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
        scoring_ranking: list[RetrievedChunk] | None = None,
    ) -> RetrievalScore:
        """Skip retrieval scoring; NarrativeQA has no gold passages."""
        # dataset: no passage annotation, retrieval never scored
        del retrieved, query
        return RetrievalScore(skipped=True)

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """Max token-F1 over the references; abstention is metadata only."""
        refs = tuple(g.free_form for g in query.gold_answers if g.free_form)
        if not refs:
            # Unreachable: the loader drops questions without references
            # and asserts every gold is non-empty. A guard, not an
            # exception, so one bad row cannot end a cell.
            return AnswerScore(value=0.0, method="no_references")
        # Score against each reference and keep the max. The abstention
        # flag is recorded and never touches the value.
        # NarrativeQA paper: two references per question, max over references
        per_ref = tuple(token_f1(predicted, r) for r in refs)
        return AnswerScore(
            value=extractive_max_f1(predicted, refs),
            method="token_f1",
            per_annotator=per_ref,
            metadata={
                "abstained": is_abstention(predicted),
                "n_references": len(refs),
            },
        )


__all__ = ["NarrativeQABenchmark"]
