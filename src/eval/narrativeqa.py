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
small-sample gate should run CHEAP SYSTEMS FIRST (M1/M2/M3) and only
then M4, so loader/scorer bugs surface before the substrate spend.

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


HF_REPO = "deepmind/narrativeqa"
# QASPER precedent: prefer the parquet auto-conversion branch (immune
# to dataset-script deprecation); fall back to the default revision if
# the branch is absent (repo migrated to native parquet).
HF_REVISION = "refs/convert/parquet"

VALID_SPLITS = ("train", "validation", "test")


# Stories in a matrix CELL — P7's seeded draw of 40 from the 115-story
# validation split.
#
# THE SINGLE SOURCE, and it needed one. This number used to live only in
# `scripts/probe_cell_costs.py`, so every other caller had to remember it
# and one — the leaf inventory — did not, enumerating all 115 and
# describing a population no cell builds.
CELL_UNITS = 40


def select_units(order: list, max_units: int | None) -> list:
    """Apply P7's seeded draw. **None means the CELL, not everything.**

    THE DEFECT THIS FIXES, and it is the seventh of its kind. P7 declared
    a seeded 40-story sample and shipped a seeded SAMPLER: the draw was
    applied only when `max_units` was passed, and `--max-units` defaults
    to None, so a cell launched without the flag ran all 115 stories and
    3,461 questions — and succeeded. The population of a cell is now a
    property of this function rather than of an operator's memory.

    `max_units` is NOT a cap. `subsample_indices(115, n)` selects a
    DIFFERENT SET for each n, so asking for fewer does not narrow the
    same sample, it takes another one. That is why a forgotten flag
    changed WHICH stories ran, not merely how many — and why a story
    picked as "largest" from the wrong draw might never be built at all.

    Explicit values are still honoured by the loader, including the full
    split (`115` drops nothing) — reachable from tests only: the runner
    passes no cap since the repo reduction. Explicit is possible; silent
    is not.
    """
    effective = CELL_UNITS if max_units is None else max_units
    if effective is not None and effective < len(order):
        picked = subsample_indices(len(order), effective)
        return [order[i] for i in picked]
    return list(order)



class NarrativeQABenchmark:
    """Iterable over per-story NarrativeQA EvalUnits + free-form scorer."""

    name = "narrativeqa"
    # Units a CELL resolves to, declared so the runner can check the
    # resolved population against a stated number rather than against
    # nothing. See `select_units` for why this is not merely a cap.
    cell_units = CELL_UNITS

    def __init__(self) -> None:
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
                    # summary is nested under document in the HF schema
                    # (top-level row has only document/question/answers),
                    # so the title lives at document.summary.title.
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

        # SEEDED SAMPLE, not the head of the split (P7), and NOT the
        # whole split either: `select_units` resolves None to CELL_UNITS,
        # so the draw is a property of the code rather than of a flag the
        # operator has to remember. A prefix was the first defect here; a
        # silent full split was the second.
        order = select_units(order, max_units)
        self.stats["subsample_seed"] = SUBSAMPLE_SEED
        self.stats["n_units_requested"] = (
            CELL_UNITS if max_units is None else max_units
        )
        # RECORDED so the draw is reproducible AND inspectable: the run
        # summary carries benchmark stats verbatim, so the exact stories
        # behind a cell are in its provenance rather than re-derivable
        # only by rerunning the sampler.
        self.stats["sampled_story_ids"] = list(order)

        n_questions = sum(len(s["qa"]) for s in stories.values() if True)
        print(
            f"[narrativeqa] split {split!r}: {len(order)} stories, "
            f"{n_questions} questions (full-text variant)"
        )

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
        scoring_ranking: list[RetrievedChunk] | None = None,
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
        convention.

        THE SCORING CONTRACT (identical in multihop.py and hotpotqa.py):
        the score is ALWAYS the computed token-F1. Abstention detection
        is recorded in metadata.abstained and never reaches a value. The
        gate this replaced forced 0.0 on any hedged prediction, which
        discarded real overlap -- see docs/EVAL_AUDIT.md ISSUE-1.

        NarrativeQA's own paper reports BLEU-1/4, METEOR and ROUGE-L.
        Token-F1 is primary here for cross-benchmark consistency;
        ROUGE-L is computed post-hoc as a secondary column so the numbers
        can be read against the published ones.
        """
        refs = tuple(g.free_form for g in query.gold_answers if g.free_form)
        if not refs:
            # Unreachable given the loader skips a story-question with no
            # references. Kept as an explicit guard rather than an
            # exception so one malformed row cannot kill a 1,208-query
            # cell; the loader-side assertion is the real defence.
            return AnswerScore(value=0.0, method="no_references")
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
