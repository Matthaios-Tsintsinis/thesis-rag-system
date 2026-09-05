"""Pins that a resumed run() indexes only units with queries left to answer.

The tests drive run() end to end and read which units it indexed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.chunking import Chunk
from src.config import DEFAULT_CONFIG
from src.eval.base import BenchmarkRunner
from src.eval.types import (
    ANSWER_TYPE_FREE_FORM,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)
from src.retrievers.base import BaseSystem, RetrievedChunk

N_UNITS = 3
QUERIES_PER_UNIT = 2


class CountingSystem(BaseSystem):
    """Fake system that records which units it indexes, in call order."""

    system_id = "FAKE"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.indexed_units: list[str] = []

    def index(self, corpus_path):  # pragma: no cover - unused
        raise AssertionError("the runner must not use the path-based index()")

    def index_items(self, items):
        # Recover the unit id from the item ids; only the corpus is passed in.
        items = list(items)
        self.indexed_units.append(items[0].item_id.split("-")[0])
        self._indexed = True
        self.chunks = [
            Chunk(chunk_id=i.item_id, doc_id=i.item_id, text=i.text,
                  n_words=len(i.text.split()), position=n,
                  gold_provenance=((i.parent_id, i.span_id),))
            for n, i in enumerate(items)
        ]

    def retrieve(self, query: str, k: int | None = None):
        return [
            RetrievedChunk(chunk=c, score=1.0 / (j + 1), rank=j)
            for j, c in enumerate(self.chunks[:2])
        ]


class MultiUnitBenchmark:
    """Fake benchmark with N units, each with its own corpus."""

    name = "fake_multi"

    def iter_eval_units(self, *, split: str, max_units=None):
        units = []
        for u in range(N_UNITS):
            corpus = tuple(
                CorpusItem(item_id=f"unit{u}-item{i}", parent_id=f"P{u}",
                           span_id=f"s{i}", text=f"body of unit {u} item {i}")
                for i in range(2)
            )
            queries = tuple(
                EvalQuery(
                    query_id=f"u{u}q{i}",
                    question_text=f"question {i} about unit {u}",
                    parent_scope=None,
                    gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                             free_form="gold"),),
                    gold_passage_sets=(frozenset({(f"P{u}", "s0")}),),
                    question_type="free_form",
                )
                for i in range(QUERIES_PER_UNIT)
            )
            units.append(EvalUnit(corpus_id=f"unit{u}", corpus=corpus,
                                  queries=queries))
        yield from units[: max_units if max_units is not None else N_UNITS]

    def score_retrieval(self, retrieved, query,
                        scoring_ranking=None) -> RetrievalScore:
        return RetrievalScore(skipped=False, f1=float(len(retrieved)))

    def score_answer(self, predicted, query) -> AnswerScore:
        return AnswerScore(value=1.0, method="token_f1")


def _fake_generate(system_prompt, user_prompt, cfg=None):
    return f"ANSWER<{user_prompt[-24:]}>"


def _bank(path: Path, query_ids) -> None:
    """Write one banked row per query id, as an interrupted pass would."""
    with path.open("w", encoding="utf-8") as f:
        for qid in query_ids:
            f.write(json.dumps({"query_id": qid}) + "\n")


def _run(*, banked=(), resume=False):
    """Drive run() to completion and return (indexed_units, scored_ids)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "o.jsonl"
        if banked:
            _bank(out, banked)
        system = CountingSystem(config=DEFAULT_CONFIG)
        runner = BenchmarkRunner(output_path=out, verbose=False,
                                 resume=resume)
        with mock.patch("src.models.generate", _fake_generate):
            rows = list(runner.run(system, MultiUnitBenchmark(),
                                   split="validation"))
        return system.indexed_units, [r.query_id for r in rows]


def _all_ids():
    return [f"u{u}q{i}" for u in range(N_UNITS) for i in range(QUERIES_PER_UNIT)]


class TestResumeSkipsIndexing(unittest.TestCase):
    def test_fully_done_unit_is_not_indexed(self):
        """A unit whose every query is banked is not indexed; the rest are."""
        indexed, scored = _run(banked=["u0q0", "u0q1"], resume=True)
        self.assertNotIn("unit0", indexed)
        self.assertEqual(indexed, ["unit1", "unit2"])
        self.assertEqual(sorted(scored),
                         sorted(["u1q0", "u1q1", "u2q0", "u2q1"]))

    def test_partially_done_unit_is_still_indexed(self):
        """One outstanding query is enough to index the unit."""
        indexed, scored = _run(banked=["u0q0"], resume=True)
        self.assertEqual(indexed, ["unit0", "unit1", "unit2"])
        self.assertIn("u0q1", scored)
        self.assertNotIn("u0q0", scored)

    def test_complete_cell_on_resume_indexes_nothing(self):
        """Resuming a finished cell indexes nothing and scores nothing."""
        indexed, scored = _run(banked=_all_ids(), resume=True)
        self.assertEqual(indexed, [])
        self.assertEqual(scored, [])

    def test_without_resume_every_unit_is_indexed(self):
        """A first pass with nothing banked indexes every unit."""
        indexed, scored = _run()
        self.assertEqual(indexed, ["unit0", "unit1", "unit2"])
        self.assertEqual(sorted(scored), sorted(_all_ids()))

    def test_banked_rows_are_ignored_without_the_resume_flag(self):
        """Without --resume, banked rows suppress no indexing."""
        indexed, scored = _run(banked=_all_ids(), resume=False)
        self.assertEqual(indexed, ["unit0", "unit1", "unit2"])
        self.assertEqual(sorted(scored), sorted(_all_ids()))


if __name__ == "__main__":
    unittest.main()
