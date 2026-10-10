"""Pins that a cell interrupted in one process and resumed in another ends
with one complete summary: the population counts units banked earlier, the
summary reads every row in the file, and a torn last line is cut first."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.chunking import Chunk
from src.config import DEFAULT_CONFIG
from src.eval import runner as runner_mod
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
from src.retrievers.base import AnswerResult, BaseSystem, RetrievedChunk

N_UNITS = 3
QUERIES_PER_UNIT = 2
# Awkward floats, so a summary that summed in another order would differ.
SCORES = [0.1, 1 / 3, 2 / 7, 0.7, 1 / 9, 0.3]


class _Bench:
    """Three units of two queries; scores come from SCORES in query order."""

    name = "spans"
    cell_units = N_UNITS

    def __init__(self) -> None:
        self.stats = {"n_queries": N_UNITS * QUERIES_PER_UNIT}
        self.returned: list[float] = []

    def iter_eval_units(self, *, split, max_units=None):
        for u in range(N_UNITS):
            corpus = (CorpusItem(item_id=f"d{u}::<whole>", parent_id=f"d{u}",
                                 span_id="<whole>", text=f"text of doc {u}"),)
            queries = tuple(
                EvalQuery(
                    query_id=f"u{u}q{i}", question_text=f"q {u} {i}",
                    parent_scope=None,
                    gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                             free_form="gold"),),
                    gold_passage_sets=(frozenset({(f"d{u}", "<whole>")}),),
                    question_type="free_form",
                )
                for i in range(QUERIES_PER_UNIT)
            )
            yield EvalUnit(corpus_id=f"unit{u}", corpus=corpus, queries=queries)

    def score_retrieval(self, retrieved, query, scoring_ranking=None):
        return RetrievalScore(skipped=False, f1=SCORES[_index(query.query_id)])

    def score_answer(self, predicted, query):
        value = SCORES[_index(query.query_id)]
        self.returned.append(value)
        return AnswerScore(value=value, method="token_f1")


def _index(query_id: str) -> int:
    """Position of a query in the cell: u1q0 -> 2."""
    return int(query_id[1]) * QUERIES_PER_UNIT + int(query_id[3])


class _System(BaseSystem):
    """Answers without a model; dies after `die_after` answers if set."""

    system_id = "SPAN"
    die_after: int | None = None
    answered = 0

    def index(self, corpus_path):  # pragma: no cover - unused
        raise AssertionError("the runner indexes through index_items")

    def index_items(self, items):
        self.chunks = [Chunk(chunk_id=i.item_id, doc_id=i.item_id, text=i.text,
                             n_words=3, position=0,
                             gold_provenance=((i.parent_id, i.span_id),))
                       for i in items]
        self._indexed = True

    def retrieve(self, query, k=None):
        return [RetrievedChunk(chunk=c, score=1.0, rank=0) for c in self.chunks]

    def answer(self, query, k=None):
        cls = type(self)
        if cls.die_after is not None and cls.answered >= cls.die_after:
            raise RuntimeError("session died")
        cls.answered += 1
        r = self.retrieve(query)
        return AnswerResult(query=query, answer="an answer", retrieved=r,
                            packed=r, extra={})


class _Case(unittest.TestCase):
    """Drives main() with the stub system and benchmark patched in."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.out = self.dir / "spans_SPAN_validation.jsonl"
        _System.die_after = None
        _System.answered = 0
        self.bench = _Bench()
        self._patches = [
            mock.patch.dict(runner_mod.SYSTEM_REGISTRY, {"SPAN": _System}),
            mock.patch.dict(runner_mod.BENCHMARK_REGISTRY,
                            {"spans": lambda: self.bench}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.td.cleanup()

    def _main(self, *extra, out=None):
        """Run main() once, as one process would."""
        lock = self.dir / "requirements.lock"
        lock.write_text("# lock\nnumpy==0.0.0\n", encoding="utf-8")
        argv = ["runner", "--system", "SPAN", "--benchmark", "spans",
                "--split", "validation", "--output", str(out or self.out),
                "--lockfile", str(lock), *extra]
        with mock.patch("scripts.pin_environment.check_lockfile",
                        return_value=0), \
                mock.patch.object(sys, "argv", argv):
            runner_mod.main()

    def _summary(self, out=None) -> dict:
        path = (out or self.out).with_suffix(".summary.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def _interrupt_then_resume(self) -> None:
        """Die after three answers, tear the last line, resume in a new run."""
        _System.die_after = 3
        with self.assertRaises(RuntimeError):
            self._main()
        self.assertFalse(self.out.with_suffix(".summary.json").exists())
        with self.out.open("a", encoding="utf-8") as f:
            f.write('{"query_id": "u1q1", "answ')
        _System.die_after = None
        self._main("--resume")


SCORED_FIELDS = ("n_queries_scored", "n_retrieval_skipped",
                 "n_retrieval_scored", "mean_retrieval_f1",
                 "mean_answer_score", "n_answerable",
                 "mean_answer_score_answerable", "n_null_queries",
                 "mean_answer_score_null", "expected_n_queries",
                 "partial_run")


class TestResumeAcrossSessions(_Case):
    """A cell resumed in a second process is summarised whole."""

    def test_resumed_summary_equals_the_uninterrupted_one(self):
        """Every scored field matches a run that was never interrupted."""
        whole = self.dir / "whole" / "spans_SPAN_validation.jsonl"
        self._main(out=whole)
        self._interrupt_then_resume()
        a, b = self._summary(whole), self._summary()
        for key in SCORED_FIELDS:
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(b["n_queries_scored"], N_UNITS * QUERIES_PER_UNIT)

    def test_resumed_summary_says_how_many_rows_this_process_wrote(self):
        """The summary records the rows banked before and written now."""
        self._interrupt_then_resume()
        s = self._summary()
        self.assertEqual(s["resumed"], {"rows_banked_before": 3,
                                        "rows_this_process": 3})

    def test_torn_line_is_cut_and_its_query_answered_once(self):
        """Every line parses and every query appears exactly once."""
        self._interrupt_then_resume()
        lines = [x for x in self.out.read_text(encoding="utf-8").splitlines()
                 if x.strip()]
        ids = [json.loads(x)["query_id"] for x in lines]
        self.assertEqual(sorted(ids), sorted(
            f"u{u}q{i}" for u in range(N_UNITS) for i in range(QUERIES_PER_UNIT)))

    def test_the_exporter_reads_a_resumed_cell(self):
        """read_cell accepts the resumed cell with its full row count."""
        from scripts.export_comparison import read_cell

        self._interrupt_then_resume()
        generator = DEFAULT_CONFIG.generation.model
        row = read_cell(self.dir, generator, "spans", "SPAN",
                        {(generator, "spans", "SPAN"): (0, 0.0)})
        self.assertEqual(row["n_queries"], N_UNITS * QUERIES_PER_UNIT)

    def test_resuming_a_complete_cell_rewrites_the_same_summary(self):
        """A resume with nothing left indexes nothing and passes the gates."""
        self._main()
        first = self._summary()
        self._main("--resume")
        second = self._summary()
        for key in SCORED_FIELDS:
            self.assertEqual(first[key], second[key], key)

    def test_duplicate_rows_refuse_the_summary(self):
        """A query banked twice is refused before any summary is written."""
        self._main()
        self.out.with_suffix(".summary.json").unlink()
        first = self.out.read_text(encoding="utf-8").splitlines()[0]
        with self.out.open("a", encoding="utf-8") as f:
            f.write(first + "\n")
        with self.assertRaisesRegex(SystemExit, "more than once"):
            self._main("--resume")
        self.assertFalse(self.out.with_suffix(".summary.json").exists())


class TestNeverResumedRunIsUnchanged(_Case):
    """A run that never resumed is summarised exactly as before."""

    def test_means_are_the_in_memory_sums_in_row_order(self):
        """The summary mean equals the scorer's floats summed in yield order."""
        self._main()
        s = self._summary()
        total = 0.0
        for v in self.bench.returned:
            total += v
        self.assertEqual(s["mean_answer_score"], total / len(self.bench.returned))

    def test_no_resumed_block_without_a_resume(self):
        """The resumed block appears only when earlier rows were banked."""
        self._main()
        self.assertNotIn("resumed", self._summary())


class TestRunnerCountsBankedUnits(unittest.TestCase):
    """run() counts units banked by an earlier session toward the population."""

    def test_fully_banked_unit_counts_as_covered(self):
        """One fully banked unit plus two indexed units cover all three."""
        _System.die_after = None
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.jsonl"
            out.write_text('{"query_id": "u0q0"}\n{"query_id": "u0q1"}\n',
                           encoding="utf-8")
            runner = BenchmarkRunner(output_path=out, verbose=False,
                                     resume=True)
            rows = list(runner.run(_System(config=DEFAULT_CONFIG), _Bench(),
                                   split="validation"))
            self.assertEqual(len(rows), 4)
            self.assertEqual(runner.n_units_processed, N_UNITS)


if __name__ == "__main__":
    unittest.main()
