"""Pins the cold-tree gate: an M4 cell aborts on the first warm tree and
names a fresh THESIS_CACHE_DIR as the remedy."""

from __future__ import annotations

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
from src.retrievers.base import AnswerResult, BaseSystem, RetrievedChunk


class _TreeSystem(BaseSystem):
    """Stub M4 that reports a warm or cold tree and counts index calls."""

    system_id = "M4"

    def __init__(self, *, warm: bool, **kw):
        super().__init__(**kw)
        self._warm = warm
        self.tree_cache_hit: bool | None = None
        self.n_indexed = 0

    def index(self, corpus_path):  # pragma: no cover
        self._indexed = True

    def index_items(self, items):
        self.n_indexed += 1
        self.tree_cache_hit = self._warm
        self.chunks = [
            Chunk(chunk_id=i.item_id, doc_id=i.item_id, text=i.text,
                  n_words=3, position=0,
                  gold_provenance=((i.parent_id, i.span_id),))
            for i in items
        ]
        self._indexed = True

    def retrieve(self, query, k=None):
        return [RetrievedChunk(chunk=c, score=1.0, rank=n)
                for n, c in enumerate(self.chunks)]

    def answer(self, query, k=None):
        r = self.retrieve(query)
        return AnswerResult(query=query, answer="x", retrieved=r, packed=r,
                            latency_s=0.01, n_retrieval_calls=1,
                            n_input_tokens=10, evidence_tokens=5, extra={})


class _TwoUnitBenchmark:
    """Two one-query units with fixed scores."""

    name = "two"
    cell_units = 2

    def __init__(self):
        self.stats = {"n_units": 2, "n_queries": 2}

    def iter_eval_units(self, split, max_units=None):
        for u in range(2):
            yield EvalUnit(
                corpus_id=f"unit{u}",
                corpus=(CorpusItem(item_id=f"p{u}::<whole>", parent_id=f"p{u}",
                                   span_id="<whole>", text="text here"),),
                queries=(EvalQuery(
                    query_id=f"u{u}q0", question_text="q?", parent_scope=None,
                    gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                             free_form="x"),),
                    gold_passage_sets=(frozenset({(f"p{u}", "s0")}),),
                    question_type="factoid"),),
            )

    def score_retrieval(self, retrieved, query, scoring_ranking=None):
        return RetrievalScore(skipped=False, f1=1.0)

    def score_answer(self, answer, query):
        return AnswerScore(value=1.0, method="stub")


def _run(*, warm: bool, require_cold: bool):
    """Runs the two-unit benchmark through BenchmarkRunner in a temp dir."""
    with tempfile.TemporaryDirectory() as td:
        system = _TreeSystem(warm=warm, config=DEFAULT_CONFIG)
        runner = BenchmarkRunner(
            output_path=Path(td) / "o.jsonl", verbose=False,
            require_cold_tree=require_cold,
        )
        rows = list(runner.run(system, _TwoUnitBenchmark(),
                               split="validation"))
        return system, rows


class TestColdTreeGate(unittest.TestCase):
    """Pins the gate's behaviour inside BenchmarkRunner.run."""

    def test_a_warm_tree_aborts_when_cold_is_required(self):
        """A warm tree under require_cold_tree raises SystemExit."""
        with self.assertRaises(SystemExit):
            _run(warm=True, require_cold=True)

    def test_it_aborts_on_the_FIRST_unit_not_after_the_pass(self):
        """The abort lands on the first warm unit, before a second index."""
        system = None
        try:
            system, _ = _run(warm=True, require_cold=True)
        except SystemExit:
            pass
        # Drive again holding the system so the index count is readable.
        with tempfile.TemporaryDirectory() as td:
            s = _TreeSystem(warm=True, config=DEFAULT_CONFIG)
            r = BenchmarkRunner(output_path=Path(td) / "o.jsonl",
                                verbose=False, require_cold_tree=True)
            with self.assertRaises(SystemExit):
                list(r.run(s, _TwoUnitBenchmark(), split="validation"))
            self.assertEqual(s.n_indexed, 1)

    def test_a_cold_tree_passes(self):
        """A cold tree under require_cold_tree indexes and scores two units."""
        system, rows = _run(warm=False, require_cold=True)
        self.assertEqual(len(rows), 2)
        self.assertEqual(system.n_indexed, 2)

    def test_warm_is_allowed_when_not_required(self):
        """Without require_cold_tree a warm tree runs; M1/M2/M3 build none."""
        system, rows = _run(warm=True, require_cold=False)
        self.assertEqual(len(rows), 2)

    def test_the_message_names_the_rule_and_the_remedy(self):
        """The abort message names tree_cache_hit and the cache dir remedy."""
        with self.assertRaises(SystemExit) as ctx:
            _run(warm=True, require_cold=True)
        msg = str(ctx.exception).lower()
        self.assertIn("tree_cache_hit", msg)
        self.assertIn("delete", msg)
        self.assertIn("thesis_cache_dir", msg)


class TestWiring(unittest.TestCase):
    """Pins that runner.main turns the gate on for M4 end to end."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.td.cleanup()

    def _drive(self, *extra, warm: bool):
        """Runs runner.main with a stub M4 and returns the summary JSON."""
        import json
        import sys

        from src.eval import runner as runner_mod

        class _M4Stub(_TreeSystem):
            def __init__(self, **kw):
                super().__init__(warm=warm, **kw)

        out = Path(self.td.name) / "two_M4.jsonl"
        # The pin gate always runs, so pass a lockfile and stub its check
        # green; only the cold-tree gate is under test here.
        lock = Path(self.td.name) / "requirements.lock"
        lock.write_text("# lock\nnumpy==0.0.0\n", encoding="utf-8")
        argv = ["runner", "--system", "M4", "--benchmark", "two",
                "--split", "validation", "--output", str(out),
                "--lockfile", str(lock), *extra]
        with mock.patch.dict(runner_mod.SYSTEM_REGISTRY, {"M4": _M4Stub}), \
             mock.patch.dict(runner_mod.BENCHMARK_REGISTRY,
                             {"two": _TwoUnitBenchmark}), \
             mock.patch("scripts.pin_environment.check_lockfile",
                        return_value=0), \
             mock.patch.object(sys, "argv", argv):
            runner_mod.main()
        return json.loads(
            out.with_suffix(".summary.json").read_text(encoding="utf-8"))

    def test_an_M4_cell_aborts_on_a_warm_tree_by_default(self):
        """The gate is on for M4 with no flag: a warm tree exits."""
        with self.assertRaises(SystemExit):
            self._drive(warm=True)

    def test_a_cold_M4_cell_runs_and_records_no_cache_hit(self):
        """A cold M4 cell completes and the summary records no cache hit."""
        summary = self._drive(warm=False)
        self.assertFalse(summary["tree_cache_hit"])
        self.assertEqual(summary["n_queries_scored"], 2)


if __name__ == "__main__":
    unittest.main()
