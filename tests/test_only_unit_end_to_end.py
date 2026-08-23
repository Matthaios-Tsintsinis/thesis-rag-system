"""`--only-unit` drives main() end to end, and its summary must not lie.

THREE DEFECTS, all found by reading a real summary rather than the code.

1. A PARTIAL RUN'S LOADER STATS ARE NOT A CELL COUNT. `iter_eval_units`
   fills `stats` as it YIELDS, and a filtered pass stops consuming the
   generator once its cap is met — so `--only-unit` on story 12 of 40
   left `n_stories: 12`, `n_queries: 372`, and the summary recorded
   `expected_n_queries: 372`. P8's guard compares a cell's row count
   against that number, so it would have certified a third of the data
   as a complete cell.

2. THE POPULATION GATE NAMED A FLAG THAT WAS NEVER PASSED. It took an
   int, and the caller synthesised `1` from `--only-unit`, so the message
   read `--max-units 1 given explicitly` on a run whose summary recorded
   `max_units: null`. The gate's own output described a different run.

3. `evidence_budget_effective` read `retrieval_token_budget`; the field
   is `retrieval_budget_tokens`. A `getattr` with a wrong name returns
   None silently, so an M4 cell reported no budget while running a
   2,000-token one — the same shape as the two fields this file's
   predecessor already fixed.

NOT A DEFECT, recorded so it is not re-investigated: every NarrativeQA
row is `retrieval_skipped`. That benchmark ships no gold passages, so
`score_retrieval` returns `RetrievalScore(skipped=True)` unconditionally
(`narrativeqa.py:295`), on every cell, filtered or not.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.eval import runner as runner_mod
from src.eval.types import (
    ANSWER_TYPE_FREE_FORM,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)
from src.chunking import Chunk
from src.retrievers.base import AnswerResult, BaseSystem

N_UNITS = 5
QUERIES_PER_UNIT = 4


class _FiveUnitBenchmark:
    """Fills stats AS IT YIELDS, exactly as the real loaders do — which
    is what makes a partial run's stats describe the wrong population."""

    name = "five"
    cell_units = N_UNITS

    def __init__(self) -> None:
        self.stats = {"n_units": 0, "n_queries": 0}

    def iter_eval_units(self, split, max_units=None):
        for u in range(N_UNITS):
            queries = tuple(
                EvalQuery(
                    query_id=f"u{u}q{i}",
                    question_text=f"question {i} of unit {u}",
                    parent_scope=None,
                    gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                             free_form="Paris"),),
                    gold_passage_sets=(frozenset({(f"p{u}", "s0")}),),
                    question_type="factoid",
                )
                for i in range(QUERIES_PER_UNIT)
            )
            self.stats["n_units"] += 1
            self.stats["n_queries"] += len(queries)
            yield EvalUnit(
                corpus_id=f"story{u:02d}",
                corpus=(CorpusItem(item_id=f"p{u}::<whole>", parent_id=f"p{u}",
                                   span_id="<whole>", text="Paris is a city."),),
                queries=queries,
            )

    def score_retrieval(self, retrieved, query, scoring_ranking=None):
        return RetrievalScore(skipped=False, recall=1.0, precision=1.0,
                              f1=1.0, n_gold=1, n_covered=1,
                              n_retrieved_atoms=1)

    def score_answer(self, answer, query):
        return AnswerScore(value=1.0, method="stub")


class _Stub(BaseSystem):
    system_id = "STUB"
    supports_batched_answer = False

    def index(self, corpus_path):  # pragma: no cover
        self._indexed = True

    def index_items(self, items):
        self.chunks = [
            Chunk(chunk_id=i.item_id, doc_id=i.item_id, text=i.text,
                  n_words=3, position=0,
                  gold_provenance=((i.parent_id, i.span_id),))
            for i in items
        ]
        self._indexed = True

    def retrieve(self, query, k=None):
        from src.retrievers.base import RetrievedChunk

        return [RetrievedChunk(chunk=c, score=1.0, rank=n)
                for n, c in enumerate(self.chunks)]

    def answer(self, query, k=None):
        retrieved = self.retrieve(query)
        return AnswerResult(query=query, answer="Paris", retrieved=retrieved,
                            packed=retrieved, latency_s=0.01,
                            n_retrieval_calls=1, n_input_tokens=10,
                            evidence_tokens=5, extra={})


class TestOnlyUnit(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self._p = [
            mock.patch.dict(runner_mod.SYSTEM_REGISTRY, {"STUB": _Stub}),
            mock.patch.dict(runner_mod.BENCHMARK_REGISTRY,
                            {"five": _FiveUnitBenchmark}),
        ]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in self._p:
            p.stop()
        self.td.cleanup()

    def _run(self, *extra):
        out = self.dir / "five_STUB.jsonl"
        argv = ["runner", "--system", "STUB", "--benchmark", "five",
                "--split", "validation", "--output", str(out),
                "--allow-unpinned", *extra]
        buf: list[str] = []
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("builtins.print", lambda *a, **k: buf.append(
                 " ".join(str(x) for x in a))):
            runner_mod.main()
        summary = json.loads(
            out.with_suffix(".summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(l) for l in
                out.read_text(encoding="utf-8").splitlines() if l.strip()]
        return summary, rows, "\n".join(buf)

    def test_it_filters_to_the_unit_rather_than_truncating_at_it(self):
        summary, rows, _ = self._run("--only-unit", "story03")
        self.assertEqual(len(rows), QUERIES_PER_UNIT)
        self.assertTrue(all(r["query_id"].startswith("u3") for r in rows))

    def test_rows_record_the_packed_set_identity(self):
        """packed_ids: the field added after a per-row M2-vs-M3 set
        comparison proved unrecoverable from banked rows. Driven through
        the REAL runner: every row must carry the ids, in order, matching
        n_packed — the pipeline reads the chunks, not a constant."""
        _, rows, _ = self._run("--only-unit", "u1")
        for r in rows:
            ids = r["metadata"]["packed_ids"]
            self.assertEqual(len(ids), r["n_packed"])
            self.assertTrue(all(isinstance(i, str) and i for i in ids))

    def test_retrieval_is_not_skipped(self):
        summary, rows, _ = self._run("--only-unit", "story03")
        self.assertEqual(summary["n_retrieval_skipped"], 0)
        self.assertEqual(summary["mean_retrieval_f1"], 1.0)

    def test_a_partial_run_never_reports_a_cell_count(self):
        """THE DANGEROUS ONE. Loader stats after a filtered pass describe
        the units consumed, not the cell, and P8 compares a cell's rows
        against this number."""
        summary, _, _ = self._run("--only-unit", "story03")
        self.assertTrue(summary["partial_run"])
        self.assertIsNone(summary["expected_n_queries"])
        self.assertIn("PARTIAL", summary["expected_n_queries_scope"])

    def test_a_full_run_reports_the_whole_cell(self):
        summary, rows, _ = self._run()
        self.assertFalse(summary["partial_run"])
        self.assertEqual(summary["expected_n_queries"],
                         N_UNITS * QUERIES_PER_UNIT)
        self.assertEqual(len(rows), N_UNITS * QUERIES_PER_UNIT)
        self.assertEqual(summary["expected_n_queries_scope"], "full cell")

    def test_the_population_message_names_the_flag_actually_given(self):
        _, _, out = self._run("--only-unit", "story03")
        self.assertIn("--only-unit", out)
        self.assertNotIn("--max-units 1 given explicitly", out)

    def test_max_units_still_names_itself(self):
        _, _, out = self._run("--max-units", "2")
        self.assertIn("--max-units 2 given explicitly", out)

    def test_only_unit_is_recorded_in_the_summary(self):
        summary, _, _ = self._run("--only-unit", "story03")
        self.assertEqual(summary["only_unit"], "story03")
        self.assertIsNone(summary["max_units"])


if __name__ == "__main__":
    unittest.main()
