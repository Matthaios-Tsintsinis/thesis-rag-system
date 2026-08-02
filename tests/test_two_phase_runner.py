"""Two-phase runner: batching must be a throughput change, nothing else.

The invariant under test is that `--batch-size N` produces the SAME
rows, in the SAME order, with the same scores, as the sequential path.
If batching can reorder or mismatch answers to queries, every downstream
number is quietly wrong — and the failure looks like plausible answers
attached to the wrong questions, not like an exception.
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
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
    AnswerScore,
)
from src.retrievers.base import BaseSystem, RetrievedChunk


class FakeSystem(BaseSystem):
    """Retrieval is deterministic and cheap; generation is what we batch."""

    system_id = "FAKE"

    def index(self, corpus_path):  # pragma: no cover - unused
        self._indexed = True

    def index_items(self, items):
        self._indexed = True
        self.chunks = [
            Chunk(chunk_id=i.item_id, doc_id=i.item_id, text=i.text,
                  n_words=len(i.text.split()), position=n,
                  gold_provenance=((i.parent_id, i.span_id),))
            for n, i in enumerate(items)
        ]

    def retrieve(self, query: str, k: int | None = None):
        # Deterministic, query-dependent, and of VARYING length so the
        # length-sorted batching actually reorders things.
        n = 1 + (len(query) % 3)
        return [
            RetrievedChunk(chunk=c, score=1.0 / (j + 1), rank=j)
            for j, c in enumerate(self.chunks[:n])
        ]


class FakeBenchmark:
    name = "fake"

    def __init__(self, n_queries: int = 7):
        self.n_queries = n_queries

    def iter_eval_units(self, *, split: str, max_units=None):
        corpus = tuple(
            CorpusItem(item_id=f"item{i}", parent_id="P", span_id=f"s{i}",
                       text=f"body text number {i} " * (i + 1))
            for i in range(4)
        )
        queries = tuple(
            EvalQuery(
                query_id=f"q{i}",
                # Deliberately varied lengths -> length sorting permutes.
                question_text="?" * (i % 4) + f" question number {i}",
                parent_scope=None,
                gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                         free_form=f"gold{i}"),),
                gold_passage_sets=(frozenset({("P", "s0")}),),
                question_type="free_form",
            )
            for i in range(self.n_queries)
        )
        yield EvalUnit(corpus_id="unit0", corpus=corpus, queries=queries)

    def score_retrieval(self, retrieved, query) -> RetrievalScore:
        return RetrievalScore(skipped=False, f1=float(len(retrieved)))

    def score_answer(self, predicted, query) -> AnswerScore:
        return AnswerScore(value=1.0 if query.query_id in predicted else 0.0,
                           method="contains_id")


def _fake_generate(system_prompt, user_prompt, cfg=None):
    """Answer encodes its own prompt so mis-pairing is detectable."""
    return f"ANSWER<{user_prompt[-24:]}>"


def _fake_generate_batch(system_prompts, user_prompts, cfg=None, **kw):
    return [_fake_generate(s, u) for s, u in zip(system_prompts, user_prompts)]


def _run(batch_size):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "o.jsonl"
        sysm = FakeSystem(config=DEFAULT_CONFIG)
        runner = BenchmarkRunner(output_path=out, verbose=False,
                                 batch_size=batch_size)
        with mock.patch("src.models.generate", _fake_generate), \
             mock.patch("src.models.generate_batch", _fake_generate_batch):
            list(runner.run(sysm, FakeBenchmark(), split="validation"))
        return [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]


class TestTwoPhaseEquivalence(unittest.TestCase):
    def test_batched_matches_sequential_exactly(self):
        seq = _run(None)
        bat = _run(3)
        self.assertEqual(len(seq), len(bat))
        for a, b in zip(seq, bat):
            self.assertEqual(a["query_id"], b["query_id"])
            self.assertEqual(a["predicted_answer"], b["predicted_answer"])
            self.assertEqual(a["retrieval"]["f1"], b["retrieval"]["f1"])
            self.assertEqual(a["answer"]["value"], b["answer"]["value"])

    def test_output_stays_in_query_order(self):
        rows = _run(3)
        self.assertEqual([r["query_id"] for r in rows],
                         [f"q{i}" for i in range(7)])

    def test_answers_are_paired_with_their_own_prompts(self):
        """The failure mode batching introduces: right answers, wrong rows."""
        for row in _run(2):
            self.assertIn(row["question_text"][-12:], row["predicted_answer"])

    def test_batch_larger_than_unit_is_fine(self):
        self.assertEqual(len(_run(999)), 7)

    def test_batch_of_one_is_fine(self):
        self.assertEqual([r["query_id"] for r in _run(1)],
                         [f"q{i}" for i in range(7)])


class TestFallback(unittest.TestCase):
    def test_unconverted_systems_fall_back_to_sequential(self):
        """M1/M7 still override answer(); batching must not silently
        bypass those overrides."""
        class Unconverted(FakeSystem):
            supports_batched_answer = False

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.jsonl"
            sysm = Unconverted(config=DEFAULT_CONFIG)
            runner = BenchmarkRunner(output_path=out, verbose=False,
                                     batch_size=4)
            with mock.patch("src.models.generate", _fake_generate), \
                 mock.patch("src.models.generate_batch") as gb:
                list(runner.run(sysm, FakeBenchmark(), split="validation"))
            gb.assert_not_called()

    def test_max_queries_still_caps(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.jsonl"
            runner = BenchmarkRunner(output_path=out, verbose=False,
                                     batch_size=3)
            with mock.patch("src.models.generate", _fake_generate), \
                 mock.patch("src.models.generate_batch", _fake_generate_batch):
                rows = list(runner.run(FakeSystem(config=DEFAULT_CONFIG),
                                       FakeBenchmark(), split="validation",
                                       max_queries=4))
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
