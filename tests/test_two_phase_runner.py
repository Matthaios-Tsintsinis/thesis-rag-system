"""Two-phase runner: batching must be a throughput change, nothing else.

The invariant under test is that `--batch-size N` produces the SAME
rows, with the same scores, as the sequential path. If batching can
mismatch answers to queries, every downstream number is quietly wrong —
and the failure looks like plausible answers attached to the wrong
questions, not like an exception.

FILE ORDER IS NOT PART OF THE INVARIANT under batching. Rows are
emitted length-sorted so each batch is flushed as it finishes, which is
what makes a dead session cost a batch instead of a pass. Every
consumer parses per line into dicts and never depends on order, so the
guarantee is completeness plus correct pairing, not sequence. The
sequential path still emits in query order and that is asserted
separately.
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


def _by_id(rows):
    return {r["query_id"]: r for r in rows}


class TestTwoPhaseEquivalence(unittest.TestCase):
    def test_batched_matches_sequential_exactly(self):
        """Same rows, same content — compared BY query_id, because
        batched output is emitted in length-sorted order (see below)."""
        seq, bat = _by_id(_run(None)), _by_id(_run(3))
        self.assertEqual(set(seq), set(bat))
        for qid, a in seq.items():
            b = bat[qid]
            self.assertEqual(a["predicted_answer"], b["predicted_answer"])
            self.assertEqual(a["retrieval"]["f1"], b["retrieval"]["f1"])
            self.assertEqual(a["answer"]["value"], b["answer"]["value"])

    def test_every_query_appears_exactly_once(self):
        """The guarantee under batching is COMPLETENESS, not file order.

        Rows are emitted length-sorted so each batch can be flushed as
        it finishes; every consumer (analyse, aggregate, the
        significance diagnostic) parses per line into dicts and never
        depends on order. Completeness and correct pairing are what
        must hold.
        """
        ids = [r["query_id"] for r in _run(3)]
        self.assertEqual(sorted(ids), sorted(f"q{i}" for i in range(7)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_sequential_path_still_emits_in_query_order(self):
        self.assertEqual([r["query_id"] for r in _run(None)],
                         [f"q{i}" for i in range(7)])

    def test_answers_are_paired_with_their_own_prompts(self):
        """The failure mode batching introduces: right answers, wrong rows."""
        for row in _run(2):
            self.assertIn(row["question_text"][-12:], row["predicted_answer"])

    def test_batch_larger_than_unit_is_fine(self):
        self.assertEqual(len(_run(999)), 7)

    def test_batch_of_one_is_fine(self):
        self.assertEqual(sorted(r["query_id"] for r in _run(1)),
                         sorted(f"q{i}" for i in range(7)))


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


class TestDurabilityAndResume(unittest.TestCase):
    """A dead session must cost a batch, not a pass.

    This project's sessions have died to a reclaimed runtime, a Drive
    disconnect, an RPD ceiling, and a torch install broken by vLLM.
    Generating a whole unit before writing anything would mean MultiHop
    (one unit, 2,556 queries) loses over an hour to any of them.
    """

    def _crashing_batch(self, fail_on_call):
        state = {"n": 0}

        def gb(system_prompts, user_prompts, cfg=None, **kw):
            state["n"] += 1
            if state["n"] >= fail_on_call:
                raise RuntimeError("simulated runtime death")
            return _fake_generate_batch(system_prompts, user_prompts, cfg)

        return gb

    @staticmethod
    def _rows(path):
        return [
            json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    def test_completed_batches_survive_a_mid_pass_crash(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.jsonl"
            runner = BenchmarkRunner(
                output_path=out, verbose=False, batch_size=2
            )
            with mock.patch("src.models.generate", _fake_generate), mock.patch(
                "src.models.generate_batch", self._crashing_batch(3)
            ):
                with self.assertRaises(RuntimeError):
                    list(
                        runner.run(
                            FakeSystem(config=DEFAULT_CONFIG),
                            FakeBenchmark(),
                            split="validation",
                        )
                    )
            rows = self._rows(out)
        self.assertEqual(len(rows), 4, "completed batches were not flushed")

    def test_resume_skips_banked_queries_and_appends(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.jsonl"
            with mock.patch("src.models.generate", _fake_generate), mock.patch(
                "src.models.generate_batch", self._crashing_batch(3)
            ):
                with self.assertRaises(RuntimeError):
                    list(
                        BenchmarkRunner(
                            output_path=out, verbose=False, batch_size=2
                        ).run(
                            FakeSystem(config=DEFAULT_CONFIG),
                            FakeBenchmark(),
                            split="validation",
                        )
                    )
            partial = self._rows(out)

            with mock.patch("src.models.generate", _fake_generate), mock.patch(
                "src.models.generate_batch", _fake_generate_batch
            ):
                list(
                    BenchmarkRunner(
                        output_path=out,
                        verbose=False,
                        batch_size=2,
                        resume=True,
                    ).run(
                        FakeSystem(config=DEFAULT_CONFIG),
                        FakeBenchmark(),
                        split="validation",
                    )
                )
            rows = self._rows(out)

        ids = [r["query_id"] for r in rows]
        self.assertEqual(sorted(ids), sorted(f"q{i}" for i in range(7)))
        self.assertEqual(len(ids), len(set(ids)), "resume duplicated a query")
        self.assertGreater(len(rows), len(partial))

    def test_without_resume_a_rerun_truncates(self):
        """Historic behaviour, preserved as the default."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.jsonl"
            out.write_text('{"query_id": "stale"}\n', encoding="utf-8")
            with mock.patch("src.models.generate", _fake_generate), mock.patch(
                "src.models.generate_batch", _fake_generate_batch
            ):
                list(
                    BenchmarkRunner(
                        output_path=out, verbose=False, batch_size=2
                    ).run(
                        FakeSystem(config=DEFAULT_CONFIG),
                        FakeBenchmark(),
                        split="validation",
                    )
                )
            text = out.read_text(encoding="utf-8")
        self.assertNotIn("stale", text)

    def test_resume_tolerates_a_torn_final_line(self):
        """A session killed mid-write leaves partial JSON — the exact
        case resume exists for, so it must not raise."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.jsonl"
            out.write_text(
                '{"query_id": "q0", "x": 1}\n{"query_id": "q1"',
                encoding="utf-8",
            )
            runner = BenchmarkRunner(
                output_path=out, verbose=False, batch_size=2, resume=True
            )
            self.assertEqual(runner._existing_query_ids(), {"q0"})


if __name__ == "__main__":
    unittest.main()
