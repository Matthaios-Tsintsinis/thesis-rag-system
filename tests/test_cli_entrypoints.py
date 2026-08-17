"""End-to-end smoke for the three CLI entry points. No GPU, no model.

WHY THIS FILE EXISTS. `b3d3df3` shipped with `runner.main()` raising
`NameError: name 'system_cls' is not defined` on its very first use — the
registry lookup was dropped while CLI flags were added around it. The
full suite was 215/215 green, because NOTHING under tests/ called
`main()`. A broken runner costs a Colab session and a 15GB model download
to discover.

An audit at that point found the same gap in all three CLIs:

    runner     no test imported it at all
    analyse    two tests, both of private helpers (_aggregate); main() never called
    aggregate  no test touched it in any form

These tests exercise the WIRING — argument parsing, registry lookup,
config construction, output paths, summary shape, and the handoff from
runner to analyse to aggregate. They deliberately stub the system so no
model loads: the failure being guarded against lives in the plumbing, and
a test that needed a GPU would not run often enough to catch it.

Any commit touching runner.py / analyse.py / aggregate.py has to pass
this.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import DEFAULT_CONFIG
from src.eval import aggregate as aggregate_mod
from src.eval import analyse as analyse_mod
from src.config import MATRIX_BATCH_SIZE
from src.eval import runner as runner_mod
from src.eval.types import (
    ANSWER_TYPE_FREE_FORM,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)
from src.retrievers.base import AnswerResult, BaseSystem, RetrievedChunk
from src.chunking import Chunk


# --- a benchmark and a system small enough to run anywhere ---------------


class _TinyBenchmark:
    name = "tiny"

    def __init__(self) -> None:
        # `n_queries` is part of the loader contract, not decoration:
        # the runner asserts an uncapped cell carries a loader-derived
        # count, because a null there disarms P8's short-cell guard
        # without removing the appearance of it. A stub that omitted it
        # was a stub that could not have caught the HotpotQA null.
        self.stats = {"n_units": 1, "n_queries": 2}  # the two below

    def iter_eval_units(self, split, max_units=None):
        corpus = (
            CorpusItem(item_id="d1::<whole>", parent_id="d1", span_id="<whole>",
                       text="Paris is the capital of France."),
        )
        queries = (
            EvalQuery(
                query_id="q1", question_text="What is the capital of France?",
                parent_scope=None,
                gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                         free_form="Paris"),),
                gold_passage_sets=(frozenset({("d1", "<whole>")}),),
                question_type="factoid",
            ),
            EvalQuery(
                query_id="q2", question_text="What is the capital of Spain?",
                parent_scope=None,
                gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                         free_form="Madrid"),),
                gold_passage_sets=(frozenset({("d1", "<whole>")}),),
                question_type="factoid",
            ),
        )
        yield EvalUnit(corpus_id="tiny", corpus=corpus, queries=queries)

    def score_retrieval(self, retrieved, query, scoring_ranking=None):
        # P6: the Benchmark protocol takes the fixed-depth scoring
        # ranking alongside the reader context. A stub that did not
        # accept it would let the runner pass a protocol it does not
        # implement, which is what this end-to-end test exists to catch.
        self.saw_scoring_ranking = scoring_ranking is not None
        return RetrievalScore(skipped=False, recall=1.0, precision=1.0, f1=1.0,
                              n_gold=1, n_covered=1, n_retrieved_atoms=1)

    def score_answer(self, answer, query):
        from src.eval.types import AnswerScore

        return AnswerScore(value=1.0 if "Paris" in answer else 0.0,
                           method="stub")


class _StubSystem(BaseSystem):
    """Answers without generating. The plumbing is the subject here."""

    system_id = "STUB"
    supports_batched_answer = False
    seen_max_new_tokens: list[int] = []

    def index(self, corpus_path):  # pragma: no cover - unused
        self._indexed = True

    def index_items(self, items):
        self.chunks = [
            Chunk(chunk_id=i.item_id, doc_id=i.item_id, text=i.text,
                  n_words=len(i.text.split()), position=0,
                  gold_provenance=((i.parent_id, i.span_id),))
            for i in items
        ]
        self._indexed = True

    def retrieve(self, query, k=None):
        return [RetrievedChunk(chunk=c, score=1.0, rank=n,
                               source_unit_type="chunk")
                for n, c in enumerate(self.chunks)]

    def answer(self, query, k=None):
        # Record what the CLI actually handed us — this is the assertion
        # that --max-new-tokens reaches the object generation consumes.
        type(self).seen_max_new_tokens.append(
            self.config.generation.max_new_tokens
        )
        retrieved = self.retrieve(query)
        return AnswerResult(
            query=query, answer="Paris", retrieved=retrieved,
            packed=retrieved, latency_s=0.01, n_retrieval_calls=1,
            n_input_tokens=10, evidence_tokens=5, extra={},
        )


class _CliCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        _StubSystem.seen_max_new_tokens = []
        self._patches = [
            mock.patch.dict(runner_mod.SYSTEM_REGISTRY, {"STUB": _StubSystem}),
            mock.patch.dict(runner_mod.BENCHMARK_REGISTRY,
                            {"tiny": _TinyBenchmark}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.td.cleanup()

    def _run(self, *extra):
        out = self.dir / "tiny_STUB.jsonl"
        # --allow-unpinned because these are CLI-surface tests on a host
        # with no requirements.lock. The lockfile gate is deliberately a
        # HARD abort (it guards a matrix split that is undetectable after
        # the fact), so every non-P10 caller has to say so explicitly.
        # That the flag is needed HERE is the gate proving it is wired:
        # dropping it makes these tests fail, which is the check the
        # gate's own test cannot make about main().
        argv = ["runner", "--system", "STUB", "--benchmark", "tiny",
                "--split", "validation", "--output", str(out),
                "--allow-unpinned", *extra]
        with mock.patch.object(sys, "argv", argv):
            runner_mod.main()
        return out

    def test_the_lockfile_gate_is_wired_into_main(self):
        """END TO END, not a unit test of the gate function.

        Without --allow-unpinned and with no lockfile present, main() must
        abort BEFORE it writes any output. This is the difference between
        "the gate exists" and "the runner calls the gate" — the exact
        distinction that made MATRIX_BATCH_SIZE inert through a whole
        release.
        """
        out = self.dir / "gated.jsonl"
        argv = ["runner", "--system", "STUB", "--benchmark", "tiny",
                "--split", "validation", "--output", str(out),
                "--lockfile", str(self.dir / "definitely-absent.lock")]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                runner_mod.main()
        self.assertFalse(out.exists(), "the gate fired too late")


class TestBatchSizeDefault(_CliCase):
    """The runner READS config.MATRIX_BATCH_SIZE — the property the P9
    commit claimed and left inert.

    The test P9 shipped asserted MATRIX_BATCH_SIZE == 16, which is a
    tautology about a constant and proves nothing about the pipeline: the
    flag still defaulted to None, so a cell launched without an explicit
    --batch-size fell back to SEQUENTIAL answering. Batch composition can
    move generated text at temperature 0, so that cell would not have
    been strictly comparable to the rest of the matrix, and nothing in
    the run would have said so.

    These assertions read what the CLI RESOLVED, from the artifact a real
    run leaves behind.
    """

    def _summary(self, *extra):
        out = self._run(*extra)
        return json.loads(
            out.with_suffix(".summary.json").read_text(encoding="utf-8"))

    def test_no_flag_resolves_to_the_configured_matrix_batch_size(self):
        self.assertEqual(self._summary()["batch_size"], MATRIX_BATCH_SIZE)

    def test_the_flag_still_overrides_downward_for_a_cost_probe(self):
        self.assertEqual(self._summary("--batch-size", "4")["batch_size"], 4)

    def test_zero_is_the_explicit_sequential_opt_out(self):
        """An escape hatch reachable only by editing config is not an
        escape hatch."""
        self.assertIsNone(self._summary("--batch-size", "0")["batch_size"])


class TestRunnerMain(_CliCase):
    def test_main_runs_and_writes_both_artifacts(self):
        """The regression: main() raised NameError on the registry lookup
        for every system, with and without the new flags."""
        out = self._run()
        self.assertTrue(out.exists(), "no JSONL written")
        summary = out.with_suffix(".summary.json")
        self.assertTrue(summary.exists(), "no summary.json written")

        rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["query_id"] for r in rows}, {"q1", "q2"})

    def test_summary_carries_run_provenance(self):
        out = self._run()
        s = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
        for key in ("system", "benchmark", "split", "n_queries_scored",
                    "generator", "git_commit", "max_new_tokens"):
            self.assertIn(key, s, f"summary lost the {key!r} field")
        self.assertEqual(s["n_queries_scored"], 2)

    def test_max_new_tokens_reaches_the_system_config(self):
        """The whole point of the flag: rebinding the module constant does
        NOT work, so the CLI must construct the config explicitly."""
        self._run("--max-new-tokens", "7")
        self.assertTrue(_StubSystem.seen_max_new_tokens)
        self.assertEqual(set(_StubSystem.seen_max_new_tokens), {7})

    def test_default_leaves_the_configured_cap_alone(self):
        self._run()
        self.assertEqual(
            set(_StubSystem.seen_max_new_tokens),
            {DEFAULT_CONFIG.generation.max_new_tokens},
        )

    def test_max_queries_caps_the_run(self):
        out = self._run("--max-queries", "1")
        rows = [x for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(rows), 1)

    def test_a_bad_cap_is_rejected_not_silently_ignored(self):
        with self.assertRaises(SystemExit):
            self._run("--max-new-tokens", "0")

    def test_preflight_runs_before_the_generator_is_loaded(self):
        """Cheap preconditions before expensive ones.

        A HotpotQA run died on an unresolvable dataset id at the first
        iter_eval_units call -- AFTER --prewarm had pulled 15 GB of Qwen
        into VRAM. A two-second metadata check would have failed in two
        seconds. This pins the ORDER, because a later refactor that moves
        benchmark construction back below prewarm would silently restore
        the two-minute failure.
        """
        order: list[str] = []

        class _Preflighting(_TinyBenchmark):
            def preflight(self):
                order.append("preflight")

        def _fake_load(*a, **kw):
            order.append("load_generator")
            return (None, None)

        with mock.patch.dict(runner_mod.BENCHMARK_REGISTRY,
                             {"tiny": _Preflighting}), \
                mock.patch("src.models.load_generator", _fake_load):
            self._run("--prewarm")

        self.assertEqual(order, ["preflight", "load_generator"])

    def test_a_failing_preflight_aborts_before_anything_expensive(self):
        class _Doomed(_TinyBenchmark):
            def preflight(self):
                raise RuntimeError("dataset id unresolvable")

        with mock.patch.dict(runner_mod.BENCHMARK_REGISTRY, {"tiny": _Doomed}):
            with self.assertRaises(RuntimeError):
                self._run()

    def test_a_benchmark_without_preflight_still_runs(self):
        """The hook is optional: benchmarks that have not added one are
        simply not checked, never broken."""
        self.assertFalse(hasattr(_TinyBenchmark, "preflight"))
        self.assertTrue(self._run().exists())


class TestAnalyseMain(_CliCase):
    def test_analyse_main_reads_the_runner_output(self):
        """analyse had two tests, both of _aggregate. main() was never
        called, so its argument wiring was as unguarded as runner's."""
        out = self._run()
        dump = self.dir / "agg.json"
        argv = ["analyse", str(out), "--output", str(dump)]
        with mock.patch.object(sys, "argv", argv):
            analyse_mod.main()
        self.assertTrue(dump.exists())
        rollup = json.loads(dump.read_text(encoding="utf-8"))
        self.assertEqual(rollup["n_total_records"], 2)
        self.assertIn("STUB", rollup["systems"])

    def test_by_type_slice_does_not_crash(self):
        out = self._run()
        with mock.patch.object(sys, "argv", ["analyse", str(out), "--by-type"]):
            analyse_mod.main()


class TestAggregateMain(_CliCase):
    def test_aggregate_main_renders_a_table(self):
        """aggregate was touched by NO test in any form."""
        self._run()
        outdir = self.dir / "agg"
        argv = ["aggregate", str(self.dir), "--output-dir", str(outdir),
                "--no-deep"]
        with mock.patch.object(sys, "argv", argv):
            aggregate_mod.main()
        written = list(outdir.glob("results_*.md")) + list(outdir.glob("results_*.csv"))
        self.assertTrue(written, "aggregate wrote nothing")

    def test_aggregate_exits_cleanly_when_there_is_nothing_to_read(self):
        empty = self.dir / "empty"
        empty.mkdir()
        argv = ["aggregate", str(empty), "--no-deep"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                aggregate_mod.main()


if __name__ == "__main__":
    unittest.main()
