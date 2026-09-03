"""End-to-end smoke for the runner CLI. No GPU, no model.

WHY THIS FILE EXISTS. `b3d3df3` shipped with `runner.main()` raising
`NameError: name 'system_cls' is not defined` on its very first use — the
registry lookup was dropped while CLI flags were added around it. The
full suite was 215/215 green, because NOTHING under tests/ called
`main()`. A broken runner costs a Colab session and a 15GB model download
to discover.

These tests exercise the WIRING — argument parsing, registry lookup,
config construction, output paths and summary shape. They deliberately
stub the system so no model loads: the failure being guarded against
lives in the plumbing, and a test that needed a GPU would not run often
enough to catch it.

Any commit touching runner.py has to pass this.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config import DEFAULT_CONFIG
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
        # Record what the CLI actually handed us — the assertion that
        # the configured cap reaches the object generation consumes.
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
        # The pin gate has no escape (the reduction removed it), so the
        # CLI drive carries a lockfile and stubs the version check green;
        # `test_the_lockfile_gate_is_wired_into_main` below drives main()
        # WITHOUT one and proves the gate fires first.
        lock = self.dir / "requirements.lock"
        lock.write_text("# lock\nnumpy==0.0.0\n", encoding="utf-8")
        argv = ["runner", "--system", "STUB", "--benchmark", "tiny",
                "--split", "validation", "--output", str(out),
                "--lockfile", str(lock), *extra]
        with mock.patch("scripts.pin_environment.check_lockfile",
                        return_value=0), \
                mock.patch.object(sys, "argv", argv):
            runner_mod.main()
        return out

    def test_the_lockfile_gate_is_wired_into_main(self):
        """END TO END, not a unit test of the gate function.

        With no lockfile present, main() must abort BEFORE it writes any
        output. This is the difference between
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


class TestHelpRenders(unittest.TestCase):
    """`--help` is the reduced CLI surface a stranger reads first.

    It CRASHED at the tag: the --benchmark help text carried bare percent
    signs ("91.7%"), which argparse expands as format specifiers, and
    nothing ever called --help. Driven end to end here, and the kept
    flags are asserted so a re-added flag or a lost one is visible.
    """

    def test_help_exits_zero_and_names_exactly_the_kept_flags(self):
        import contextlib
        import io

        buf = io.StringIO()
        with mock.patch.object(sys, "argv", ["runner", "--help"]), \
                contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                runner_mod.main()
        self.assertEqual(cm.exception.code, 0)
        text = buf.getvalue()
        flags = set(re.findall(r"^\s+(--[a-z-]+)", text, re.M))
        self.assertEqual(flags, {"--lockfile", "--system", "--benchmark",
                                 "--split", "--output", "--generator",
                                 "--resume"})


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

    def test_summary_provenance_hashes_the_lockfile_the_gate_checked(self):
        """--lockfile names the gate's lock; the provenance block must hash
        THAT file, not a hardcoded ./requirements.lock (which banked
        lockfile_hash=null whenever the two differed)."""
        from scripts.pin_environment import lockfile_hash

        out = self._run()
        s = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
        lock_text = (self.dir / "requirements.lock").read_text(encoding="utf-8")
        self.assertEqual(s["environment"]["lockfile_hash"], lockfile_hash(lock_text))

    def test_default_leaves_the_configured_cap_alone(self):
        self._run()
        self.assertEqual(
            set(_StubSystem.seen_max_new_tokens),
            {DEFAULT_CONFIG.generation.max_new_tokens},
        )

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


if __name__ == "__main__":
    unittest.main()
