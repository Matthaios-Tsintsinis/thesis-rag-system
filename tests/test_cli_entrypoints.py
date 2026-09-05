"""End-to-end tests of the runner CLI: flags, registry lookup, the lockfile
gate, output paths and summary shape. The system is a stub, so no model
loads and no GPU is needed."""

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
    """One document and two queries, enough to drive the whole runner."""

    name = "tiny"

    def __init__(self) -> None:
        # The runner checks the scored count against the loader's n_queries,
        # so the stub carries one that matches its two queries.
        self.stats = {"n_units": 1, "n_queries": 2}

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
        # Record whether the runner passed the depth-50 scoring ranking
        # beside the reader context.
        self.saw_scoring_ranking = scoring_ranking is not None
        return RetrievalScore(skipped=False, recall=1.0, precision=1.0, f1=1.0,
                              n_gold=1, n_covered=1, n_retrieved_atoms=1)

    def score_answer(self, answer, query):
        from src.eval.types import AnswerScore

        return AnswerScore(value=1.0 if "Paris" in answer else 0.0,
                           method="stub")


class _StubSystem(BaseSystem):
    """Answers "Paris" without a model so only the runner plumbing runs."""

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
        # Record the cap the CLI handed this system.
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
    """Base case: patches the stub system and benchmark into the registries."""

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

    def _run(self, *extra, out=None):
        """Drive main() end to end with a lockfile and a stubbed pin check."""
        out = out or self.dir / "tiny_STUB.jsonl"
        # The pin gate always runs, so every drive carries a lockfile and
        # stubs the version check green.
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
        """main() aborts on a missing lockfile before writing any output."""
        out = self.dir / "gated.jsonl"
        argv = ["runner", "--system", "STUB", "--benchmark", "tiny",
                "--split", "validation", "--output", str(out),
                "--lockfile", str(self.dir / "definitely-absent.lock")]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                runner_mod.main()
        self.assertFalse(out.exists(), "the gate fired too late")


class TestHelpRenders(unittest.TestCase):
    """--help renders and lists exactly the seven runner flags."""

    def test_help_exits_zero_and_names_exactly_the_kept_flags(self):
        """--help exits 0 and names the seven flags, no more, no fewer."""
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
    """main() produces the artifacts the rest of the pipeline reads."""

    def test_main_runs_and_writes_both_artifacts(self):
        """main() writes the JSONL and the summary with one row per query."""
        out = self._run()
        self.assertTrue(out.exists(), "no JSONL written")
        summary = out.with_suffix(".summary.json")
        self.assertTrue(summary.exists(), "no summary.json written")

        rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["query_id"] for r in rows}, {"q1", "q2"})

    def test_summary_carries_run_provenance(self):
        """The summary carries the provenance fields and the scored count."""
        out = self._run()
        s = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
        for key in ("system", "benchmark", "split", "n_queries_scored",
                    "generator", "git_commit", "max_new_tokens"):
            self.assertIn(key, s, f"summary lost the {key!r} field")
        self.assertEqual(s["n_queries_scored"], 2)

    def test_the_summary_main_writes_is_the_one_the_exporter_reads(self):
        """The exporter's read_cell reads the summary and JSONL of main()."""
        from scripts.export_comparison import read_cell

        out = self._run(out=self.dir / "tiny_STUB_validation.jsonl")
        generator = DEFAULT_CONFIG.generation.model
        # The stub never abstains, so the credited-refusal entry is (0, 0.0).
        row = read_cell(self.dir, generator, "tiny", "STUB",
                        {(generator, "tiny", "STUB"): (0, 0.0)})
        self.assertEqual(row["n_queries"], 2)
        self.assertEqual(row["n_credited"], 0)
        self.assertEqual(row["generator"], generator)
        self.assertEqual(float(row["mean_answer_score_answerable"]),
                         float(json.loads((self.dir / "tiny_STUB_validation.summary.json")
                                          .read_text(encoding="utf-8"))
                               ["mean_answer_score_answerable"]))

    def test_summary_provenance_hashes_the_lockfile_the_gate_checked(self):
        """The provenance block hashes the --lockfile the gate checked."""
        from scripts.pin_environment import lockfile_hash

        out = self._run()
        s = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
        lock_text = (self.dir / "requirements.lock").read_text(encoding="utf-8")
        self.assertEqual(s["environment"]["lockfile_hash"], lockfile_hash(lock_text))

    def test_default_leaves_the_configured_cap_alone(self):
        """The system sees the configured max_new_tokens unchanged."""
        self._run()
        self.assertEqual(
            set(_StubSystem.seen_max_new_tokens),
            {DEFAULT_CONFIG.generation.max_new_tokens},
        )

    def test_a_failing_preflight_aborts_before_anything_expensive(self):
        """A preflight error propagates out of main() before indexing."""
        class _Doomed(_TinyBenchmark):
            def preflight(self):
                raise RuntimeError("dataset id unresolvable")

        with mock.patch.dict(runner_mod.BENCHMARK_REGISTRY, {"tiny": _Doomed}):
            with self.assertRaises(RuntimeError):
                self._run()

    def test_a_benchmark_without_preflight_still_runs(self):
        """A benchmark with no preflight hook still runs."""
        self.assertFalse(hasattr(_TinyBenchmark, "preflight"))
        self.assertTrue(self._run().exists())


if __name__ == "__main__":
    unittest.main()
