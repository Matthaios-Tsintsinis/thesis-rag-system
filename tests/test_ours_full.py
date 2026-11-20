"""The full round (OURS_VERSIONS "Full splits"): round 2's model on every question of
NarrativeQA's test split and HotpotQA's dev set (distractor), ours first. Pooled HotpotQA is
not in it (the operator, 2026-11-20).

The three full benchmarks read every story or question, never a draw, and declare the counts
the runner checks. Ours' cells run them into the full round's bank; ours builds every tree
of the full round itself, in a cache of its own, so M4's banked cache keeps only trees M4
built and M4's full cells can later build theirs cold."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ours import config as settings
from ours import model as om
from scripts import dev_ours
from src.eval import hotpotqa as hq
from src.eval import narrativeqa as nq
from src.retrievers.m4_raptor import M4_SUBSTRATE_NAMESPACE
from tests.test_hotpotqa import _FakeDS, _fake_rows
from tests.test_ours_generators import _MixedUnits, tree_dirs
from tests.test_ours_smoke import ANSWERS, DECOMPOSITIONS, FakeReader, llama_config, ours_fakes


def nqa_rows(n_stories: int, per_story: int = 2) -> list[dict]:
    """NarrativeQA's HF rows: one per question, the story repeated on each."""
    rows = []
    for s in range(n_stories):
        for q in range(per_story):
            rows.append({"document": {"id": f"doc{s:03d}", "text": f"Story {s}.", "kind": "gutenberg",
                                      "url": "", "summary": {"title": f"Title {s}"}},
                         "question": {"text": f"Question {q} about story {s}?"},
                         "answers": [{"text": "yes"}, {"text": "aye"}]})
    return rows


def splits_read(bench, rows) -> list[str]:
    """Serve `rows` as whatever split the loader asks for, and record which it asked for."""
    asked = []

    def load(split):
        asked.append(split)
        return rows
    bench._load_split = load
    return asked


class TestNarrativeQATest(unittest.TestCase):
    """Every story of the test split, the split the papers report; none of it ever read in development."""

    def test_every_story_not_the_draw(self):
        bench = nq.NarrativeQATestBenchmark()
        asked = splits_read(bench, nqa_rows(50))
        units = list(bench.iter_eval_units(split="test"))
        self.assertEqual(asked, ["test"])
        self.assertEqual([u.corpus_id for u in units], [f"doc{s:03d}" for s in range(50)])
        self.assertEqual(sum(len(u.queries) for u in units), 100)
        self.assertEqual(bench.stats["n_units_requested"], "all")
        self.assertEqual(bench.stats["sampled_story_ids"], [f"doc{s:03d}" for s in range(50)])

    def test_the_registered_loader_still_draws_its_forty(self):
        bench = nq.NarrativeQABenchmark()
        splits_read(bench, nqa_rows(50))
        self.assertEqual(len(list(bench.iter_eval_units(split="validation"))), nq.CELL_UNITS)

    def test_only_the_test_split(self):
        bench = nq.NarrativeQATestBenchmark()
        splits_read(bench, nqa_rows(3))
        for split in ("validation", "train"):
            with self.assertRaisesRegex(ValueError, "test"):
                list(bench.iter_eval_units(split=split))

    def test_it_declares_the_splits_355_stories(self):
        self.assertEqual(nq.NarrativeQATestBenchmark.name, "narrativeqa_test")
        self.assertEqual(nq.NarrativeQATestBenchmark().cell_units, 355)

    def test_all_is_a_cap_only_the_loader_can_ask_for(self):
        order = [f"s{i}" for i in range(500)]
        self.assertEqual(nq.select_units(order, nq.ALL_UNITS), order)


@contextlib.contextmanager
def hf_rows(rows):
    """datasets.load_dataset serving `rows`; the measurements, which count tokens, skipped."""
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: rows
    with mock.patch.dict(sys.modules, {"datasets": fake}), \
            mock.patch.object(hq.HotpotQABenchmark, "_measure", lambda self, ds: None):
        yield


class TestHotpotQAFull(unittest.TestCase):
    """Every dev question, distractor setting; the registered loader still draws its 1,000."""

    def test_every_question_not_the_draw(self):
        rows = _fake_rows(1200)
        with hf_rows(rows):
            self.assertEqual(len(hq.HotpotQAFullBenchmark()._load()), 1200)
            self.assertEqual(len(hq.HotpotQABenchmark()._load()), hq.PREREGISTERED_Q)
            self.assertEqual(len(hq.HotpotQAPooledBenchmark()._load()), hq.PREREGISTERED_Q)

    def test_they_declare_the_dev_sets_counts(self):
        self.assertEqual((hq.HotpotQAFullBenchmark.name, hq.HotpotQAFullBenchmark().cell_units),
                         ("hotpotqa_full", 7405))
        self.assertEqual(hq.HotpotQAFullBenchmark().variant, "distractor")
        self.assertFalse(hasattr(hq, "HotpotQAPooledFullBenchmark"))


class TestRegistry(unittest.TestCase):
    def test_the_runner_knows_the_full_benchmarks(self):
        from src.eval.runner import BENCHMARK_REGISTRY

        self.assertIs(BENCHMARK_REGISTRY["narrativeqa_test"], nq.NarrativeQATestBenchmark)
        self.assertIs(BENCHMARK_REGISTRY["hotpotqa_full"], hq.HotpotQAFullBenchmark)
        self.assertNotIn("hotpotqa_pooled_full", BENCHMARK_REGISTRY)
        self.assertIs(BENCHMARK_REGISTRY["narrativeqa"], nq.NarrativeQABenchmark)

    def test_the_full_rounds_benchmarks_and_splits(self):
        self.assertEqual(dev_ours.FULL_BENCHMARKS, ("narrativeqa_test", "hotpotqa_full"))
        self.assertEqual([dev_ours.split_of(b) for b in dev_ours.FULL_BENCHMARKS],
                         ["test", "validation"])
        self.assertEqual({dev_ours.split_of(b) for b in dev_ours.SCORED_BENCHMARKS}, {"validation"})

    def test_the_full_round_runs_round_twos_model(self):
        from scripts.report_ours import FROZEN_CODE, FULL_CODE

        self.assertEqual(FULL_CODE, {"ours_v20_full": FROZEN_CODE["ours_v20"]})


class TestFullCell(unittest.TestCase):
    """A full cell: the runner's own command with the reader, the benchmark's split, into the
    full round's bank, in ours' own cache only."""

    def setUp(self):
        from ours.model import CODE_FINGERPRINT

        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.bank, self.local = self.root / "drive" / "ours_v99_full", self.root / "local"
        self.code = CODE_FINGERPRINT
        self.cache = self.root / "cache_ours_full"
        self.patches = [mock.patch.dict("scripts.report_ours.FULL_CODE", {"ours_v99_full": "f" * 16}),
                        mock.patch.dict("scripts.report_ours.SAME_OUTPUTS", {self.code: "f" * 16}),
                        mock.patch("src.paths.cache_dir", lambda: self.cache)]
        for p in self.patches:
            p.start()
        self.commands = []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.td.cleanup()

    def _runner(self):
        def run(command, cwd):
            self.commands.append(command)
            out = Path(command[command.index("--output") + 1])
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"query_id": "q0", "metadata": {"ours_code": self.code}}) + "\n")
            out.with_suffix(".summary.json").write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        return run

    def _cell(self, benchmark, bank=None, generator=None):
        with contextlib.redirect_stdout(io.StringIO()):
            dev_ours.scored_cell(benchmark, bank or self.bank, self.local, run=self._runner(),
                                 generator=generator)

    def test_narrativeqa_test_reads_the_test_split_with_the_reader(self):
        self._cell("narrativeqa_test")
        command = self.commands[0]
        self.assertEqual(command[command.index("--benchmark") + 1], "narrativeqa_test")
        self.assertEqual(command[command.index("--split") + 1], "test")
        self.assertEqual(command[command.index("--generator") + 1], settings.READER)
        self.assertEqual(command[-1], "--resume")
        self.assertEqual(Path(command[command.index("--output") + 1]),
                         self.local / "ours_v99_full" / "narrativeqa_test_OURS_test.jsonl")
        self.assertEqual(sorted(p.name for p in self.bank.iterdir()),
                         ["narrativeqa_test_OURS_test.jsonl", "narrativeqa_test_OURS_test.summary.json"])

    def test_hotpotqa_reads_the_dev_set_in_any_order(self):
        self._cell("hotpotqa_full")
        self._cell("narrativeqa_test")
        self.assertEqual([c[c.index("--split") + 1] for c in self.commands], ["validation", "test"])
        self.assertIn("hotpotqa_full_OURS_validation.summary.json",
                      [p.name for p in self.bank.iterdir()])

    def test_a_finished_cell_is_not_run_again(self):
        self._cell("hotpotqa_full")
        self._cell("hotpotqa_full")
        self.assertEqual(len(self.commands), 1)

    def test_only_the_full_benchmarks(self):
        for benchmark in dev_ours.SCORED_BENCHMARKS:
            with self.assertRaisesRegex(SystemExit, "the full round runs"):
                self._cell(benchmark)
        self.assertEqual(self.commands, [])

    def test_a_rounds_bank_runs_its_registered_benchmarks_only(self):
        with mock.patch.dict("scripts.report_ours.FROZEN_CODE", {"ours_v99": self.code}):
            for benchmark in dev_ours.FULL_BENCHMARKS:
                with self.assertRaisesRegex(SystemExit, "a round runs"):
                    self._cell(benchmark, bank=self.root / "drive" / "ours_v99")
        self.assertEqual(self.commands, [])

    def test_only_in_ours_own_cache(self):
        for cache in (self.root / "cache", self.root / "cache_ours_dev"):
            with mock.patch("src.paths.cache_dir", lambda: cache):
                with self.assertRaisesRegex(SystemExit, "cache_ours_full"):
                    self._cell("hotpotqa_full")
        self.assertEqual(self.commands, [])

    def test_only_round_twos_model_and_the_reader(self):
        with mock.patch.dict("scripts.report_ours.SAME_OUTPUTS", {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "Check out its commit"):
                self._cell("hotpotqa_full")
        with self.assertRaisesRegex(SystemExit, "the sweep runs narrativeqa and multihop_rag"):
            self._cell("narrativeqa_test", generator="Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(self.commands, [])


class _SplitUnits(_MixedUnits):
    """The mixed units, recording the split each pass asks for."""

    asked: list = []

    def iter_eval_units(self, *, split, max_units=None):
        type(self).asked.append(split)
        yield from self.units


class TestFullTrees(unittest.TestCase):
    """Ahead of a full cell, every tree it will read, M4's included, built in ours' own cache."""

    def _build(self, cache, benchmark="narrativeqa_test", generator=None):
        with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)), \
                mock.patch.dict("src.eval.runner.BENCHMARK_REGISTRY", {benchmark: _SplitUnits}), \
                contextlib.redirect_stdout(io.StringIO()):
            return dev_ours.build_own_trees(benchmark, check_env=lambda lockfile: None,
                                            generator=generator)

    def test_every_tree_built_once_and_the_cell_reads_them_warm(self):
        _SplitUnits.asked = []
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache_ours_full"
            self.assertEqual(self._build(cache), {"units": 2, "warm": 0, "built": 2})
            self.assertEqual(self._build(cache), {"units": 2, "warm": 2, "built": 0})
            # M4's builder and key, for the unit M4's filter leaves whole; ours' own tree for the other.
            self.assertEqual(len(tree_dirs(cache, M4_SUBSTRATE_NAMESPACE)), 1)
            self.assertEqual(len(tree_dirs(cache, settings.TREE_NAMESPACE)), 1)
            with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
                system = om.OursSystem(config=llama_config())
                for unit in _SplitUnits().units:
                    system.index_items(unit.corpus)
                    self.assertTrue(system.tree_cache_hit)
        self.assertEqual(_SplitUnits.asked, ["test", "test"])

    def test_a_split_of_another_size_is_caught_before_the_cell(self):
        class Declares3(_SplitUnits):
            cell_units = 3

        class Declares2(_SplitUnits):
            cell_units = 2

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache_ours_full"
            with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)), \
                    contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.dict("src.eval.runner.BENCHMARK_REGISTRY", {"narrativeqa_test": Declares3}):
                    with self.assertRaisesRegex(SystemExit, "has 2 units, not the 3"):
                        dev_ours.build_own_trees("narrativeqa_test", check_env=lambda lockfile: None)
                with mock.patch.dict("src.eval.runner.BENCHMARK_REGISTRY", {"narrativeqa_test": Declares2}):
                    counts = dev_ours.build_own_trees("narrativeqa_test", check_env=lambda lockfile: None)
        # The trees built before the refusal are kept and read warm.
        self.assertEqual(counts, {"units": 2, "warm": 2, "built": 0})

    def test_hotpotqa_full_reads_validation(self):
        _SplitUnits.asked = []
        with tempfile.TemporaryDirectory() as td:
            self._build(Path(td) / "cache_ours_full", benchmark="hotpotqa_full")
        self.assertEqual(_SplitUnits.asked, ["validation"])

    def test_only_in_ours_own_cache_and_with_the_reader(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(SystemExit, "cache_ours_full"):
                self._build(Path(td) / "cache")
            with self.assertRaisesRegex(SystemExit, "the full round runs the reader"):
                self._build(Path(td) / "cache_ours_full", generator="Qwen/Qwen2.5-3B-Instruct")
            self.assertFalse((Path(td) / "cache" / M4_SUBSTRATE_NAMESPACE).exists())

    def test_a_rounds_trees_are_still_ours_own_only(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache"
            self.assertEqual(self._build(cache, benchmark="narrativeqa"),
                             {"units": 1, "warm": 0, "built": 1})
            self.assertFalse((cache / M4_SUBSTRATE_NAMESPACE).exists())


class TestCli(unittest.TestCase):
    def test_scored_and_trees_take_the_full_benchmarks(self):
        seen = []
        with mock.patch.object(dev_ours, "scored_cell", lambda *a, **k: seen.append(("scored", a[0]))), \
                mock.patch.object(dev_ours, "build_own_trees", lambda *a, **k: seen.append(("trees", a[0]))):
            for argv in (["scored", "--benchmark", "narrativeqa_test", "--bank", "x/ours_v20_full"],
                         ["trees", "--benchmark", "hotpotqa_full"]):
                with mock.patch.object(sys, "argv", ["dev_ours", *argv]):
                    dev_ours.main()
        self.assertEqual(seen, [("scored", "narrativeqa_test"), ("trees", "hotpotqa_full")])
        with mock.patch.object(sys, "argv", ["dev_ours", "scored", "--benchmark", "hotpotqa_pooled_full",
                                             "--bank", "x/ours_v20_full"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as refused:
            dev_ours.main()
        # argparse's own refusal: pooled HotpotQA is no choice of the full round.
        self.assertEqual(refused.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
