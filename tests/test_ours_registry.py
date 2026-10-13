"""Pins ours as the runner's fifth system: `--system OURS` runs a cell end to
end, the summary names ours' reader and reranker, and M4's cold-tree gate
does not apply, so a second run over warm trees is not refused."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ours import config as settings
from ours.model import OursSystem
from src.eval import runner as runner_mod
from tests.test_ours_smoke import (
    ANSWERS,
    DECOMPOSITIONS,
    FakeReader,
    TwoHotpotUnits,
    ours_fakes,
)


class TestRegistry(unittest.TestCase):
    """The runner resolves OURS and runs it like any other system."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _main(self, out: Path):
        lock = self.dir / "requirements.lock"
        lock.write_text("# lock\nnumpy==0.0.0\n", encoding="utf-8")
        argv = ["runner", "--system", "OURS", "--benchmark", "hotpotqa",
                "--split", "validation", "--output", str(out),
                "--lockfile", str(lock), "--generator", settings.READER]
        hub = mock.Mock()
        hub.model_info.return_value = mock.Mock(sha="hubsha")
        with ours_fakes(self.dir / "cache", FakeReader(DECOMPOSITIONS, ANSWERS)), \
                mock.patch.dict(runner_mod.BENCHMARK_REGISTRY,
                                {"hotpotqa": TwoHotpotUnits}), \
                mock.patch("scripts.pin_environment.check_lockfile", return_value=0), \
                mock.patch("huggingface_hub.hf_hub_download", return_value="x"), \
                mock.patch("huggingface_hub.HfApi", return_value=hub), \
                mock.patch.object(sys, "argv", argv):
            runner_mod.main()
        return json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))

    def test_registry_resolves_ours(self):
        self.assertIs(runner_mod.SYSTEM_REGISTRY["OURS"], OursSystem)

    def test_a_cell_runs_and_its_summary_names_ours(self):
        summary = self._main(self.dir / "ours" / "hotpotqa_OURS_validation.jsonl")
        self.assertEqual(summary["system"], "OURS")
        self.assertEqual(summary["n_queries_scored"], 2)
        self.assertEqual(summary["generator"], settings.READER)
        self.assertEqual(summary["index_llm"], settings.READER)
        self.assertEqual(summary["model_revisions"]["reranker"], settings.RERANKER)
        self.assertEqual(summary["chunking_strategy"], "raptor_100tok")
        self.assertIs(summary["tree_cache_hit"], False)

    def test_warm_trees_are_not_refused(self):
        """A second cell over the trees the first built runs to its summary."""
        self._main(self.dir / "a" / "hotpotqa_OURS_validation.jsonl")
        summary = self._main(self.dir / "b" / "hotpotqa_OURS_validation.jsonl")
        self.assertIs(summary["tree_cache_hit"], True)


if __name__ == "__main__":
    unittest.main()
