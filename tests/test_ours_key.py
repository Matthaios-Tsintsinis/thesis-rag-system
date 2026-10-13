"""Pins that ours finds M4's trees under M4's own cache key and writes
nothing else into the cache, that section settings never move that key, and
that ours refuses a configuration it was not built for."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from ours import config as settings
from ours import model as om
from src.retrievers.m4_raptor import RaptorSystem
from tests.test_ours_smoke import (
    ANSWERS,
    DECOMPOSITIONS,
    FakeReader,
    hotpot_unit,
    llama_config,
    ours_fakes,
)


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)).replace("\\", "/")
                  for p in root.rglob("*") if p.is_file())


class TestCacheKey(unittest.TestCase):
    """Ours indexes through M4's code and lands in M4's directory."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.cache = Path(cls.td.name)
        cls.items = hotpot_unit(0).corpus
        with ours_fakes(cls.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            cls.ours = om.OursSystem(config=llama_config())
            cls.ours.index_items(cls.items)
            cls.files_after_index = _files(cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def _m4_warm_path(self):
        """Where plain M4, under the runner's config, finds this unit's tree."""
        with ours_fakes(self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            m4 = RaptorSystem(config=llama_config())
            m4.topology_env_override = settings.TREE_BUILD_ENV
            return m4.substrate_warm_path(self.items)

    def test_the_tree_ours_built_is_where_m4_looks(self):
        warm = self._m4_warm_path()
        self.assertIsNotNone(warm)
        self.assertTrue(all(f.startswith("M4_RAPTOR/" + Path(warm).name + "/")
                            for f in self.files_after_index), self.files_after_index)

    def test_m4_inside_ours_runs_with_the_runners_config(self):
        self.assertEqual(self.ours._m4.config, llama_config())

    def test_section_settings_do_not_move_the_key(self):
        """Changing every section setting leaves the cache untouched."""
        before = self._m4_warm_path()
        with mock.patch.multiple(settings, SECTION_SIZE_CHUNKS=2,
                                 MIN_HEADINGS=1, MIN_SECTION_CHUNKS=1):
            after = self._m4_warm_path()
            with ours_fakes(self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
                again = om.OursSystem(config=llama_config())
                again.index_items(self.items)
        self.assertEqual(before, after)
        self.assertTrue(again.tree_cache_hit)
        self.assertEqual(_files(self.cache), self.files_after_index)

    def test_summary_sees_the_reranker_and_m4s_embedder(self):
        resolved = self.ours.resolved_components
        self.assertEqual(resolved.reranker_id, settings.RERANKER)
        self.assertEqual(resolved.embedder_id, settings.EMBEDDER)


class TestGates(unittest.TestCase):
    """Ours refuses what it was not built for, before any work."""

    def test_another_reader_is_refused(self):
        from src.config import DEFAULT_CONFIG
        with self.assertRaises(SystemExit):
            om.OursSystem(config=DEFAULT_CONFIG)

    def test_sampling_is_refused(self):
        cfg = llama_config()
        cfg = replace(cfg, generation=replace(cfg.generation, temperature=0.7))
        with self.assertRaises(SystemExit):
            om.OursSystem(config=cfg)

    def test_another_embedder_is_refused(self):
        cfg = llama_config()
        cfg = replace(cfg, m4=replace(cfg.m4, embedder="BAAI/bge-m3"))
        with self.assertRaises(SystemExit):
            om.OursSystem(config=cfg)

    def test_no_tree_env_is_refused_before_indexing(self):
        system = om.OursSystem(config=llama_config())
        with mock.patch.object(settings, "TREE_BUILD_ENV", None), \
                mock.patch("ours.model.check_model_revisions", lambda: None), \
                mock.patch.object(system._m4, "index_items") as m4_index:
            with self.assertRaises(SystemExit):
                system.index_items(hotpot_unit(0).corpus)
        m4_index.assert_not_called()

    def test_hub_revision_must_match_the_pin(self):
        class _Info:
            def __init__(self, sha):
                self.sha = sha

        def info_for(sha_by_model):
            api = mock.Mock()
            api.model_info.side_effect = lambda m: _Info(sha_by_model[m])
            return api

        pinned = {settings.READER: settings.READER_REVISION,
                  settings.EMBEDDER: settings.EMBEDDER_REVISION}
        with mock.patch("huggingface_hub.HfApi", return_value=info_for(pinned)):
            om.check_model_revisions()
        moved = dict(pinned, **{settings.READER: "0" * 40})
        with mock.patch("huggingface_hub.HfApi", return_value=info_for(moved)):
            with self.assertRaises(SystemExit):
                om.check_model_revisions()


if __name__ == "__main__":
    unittest.main()
