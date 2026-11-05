"""Pins that ours builds its own trees under its own cache namespace and
touches nothing of M4's, that its leaf settings move the key and its section
settings do not, and that ours refuses a configuration it was not built for."""

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
    """Ours' trees stand apart from M4's, whatever is in the cache."""

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

    def _warm_path(self, cache=None):
        """Where ours, under the runner's config, finds this unit's tree."""
        with ours_fakes(cache or self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            tree = om.OursTree(llama_config())
            tree.topology_env_override = settings.TREE_BUILD_ENV
            return tree.substrate_warm_path(self.items)

    def _m4(self):
        m4 = RaptorSystem(config=llama_config())
        m4.topology_env_override = settings.TREE_BUILD_ENV
        return m4

    def test_the_tree_lands_in_ours_namespace_and_nowhere_else(self):
        warm = self._warm_path()
        self.assertIsNotNone(warm)
        self.assertTrue(self.files_after_index)
        self.assertTrue(all(f.startswith(f"{settings.TREE_NAMESPACE}/{Path(warm).name}/")
                            for f in self.files_after_index), self.files_after_index)

    def test_m4_finds_nothing_in_a_cache_only_ours_built(self):
        with ours_fakes(self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            self.assertIsNone(self._m4().substrate_warm_path(self.items))

    def test_a_tree_of_m4s_is_neither_read_nor_touched(self):
        """Beside M4's tree for the same corpus, ours still builds its own."""
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
                self._m4().index_items(self.items)
                banked = {f: (cache / f).read_bytes() for f in _files(cache)}
                self.assertTrue(banked)
                ours = om.OursSystem(config=llama_config())
                ours.index_items(self.items)
                self.assertIs(ours.tree_cache_hit, False)
            for f, data in banked.items():
                self.assertEqual((cache / f).read_bytes(), data, f)
            added = set(_files(cache)) - set(banked)
            self.assertTrue(added)
            self.assertTrue(all(f.startswith(settings.TREE_NAMESPACE + "/") for f in added))

    def test_the_builder_runs_with_the_runners_config_but_for_its_leaves(self):
        expected = llama_config()
        expected = replace(expected, m4=replace(expected.m4, chunker=om.semantic_chunker()))
        self.assertEqual(self.ours._builder.config, expected)
        self.assertEqual(self.ours._builder.config.m4.chunker.strategy, "ours_semantic")

    def test_a_built_tree_is_read_back_the_same_as_a_cached_one(self):
        with ours_fakes(self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            again = om.OursSystem(config=llama_config())
            again.index_items(self.items)
        self.assertIs(self.ours.tree_cache_hit, False)
        self.assertIs(again.tree_cache_hit, True)
        self.assertEqual([c.text for c in again.chunks], [c.text for c in self.ours.chunks])
        self.assertEqual([c.gold_provenance for c in again.chunks],
                         [c.gold_provenance for c in self.ours.chunks])
        self.assertEqual(sorted(again._builder._tree.nodes),
                         sorted(self.ours._builder._tree.nodes))

    def test_every_leaf_setting_moves_the_key(self):
        before = self._warm_path()
        for name, value in (("SENTENCE_BUFFER", 2), ("BREAKPOINT_PERCENTILE", 80.0),
                            ("BREAKPOINT_MAX_DISTANCE", 0.4), ("MIN_CHUNK_WORDS", 60),
                            ("MAX_CHUNK_WORDS", 300), ("MAX_MERGED_CHUNK_WORDS", 450)):
            with mock.patch.object(settings, name, value):
                self.assertIsNone(self._warm_path(), name)
        self.assertEqual(self._warm_path(), before)

    def test_every_tree_setting_moves_the_key(self):
        before = self._warm_path()
        for name, value in (("TREE_BRANCHING_FACTOR", 3), ("TREE_MIN_CLUSTER_SIZE", 12),
                            ("TREE_MAX_DEPTH", 3), ("KMEANS_BATCH_SIZE", 1024),
                            ("KMEANS_N_INIT", 5), ("KMEANS_RANDOM_STATE", 7),
                            ("SUMMARY_PROMPT", "Summarize."), ("SUMMARY_MAX_TOKENS", 100),
                            ("SUMMARY_CONTEXT_TOKENS", 2000)):
            with mock.patch.object(settings, name, value):
                self.assertIsNone(self._warm_path(), name)
        self.assertEqual(self._warm_path(), before)

    def test_section_settings_do_not_move_the_key(self):
        """Changing every section setting leaves the cache untouched."""
        before = self._warm_path()
        with mock.patch.multiple(settings, SECTION_SIZE_CHUNKS=2,
                                 MIN_HEADINGS=1, MIN_SECTION_CHUNKS=1):
            after = self._warm_path()
            with ours_fakes(self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
                again = om.OursSystem(config=llama_config())
                again.index_items(self.items)
        self.assertEqual(before, after)
        self.assertTrue(again.tree_cache_hit)
        self.assertEqual(_files(self.cache), self.files_after_index)

    def test_summary_sees_the_reranker_the_embedder_and_the_leaves(self):
        resolved = self.ours.resolved_components
        self.assertEqual(resolved.reranker_id, settings.RERANKER)
        self.assertEqual(resolved.embedder_id, settings.EMBEDDER)
        self.assertEqual(resolved.chunker_config, om.semantic_chunker())


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
                mock.patch.object(system._builder, "index_items") as tree_index:
            with self.assertRaises(SystemExit):
                system.index_items(hotpot_unit(0).corpus)
        tree_index.assert_not_called()

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
