"""Pins that ours reads the documents M4's filter leaves out: a unit holding one
gets a tree from M4's builder with the filter lifted, in ours' own cache, with
M4's bank untouched, and every other unit still reads M4's tree."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ours import config as settings
from ours import model as om
from src.eval.base import BenchmarkRunner
from src.eval.types import CorpusItem
from src.retrievers.m4_raptor import M4_SUBSTRATE_NAMESPACE, RaptorSystem
from tests.test_ours_smoke import (
    ANSWERS,
    DECOMPOSITIONS,
    FakeReader,
    TwoHotpotUnits,
    hotpot_unit,
    llama_config,
    ours_fakes,
)

SHORT_TITLE = "Tiny Place"
SHORT_TEXT = "Tiny Place is a village of forty people on the topic0x0 river."


def _short_item(text=SHORT_TEXT, title=SHORT_TITLE):
    return CorpusItem(item_id=f"{title}::sent0", parent_id=title, span_id="sent0", text=text,
                      metadata={"title": title, "sent_id": 0})


def unit_with_short_document(k=0):
    """hotpot_unit(k) with one more document, under M4's filter."""
    unit = hotpot_unit(k)
    return replace(unit, corpus=tuple(unit.corpus) + (_short_item(),))


def _bytes(root: Path) -> dict:
    return {str(p.relative_to(root)).replace("\\", "/"): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


class TestShortDocumentsCount(unittest.TestCase):
    """The count judges a document as M4's filter does."""

    def setUp(self):
        self.m4 = RaptorSystem(llama_config())

    def test_a_whole_unit_has_none(self):
        self.assertEqual(om.short_documents(hotpot_unit(0).corpus, self.m4), 0)

    def test_the_boundary_is_m4s(self):
        limit = llama_config().m4.chunker.min_chars_per_doc
        self.assertEqual(limit, 200)
        just_under = hotpot_unit(0).corpus + (_short_item("x" * (limit - 1)),)
        exactly = hotpot_unit(0).corpus + (_short_item("x" * limit),)
        self.assertEqual(om.short_documents(just_under, self.m4), 1)
        self.assertEqual(om.short_documents(exactly, self.m4), 0)

    def test_a_document_counts_by_its_whole_text_not_its_sentences(self):
        sentence = "This sentence about the village runs to about seventy characters long."
        items = [replace(_short_item(sentence), item_id=f"{SHORT_TITLE}::sent{i}",
                         span_id=f"sent{i}", metadata={"title": SHORT_TITLE, "sent_id": i})
                 for i in range(4)]
        self.assertEqual(om.short_documents(tuple(items), self.m4), 0)
        self.assertEqual(om.short_documents(tuple(items[:2]), self.m4), 1)


class TestUnitWithAShortDocument(unittest.TestCase):
    """M4's tree for the same corpus stands beside ours; ours neither reads nor touches it."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.cache = Path(cls.td.name)
        cls.items = unit_with_short_document().corpus
        with ours_fakes(cls.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            cls.m4 = RaptorSystem(config=llama_config())
            cls.m4.topology_env_override = settings.TREE_BUILD_ENV
            cls.m4.index_items(cls.items)
            cls.m4_files = _bytes(cls.cache)
            cls.ours = om.OursSystem(config=llama_config())
            cls.ours.index_items(cls.items)
            cls.files_after_index = _bytes(cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def _leaves_of_short_document(self, system):
        return [c for c in system.chunks if c.text == SHORT_TEXT]

    def test_ours_reads_the_document_m4_leaves_out(self):
        self.assertEqual(self._leaves_of_short_document(self.m4), [])
        self.assertEqual(len(self._leaves_of_short_document(self.ours)), 1)

    def test_only_the_short_document_is_added(self):
        self.assertEqual(sorted(c.text for c in self.m4.chunks),
                         sorted(c.text for c in self.ours.chunks if c.text != SHORT_TEXT))

    def test_the_leaf_carries_gold_provenance(self):
        leaf = self._leaves_of_short_document(self.ours)[0]
        self.assertEqual(leaf.gold_provenance, ((SHORT_TITLE, "sent0"),))

    def test_its_tree_is_built_not_read_from_m4(self):
        self.assertIs(self.ours._tree_system, self.ours._whole)
        self.assertFalse(self.ours.tree_cache_hit)

    def test_its_tree_is_in_ours_namespace(self):
        new = set(self.files_after_index) - set(self.m4_files)
        self.assertTrue(new)
        self.assertTrue(all(f.startswith(settings.TREE_NAMESPACE + "/") for f in new), new)

    def test_m4s_bank_is_untouched(self):
        for name, content in self.m4_files.items():
            self.assertTrue(name.startswith(M4_SUBSTRATE_NAMESPACE + "/"), name)
            self.assertEqual(self.files_after_index[name], content, name)
        self.assertEqual(len([f for f in self.files_after_index
                              if f.startswith(M4_SUBSTRATE_NAMESPACE + "/")]), len(self.m4_files))

    def test_m4_itself_keeps_its_filter(self):
        self.assertEqual(self.ours._m4.config, llama_config())
        self.assertEqual(self.ours._m4.config.m4.chunker.min_chars_per_doc, 200)
        self.assertEqual(self.ours._whole.config.m4.chunker.min_chars_per_doc, 0)

    def test_a_second_pass_reads_its_own_tree_and_writes_nothing(self):
        with ours_fakes(self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            again = om.OursSystem(config=llama_config())
            again.index_items(self.items)
        self.assertTrue(again.tree_cache_hit)
        self.assertEqual(_bytes(self.cache), self.files_after_index)
        self.assertEqual(len(self._leaves_of_short_document(again)), 1)


class TestUnitWithNoShortDocument(unittest.TestCase):
    """Every other unit reads M4's tree as v8 did; ours' namespace stays empty."""

    def test_it_uses_m4_and_writes_nothing_of_ours(self):
        with tempfile.TemporaryDirectory() as td:
            with ours_fakes(td, FakeReader(DECOMPOSITIONS, ANSWERS)):
                ours = om.OursSystem(config=llama_config())
                ours.index_items(hotpot_unit(0).corpus)
            self.assertIs(ours._tree_system, ours._m4)
            self.assertFalse((Path(td) / settings.TREE_NAMESPACE).exists())
            self.assertTrue((Path(td) / M4_SUBSTRATE_NAMESPACE).exists())


class _OneOfEach(TwoHotpotUnits):
    """A unit with a short document, then one without."""

    def __init__(self):
        super().__init__()
        self.units = [unit_with_short_document(0), hotpot_unit(1)]


def _run(cache, out):
    with ours_fakes(cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
        system = om.OursSystem(config=llama_config())
        list(BenchmarkRunner(output_path=Path(out), verbose=False).run(
            system, _OneOfEach(), split="validation"))
    return [json.loads(x) for x in Path(out).read_text(encoding="utf-8").splitlines() if x.strip()]


class TestRows(unittest.TestCase):
    """Through the runner: the rows say which tree they read, and a warm pass changes nothing."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        root = Path(cls.td.name)
        cls.cold = _run(root / "cache", root / "cold.jsonl")
        cls.warm = _run(root / "cache", root / "warm.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_each_row_counts_its_units_short_documents(self):
        self.assertEqual([r["metadata"]["ours_n_short_documents"] for r in self.cold], [1, 0])

    def test_the_short_document_reaches_the_ranking(self):
        self.assertIn(SHORT_TITLE, self.cold[0]["metadata"]["ours_doc_ranking"])

    def test_cold_then_warm(self):
        self.assertEqual([r["metadata"]["ours_tree_cache_hit"] for r in self.cold], [False, False])
        self.assertEqual([r["metadata"]["ours_tree_cache_hit"] for r in self.warm], [True, True])

    def test_a_warm_row_equals_a_cold_row(self):
        def strip(row):
            row = dict(row, metadata={k: v for k, v in row["metadata"].items()
                                      if k != "ours_tree_cache_hit"})
            row.pop("latency_s")
            return row
        self.assertEqual([strip(r) for r in self.cold], [strip(r) for r in self.warm])


if __name__ == "__main__":
    unittest.main()
