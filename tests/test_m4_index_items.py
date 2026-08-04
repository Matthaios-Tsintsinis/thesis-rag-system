"""M4's per-parent corpus layout (commit 4) and the cache-key promise it makes.

The layout matters twice. Semantically, it is what lets the paper's
contiguous 100-token chunker run over the DOCUMENT rather than over
per-item fragments. Mechanically, `corpus_content_hash` is computed over
the temp directory `index_items` writes, so the layout is a cache-key
input — and the single-item rule (rule B) is the promise that keeps 1:1
benchmarks off a rebuild.

Both halves of that promise are tested, because both were measured to
matter and neither is obvious:

  * the FILENAME is folded into corpus_content_hash alongside the bytes,
    so a parent-derived name moves NarrativeQA and QuALITY even though
    they are strictly 1:1 (their item_id is "{parent}::<whole>", not the
    bare parent id). MultiHop escapes only by the coincidence that its
    item_id IS its parent_id.
  * the RAW BYTES matter because parsing.clean_text is not the identity;
    a multi-item parent is written pre-cleaned, and doing that to a
    single-item parent would move its hash via the content instead.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.cache import corpus_content_hash
from src.chunking import chunk_corpus
from src.config import DEFAULT_CONFIG
from src.eval.types import CorpusItem
from src.parsing import clean_text, walk_corpus
from src.retrievers import m4_raptor
from src.retrievers.base import BaseSystem
from src.retrievers.m4_raptor import RaptorSystem, build_parent_payload


def _items_multihop(n: int = 3) -> list[CorpusItem]:
    """item_id == parent_id. The shape that survives on a coincidence."""
    return [
        CorpusItem(item_id=f"https://x.test/{i}", parent_id=f"https://x.test/{i}",
                   span_id="<whole>", text=f"Article {i} body text. " * 30)
        for i in range(n)
    ]


def _items_whole(n: int = 3) -> list[CorpusItem]:
    """NarrativeQA / QuALITY shape: 1:1, but item_id != parent_id."""
    return [
        CorpusItem(item_id=f"story{i}::<whole>", parent_id=f"story{i}",
                   span_id="<whole>", text=f"Story {i} full text. " * 30)
        for i in range(n)
    ]


def _items_qasper(n_paras: int = 6) -> list[CorpusItem]:
    """Many items per parent — the case the concatenation exists for."""
    return [
        CorpusItem(item_id=f"paper1::sec1.para{i}", parent_id="paper1",
                   span_id=f"sec1.para{i}",
                   text=f"Paragraph {i} says something about topic {i}. " * 4)
        for i in range(n_paras)
    ]


class _CaptureMixin:
    """Records the corpus hash + file list, then fakes a chunking pass.

    index() is replaced by real parsing + real chunking and nothing else:
    the clean_text trap this module guards against lives in walk_corpus,
    so it must be exercised, while the embeddings and the UMAP/GMM tree
    would only make the test slow.
    """

    def index(self, corpus_path):  # noqa: D102
        corpus_path = Path(corpus_path)
        self.captured_hash = corpus_content_hash(corpus_path)
        self.captured_files = sorted(
            p.name for p in corpus_path.rglob("*") if p.is_file()
        )
        cfg = DEFAULT_CONFIG.m4.chunker
        docs = list(walk_corpus(corpus_path, min_chars=cfg.min_chars_per_doc))
        self.chunks = chunk_corpus(docs, cfg)
        self._indexed = True

    def retrieve(self, query, k=None):  # noqa: D102
        return []


class _BaseLayout(_CaptureMixin, BaseSystem):
    """The per-item layout, i.e. BaseSystem.index_items unmodified."""

    system_id = "BASE"


class _M4Layout(_CaptureMixin, RaptorSystem):
    """The per-parent layout under test."""


def _hash_and_files(cls, items):
    sysm = cls(config=DEFAULT_CONFIG)
    sysm.index_items(items)
    return sysm.captured_hash, sysm.captured_files, sysm


class TestSingleItemParentIsByteIdentical(unittest.TestCase):
    """Rule B. The promise that keeps 1:1 benchmarks off a rebuild."""

    def test_multihop_shape_hash_is_unmoved(self):
        base, _, _ = _hash_and_files(_BaseLayout, _items_multihop())
        m4, _, _ = _hash_and_files(_M4Layout, _items_multihop())
        self.assertEqual(base, m4)

    def test_narrativeqa_quality_shape_hash_is_unmoved(self):
        """The case a parent-derived filename would have broken.

        These are strictly 1:1, so the intuition "one item per parent
        means nothing changes" says they are safe. They are not: their
        item_id is "{parent}::<whole>", so naming the file after the
        parent moves the hash through the PATH while the bytes stay
        identical. Rule B is what makes them safe.
        """
        base, _, _ = _hash_and_files(_BaseLayout, _items_whole())
        m4, _, _ = _hash_and_files(_M4Layout, _items_whole())
        self.assertEqual(base, m4)

    def test_uncleaned_text_is_preserved_verbatim(self):
        """clean_text is NOT the identity, so pre-cleaning a single-item
        parent would move the hash through the CONTENT instead."""
        dirty = "Article  with  double  spaces.  " * 30
        self.assertNotEqual(clean_text(dirty), dirty)
        items = [CorpusItem(item_id="a::<whole>", parent_id="a",
                            span_id="<whole>", text=dirty)]
        base, _, _ = _hash_and_files(_BaseLayout, items)
        m4, _, _ = _hash_and_files(_M4Layout, items)
        self.assertEqual(base, m4)

    def test_one_file_per_parent(self):
        _, files, _ = _hash_and_files(_M4Layout, _items_qasper())
        self.assertEqual(len(files), 1, "6 paragraphs of one paper -> 1 file")

    def test_many_item_parent_hash_moves(self):
        """Stated, not hidden: QASPER-shaped corpora DO rebuild. That is
        the price of chunking the document instead of its fragments."""
        base, _, _ = _hash_and_files(_BaseLayout, _items_qasper())
        m4, _, _ = _hash_and_files(_M4Layout, _items_qasper())
        self.assertNotEqual(base, m4)


class TestProvenance(unittest.TestCase):
    def test_single_item_parent_stamps_its_own_atom(self):
        _, _, sysm = _hash_and_files(_M4Layout, _items_whole(2))
        self.assertTrue(sysm.chunks)
        for chunk in sysm.chunks:
            self.assertEqual(len(chunk.gold_provenance), 1)
            parent, span = chunk.gold_provenance[0]
            self.assertTrue(parent.startswith("story"))
            self.assertEqual(span, "<whole>")

    def test_multi_item_parent_derives_atoms_from_offsets(self):
        _, _, sysm = _hash_and_files(_M4Layout, _items_qasper())
        self.assertTrue(sysm.chunks)
        seen: set[str] = set()
        for chunk in sysm.chunks:
            self.assertTrue(
                chunk.gold_provenance,
                f"chunk {chunk.chunk_id} intersected no source item",
            )
            for parent, span in chunk.gold_provenance:
                self.assertEqual(parent, "paper1")
                seen.add(span)
        # Every paragraph is covered by at least one chunk: contiguous
        # non-overlapping chunking partitions the document.
        self.assertEqual(seen, {f"sec1.para{i}" for i in range(6)})

    def test_a_boundary_crossing_chunk_carries_both_atoms(self):
        """The point of contiguous chunking. Short items pack several to
        a 100-token chunk, and CK-2 should credit every one it covers."""
        _, _, sysm = _hash_and_files(_M4Layout, _items_qasper())
        self.assertTrue(
            any(len(c.gold_provenance) > 1 for c in sysm.chunks),
            "no chunk spanned an item boundary; the fixture is too coarse "
            "to exercise multi-atom provenance",
        )

    def test_provenance_is_ordered_by_member_position(self):
        _, _, sysm = _hash_and_files(_M4Layout, _items_qasper())
        order = {f"sec1.para{i}": i for i in range(6)}
        for chunk in sysm.chunks:
            spans = [order[s] for _, s in chunk.gold_provenance]
            self.assertEqual(spans, sorted(spans))


class TestBuildParentPayload(unittest.TestCase):
    def test_spans_index_the_payload_they_are_returned_with(self):
        items = _items_qasper(3)
        payload, spans = build_parent_payload(items)
        self.assertEqual(len(spans), 3)
        for (lo, hi, span_id), item in zip(spans, items):
            self.assertEqual(payload[lo:hi], clean_text(item.text))
            self.assertEqual(span_id, item.span_id)

    def test_payload_is_clean_text_stable(self):
        """The invariant the override asserts at runtime: reading the file
        back must be a no-op, or the offsets index the wrong string."""
        payload, _ = build_parent_payload(_items_qasper(4))
        self.assertEqual(clean_text(payload), payload)

    def test_dirty_members_are_cleaned_before_measuring(self):
        items = [
            CorpusItem(item_id="p::a", parent_id="p", span_id="a",
                       text="one  two\t\tthree  "),
            CorpusItem(item_id="p::b", parent_id="p", span_id="b",
                       text="  four   five"),
        ]
        payload, spans = build_parent_payload(items)
        self.assertEqual(payload, "one two three\n\nfour five")
        self.assertEqual(payload[spans[0][0]:spans[0][1]], "one two three")
        self.assertEqual(payload[spans[1][0]:spans[1][1]], "four five")

    def test_empty_members_are_skipped_not_offset(self):
        items = [
            CorpusItem(item_id="p::a", parent_id="p", span_id="a", text="alpha"),
            CorpusItem(item_id="p::b", parent_id="p", span_id="b", text="   "),
            CorpusItem(item_id="p::c", parent_id="p", span_id="c", text="gamma"),
        ]
        payload, spans = build_parent_payload(items)
        self.assertEqual(payload, "alpha\n\ngamma")
        self.assertEqual([s[2] for s in spans], ["a", "c"])
        for lo, hi, _ in spans:
            self.assertTrue(payload[lo:hi].strip())

    def test_grouping_preserves_first_appearance_order(self):
        items = [
            CorpusItem(item_id="b::1", parent_id="b", span_id="1", text="x"),
            CorpusItem(item_id="a::1", parent_id="a", span_id="1", text="y"),
            CorpusItem(item_id="b::2", parent_id="b", span_id="2", text="z"),
        ]
        groups = m4_raptor.group_items_by_parent(items)
        self.assertEqual(list(groups), ["b", "a"])
        self.assertEqual([i.span_id for i in groups["b"]], ["1", "2"])


class TestIdempotenceGuard(unittest.TestCase):
    def test_non_idempotent_clean_text_is_refused(self):
        """If clean_text ever stops being idempotent, the offsets index a
        string the chunker never sees and provenance drifts across the
        document — silently. Refuse instead."""
        original = m4_raptor.clean_text
        try:
            # Appending unconditionally is non-idempotent for ANY input,
            # which a substitution-based fake is not: it would collapse
            # on the first pass and then agree with itself.
            m4_raptor.clean_text = lambda t: (t or "") + " ."
            sysm = _M4Layout(config=DEFAULT_CONFIG)
            with self.assertRaises(RuntimeError):
                sysm.index_items(_items_qasper())
        finally:
            m4_raptor.clean_text = original

    def test_missing_offsets_are_refused_for_a_multi_item_parent(self):
        """A chunker override without start_char/end_char cannot support
        a multi-item parent; failing loudly beats empty provenance."""
        class _NoOffsets(_M4Layout):
            def index(self, corpus_path):
                super().index(corpus_path)
                for c in self.chunks:
                    c.metadata.pop("start_char", None)

        sysm = _NoOffsets(config=DEFAULT_CONFIG)
        with self.assertRaises(RuntimeError):
            sysm.index_items(_items_qasper())


class TestTempDirIsCleanedUp(unittest.TestCase):
    def test_no_temp_directory_survives(self):
        before = set(Path(tempfile.gettempdir()).glob("M4_corpus_*"))
        _hash_and_files(_M4Layout, _items_multihop())
        after = set(Path(tempfile.gettempdir()).glob("M4_corpus_*"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
