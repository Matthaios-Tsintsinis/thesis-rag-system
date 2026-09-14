"""Pins the per-parent corpus layout of BaseSystem.index_items and its
cache-key promise: a single-item parent hashes as one file per item.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.cache import corpus_content_hash
from src.chunking import ChunkingConfig, chunk_corpus
from src.config import DEFAULT_CONFIG
from src.eval.types import CorpusItem
from src.parsing import ParsedDocument, clean_text, walk_corpus
from src.retrievers import base as base_mod
from src.retrievers.base import BaseSystem, build_parent_payload
from src.retrievers.m1_closedbook import ClosedBookSystem
from src.retrievers.m4_raptor import RaptorSystem


WORD_WINDOW = DEFAULT_CONFIG.chunking          # M2 / M3
RAPTOR_100TOK = DEFAULT_CONFIG.m4.chunker      # M4


def _items_multihop(n: int = 3) -> list[CorpusItem]:
    """MultiHop shape: 1:1 with item_id == parent_id."""
    return [
        CorpusItem(item_id=f"https://x.test/{i}", parent_id=f"https://x.test/{i}",
                   span_id="<whole>", text=f"Article {i} body text. " * 30)
        for i in range(n)
    ]


def _items_whole(n: int = 3) -> list[CorpusItem]:
    """NarrativeQA shape: 1:1, but item_id is "{parent}::<whole>"."""
    return [
        CorpusItem(item_id=f"story{i}::<whole>", parent_id=f"story{i}",
                   span_id="<whole>", text=f"Story {i} full text. " * 30)
        for i in range(n)
    ]


def _items_qasper(n_paras: int = 6) -> list[CorpusItem]:
    """Many items per parent: the shape the concatenation exists for."""
    return [
        CorpusItem(item_id=f"paper1::sec1.para{i}", parent_id="paper1",
                   span_id=f"sec1.para{i}",
                   text=f"Paragraph {i} says something about topic {i}. " * 4)
        for i in range(n_paras)
    ]


# HotpotQA sentence sizes come from the dataset: paragraphs average ~511
# chars over ~4 sentences, so one sentence (~124 chars) sits under
# min_chars_per_doc = 200 while the whole paragraph sits over it.
_HOTPOT_SENTENCE_CHARS = 124
_HOTPOT_SENTS_PER_TITLE = 4


def _items_hotpot(n_titles: int = 3) -> list[CorpusItem]:
    """HotpotQA shape: sentence items under a paragraph title."""
    items: list[CorpusItem] = []
    for t in range(n_titles):
        for s in range(_HOTPOT_SENTS_PER_TITLE):
            stem = f"Sentence {s} of title {t} states a fact about the topic, "
            text = (stem + "and continues with filler words. " * 4)[
                :_HOTPOT_SENTENCE_CHARS
            ].strip() + "."
            items.append(CorpusItem(
                item_id=f"Title {t}::sent{s}", parent_id=f"Title {t}",
                span_id=f"sent{s}", text=text))
    return items


class _CaptureMixin:
    """Records the corpus hash and file list, then parses and chunks."""

    chunker: ChunkingConfig = RAPTOR_100TOK

    def index(self, corpus_path):  # noqa: D102
        corpus_path = Path(corpus_path)
        self.captured_hash = corpus_content_hash(corpus_path)
        self.captured_files = sorted(
            p.name for p in corpus_path.rglob("*") if p.is_file()
        )
        cfg = self.chunker
        docs = list(walk_corpus(corpus_path, min_chars=cfg.min_chars_per_doc))
        self.chunks = chunk_corpus(docs, cfg)
        self._indexed = True

    def retrieve(self, query, k=None):  # noqa: D102
        return []


class _LegacyPerItemLayout(_CaptureMixin, BaseSystem):
    """Reference layout: one raw file per item, named from item_id."""

    system_id = "LEGACY"

    def index_items(self, items):  # noqa: D102
        item_by_doc_id: dict[str, CorpusItem] = {}
        with tempfile.TemporaryDirectory(prefix=f"{self.system_id}_corpus_") as td:
            td_path = Path(td)
            # Write one file per item, deduplicating filename collisions.
            for item in items:
                safe = hashlib.sha1(item.item_id.encode("utf-8")).hexdigest()[:16]
                filename = f"{safe}.txt"
                if filename in item_by_doc_id:
                    n = 1
                    while f"{safe}_{n}.txt" in item_by_doc_id:
                        n += 1
                    filename = f"{safe}_{n}.txt"
                (td_path / filename).write_text(item.text, encoding="utf-8")
                item_by_doc_id[filename] = item
            self.index(td_path)
        # Stamp each chunk with its item's (parent_id, span_id).
        for chunk in self.chunks:
            item = item_by_doc_id.get(chunk.doc_id)
            if item is not None:
                chunk.gold_provenance = ((item.parent_id, item.span_id),)


class _PromotedLayout(_CaptureMixin, BaseSystem):
    """The base per-parent layout as M2/M3 get it."""

    system_id = "PROMOTED"


class _PromotedWordWindow(_PromotedLayout):
    """The base layout under M2/M3's word-window chunker."""

    system_id = "PROMOTED_WW"
    chunker = WORD_WINDOW


class _M4Layout(_CaptureMixin, RaptorSystem):
    """The base layout as M4 inherits it."""


def _hash_and_files(cls, items):
    """Indexes items with cls and returns the hash, file list and system."""
    sysm = cls(config=DEFAULT_CONFIG)
    sysm.index_items(items)
    return sysm.captured_hash, sysm.captured_files, sysm


class TestPromotionIsWhatItClaims(unittest.TestCase):
    """Pins that every system shares the one base layout."""

    def test_m4_inherits_the_base_implementation(self):
        """Pins that M4 uses the base index_items unchanged."""
        self.assertIs(RaptorSystem.index_items, BaseSystem.index_items)

    def test_a_plain_base_subclass_writes_one_file_per_parent(self):
        """Pins that the base layout writes one file per parent."""
        _, files, _ = _hash_and_files(_PromotedLayout, _items_qasper())
        self.assertEqual(len(files), 1, "6 paragraphs of one paper -> 1 file")

    def test_the_legacy_reference_still_writes_one_file_per_item(self):
        """Pins that the reference layout writes one file per item."""
        _, files, _ = _hash_and_files(_LegacyPerItemLayout, _items_qasper())
        self.assertEqual(len(files), 6)


class TestSingleItemParentIsByteIdentical(unittest.TestCase):
    """Pins that a single-item parent hashes as one file per item does."""

    def test_multihop_shape_hash_is_unmoved(self):
        """Pins that the 1:1 MultiHop shape hashes the same either way."""
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, _items_multihop())
        promoted, _, _ = _hash_and_files(_PromotedLayout, _items_multihop())
        self.assertEqual(legacy, promoted)

    def test_narrativeqa_quality_shape_hash_is_unmoved(self):
        """Pins that the file name comes from item_id and enters the hash."""
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, _items_whole())
        promoted, _, _ = _hash_and_files(_PromotedLayout, _items_whole())
        self.assertEqual(legacy, promoted)

    def test_uncleaned_text_is_preserved_verbatim(self):
        """Pins that a single-item parent is written raw, not cleaned."""
        dirty = "Article  with  double  spaces.  " * 30
        self.assertNotEqual(clean_text(dirty), dirty)
        items = [CorpusItem(item_id="a::<whole>", parent_id="a",
                            span_id="<whole>", text=dirty)]
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, items)
        promoted, _, _ = _hash_and_files(_PromotedLayout, items)
        self.assertEqual(legacy, promoted)

    def test_every_system_lands_on_the_same_corpus_hash(self):
        """Pins that the base subclass and M4 write identical temp dirs."""
        base_hash, _, _ = _hash_and_files(_PromotedLayout, _items_whole())
        m4_hash, _, _ = _hash_and_files(_M4Layout, _items_whole())
        self.assertEqual(base_hash, m4_hash)

    def test_many_item_parent_hash_moves(self):
        """Pins that a multi-item parent hashes unlike one file per item."""
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, _items_qasper())
        promoted, _, _ = _hash_and_files(_PromotedLayout, _items_qasper())
        self.assertNotEqual(legacy, promoted)


class TestTheGranularityBugIsFixed(unittest.TestCase):
    """Pins that HotpotQA sentence items only chunk when grouped by parent."""

    def test_per_item_sentences_are_dropped_by_the_min_chars_filter(self):
        """Pins that ~124-char sentences fall under min_chars_per_doc = 200."""
        _, _, sysm = _hash_and_files(_LegacyPerItemLayout, _items_hotpot())
        self.assertEqual(sysm.chunks, [])

    def test_per_parent_paragraphs_survive_and_carry_every_atom(self):
        """Pins that whole paragraphs chunk and keep every sentence atom."""
        _, files, sysm = _hash_and_files(_PromotedWordWindow, _items_hotpot())
        self.assertEqual(len(files), 3, "3 titles -> 3 paragraph files")
        self.assertTrue(sysm.chunks)
        seen = {span for c in sysm.chunks for _, span in c.gold_provenance}
        self.assertEqual(seen, {f"sent{s}" for s in range(_HOTPOT_SENTS_PER_TITLE)})


class TestProvenance(unittest.TestCase):
    """Pins the gold provenance stamped on each chunk."""

    def test_single_item_parent_stamps_its_own_atom(self):
        """Pins that a single-item parent stamps its one (parent, span)."""
        _, _, sysm = _hash_and_files(_PromotedLayout, _items_whole(2))
        self.assertTrue(sysm.chunks)
        for chunk in sysm.chunks:
            self.assertEqual(len(chunk.gold_provenance), 1)
            parent, span = chunk.gold_provenance[0]
            self.assertTrue(parent.startswith("story"))
            self.assertEqual(span, "<whole>")

    def test_multi_item_parent_derives_atoms_from_offsets(self):
        """Pins that a multi-item parent stamps atoms from char offsets."""
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
        # Contiguous chunking partitions the document, so every paragraph
        # lands in at least one chunk.
        self.assertEqual(seen, {f"sec1.para{i}" for i in range(6)})

    def test_word_window_multi_item_parent_derives_atoms_too(self):
        """Pins that overlapping word windows still cover every atom."""
        _, _, sysm = _hash_and_files(_PromotedWordWindow, _items_qasper())
        self.assertTrue(sysm.chunks)
        seen = {span for c in sysm.chunks for _, span in c.gold_provenance}
        self.assertEqual(seen, {f"sec1.para{i}" for i in range(6)})

    def test_a_boundary_crossing_chunk_carries_both_atoms(self):
        """Pins that a chunk spanning two short items carries both atoms."""
        _, _, sysm = _hash_and_files(_M4Layout, _items_qasper())
        self.assertTrue(
            any(len(c.gold_provenance) > 1 for c in sysm.chunks),
            "no chunk spanned an item boundary; the fixture is too coarse "
            "to exercise multi-atom provenance",
        )

    def test_provenance_is_ordered_by_member_position(self):
        """Pins that a chunk's atoms follow the members' document order."""
        _, _, sysm = _hash_and_files(_M4Layout, _items_qasper())
        order = {f"sec1.para{i}": i for i in range(6)}
        for chunk in sysm.chunks:
            spans = [order[s] for _, s in chunk.gold_provenance]
            self.assertEqual(spans, sorted(spans))


class TestWordWindowOffsets(unittest.TestCase):
    """Pins the word-window char offsets and the chunk text they index."""

    def _doc(self, text: str) -> ParsedDocument:
        """Wraps text in a one-document corpus."""
        return ParsedDocument(doc_id="d.txt", path=Path("d.txt"), text=text)

    def test_offsets_index_the_source_text(self):
        """Pins that start_char/end_char slice the words of each chunk."""
        text = "alpha beta gamma\n\ndelta epsilon zeta eta theta iota kappa"
        cfg = ChunkingConfig(strategy="word_window", chunk_words=4, overlap_words=1)
        chunks = chunk_corpus([self._doc(text)], cfg)
        self.assertTrue(chunks)
        for c in chunks:
            lo, hi = c.metadata["start_char"], c.metadata["end_char"]
            self.assertEqual(text[lo:hi].split(), c.text.split())

    def test_chunk_text_is_unchanged_by_the_offsets(self):
        """Pins that chunk text matches a plain word-window stride."""
        text = "  one two\tthree\n\nfour   five six seven eight  "
        cfg = ChunkingConfig(strategy="word_window", chunk_words=3, overlap_words=1)
        chunks = chunk_corpus([self._doc(text)], cfg)
        # Build the expected texts with a plain stride over the words.
        words = text.split()
        stride = cfg.chunk_words - cfg.overlap_words
        expected = []
        for start in range(0, len(words), stride):
            window = words[start:start + cfg.chunk_words]
            if not window:
                break
            expected.append(" ".join(window))
            if start + cfg.chunk_words >= len(words):
                break
        self.assertEqual([c.text for c in chunks], expected)

    def test_offsets_are_monotonic_and_within_bounds(self):
        """Pins that offsets increase and stay inside the text."""
        text = " ".join(f"w{i}" for i in range(50))
        cfg = ChunkingConfig(strategy="word_window", chunk_words=10, overlap_words=2)
        chunks = chunk_corpus([self._doc(text)], cfg)
        last = -1
        for c in chunks:
            lo, hi = c.metadata["start_char"], c.metadata["end_char"]
            self.assertLess(lo, hi)
            self.assertLessEqual(hi, len(text))
            self.assertGreater(lo, last)
            last = lo


class TestBuildParentPayload(unittest.TestCase):
    """Pins how build_parent_payload joins items and reports spans."""

    def test_spans_index_the_payload_they_are_returned_with(self):
        """Pins that each span slices its item's cleaned text."""
        items = _items_qasper(3)
        payload, spans = build_parent_payload(items)
        self.assertEqual(len(spans), 3)
        for (lo, hi, span_id), item in zip(spans, items):
            self.assertEqual(payload[lo:hi], clean_text(item.text))
            self.assertEqual(span_id, item.span_id)

    def test_payload_is_clean_text_stable(self):
        """Pins that clean_text leaves the payload unchanged."""
        payload, _ = build_parent_payload(_items_qasper(4))
        self.assertEqual(clean_text(payload), payload)

    def test_dirty_members_are_cleaned_before_measuring(self):
        """Pins that members are cleaned before their spans are measured."""
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
        """Pins that blank members produce no span and no separator."""
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
        """Pins that parents keep first-appearance order when grouped."""
        items = [
            CorpusItem(item_id="b::1", parent_id="b", span_id="1", text="x"),
            CorpusItem(item_id="a::1", parent_id="a", span_id="1", text="y"),
            CorpusItem(item_id="b::2", parent_id="b", span_id="2", text="z"),
        ]
        groups = base_mod.group_items_by_parent(items)
        self.assertEqual(list(groups), ["b", "a"])
        self.assertEqual([i.span_id for i in groups["b"]], ["1", "2"])


class TestIdempotenceGuard(unittest.TestCase):
    """Pins the guards that refuse a layout whose offsets could drift."""

    def test_non_idempotent_clean_text_is_refused(self):
        """Pins that index_items raises when clean_text is not idempotent."""
        original = base_mod.clean_text
        try:
            # An unconditional append is non-idempotent for every input.
            base_mod.clean_text = lambda t: (t or "") + " ."
            sysm = _PromotedLayout(config=DEFAULT_CONFIG)
            with self.assertRaises(RuntimeError):
                sysm.index_items(_items_qasper())
        finally:
            base_mod.clean_text = original

    def test_missing_offsets_are_refused_for_a_multi_item_parent(self):
        """Pins that missing char offsets raise on a multi-item parent."""
        class _NoOffsets(_PromotedLayout):
            def index(self, corpus_path):
                super().index(corpus_path)
                for c in self.chunks:
                    c.metadata.pop("start_char", None)

        sysm = _NoOffsets(config=DEFAULT_CONFIG)
        with self.assertRaises(RuntimeError):
            sysm.index_items(_items_qasper())


class TestM1IsUnaffected(unittest.TestCase):
    """Pins that M1 has no index, so the layout cannot reach its answers."""

    def test_index_ignores_the_corpus_path_entirely(self):
        """Pins that M1's index never reads the corpus path."""
        sysm = ClosedBookSystem(DEFAULT_CONFIG)
        # The path does not exist; anything reading the corpus raises here.
        sysm.index(Path("no-such-directory-anywhere"))
        self.assertTrue(sysm._indexed)

    def test_index_items_produces_no_chunks_and_no_retrieval(self):
        """Pins that M1 keeps no chunks and retrieves nothing."""
        sysm = ClosedBookSystem(DEFAULT_CONFIG)
        sysm.index_items(_items_hotpot())
        self.assertEqual(sysm.chunks, [])
        self.assertEqual(sysm.retrieve("anything"), [])

    def test_the_prompt_is_identical_under_either_layout(self):
        """Pins that an M1 prompt is the same under either layout."""
        legacy = ClosedBookSystem(DEFAULT_CONFIG)
        legacy.index_items = _LegacyPerItemLayout.index_items.__get__(legacy)
        legacy.index_items(_items_hotpot())
        promoted = ClosedBookSystem(DEFAULT_CONFIG)
        promoted.index_items(_items_hotpot())
        a = legacy.prepare("who wrote it?")
        b = promoted.prepare("who wrote it?")
        self.assertEqual((a.system_prompt, a.user_prompt, a.n_input_tokens),
                         (b.system_prompt, b.user_prompt, b.n_input_tokens))


class TestTempDirIsCleanedUp(unittest.TestCase):
    """Pins that the temp corpus directory is removed after indexing."""

    def test_no_temp_directory_survives(self):
        """Pins that no M4_corpus_* directory outlives index_items."""
        before = set(Path(tempfile.gettempdir()).glob("M4_corpus_*"))
        _hash_and_files(_M4Layout, _items_multihop())
        after = set(Path(tempfile.gettempdir()).glob("M4_corpus_*"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
