"""The per-parent corpus layout — now `BaseSystem.index_items` — and the
cache-key promise it makes to every system that uses it.

The layout matters twice. Semantically, it is what lets a retriever chunk
the DOCUMENT rather than per-item annotation fragments. Mechanically,
`corpus_content_hash` is computed over the temp directory `index_items`
writes, so the layout is a cache-key input for M2, M3, M4, M9 and the
FROZEN M7 — and the single-item rule (rule B) is the promise that keeps
1:1 benchmarks off a rebuild.

WHY THIS FILE CARRIES A FROZEN COPY OF THE OLD LAYOUT. The layout was
M4-local (commit cb5c8ed) and rule B was tested by comparing M4's
override against the inherited per-item base. Promoting it to the base
made that comparison compare a function with itself — a test that cannot
fail is not a test. `_LegacyPerItemLayout` below is therefore a verbatim
frozen copy of the pre-promotion `BaseSystem.index_items`, and rule B is
tested against IT. It is a fixed historical reference, not a live
implementation: if the promoted layout is ever changed, this disagrees,
and that disagreement is the point. DO NOT "sync" it.

Both halves of rule B are tested, because both were measured to matter
and neither is obvious:

  * the FILENAME is folded into corpus_content_hash alongside the bytes,
    so a parent-derived name moves NarrativeQA and QuALITY even though
    they are strictly 1:1 (their item_id is "{parent}::<whole>", not the
    bare parent id). MultiHop escapes only by the coincidence that its
    item_id IS its parent_id.
  * the RAW BYTES matter because parsing.clean_text is not the identity;
    a multi-item parent is written pre-cleaned, and doing that to a
    single-item parent would move its hash via the content instead.

The claim these tests support is about fixtures. The claim about the
REAL banked corpora is measured by
`scripts/prove_index_layout_key_invariance.py`, which computes every
system's substrate key on the real MultiHop and NarrativeQA corpora under
both layouts.
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


WORD_WINDOW = DEFAULT_CONFIG.chunking          # M2 / M3 / M7 / M9
RAPTOR_100TOK = DEFAULT_CONFIG.m4.chunker      # M4


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


# HotpotQA's MEASURED shape, not a guessed one. A fixture whose
# parameters are invented tests the assumption rather than the world (it
# has already happened once in this project), so these come from the
# measurements in the handoff: paragraphs average 127.7 tiktoken tokens
# (~511 chars) over 4.12 sentences, i.e. a sentence is ~124 chars —
# comfortably under `min_chars_per_doc = 200`, while the paragraph is
# comfortably over it. That gap IS the bug.
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
    """Records the corpus hash + file list, then fakes a chunking pass.

    index() is replaced by real parsing + real chunking and nothing else:
    the clean_text trap this module guards against lives in walk_corpus,
    so it must be exercised, while the embeddings and the UMAP/GMM tree
    would only make the test slow.
    """

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
    """FROZEN copy of `BaseSystem.index_items` as it stood at e7f41a3.

    One file per CorpusItem, named from item_id, raw bytes, provenance
    stamped from the item's own (parent_id, span_id). This is the layout
    every banked MultiHop and NarrativeQA cell was built under, so it is
    the reference rule B must match. It is HISTORY: do not update it to
    track changes in the live implementation.
    """

    system_id = "LEGACY"

    def index_items(self, items):  # noqa: D102
        item_by_doc_id: dict[str, CorpusItem] = {}
        with tempfile.TemporaryDirectory(prefix=f"{self.system_id}_corpus_") as td:
            td_path = Path(td)
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
        for chunk in self.chunks:
            item = item_by_doc_id.get(chunk.doc_id)
            if item is not None:
                chunk.gold_provenance = ((item.parent_id, item.span_id),)


class _PromotedLayout(_CaptureMixin, BaseSystem):
    """A plain BaseSystem subclass — i.e. the promoted layout as M2/M3 get it."""

    system_id = "PROMOTED"


class _PromotedWordWindow(_PromotedLayout):
    """M2/M3/M7/M9's real chunker, which is the one the promotion newly exposes
    to multi-item parents."""

    system_id = "PROMOTED_WW"
    chunker = WORD_WINDOW


class _M4Layout(_CaptureMixin, RaptorSystem):
    """M4, which now INHERITS the layout it used to own."""


def _hash_and_files(cls, items):
    sysm = cls(config=DEFAULT_CONFIG)
    sysm.index_items(items)
    return sysm.captured_hash, sysm.captured_files, sysm


class TestPromotionIsWhatItClaims(unittest.TestCase):
    """The promotion itself, asserted rather than assumed."""

    def test_m4_inherits_the_base_implementation(self):
        self.assertIs(RaptorSystem.index_items, BaseSystem.index_items)

    def test_a_plain_base_subclass_writes_one_file_per_parent(self):
        _, files, _ = _hash_and_files(_PromotedLayout, _items_qasper())
        self.assertEqual(len(files), 1, "6 paragraphs of one paper -> 1 file")

    def test_the_legacy_reference_still_writes_one_file_per_item(self):
        """Guards the guard: if this ever passes trivially, every rule-B
        test below is comparing the promoted layout with itself."""
        _, files, _ = _hash_and_files(_LegacyPerItemLayout, _items_qasper())
        self.assertEqual(len(files), 6)


class TestSingleItemParentIsByteIdentical(unittest.TestCase):
    """Rule B. The promise that keeps the banked 1:1 cells off a rebuild."""

    def test_multihop_shape_hash_is_unmoved(self):
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, _items_multihop())
        promoted, _, _ = _hash_and_files(_PromotedLayout, _items_multihop())
        self.assertEqual(legacy, promoted)

    def test_narrativeqa_quality_shape_hash_is_unmoved(self):
        """The case a parent-derived filename would have broken.

        These are strictly 1:1, so the intuition "one item per parent
        means nothing changes" says they are safe. They are not: their
        item_id is "{parent}::<whole>", so naming the file after the
        parent moves the hash through the PATH while the bytes stay
        identical. Rule B is what makes them safe.
        """
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, _items_whole())
        promoted, _, _ = _hash_and_files(_PromotedLayout, _items_whole())
        self.assertEqual(legacy, promoted)

    def test_uncleaned_text_is_preserved_verbatim(self):
        """clean_text is NOT the identity, so pre-cleaning a single-item
        parent would move the hash through the CONTENT instead."""
        dirty = "Article  with  double  spaces.  " * 30
        self.assertNotEqual(clean_text(dirty), dirty)
        items = [CorpusItem(item_id="a::<whole>", parent_id="a",
                            span_id="<whole>", text=dirty)]
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, items)
        promoted, _, _ = _hash_and_files(_PromotedLayout, items)
        self.assertEqual(legacy, promoted)

    def test_every_system_lands_on_the_same_corpus_hash(self):
        """M2/M3/M9's base subclass and M4 must write byte-identical temp
        dirs, or the promotion moved one of them relative to the other."""
        base_hash, _, _ = _hash_and_files(_PromotedLayout, _items_whole())
        m4_hash, _, _ = _hash_and_files(_M4Layout, _items_whole())
        self.assertEqual(base_hash, m4_hash)

    def test_many_item_parent_hash_moves(self):
        """Stated, not hidden: QASPER- and HotpotQA-shaped corpora DO
        rebuild. That is the price of chunking the document instead of its
        fragments, and it is why the promotion was gated on a measurement
        that no 1:1 benchmark moved."""
        legacy, _, _ = _hash_and_files(_LegacyPerItemLayout, _items_qasper())
        promoted, _, _ = _hash_and_files(_PromotedLayout, _items_qasper())
        self.assertNotEqual(legacy, promoted)


class TestTheGranularityBugIsFixed(unittest.TestCase):
    """The reason the promotion happened: HotpotQA's sentence items."""

    def test_per_item_sentences_are_dropped_by_the_min_chars_filter(self):
        """The measured failure. Sentences average ~124 chars against
        min_chars_per_doc=200, so the per-item layout hands the chunker
        almost nothing — and on the unit where NOTHING cleared the bar,
        M2 and M3 raised 'No chunks produced'."""
        _, _, sysm = _hash_and_files(_LegacyPerItemLayout, _items_hotpot())
        self.assertEqual(sysm.chunks, [])

    def test_per_parent_paragraphs_survive_and_carry_every_atom(self):
        _, files, sysm = _hash_and_files(_PromotedWordWindow, _items_hotpot())
        self.assertEqual(len(files), 3, "3 titles -> 3 paragraph files")
        self.assertTrue(sysm.chunks)
        seen = {span for c in sysm.chunks for _, span in c.gold_provenance}
        self.assertEqual(seen, {f"sent{s}" for s in range(_HOTPOT_SENTS_PER_TITLE)})


class TestProvenance(unittest.TestCase):
    def test_single_item_parent_stamps_its_own_atom(self):
        _, _, sysm = _hash_and_files(_PromotedLayout, _items_whole(2))
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

    def test_word_window_multi_item_parent_derives_atoms_too(self):
        """M2/M3's chunker on the same shape. Overlapping windows mean a
        span can be claimed by two chunks; every atom must still appear."""
        _, _, sysm = _hash_and_files(_PromotedWordWindow, _items_qasper())
        self.assertTrue(sysm.chunks)
        seen = {span for c in sysm.chunks for _, span in c.gold_provenance}
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


class TestWordWindowOffsets(unittest.TestCase):
    """The offsets the promotion needed, and the invariance it needed them
    to preserve."""

    def _doc(self, text: str) -> ParsedDocument:
        return ParsedDocument(doc_id="d.txt", path=Path("d.txt"), text=text)

    def test_offsets_index_the_source_text(self):
        text = "alpha beta gamma\n\ndelta epsilon zeta eta theta iota kappa"
        cfg = ChunkingConfig(strategy="word_window", chunk_words=4, overlap_words=1)
        chunks = chunk_corpus([self._doc(text)], cfg)
        self.assertTrue(chunks)
        for c in chunks:
            lo, hi = c.metadata["start_char"], c.metadata["end_char"]
            self.assertEqual(text[lo:hi].split(), c.text.split())

    def test_chunk_text_is_unchanged_by_the_offsets(self):
        """The cache-safety half: adding metadata must not change chunk
        TEXT, or every warm chunks.jsonl disagrees with a rebuild."""
        text = "  one two\tthree\n\nfour   five six seven eight  "
        cfg = ChunkingConfig(strategy="word_window", chunk_words=3, overlap_words=1)
        chunks = chunk_corpus([self._doc(text)], cfg)
        # What the pre-offsets implementation produced, by construction:
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
    def test_spans_index_the_payload_they_are_returned_with(self):
        items = _items_qasper(3)
        payload, spans = build_parent_payload(items)
        self.assertEqual(len(spans), 3)
        for (lo, hi, span_id), item in zip(spans, items):
            self.assertEqual(payload[lo:hi], clean_text(item.text))
            self.assertEqual(span_id, item.span_id)

    def test_payload_is_clean_text_stable(self):
        """The invariant index_items asserts at runtime: reading the file
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
        groups = base_mod.group_items_by_parent(items)
        self.assertEqual(list(groups), ["b", "a"])
        self.assertEqual([i.span_id for i in groups["b"]], ["1", "2"])


class TestIdempotenceGuard(unittest.TestCase):
    def test_non_idempotent_clean_text_is_refused(self):
        """If clean_text ever stops being idempotent, the offsets index a
        string the chunker never sees and provenance drifts across the
        document — silently. Refuse instead."""
        original = base_mod.clean_text
        try:
            # Appending unconditionally is non-idempotent for ANY input,
            # which a substitution-based fake is not: it would collapse
            # on the first pass and then agree with itself.
            base_mod.clean_text = lambda t: (t or "") + " ."
            sysm = _PromotedLayout(config=DEFAULT_CONFIG)
            with self.assertRaises(RuntimeError):
                sysm.index_items(_items_qasper())
        finally:
            base_mod.clean_text = original

    def test_missing_offsets_are_refused_for_a_multi_item_parent(self):
        """A chunker without start_char/end_char (the semantic strategy)
        cannot support a multi-item parent; failing loudly beats empty
        provenance, which CK-2 would read as a retrieval miss."""
        class _NoOffsets(_PromotedLayout):
            def index(self, corpus_path):
                super().index(corpus_path)
                for c in self.chunks:
                    c.metadata.pop("start_char", None)

        sysm = _NoOffsets(config=DEFAULT_CONFIG)
        with self.assertRaises(RuntimeError):
            sysm.index_items(_items_qasper())


class TestM1IsUnaffected(unittest.TestCase):
    """M1 has no index, so the layout cannot reach it. Verified, not assumed —
    it is the reason the banked hotpotqa_M1 cell does not need re-running."""

    def test_index_ignores_the_corpus_path_entirely(self):
        sysm = ClosedBookSystem(DEFAULT_CONFIG)
        # A path that does not exist. Anything that read the corpus would
        # raise here.
        sysm.index(Path("no-such-directory-anywhere"))
        self.assertTrue(sysm._indexed)

    def test_index_items_produces_no_chunks_and_no_retrieval(self):
        sysm = ClosedBookSystem(DEFAULT_CONFIG)
        sysm.index_items(_items_hotpot())
        self.assertEqual(sysm.chunks, [])
        self.assertEqual(sysm.retrieve("anything"), [])

    def test_the_prompt_is_identical_under_either_layout(self):
        """What a HotpotQA M1 row actually depends on."""
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
    def test_no_temp_directory_survives(self):
        before = set(Path(tempfile.gettempdir()).glob("M4_corpus_*"))
        _hash_and_files(_M4Layout, _items_multihop())
        after = set(Path(tempfile.gettempdir()).glob("M4_corpus_*"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
