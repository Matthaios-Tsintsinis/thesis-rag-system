"""Pins the semantic leaves: where sentences end, where a document is cut, how
pieces are sized, and that every word of a document lands in one leaf with the
offsets gold provenance is stamped from."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from tests.test_ours_smoke import (
    ANSWERS,
    DECOMPOSITIONS,
    FakeReader,
    fake_embed,
    hotpot_unit,
    llama_config,
    ours_fakes,
)


def _units(*which):
    """Unit vectors along the axes named, one row per sentence."""
    out = np.zeros((len(which), 4), dtype=np.float32)
    for row, axis in enumerate(which):
        out[row, axis] = 1.0
    return out


def _spans(n):
    """n five-character sentences a space apart."""
    return [(6 * i, 6 * i + 5) for i in range(n)]


class TestSentences(unittest.TestCase):

    def _texts(self, text):
        return [text[lo:hi] for lo, hi in om.split_sentences(text)]

    def test_the_designs_terminators_and_line_breaks_end_a_sentence(self):
        self.assertEqual(self._texts("One. Two! Three? Four; five\n\nSix"),
                         ["One.", "Two!", "Three?", "Four;", "five", "Six"])

    def test_a_terminator_inside_a_word_does_not(self):
        self.assertEqual(self._texts("Version 2.5 is out. Yes."),
                         ["Version 2.5 is out.", "Yes."])

    def test_no_empty_sentence_and_no_sentence_at_all(self):
        self.assertEqual(self._texts("One.\n\n\n  \n\nTwo."), ["One.", "Two."])
        self.assertEqual(om.split_sentences("   \n "), [])

    def test_each_sentence_sees_its_buffer(self):
        text = "One. Two. Three. Four."
        self.assertEqual(om.buffered_sentences(text, om.split_sentences(text)),
                         ["One. Two.", "One. Two. Three.", "Two. Three. Four.", "Three. Four."])

    def test_a_wider_buffer_reaches_further(self):
        text = "One. Two. Three. Four."
        with mock.patch.object(settings, "SENTENCE_BUFFER", 2):
            self.assertEqual(om.buffered_sentences(text, om.split_sentences(text))[0],
                             "One. Two. Three.")


class TestCuts(unittest.TestCase):

    def test_cut_where_the_topic_moves(self):
        got = om.topic_segments(_spans(6), _units(0, 0, 0, 1, 1, 1))
        self.assertEqual(got, [(0, 17), (18, 35)])

    def test_one_topic_is_one_piece(self):
        self.assertEqual(om.topic_segments(_spans(4), _units(0, 0, 0, 0)), [(0, 23)])

    def test_one_sentence_or_none(self):
        self.assertEqual(om.topic_segments(_spans(1), _units(0)), [(0, 5)])
        self.assertEqual(om.topic_segments([], np.zeros((0, 4))), [])

    def test_the_percentile_decides_below_the_absolute_cap(self):
        """Close sentences are still cut at their largest gap."""
        a, b, c = np.array([1, 0]), np.array([0.99, 0.141]), np.array([0.95, 0.312])
        vectors = np.vstack([v / np.linalg.norm(v) for v in (a, a, a, b, c, c)])
        got = om.topic_segments(_spans(6), vectors)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0], (0, 23))

    def test_the_absolute_cap_cuts_every_gap_above_it(self):
        """Half the gaps are wide: the 90th percentile alone would cut none."""
        vectors = _units(0, 1, 0, 1, 0)
        self.assertEqual(len(om.topic_segments(_spans(5), vectors)), 5)


class TestSizes(unittest.TestCase):

    @staticmethod
    def _words(n, start=0):
        return " ".join(f"w{start + i}" for i in range(n))

    def _sized(self, *counts):
        """Pieces of the given word counts, end to end."""
        parts, segments, cursor, seen = [], [], 0, 0
        for n in counts:
            part = self._words(n, seen)
            parts.append(part)
            segments.append((cursor, cursor + len(part)))
            cursor += len(part) + 1
            seen += n
        return " ".join(parts), om.sized_segments(" ".join(parts), segments)

    def test_a_long_piece_is_split_into_windows(self):
        with mock.patch.multiple(settings, MAX_CHUNK_WORDS=10, MIN_CHUNK_WORDS=2,
                                 MAX_MERGED_CHUNK_WORDS=12):
            text, got = self._sized(25)
        self.assertEqual([n for *_, n in got], [10, 10, 5])
        self.assertEqual(text[got[1][0]:got[1][1]], self._words(10, 10))

    def test_a_short_piece_joins_the_one_before(self):
        with mock.patch.multiple(settings, MAX_CHUNK_WORDS=10, MIN_CHUNK_WORDS=5,
                                 MAX_MERGED_CHUNK_WORDS=12):
            text, got = self._sized(6, 3)
        self.assertEqual([n for *_, n in got], [9])
        self.assertEqual(text[got[0][0]:got[0][1]], text)

    def test_a_join_stops_at_the_merged_cap(self):
        with mock.patch.multiple(settings, MAX_CHUNK_WORDS=10, MIN_CHUNK_WORDS=5,
                                 MAX_MERGED_CHUNK_WORDS=12):
            _, got = self._sized(10, 3, 4)
        self.assertEqual([n for *_, n in got], [10, 7])

    def test_short_pieces_build_up_to_the_cap(self):
        with mock.patch.multiple(settings, MAX_CHUNK_WORDS=10, MIN_CHUNK_WORDS=5,
                                 MAX_MERGED_CHUNK_WORDS=12):
            _, got = self._sized(4, 4, 4, 4)
        self.assertEqual([n for *_, n in got], [12, 4])

    def test_a_short_first_piece_stays_a_leaf(self):
        """Nothing stands before it to join; the design's rule looks back only."""
        with mock.patch.multiple(settings, MAX_CHUNK_WORDS=10, MIN_CHUNK_WORDS=5,
                                 MAX_MERGED_CHUNK_WORDS=12):
            _, got = self._sized(3, 8)
        self.assertEqual([n for *_, n in got], [3, 8])

    def test_the_design_values_are_the_harness_records(self):
        """The settings are the chunker record's defaults, the design's own values."""
        from src.config import ChunkingConfig

        record = ChunkingConfig()
        self.assertEqual(
            (settings.SENTENCE_BUFFER, settings.BREAKPOINT_PERCENTILE,
             settings.BREAKPOINT_MAX_DISTANCE, settings.MIN_CHUNK_WORDS,
             settings.MAX_CHUNK_WORDS, settings.MAX_MERGED_CHUNK_WORDS),
            (record.buffer_size, record.breakpoint_percentile, record.absolute_threshold,
             record.min_words, record.max_words, record.max_if_min_words))


class TestSemanticChunks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rivers = " ".join(f"The river number {i} runs past the old mill and the town." for i in range(30))
        kings = " ".join(f"The king number {i} ruled the northern land for many years." for i in range(30))
        cls.docs = [SimpleNamespace(doc_id="a.txt", text=f"{rivers}\n\n{kings}"),
                    SimpleNamespace(doc_id="b.txt", text="A short note. Only two sentences."),
                    SimpleNamespace(doc_id="c.txt", text=" ".join(["word"] * 900))]
        cls.calls = []

        def embed(texts, model_name=None):
            cls.calls.append((len(texts), model_name))
            return fake_embed(texts, model_name)

        with mock.patch("src.models.embed_texts", embed):
            cls.chunks = om.semantic_chunks(cls.docs)

    def _of(self, doc_id):
        return [c for c in self.chunks if c.doc_id == doc_id]

    def test_every_word_lands_in_one_leaf_in_order(self):
        for doc in self.docs:
            self.assertEqual(" ".join(c.text for c in self._of(doc.doc_id)),
                             " ".join(doc.text.split()))

    def test_offsets_slice_back_to_the_leaf_and_do_not_overlap(self):
        for doc in self.docs:
            mine = self._of(doc.doc_id)
            for c in mine:
                piece = doc.text[c.metadata["start_char"]:c.metadata["end_char"]]
                self.assertEqual(" ".join(piece.split()), c.text)
            for a, b in zip(mine, mine[1:]):
                self.assertLessEqual(a.metadata["end_char"], b.metadata["start_char"])

    def test_ids_counts_and_order(self):
        for doc in self.docs:
            for i, c in enumerate(self._of(doc.doc_id)):
                self.assertEqual((c.chunk_id, c.position, c.n_words),
                                 (f"{doc.doc_id}::{i:04d}", i, len(c.text.split())))

    def test_a_leaf_has_no_line_break_and_stays_under_the_cap(self):
        for c in self.chunks:
            self.assertNotIn("\n", c.text)
            self.assertLessEqual(c.n_words, settings.MAX_MERGED_CHUNK_WORDS)

    def test_a_short_document_is_one_leaf_and_a_long_one_is_cut(self):
        self.assertEqual(len(self._of("b.txt")), 1)
        self.assertGreater(len(self._of("a.txt")), 1)
        self.assertEqual([c.n_words for c in self._of("c.txt")], [400, 400, 100])

    def test_sentences_are_embedded_once_for_the_corpus_with_ours_embedder(self):
        self.assertEqual(self.calls, [(63, settings.EMBEDDER)])

    def test_no_document_no_leaf(self):
        with mock.patch("src.models.embed_texts", fake_embed):
            self.assertEqual(om.semantic_chunks([]), [])


class TestProvenance(unittest.TestCase):
    """The harness stamps gold provenance from the leaves' offsets."""

    def test_a_leaf_carries_the_sentences_it_covers(self):
        unit = hotpot_unit(0)
        with tempfile.TemporaryDirectory() as td, \
                ours_fakes(Path(td), FakeReader(DECOMPOSITIONS, ANSWERS)):
            ours = om.OursSystem(config=llama_config())
            ours.index_items(unit.corpus)
        by_title: dict = {}
        for item in unit.corpus:
            by_title.setdefault(item.parent_id, []).append(item.span_id)
        for title, spans in by_title.items():
            leaves = [c for c in ours.chunks if c.gold_provenance
                      and c.gold_provenance[0][0] == title]
            self.assertTrue(leaves, title)
            covered = {span for c in leaves for _, span in c.gold_provenance}
            self.assertEqual(covered, set(spans))
        self.assertTrue(all(re.search(r"\w", c.text) for c in ours.chunks))


if __name__ == "__main__":
    unittest.main()
