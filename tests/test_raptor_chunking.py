"""Tests for the RAPTOR chunker in src/raptor_paper.py: the 100-token soft
bound, sentence preservation, zero overlap, the kept terminators, and the
ChunkingConfig schema that every substrate cache key folds in.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from pathlib import Path

from src.chunking import chunk_document
from src.config import ChunkingConfig
from src.parsing import ParsedDocument
from src.raptor_paper import (
    count_tokens_reference,
    split_text_raptor,
)


RAPTOR_CFG = ChunkingConfig(
    strategy="raptor_100tok", chunk_words=100, overlap_words=0
)


def _doc(text: str, doc_id: str = "d0") -> ParsedDocument:
    """Wrap text in a ParsedDocument with a fixed id."""
    return ParsedDocument(
        doc_id=doc_id, path=Path(f"{doc_id}.txt"), text=text, metadata={}
    )


class TestTerminatorPreservation(unittest.TestCase):
    """Sentence delimiters stay attached to their sentence."""

    # deviation from ref (ref's re.split drops . ! ? \n): see METHODS §A.4.4 ruling 1

    def test_sentence_terminators_are_kept(self):
        """. ! ? stay on the sentence they end."""
        spans = split_text_raptor("Alpha beta. Gamma delta! Epsilon zeta?")
        joined = " ".join(s.text for s in spans)
        self.assertIn("beta.", joined)
        self.assertIn("delta!", joined)
        self.assertIn("zeta?", joined)

    def test_terminator_runs_collapse_to_one_boundary(self):
        """A run like ... or ?! is one boundary and leaves no empty segment."""
        spans = split_text_raptor("Wait... Really?! Yes.")
        joined = " ".join(s.text for s in spans)
        self.assertIn("Wait...", joined)
        self.assertIn("Really?!", joined)
        self.assertNotIn("  ", joined)

    def test_newlines_are_boundaries_not_content(self):
        """A newline splits sentences and collapses to a space."""
        spans = split_text_raptor("Line one\n\nLine two")
        joined = " ".join(s.text for s in spans)
        self.assertNotIn("\n", joined)
        self.assertIn("Line one", joined)
        self.assertIn("Line two", joined)

    def test_subphrase_delimiters_are_kept(self):
        """, ; : survive the sub-split of an over-long sentence."""
        long_clause = " ".join(["word"] * 90)
        text = f"{long_clause}, {long_clause}; tail."
        spans = split_text_raptor(text)
        joined = " ".join(s.text for s in spans)
        self.assertIn(",", joined)
        self.assertIn(";", joined)


class TestTokenBound(unittest.TestCase):
    """The 100-token soft bound and the reference token count."""

    def test_chunks_respect_the_soft_bound(self):
        """Ordinary sentences never push a chunk past max_tokens."""
        # RAPTOR paper §3: "short, contiguous texts of length 100"
        text = " ".join(f"Sentence number {i} here." for i in range(200))
        spans = split_text_raptor(text, max_tokens=100)
        self.assertGreater(len(spans), 1)
        for s in spans:
            self.assertLessEqual(s.n_tokens, 100)

    def test_sentence_is_never_cut_mid_sentence(self):
        """An overflowing sentence moves whole to the next chunk."""
        # RAPTOR paper §3: "we move the entire sentence to the next chunk"
        text = " ".join(f"This is sentence {i} of the document." for i in range(60))
        spans = split_text_raptor(text, max_tokens=100)
        for s in spans:
            self.assertFalse(s.text.startswith("of the document"))
            # every chunk ends on a sentence boundary
            self.assertTrue(s.text.rstrip().endswith("."))

    def test_over_long_sentence_is_emitted_oversized_not_truncated(self):
        """A long sentence with no , ; : is emitted oversized, not cut."""
        # ref: raptor/utils.py::split_text @ 7da1d48a
        monster = " ".join(["token"] * 400) + "."
        spans = split_text_raptor(monster, max_tokens=100)
        self.assertEqual(len(spans), 1)
        self.assertGreater(spans[0].n_tokens, 100)
        self.assertIn("token token", spans[0].text)

    def test_over_long_sentence_falls_back_to_subphrases(self):
        """A long sentence is sub-split on , ; : before being emitted."""
        # ref: raptor/utils.py::split_text @ 7da1d48a
        clause = " ".join(["alpha"] * 80)
        text = f"{clause}, {clause}, {clause}."
        spans = split_text_raptor(text, max_tokens=100)
        self.assertGreater(len(spans), 1)

    def test_token_counting_uses_the_reference_leading_space(self):
        """Token count is len(encode(" " + sentence)) under cl100k_base."""
        # ref: raptor/utils.py::split_text @ 7da1d48a (tiktoken cl100k_base)
        self.assertEqual(
            count_tokens_reference("hello world"),
            len(__import__("tiktoken").get_encoding("cl100k_base").encode(" hello world")),
        )


class TestNoOverlap(unittest.TestCase):
    """Chunks never share content."""

    # ref: raptor/utils.py::split_text @ 7da1d48a (overlap never passed, 0)

    def test_chunks_do_not_repeat_content(self):
        """No sentence marker appears in two chunks."""
        text = " ".join(f"Unique sentence {i} marker." for i in range(80))
        spans = split_text_raptor(text, max_tokens=100)
        seen: set[str] = set()
        for s in spans:
            for word in s.text.split():
                if word.isdigit():
                    self.assertNotIn(word, seen)
                    seen.add(word)

    def test_spans_are_monotonic_and_non_overlapping(self):
        """Character spans are ordered, non-empty and disjoint."""
        text = " ".join(f"Sentence {i} body text here." for i in range(80))
        spans = split_text_raptor(text, max_tokens=100)
        for prev, nxt in zip(spans, spans[1:]):
            self.assertLessEqual(prev.end_char, nxt.start_char)
            self.assertLess(prev.start_char, prev.end_char)


class TestProvenanceSpans(unittest.TestCase):
    """Character offsets point into the original text."""

    def test_spans_point_into_the_original_text(self):
        """Each span lies inside the text and covers its first token."""
        text = "Alpha one. Beta two. Gamma three."
        spans = split_text_raptor(text, max_tokens=6)
        for s in spans:
            self.assertGreaterEqual(s.start_char, 0)
            self.assertLessEqual(s.end_char, len(text))
            # chunk text is whitespace-normalised, so check containment
            region = text[s.start_char : s.end_char]
            self.assertIn(s.text.split()[0], region)

    def test_offsets_skip_leading_whitespace(self):
        """start_char lands on the first non-space character."""
        text = "\n\n   Alpha one. Beta two."
        spans = split_text_raptor(text, max_tokens=100)
        self.assertEqual(text[spans[0].start_char], "A")


class TestDegenerate(unittest.TestCase):
    """Empty, unterminated and invalid inputs."""

    def test_empty_and_whitespace(self):
        """Empty or whitespace-only text yields no spans."""
        self.assertEqual(split_text_raptor(""), [])
        self.assertEqual(split_text_raptor("   \n\n  "), [])

    def test_no_terminator_at_all(self):
        """Text without a terminator is one span, unchanged."""
        spans = split_text_raptor("just some words with no stop")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "just some words with no stop")

    def test_rejects_non_positive_budget(self):
        """max_tokens <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            split_text_raptor("text", max_tokens=0)


class TestChunkDocumentIntegration(unittest.TestCase):
    """chunk_document dispatches the raptor_100tok strategy."""

    def test_dispatch_produces_chunks_with_offset_metadata(self):
        """Chunks carry position, id, word count and offset metadata."""
        text = " ".join(f"Sentence {i} of the doc." for i in range(60))
        chunks = chunk_document(_doc(text), RAPTOR_CFG)
        self.assertGreater(len(chunks), 1)
        for i, c in enumerate(chunks):
            self.assertEqual(c.position, i)
            self.assertEqual(c.chunk_id, f"d0::{i:04d}")
            self.assertEqual(c.n_words, len(c.text.split()))
            self.assertIn("start_char", c.metadata)
            self.assertIn("end_char", c.metadata)
            self.assertIn("n_tokens", c.metadata)
            # gold_provenance is stamped later by index_items
            self.assertEqual(c.gold_provenance, ())

    def test_non_zero_overlap_is_rejected(self):
        """The raptor strategy refuses a non-zero overlap."""
        bad = replace(RAPTOR_CFG, overlap_words=10)
        with self.assertRaises(ValueError):
            chunk_document(_doc("Alpha. Beta."), bad)

    def test_other_strategies_are_untouched(self):
        """The default word_window strategy still yields 200-word chunks."""
        # harness choice: shared default for M2/M3 (METHODS §A.2)
        text = " ".join(["word"] * 500)
        ww = chunk_document(_doc(text), ChunkingConfig())
        self.assertTrue(ww)
        self.assertEqual(ww[0].n_words, 200)


class TestCacheDiscipline(unittest.TestCase):
    """The ChunkingConfig schema is pinned; a new field moves every key."""

    # kept: part of every substrate cache key
    EXPECTED_FIELDS = {
        "strategy",
        "breakpoint_percentile",
        "absolute_threshold",
        "min_words",
        "max_words",
        "max_if_min_words",
        "buffer_size",
        "chunk_words",
        "overlap_words",
        "min_chars_per_doc",
    }

    def test_chunking_config_schema_is_frozen(self):
        """ChunkingConfig has exactly the expected fields."""
        self.assertEqual(set(asdict(ChunkingConfig())), self.EXPECTED_FIELDS)

    def test_default_config_values_are_unchanged(self):
        """The default strategy, window, overlap and doc floor are pinned."""
        # harness choice: shared default for M2/M3 (METHODS §A.2)
        d = asdict(ChunkingConfig())
        self.assertEqual(d["strategy"], "word_window")
        self.assertEqual(d["chunk_words"], 200)
        self.assertEqual(d["overlap_words"], 50)
        self.assertEqual(d["min_chars_per_doc"], 200)

    def test_raptor_config_differs_only_in_existing_fields(self):
        """RAPTOR_CFG changes only strategy, chunk_words and overlap_words."""
        base, raptor = asdict(ChunkingConfig()), asdict(RAPTOR_CFG)
        self.assertEqual(set(base), set(raptor))
        self.assertEqual(
            {k for k in base if base[k] != raptor[k]},
            {"strategy", "chunk_words", "overlap_words"},
        )


if __name__ == "__main__":
    unittest.main()
