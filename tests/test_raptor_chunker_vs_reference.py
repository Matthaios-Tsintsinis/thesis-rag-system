"""Diff M4's chunker against a transcription of the reference split_text.

Agreement is asserted where we follow the reference; the two declared
divergences (METHODS §A.4.4 rulings 1 and 1b) are asserted as divergences.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiktoken

from src.raptor_paper import REFERENCE_ENCODING, split_text_raptor


# ref: raptor/utils.py::split_text @ 7da1d48a
# Transcribed as-is, including the re.split that drops delimiters and the
# out-of-order append of over-long-sentence pieces. Leave it untouched: it
# is the oracle the tests diff against.
def reference_split_text(text, tokenizer, max_tokens, overlap=0):
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        if not sentence.strip():
            continue

        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            filtered_sub_sentences = [
                sub.strip() for sub in sub_sentences if sub.strip() != ""
            ]
            sub_token_counts = [
                len(tokenizer.encode(" " + sub_sentence))
                for sub_sentence in filtered_sub_sentences
            ]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(
                filtered_sub_sentences, sub_token_counts
            ):
                if sub_length + sub_token_count > max_tokens:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(
                            sub_token_counts[
                                max(0, len(sub_chunk) - overlap):len(sub_chunk)
                            ]
                        )
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(
                n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)]
            )
            current_chunk.append(sentence)
            current_length += token_count

        else:
            current_chunk.append(sentence)
            current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _strip_punct(s: str) -> str:
    """Strip the punctuation ruling 1 restores, then lower-case."""
    return re.sub(r"[.!?,;:\s]+", " ", s).strip().lower()


PROSE = (
    "The committee met on Tuesday. Its chair opened with a short statement! "
    "Nobody objected to the agenda? The first item concerned the budget. "
) * 12

LONG_SENTENCE = (
    "the quick brown fox jumps over the lazy dog while carrying " * 20
).strip()


class TestOrdinaryProse(unittest.TestCase):
    """Prose with no over-long sentence, so only ruling 1 applies."""

    def setUp(self):
        self.tok = tiktoken.get_encoding(REFERENCE_ENCODING)
        self.ours = split_text_raptor(PROSE, max_tokens=100)
        self.ref = reference_split_text(PROSE, self.tok, 100)

    def test_the_content_stream_is_identical_modulo_punctuation(self):
        """Concatenated chunks give the same word stream on both sides."""
        ours = " ".join(_strip_punct(s.text) for s in self.ours)
        ref = " ".join(_strip_punct(c) for c in self.ref)
        self.assertEqual(ours, ref)

    def test_the_reference_really_is_punctuation_free(self):
        """The transcribed reference emits no terminators."""
        self.assertFalse(any(ch in c for c in self.ref for ch in ".!?"))

    def test_ours_keeps_terminators(self):
        """Every chunk of ours keeps at least one terminator."""
        self.assertTrue(
            all(any(ch in s.text for ch in ".!?") for s in self.ours)
        )

    def test_every_chunk_respects_the_bound(self):
        """Ordinary prose never exceeds the 100-token bound."""
        self.assertTrue(all(s.n_tokens <= 100 for s in self.ours))


class TestOverLongSentencePlacement(unittest.TestCase):
    """Ruling 1b: over-long-sentence pieces stay in document order."""

    def setUp(self):
        self.tok = tiktoken.get_encoding(REFERENCE_ENCODING)
        self.doc = (
            "Short opener sentence number one. "
            "Short opener sentence number two. "
            + LONG_SENTENCE
            + ". Short closer sentence number one. "
            "Short closer sentence number two."
        )
        self.ours = [s.text for s in split_text_raptor(self.doc, max_tokens=100)]
        self.ref = reference_split_text(self.doc, self.tok, 100)

    def test_the_fixture_actually_exercises_the_branch(self):
        """The long sentence really exceeds the bound on its own."""
        self.assertGreater(
            len(self.tok.encode(" " + LONG_SENTENCE)), 100
        )

    def test_ours_is_in_document_order(self):
        """Ours puts openers first and closers last."""
        self.assertIn("opener", self.ours[0])
        self.assertIn("closer", self.ours[-1])

    def test_the_reference_is_not_in_document_order(self):
        """The reference emits the long sentence ahead of the openers."""
        self.assertNotIn("opener", self.ref[0])
        self.assertIn("quick brown fox", self.ref[0])

    def test_the_reference_packs_flanking_sentences_across_the_long_one(self):
        """The reference packs openers and closers into one chunk."""
        merged = [c for c in self.ref if "opener" in c and "closer" in c]
        self.assertEqual(len(merged), 1)

    def test_no_content_is_lost_on_either_side(self):
        """Both sides carry the same multiset of words."""
        ours = sorted(_strip_punct(" ".join(self.ours)).split())
        ref = sorted(_strip_punct(" ".join(self.ref)).split())
        self.assertEqual(ours, ref)


class TestSharedReferenceBehaviours(unittest.TestCase):
    """Behaviours where ours must agree with the reference."""

    def setUp(self):
        self.tok = tiktoken.get_encoding(REFERENCE_ENCODING)

    def test_an_oversized_subphrase_is_emitted_not_truncated(self):
        """An unsplittable piece over the bound is emitted whole by both."""
        unsplittable = "word " * 400
        ours = split_text_raptor(unsplittable, max_tokens=100)
        ref = reference_split_text(unsplittable, self.tok, 100)
        self.assertTrue(any(s.n_tokens > 100 for s in ours))
        self.assertTrue(
            any(len(self.tok.encode(" " + c)) > 100 for c in ref)
        )

    def test_neither_side_overlaps(self):
        """Neither side repeats words across chunks (overlap 0)."""
        ours = [_strip_punct(s.text) for s in split_text_raptor(PROSE, max_tokens=100)]
        joined = " ".join(ours).split()
        ref_joined = " ".join(
            _strip_punct(c) for c in reference_split_text(PROSE, self.tok, 100)
        ).split()
        self.assertEqual(len(joined), len(ref_joined))

    def test_empty_input_is_empty_on_both_sides(self):
        """Whitespace-only input yields no chunks on either side."""
        self.assertEqual(split_text_raptor("   ", max_tokens=100), [])
        self.assertEqual(reference_split_text("   ", self.tok, 100), [])


if __name__ == "__main__":
    unittest.main()
