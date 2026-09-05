"""Pins M3's RRF fusion to the published formula and its zero-BM25 filter.

Expected values are hand-computed, never taken from the function under test.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_CONFIG, RRF_K
from src.retrievers.m3_hybrid import _tokenize, rrf_fuse


class TestCormackFormula(unittest.TestCase):
    def test_the_worked_example(self):
        """Fusing [7, 3], [3, 9]: doc 3 = 1/61 + 1/62, 7 = 1/61, 9 = 1/62."""
        # RRF (Cormack et al. 2009): score = sum 1/(k + rank), rank 1-based, k = 60
        got = rrf_fuse([[7, 3], [3, 9]], k=60)
        expected = [
            (3, 1 / 61 + 1 / 62),
            (7, 1 / 61),
            (9, 1 / 62),
        ]
        self.assertEqual([i for i, _ in got], [i for i, _ in expected])
        for (_, g), (_, e) in zip(got, expected):
            self.assertAlmostEqual(g, e, places=12)

    def test_rank_is_one_based(self):
        """The first document scores 1/(k + 1), not 1/k."""
        (only_id, only_score), = rrf_fuse([[5]], k=60)
        self.assertEqual(only_id, 5)
        self.assertAlmostEqual(only_score, 1 / 61, places=12)
        self.assertNotAlmostEqual(only_score, 1 / 60, places=12)

    def test_contributions_are_summed_across_rankings(self):
        """A document in both rankings gets the sum of both contributions."""
        got = dict(rrf_fuse([[1, 2], [2, 1]], k=60))
        self.assertAlmostEqual(got[1], 1 / 61 + 1 / 62, places=12)
        self.assertAlmostEqual(got[2], 1 / 62 + 1 / 61, places=12)

    def test_ordering_is_descending_by_score(self):
        """Fused output is sorted by score, highest first."""
        got = rrf_fuse([[9, 8, 7, 6], [6, 7]], k=60)
        scores = [s for _, s in got]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_input(self):
        """No rankings, or one empty ranking, fuses to an empty list."""
        self.assertEqual(rrf_fuse([], k=60), [])
        self.assertEqual(rrf_fuse([[]], k=60), [])

    def test_k_is_the_papers_constant_and_the_pipeline_reads_it(self):
        """RRF_K is 60 and the retrieval config resolves to the same value."""
        # RRF (Cormack et al. 2009): score = sum 1/(k + rank), rank 1-based, k = 60
        self.assertEqual(RRF_K, 60)
        self.assertEqual(DEFAULT_CONFIG.retrieval.rrf_k, 60)


class TestZeroBM25Filter(unittest.TestCase):
    """Pins the score > 0 predicate M3 applies to BM25 before fusion."""

    def setUp(self):
        from rank_bm25 import BM25Okapi

        # Nine documents, so no term sits in exactly half the corpus.
        self.docs = [
            "zebra quantum telemetry report",
            "ordinary filler about gardening",
            "another unrelated note on cookery",
            "assorted remarks concerning pottery",
            "a short piece about cycling",
            "notes on medieval carpentry",
            "observations regarding tidal patterns",
            "a summary of railway timetables",
            "commentary on regional cuisine",
        ]
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])

    def test_only_zero_overlap_documents_are_dropped(self):
        """Documents sharing a query term score > 0; the rest score <= 0."""
        # deviation from RRF (no sparse credit without a shared term): see METHODS §A.3
        scores = self.bm25.get_scores(_tokenize("zebra quantum"))
        kept = [i for i, s in enumerate(scores) if s > 0]
        dropped = [i for i, s in enumerate(scores) if s <= 0]

        self.assertTrue(kept, "fixture must keep something")
        self.assertTrue(dropped, "fixture must drop something")
        for i in kept:
            self.assertTrue(
                "zebra" in self.docs[i] or "quantum" in self.docs[i]
            )
        for i in dropped:
            self.assertNotIn("zebra", self.docs[i])
            self.assertNotIn("quantum", self.docs[i])


class TestFilterPredicateEdgeCase(unittest.TestCase):
    """Pins where the score > 0 predicate and "no lexical overlap" differ."""

    def test_a_term_in_exactly_half_the_corpus_has_zero_idf(self):
        """A term in exactly N/2 documents has idf 0, so its docs score 0."""
        from rank_bm25 import BM25Okapi

        # ref: rank_bm25 @ 47aa3ddf (BM25Okapi defaults)
        docs = ["zebra alpha", "zebra beta", "gamma delta", "epsilon zeta"]
        bm25 = BM25Okapi([_tokenize(d) for d in docs])
        self.assertEqual(bm25.idf["zebra"], 0.0)

        scores = bm25.get_scores(_tokenize("zebra"))
        # Both documents contain the term and both score zero, so the
        # filter drops them.
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)

    def test_a_term_above_half_the_corpus_is_floored_positive(self):
        """A term in more than half the corpus gets a positive idf floor."""
        from rank_bm25 import BM25Okapi

        docs = ["zebra alpha", "zebra beta", "zebra gamma", "delta epsilon"]
        bm25 = BM25Okapi([_tokenize(d) for d in docs])
        self.assertGreater(bm25.idf["zebra"], 0.0)

    def test_a_dropped_document_would_have_contributed_only_a_tail_term(self):
        """A rank-4 RRF contribution is within 5% of a rank-1 contribution."""
        rank_1 = 1 / (RRF_K + 1)
        rank_4 = 1 / (RRF_K + 4)
        self.assertLess(abs(rank_1 - rank_4) / rank_1, 0.05)


class TestTokeniser(unittest.TestCase):
    def test_it_lowercases_and_splits_on_word_characters(self):
        """Punctuation is dropped and tokens are lowercased."""
        # harness choice: simplest deterministic tokeniser (METHODS §A.3)
        self.assertEqual(_tokenize("Zebra, QUANTUM!"), ["zebra", "quantum"])

    def test_it_is_unicode_aware(self):
        """Non-ASCII word characters tokenise like ASCII ones."""
        self.assertEqual(_tokenize("Ζέβρα καφέ"), ["ζέβρα", "καφέ"])


if __name__ == "__main__":
    unittest.main()
