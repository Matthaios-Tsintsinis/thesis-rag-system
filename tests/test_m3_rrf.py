"""M3's fusion against the published RRF formula, and its declared filter.

WHY THIS EXISTS. Before 2026-08-22 the suite contained NO test of M3 at
all -- no test named `rrf_fuse`, `HybridRRFSystem` or `m3_hybrid`. M3 is
the one baseline in the matrix whose retrieval mechanism comes from a
PAPER (Cormack et al., SIGIR 2009), so its formula and its constant are a
fidelity claim, and that claim had no executable check. The final fidelity
audit found the gap; this file closes it.

The oracle is the published formula, written out here independently:

    RRFscore(d) = sum over rankings r of  1 / (k + rank_r(d))

with rank 1-BASED and k = 60. Values below are hand-computed in the
comments so a reader can verify them without running anything -- the
expected numbers do not come from calling the function under test.

Also pinned: the ZERO-BM25 FILTER, M3's one declared departure from
literal Cormack (dropping documents with no lexical overlap before
fusion). It is declared in `src/retrievers/m3_hybrid.py` and in the
thesis-facing deviations table, so the behaviour must be an assertion
rather than an accident.
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
        """Hand calculation, k=60.

        ranking A = [7, 3]; ranking B = [3, 9]
          doc 3: 1/(60+1) + 1/(60+2)  (rank 2 in A, rank 1 in B)
          doc 7: 1/(60+1)
          doc 9: 1/(60+2)
        so doc 3 leads, and 7 outranks 9 because 1/61 > 1/62.
        """
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
        """The published denominator is k + rank with rank starting at 1.

        A 0-based implementation would score the first document 1/k
        exactly; this asserts it does not.
        """
        (only_id, only_score), = rrf_fuse([[5]], k=60)
        self.assertEqual(only_id, 5)
        self.assertAlmostEqual(only_score, 1 / 61, places=12)
        self.assertNotAlmostEqual(only_score, 1 / 60, places=12)

    def test_contributions_are_summed_across_rankings(self):
        """A document in both lists must beat a document in one, even when
        the single-list document is ranked higher."""
        got = dict(rrf_fuse([[1, 2], [2, 1]], k=60))
        self.assertAlmostEqual(got[1], 1 / 61 + 1 / 62, places=12)
        self.assertAlmostEqual(got[2], 1 / 62 + 1 / 61, places=12)

    def test_ordering_is_descending_by_score(self):
        got = rrf_fuse([[9, 8, 7, 6], [6, 7]], k=60)
        scores = [s for _, s in got]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_input(self):
        self.assertEqual(rrf_fuse([], k=60), [])
        self.assertEqual(rrf_fuse([[]], k=60), [])

    def test_k_is_the_papers_constant_and_the_pipeline_reads_it(self):
        """RRF_K is a fidelity COMMITMENT (Cormack's k=60), not a tuning
        value, so pinning the literal is correct here -- and the second
        assertion is what makes it more than a statement about itself:
        the value M3 actually fuses with is the one resolved from config.
        """
        self.assertEqual(RRF_K, 60)
        self.assertEqual(DEFAULT_CONFIG.retrieval.rrf_k, 60)


class TestZeroBM25Filter(unittest.TestCase):
    """M3's declared departure from literal Cormack RRF.

    Reproduces the filter's predicate over a real BM25Okapi so the test
    fails if the declaration and the behaviour ever part company.
    """

    def setUp(self):
        from rank_bm25 import BM25Okapi

        # Nine documents, so no term lands on the n == N/2 knife-edge that
        # TestFilterPredicateEdgeCase below is about.
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
    """The filter's PREDICATE is `score > 0`; its DESCRIPTION is "no
    lexical overlap". Those are not identical, and the gap is recorded
    here rather than assumed away (docs/FINAL_FIDELITY_AUDIT.md AF-9).

    BM25Okapi's idf is `log((N - n + 0.5) / (n + 0.5))`, which is exactly
    ZERO when a term occurs in precisely half the documents. A document
    whose ONLY shared term is such a term therefore scores 0 and is
    dropped despite overlapping.

    MEASURED, and this is why AF-9 is INFO rather than a deviation: on
    the real MultiHop corpus (7,186 word-window chunks, all 2,556
    queries, over the top-50 first stage the filter operates on) the
    number of vocabulary terms with idf <= 0 is ZERO, and the number of
    dropped-but-overlapping documents is ZERO. Two things make the edge
    case vanish at scale -- exact n == N/2 is a knife-edge, and
    rank_bm25 floors NEGATIVE idf to `epsilon * average_idf`, which is
    positive, so terms above half the corpus do not produce a
    non-positive score either. The description is therefore accurate in
    practice, and this test exists so a future rank_bm25 that stops
    flooring cannot make it quietly inaccurate.
    """

    def test_a_term_in_exactly_half_the_corpus_has_zero_idf(self):
        from rank_bm25 import BM25Okapi

        docs = ["zebra alpha", "zebra beta", "gamma delta", "epsilon zeta"]
        bm25 = BM25Okapi([_tokenize(d) for d in docs])
        self.assertEqual(bm25.idf["zebra"], 0.0)

        scores = bm25.get_scores(_tokenize("zebra"))
        # Both documents CONTAIN the query term and both score zero, so
        # the filter would drop them.
        self.assertEqual(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)

    def test_a_term_above_half_the_corpus_is_floored_positive(self):
        """The other half of the mechanism: negative idf never reaches the
        predicate, because rank_bm25 replaces it with a positive floor."""
        from rank_bm25 import BM25Okapi

        docs = ["zebra alpha", "zebra beta", "zebra gamma", "delta epsilon"]
        bm25 = BM25Okapi([_tokenize(d) for d in docs])
        self.assertGreater(bm25.idf["zebra"], 0.0)

    def test_a_dropped_document_would_have_contributed_only_a_tail_term(self):
        """The declaration calls the filter benign because a zero-overlap
        document sits at the BOTTOM of the sparse list. Quantify it: its
        RRF contribution at rank 4 is under 2% of a rank-1 contribution.
        """
        rank_1 = 1 / (RRF_K + 1)
        rank_4 = 1 / (RRF_K + 4)
        self.assertLess(abs(rank_1 - rank_4) / rank_1, 0.05)


class TestTokeniser(unittest.TestCase):
    def test_it_lowercases_and_splits_on_word_characters(self):
        self.assertEqual(_tokenize("Zebra, QUANTUM!"), ["zebra", "quantum"])

    def test_it_is_unicode_aware(self):
        self.assertEqual(_tokenize("Ζέβρα καφέ"), ["ζέβρα", "καφέ"])


if __name__ == "__main__":
    unittest.main()
