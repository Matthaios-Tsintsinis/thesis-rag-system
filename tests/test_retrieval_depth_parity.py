"""Rank-aware metrics read a depth-50 scoring ranking for every system.

The reader context stays top-15 (or M4's budget fill); only scoring
reads the deeper ranking.
"""

from __future__ import annotations

import unittest

from src.chunking import Chunk
from src.config import SCORING_RANKING_DEPTH
from src.eval.alignment import score_retrieval_rank_aware
from src.eval.multihop import RANK_K_VALUES, MultiHopBenchmark
from src.eval.types import (
    ANSWER_TYPE_FREE_FORM,
    EvalQuery,
    GoldAnswer,
)
from src.retrievers.base import RetrievedChunk


def _chunk(doc: str, rank: int) -> RetrievedChunk:
    """Make a retrieved chunk whose provenance is the whole document."""
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"c{rank}", doc_id=doc, text="t", n_words=1,
                    position=rank, gold_provenance=((doc, "<whole>"),)),
        score=1.0 - rank * 0.001, rank=rank)


def _query(gold_docs: tuple[str, ...]) -> EvalQuery:
    """Make a query whose gold passages are the given whole documents."""
    return EvalQuery(
        query_id="q1", question_text="?", parent_scope=None,
        gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                 free_form="x"),),
        gold_passage_sets=(frozenset((d, "<whole>") for d in gold_docs),),
        question_type="inference_query", metadata={})


class TestDepthParity(unittest.TestCase):
    """Two systems with opposite chunk distributions score at one depth."""

    def _concentrated(self, depth: int) -> list[RetrievedChunk]:
        """Build a ranking whose 15 reader chunks cover only 4 documents."""
        reader = [_chunk(f"doc{i % 4}", i) for i in range(15)]
        tail = [_chunk(f"docC{i}", 15 + i) for i in range(depth)]
        return (reader + tail)[:depth]

    def _spread(self, depth: int) -> list[RetrievedChunk]:
        """Build a ranking with one document per chunk."""
        return [_chunk(f"docS{i}", i) for i in range(depth)]

    def test_reader_contexts_really_do_differ_in_document_count(self):
        """The fixtures surface 4 and 15 documents in the reader context."""
        conc_docs = {r.chunk.doc_id for r in self._concentrated(50)[:15]}
        spread_docs = {r.chunk.doc_id for r in self._spread(50)[:15]}
        self.assertEqual(len(conc_docs), 4)
        self.assertEqual(len(spread_docs), 15)

    def test_every_k_in_the_grid_is_backed_by_real_candidates_for_both(self):
        """Every K in the grid is backed by K ranked documents, each system."""
        gold = _query(("docS0",))
        a = score_retrieval_rank_aware(
            self._concentrated(SCORING_RANKING_DEPTH),
            gold.gold_passage_sets[0], k_values=RANK_K_VALUES)
        b = score_retrieval_rank_aware(
            self._spread(SCORING_RANKING_DEPTH),
            gold.gold_passage_sets[0], k_values=RANK_K_VALUES)
        for res, label in ((a, "concentrated"), (b, "spread")):
            self.assertGreaterEqual(
                res["n_docs_ranked"], max(RANK_K_VALUES),
                f"{label}: K={max(RANK_K_VALUES)} is not backed by "
                f"{max(RANK_K_VALUES)} candidate documents",
            )

    def test_without_the_fix_the_depths_would_have_differed(self):
        """Scoring the reader context alone gives unequal document depths."""
        gold = _query(("docS0",))
        a = score_retrieval_rank_aware(
            self._concentrated(SCORING_RANKING_DEPTH)[:15],
            gold.gold_passage_sets[0], k_values=RANK_K_VALUES)
        b = score_retrieval_rank_aware(
            self._spread(SCORING_RANKING_DEPTH)[:15],
            gold.gold_passage_sets[0], k_values=RANK_K_VALUES)
        self.assertEqual(a["n_docs_ranked"], 4)
        self.assertEqual(b["n_docs_ranked"], 15)
        # Only the spread system reaches the largest K from the reader context.
        self.assertLess(a["n_docs_ranked"], max(RANK_K_VALUES))
        self.assertGreaterEqual(b["n_docs_ranked"], max(RANK_K_VALUES))


class TestTheBenchmarkUsesTheScoringRanking(unittest.TestCase):
    """MultiHopBenchmark scores rank-aware metrics over the scoring ranking."""

    def test_rank_aware_reads_the_scoring_ranking_not_the_reader_context(self):
        """Hit@K reads the scoring ranking, not the reader context."""
        bench = MultiHopBenchmark()
        query = _query(("docDEEP",))
        reader = [_chunk("docA", 0)]
        deep = [_chunk("docA", 0), _chunk("docDEEP", 1)]

        without = bench.score_retrieval(reader, query)
        with_deep = bench.score_retrieval(reader, query, scoring_ranking=deep)

        self.assertEqual(without.hit_at_k[10], 0.0)
        self.assertEqual(with_deep.hit_at_k[10], 1.0)

    def test_set_level_stays_over_the_reader_context(self):
        """Set-level P/R/F1 stay over the reader context at every depth."""
        bench = MultiHopBenchmark()
        query = _query(("docDEEP",))
        reader = [_chunk("docA", 0)]
        deep = [_chunk("docA", 0), _chunk("docDEEP", 1)]

        without = bench.score_retrieval(reader, query)
        with_deep = bench.score_retrieval(reader, query, scoring_ranking=deep)

        self.assertEqual(without.f1, with_deep.f1)
        self.assertEqual(without.recall, with_deep.recall)
        self.assertEqual(with_deep.recall, 0.0)

    def test_absent_scoring_ranking_falls_back_to_the_reader_context(self):
        """Without a scoring ranking, Hit@K reads the reader context."""
        bench = MultiHopBenchmark()
        query = _query(("docA",))
        reader = [_chunk("docA", 0)]
        self.assertEqual(bench.score_retrieval(reader, query).hit_at_k[1], 1.0)


class TestTheKGrid(unittest.TestCase):
    """The K grids carry the official cut-offs."""

    def test_multihop_carries_the_papers_hits_at_4(self):
        """MultiHop's grid includes 4 and 10."""
        # official: retrieval_evaluate.py @ cde8e844 (Hits@4, Hits@10); K = 1, 5 are ours
        self.assertIn(4, RANK_K_VALUES)
        self.assertIn(10, RANK_K_VALUES)

    def test_hotpotqa_keeps_hit_at_2_for_its_two_gold_titles(self):
        """HotpotQA's grid includes 2, one per gold title."""
        # harness choice: two gold titles, Hit@2 is the headline (METHODS §B.3)
        from src.eval.hotpotqa import RANK_K_VALUES as HOTPOT_K

        self.assertIn(2, HOTPOT_K)

    def test_hit_at_k_is_monotone_in_k_over_the_deeper_ranking(self):
        """Hit@K never decreases as K grows."""
        gold = _query(("docS7",))
        ranking = [_chunk(f"docS{i}", i) for i in range(20)]
        res = score_retrieval_rank_aware(
            ranking, gold.gold_passage_sets[0], k_values=(1, 4, 5, 10))
        hits = [res["hit_at_k"][k] for k in (1, 4, 5, 10)]
        self.assertEqual(hits, sorted(hits))


class TestSystemsExposeAScoringRanking(unittest.TestCase):
    """Every system asks for the configured scoring depth."""

    def test_the_base_default_asks_for_the_configured_depth(self):
        """retrieve_for_scoring defaults its depth to SCORING_RANKING_DEPTH."""
        import inspect

        from src.retrievers.base import BaseSystem

        self.assertEqual(
            inspect.signature(BaseSystem.retrieve_for_scoring)
            .parameters["depth"].default,
            SCORING_RANKING_DEPTH,
        )



if __name__ == "__main__":
    unittest.main()
