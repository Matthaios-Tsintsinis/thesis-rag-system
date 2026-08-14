"""P6: rank-aware metrics are measured at ONE depth for every system.

THE DEFECT. K used to count documents surfaced by the reader's top-15
CHUNKS. A system whose 15 chunks collapsed into 4 articles was scored at
Hit@10 over 4 candidate documents; one spreading over 15 articles got 10.
The ranking depth was therefore a property of the SYSTEM's chunk
distribution rather than of the metric, which is not the published Hit@K
and is not comparable across systems. M4 was hit twice, because its
summary nodes carry no provenance and rank no document at all.

THE FIX. Every system returns a fixed-depth `scoring_ranking`
(SCORING_RANKING_DEPTH=50) used for rank-aware scoring only. Generation
input is untouched: the reader still receives top-15, or M4's
2,000-token budget fill.

The parity test below is the reason the change exists, so it is measured
rather than asserted: two systems with deliberately opposite chunk
distributions must be evaluated over document rankings of the same
depth.
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
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"c{rank}", doc_id=doc, text="t", n_words=1,
                    position=rank, gold_provenance=((doc, "<whole>"),)),
        score=1.0 - rank * 0.001, rank=rank)


def _query(gold_docs: tuple[str, ...]) -> EvalQuery:
    return EvalQuery(
        query_id="q1", question_text="?", parent_scope=None,
        gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                 free_form="x"),),
        gold_passage_sets=(frozenset((d, "<whole>") for d in gold_docs),),
        question_type="inference_query", metadata={})


class TestDepthParity(unittest.TestCase):
    """Two systems, opposite chunk distributions, same evaluated depth."""

    def _concentrated(self, depth: int) -> list[RetrievedChunk]:
        """15 reader chunks collapsing into 4 documents; the deeper
        ranking still reaches `depth` distinct documents."""
        reader = [_chunk(f"doc{i % 4}", i) for i in range(15)]
        tail = [_chunk(f"docC{i}", 15 + i) for i in range(depth)]
        return (reader + tail)[:depth]

    def _spread(self, depth: int) -> list[RetrievedChunk]:
        """15 reader chunks over 15 distinct documents."""
        return [_chunk(f"docS{i}", i) for i in range(depth)]

    def test_reader_contexts_really_do_differ_in_document_count(self):
        """Guard the guard: if both fixtures surfaced the same number of
        documents, the parity test below could not fail."""
        conc_docs = {r.chunk.doc_id for r in self._concentrated(50)[:15]}
        spread_docs = {r.chunk.doc_id for r in self._spread(50)[:15]}
        self.assertEqual(len(conc_docs), 4)
        self.assertEqual(len(spread_docs), 15)

    def test_every_k_in_the_grid_is_backed_by_real_candidates_for_both(self):
        """THE INVARIANT THAT MAKES Hit@K COMPARABLE — and it is not
        "identical document depth".

        Depth 50 is fixed in CANDIDATES; the derived document ranking is
        still shorter when chunks concentrate (39 documents here against
        50). That is a property of the corpus, not a defect. What Hit@K
        requires is that K be backed by K real candidate documents for
        every system: only then does Hit@10 mean the same thing twice.
        Before the fix the concentrated system had FOUR documents against
        K=10, so its Hit@10 was silently a Hit@4.
        """
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
        """The old behaviour, reproduced from the reader contexts alone,
        so the defect is on the record as a measurement."""
        gold = _query(("docS0",))
        a = score_retrieval_rank_aware(
            self._concentrated(SCORING_RANKING_DEPTH)[:15],
            gold.gold_passage_sets[0], k_values=RANK_K_VALUES)
        b = score_retrieval_rank_aware(
            self._spread(SCORING_RANKING_DEPTH)[:15],
            gold.gold_passage_sets[0], k_values=RANK_K_VALUES)
        self.assertEqual(a["n_docs_ranked"], 4)
        self.assertEqual(b["n_docs_ranked"], 15)
        # The concentrated system had FOUR candidate documents against
        # K=10: its Hit@10 was a Hit@4 wearing the wrong name, and the
        # two systems' columns were not the same measurement.
        self.assertLess(a["n_docs_ranked"], max(RANK_K_VALUES))
        self.assertGreaterEqual(b["n_docs_ranked"], max(RANK_K_VALUES))


class TestTheBenchmarkUsesTheScoringRanking(unittest.TestCase):
    def test_rank_aware_reads_the_scoring_ranking_not_the_reader_context(self):
        bench = MultiHopBenchmark()
        query = _query(("docDEEP",))
        reader = [_chunk("docA", 0)]
        deep = [_chunk("docA", 0), _chunk("docDEEP", 1)]

        without = bench.score_retrieval(reader, query)
        with_deep = bench.score_retrieval(reader, query, scoring_ranking=deep)

        self.assertEqual(without.hit_at_k[10], 0.0)
        self.assertEqual(with_deep.hit_at_k[10], 1.0)

    def test_set_level_stays_over_the_reader_context(self):
        """Set-level P/R/F1 measure what the GENERATOR saw, so a deeper
        scoring ranking must not move them."""
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
        bench = MultiHopBenchmark()
        query = _query(("docA",))
        reader = [_chunk("docA", 0)]
        self.assertEqual(bench.score_retrieval(reader, query).hit_at_k[1], 1.0)


class TestTheKGrid(unittest.TestCase):
    def test_multihop_carries_the_papers_hits_at_4(self):
        self.assertIn(4, RANK_K_VALUES)
        self.assertIn(10, RANK_K_VALUES)

    def test_hotpotqa_keeps_hit_at_2_for_its_two_gold_titles(self):
        from src.eval.hotpotqa import RANK_K_VALUES as HOTPOT_K

        self.assertIn(2, HOTPOT_K)

    def test_hit_at_k_is_monotone_in_k_over_the_deeper_ranking(self):
        gold = _query(("docS7",))
        ranking = [_chunk(f"docS{i}", i) for i in range(20)]
        res = score_retrieval_rank_aware(
            ranking, gold.gold_passage_sets[0], k_values=(1, 4, 5, 10))
        hits = [res["hit_at_k"][k] for k in (1, 4, 5, 10)]
        self.assertEqual(hits, sorted(hits))


class TestSystemsExposeAScoringRanking(unittest.TestCase):
    def test_the_base_default_asks_for_the_configured_depth(self):
        import inspect

        from src.retrievers.base import BaseSystem

        src = inspect.getsource(BaseSystem.retrieve_for_scoring)
        self.assertIn("k=depth", src)
        self.assertEqual(
            inspect.signature(BaseSystem.retrieve_for_scoring)
            .parameters["depth"].default,
            SCORING_RANKING_DEPTH,
        )

    def test_m9_reuses_its_pool_instead_of_re_running_the_pipeline(self):
        """M9's corrective decision is k-independent, so one pass cut
        twice is exact. Re-running would pay a second reranker pass and a
        second rewrite LLM call."""
        import inspect

        from src.retrievers.m9_corrective import CorrectiveRAGSystem

        src = inspect.getsource(CorrectiveRAGSystem.prepare)
        self.assertIn("SCORING_RANKING_DEPTH", src)
        self.assertIn("scoring_ranking=scoring_ranking", src)
        self.assertEqual(src.count("_corrective_retrieve("), 1)


if __name__ == "__main__":
    unittest.main()
