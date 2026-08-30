"""replay_retrieval: the collapse oracle, the row gate, the bounds.

The collapse mirror is kept honest by an ORACLE: the same fixtures go
through the frozen `score_retrieval_rank_aware` and through the
mirror + `rank_stats_from_ranking`, and hit@K / MRR must agree — the
copy cannot drift without this failing. The gate and bounds are pure
functions, tested directly.
"""

from __future__ import annotations

import unittest

from src.chunking import Chunk
from src.eval.alignment import score_retrieval_rank_aware
from src.retrievers.base import RetrievedChunk
from scripts.replay_retrieval import (
    collapse_to_doc_ranking,
    compare_row,
    rank_stats_from_ranking,
)


def _rc(rank, atoms):
    c = Chunk(chunk_id=f"c{rank}", doc_id="d", text="t", n_words=1,
              position=rank, gold_provenance=tuple(atoms))
    return RetrievedChunk(chunk=c, score=1.0 / (rank + 1), rank=rank)


FIXTURES = [
    # (retrieved atom stamps in rank order, gold atoms)
    ([("a", "<w>")], {("a", "<w>")}),
    ([("a", "<w>"), ("b", "<w>"), ("a", "<w>"), ("c", "<w>")],
     {("b", "<w>"), ("c", "<w>")}),
    ([("x", "<w>"), ("y", "<w>"), ("z", "<w>")], {("q", "<w>")}),
    ([("a", "<w>"), ("b", "<w>"), ("c", "<w>"), ("d", "<w>"),
      ("e", "<w>"), ("f", "<w>")], {("f", "<w>"), ("a", "<w>")}),
]


class TestCollapseOracle(unittest.TestCase):
    def test_mirror_agrees_with_frozen_scorer(self):
        for stamps, gold in FIXTURES:
            retrieved = [_rc(i, [a]) for i, a in enumerate(stamps)]
            oracle = score_retrieval_rank_aware(
                retrieved, frozenset(gold), k_values=(1, 5, 10))
            ranking = collapse_to_doc_ranking(retrieved)
            mine = rank_stats_from_ranking(ranking, frozenset(gold),
                                           (1, 5, 10))
            self.assertEqual(mine["mrr"], oracle["mrr"], stamps)
            for k in (1, 5, 10):
                self.assertEqual(mine["hit_at_k"][k],
                                 oracle["hit_at_k"][k], (stamps, k))

    def test_recall_bounds_per_row(self):
        for stamps, gold in FIXTURES:
            retrieved = [_rc(i, [a]) for i, a in enumerate(stamps)]
            ranking = collapse_to_doc_ranking(retrieved)
            st = rank_stats_from_ranking(ranking, frozenset(gold), (5,))
            self.assertLessEqual(st["recall_at_k"][5], st["hit_at_k"][5])
            if len(gold) == 2 and st["hit_at_k"][5] == 1.0:
                self.assertGreaterEqual(st["recall_at_k"][5], 0.5)

    def test_recall_values(self):
        stamps = [("a", "<w>"), ("b", "<w>"), ("c", "<w>"), ("d", "<w>"),
                  ("e", "<w>"), ("f", "<w>")]
        gold = frozenset({("a", "<w>"), ("f", "<w>")})
        ranking = collapse_to_doc_ranking([_rc(i, [a])
                                           for i, a in enumerate(stamps)])
        st = rank_stats_from_ranking(ranking, gold, (1, 5, 10))
        self.assertEqual(st["recall_at_k"][1], 0.5)   # only "a" in top-1
        self.assertEqual(st["recall_at_k"][5], 0.5)   # "f" is rank 6
        self.assertEqual(st["recall_at_k"][10], 1.0)


class _Replayed:
    def __init__(self, **kw):
        self.skipped = kw.get("skipped", False)
        self.f1 = kw.get("f1", 0.5)
        self.recall = kw.get("recall", 0.5)
        self.precision = kw.get("precision", 0.5)
        self.mrr = kw.get("mrr", 1.0)
        self.hit_at_k = kw.get("hit_at_k", {1: 1.0})
        self.map_at_k = kw.get("map_at_k", {1: 1.0})


class TestRowGate(unittest.TestCase):
    BANKED = {"skipped": False, "f1": 0.5, "recall": 0.5, "precision": 0.5,
              "mrr": 1.0, "hit_at_k": {"1": 1.0}, "map_at_k": {"1": 1.0}}

    def test_reproducing_row_passes(self):
        self.assertEqual(compare_row(dict(self.BANKED), _Replayed()), [])

    def test_each_field_is_compared(self):
        for field, value in (("f1", 0.4), ("recall", 0.4),
                             ("precision", 0.4), ("mrr", 0.9)):
            bad = compare_row(dict(self.BANKED), _Replayed(**{field: value}))
            self.assertTrue(bad and field in bad[0], (field, bad))
        bad = compare_row(dict(self.BANKED),
                          _Replayed(hit_at_k={1: 0.0}))
        self.assertTrue(bad and "hit_at_k" in bad[0])

    def test_string_vs_int_k_keys_are_normalised(self):
        # banked dicts round-trip through JSON with string keys
        self.assertEqual(
            compare_row(dict(self.BANKED),
                        _Replayed(hit_at_k={"1": 1.0}, map_at_k={1: 1.0})),
            [])

    def test_skipped_rows_match_as_skipped(self):
        self.assertEqual(compare_row({"skipped": True},
                                     _Replayed(skipped=True)), [])
        self.assertTrue(compare_row({"skipped": True}, _Replayed()))


if __name__ == "__main__":
    unittest.main()
