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


def _rc_multi(rank, atom_lists):
    c = Chunk(chunk_id=f"c{rank}", doc_id="d", text="t", n_words=1,
              position=rank, gold_provenance=tuple(atom_lists))
    return RetrievedChunk(chunk=c, score=1.0 / (rank + 1), rank=rank)


FIXTURES = [
    # (retrieved atom stamps in rank order, gold atoms)
    ([("a", "<w>")], {("a", "<w>")}),
    # duplicate document across ranks: dedup credits it ONCE
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

    def test_summary_node_never_credited_and_dup_gold_counted_once(self):
        # rank 0: a SUMMARY NODE (no provenance -- exactly how M4's
        # summary units appear); rank 1 + rank 3: the SAME gold doc
        # twice; rank 2: a non-gold doc. Oracle + mirror must agree, the
        # summary node must contribute nothing, and the duplicated gold
        # doc must be counted at most once (recall capped at 1/n_gold
        # per doc).
        retrieved = [
            _rc_multi(0, []),                     # summary node
            _rc(1, [("g", "<w>")]),
            _rc(2, [("x", "<w>")]),
            _rc(3, [("g", "<w>")]),               # duplicate of rank 1
        ]
        gold = frozenset({("g", "<w>"), ("h", "<w>")})
        oracle = score_retrieval_rank_aware(retrieved, gold,
                                            k_values=(1, 5, 10))
        ranking = collapse_to_doc_ranking(retrieved)
        self.assertEqual(ranking, [("g", "<w>"), ("x", "<w>")])
        mine = rank_stats_from_ranking(ranking, gold, (1, 5, 10))
        self.assertEqual(mine["mrr"], oracle["mrr"])
        for k in (1, 5, 10):
            self.assertEqual(mine["hit_at_k"][k], oracle["hit_at_k"][k])
        # dup gold counted once: recall@5 = 1/2, never 2/2
        self.assertEqual(mine["recall_at_k"][5], 0.5)

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


from src.eval.types import RetrievalScore


def _Replayed(**kw):
    """The REAL production dataclass (the GoldAnswer standard), with the
    row-gate fields defaulted to the banked fixture's values."""
    base = dict(skipped=False, f1=0.5, recall=0.5, precision=0.5,
                mrr=1.0, hit_at_k={1: 1.0}, map_at_k={1: 1.0})
    base.update(kw)
    return RetrievalScore(**base)


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
