"""Tests for scripts/replay_retrieval.py: the document collapse agrees
with the frozen scorer, recall@K stays within bounds, and the row gate
compares every field."""

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
    # the same document at two ranks is credited once
    ([("a", "<w>"), ("b", "<w>"), ("a", "<w>"), ("c", "<w>")],
     {("b", "<w>"), ("c", "<w>")}),
    ([("x", "<w>"), ("y", "<w>"), ("z", "<w>")], {("q", "<w>")}),
    ([("a", "<w>"), ("b", "<w>"), ("c", "<w>"), ("d", "<w>"),
      ("e", "<w>"), ("f", "<w>")], {("f", "<w>"), ("a", "<w>")}),
]


class TestCollapseOracle(unittest.TestCase):
    def test_mirror_agrees_with_frozen_scorer(self):
        """Collapse + rank stats give the same hit@K and MRR as the scorer."""
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
        """recall@5 never exceeds hit@5 and is at least 1/n_gold on a hit."""
        for stamps, gold in FIXTURES:
            retrieved = [_rc(i, [a]) for i, a in enumerate(stamps)]
            ranking = collapse_to_doc_ranking(retrieved)
            st = rank_stats_from_ranking(ranking, frozenset(gold), (5,))
            self.assertLessEqual(st["recall_at_k"][5], st["hit_at_k"][5])
            if len(gold) == 2 and st["hit_at_k"][5] == 1.0:
                self.assertGreaterEqual(st["recall_at_k"][5], 0.5)

    def test_summary_node_never_credited_and_dup_gold_counted_once(self):
        """A summary node earns no credit; a repeated gold doc counts once."""
        # rank 0 is a summary node with no provenance, as M4 returns
        # them; ranks 1 and 3 are the same gold doc; rank 2 is non-gold
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
        # the duplicated gold doc gives recall@5 = 1/2, not 2/2
        self.assertEqual(mine["recall_at_k"][5], 0.5)

    def test_recall_values(self):
        """recall@K counts the gold docs found within the top K."""
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
    """Build a RetrievalScore whose fields default to the banked fixture."""
    base = dict(skipped=False, f1=0.5, recall=0.5, precision=0.5,
                mrr=1.0, hit_at_k={1: 1.0}, map_at_k={1: 1.0})
    base.update(kw)
    return RetrievalScore(**base)


class TestRowGate(unittest.TestCase):
    BANKED = {"skipped": False, "f1": 0.5, "recall": 0.5, "precision": 0.5,
              "mrr": 1.0, "hit_at_k": {"1": 1.0}, "map_at_k": {"1": 1.0}}

    def test_reproducing_row_passes(self):
        """A replayed row equal to the banked row raises no mismatch."""
        self.assertEqual(compare_row(dict(self.BANKED), _Replayed()), [])

    def test_each_field_is_compared(self):
        """Changing any one gated field is reported by name."""
        for field, value in (("f1", 0.4), ("recall", 0.4),
                             ("precision", 0.4), ("mrr", 0.9)):
            bad = compare_row(dict(self.BANKED), _Replayed(**{field: value}))
            self.assertTrue(bad and field in bad[0], (field, bad))
        bad = compare_row(dict(self.BANKED),
                          _Replayed(hit_at_k={1: 0.0}))
        self.assertTrue(bad and "hit_at_k" in bad[0])

    def test_string_vs_int_k_keys_are_normalised(self):
        """String K keys from JSON and int K keys compare equal."""
        self.assertEqual(
            compare_row(dict(self.BANKED),
                        _Replayed(hit_at_k={"1": 1.0}, map_at_k={1: 1.0})),
            [])

    def test_skipped_rows_match_as_skipped(self):
        """A skipped banked row matches only a skipped replayed row."""
        self.assertEqual(compare_row({"skipped": True},
                                     _Replayed(skipped=True)), [])
        self.assertTrue(compare_row({"skipped": True}, _Replayed()))


if __name__ == "__main__":
    unittest.main()
