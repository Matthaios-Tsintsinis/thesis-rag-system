"""Rank-aware retrieval metric tests (no GPU/API/network).

Guards the document-level MAP@K / Hit@K / MRR computation against the
chunk-multiplicity bug that pushed MAP above 1.0 on MultiHop (one gold
article contributing many "relevant" chunk positions while the AP
denominator normalised by distinct gold atoms).

Run from the repo root:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.eval.alignment import score_retrieval_rank_aware


def _chunks(*atom_lists):
    """Build fake RetrievedChunks from per-chunk atom lists.

    Each atom is a (parent, span) tuple; the function reads only
    r.chunk.gold_provenance, so SimpleNamespace stand-ins suffice.
    """
    out = []
    for rank, atoms in enumerate(atom_lists):
        out.append(
            SimpleNamespace(
                chunk=SimpleNamespace(gold_provenance=tuple(atoms)),
                score=1.0,
                rank=rank,
            )
        )
    return out


def _a(name: str) -> tuple[str, str]:
    return (name, "<whole>")


class TestMapBugRepro(unittest.TestCase):
    def test_chunk_multiplicity_no_longer_exceeds_one(self) -> None:
        # The exact failure shape: gold = {A, B}, top-10 chunks all from
        # the two gold articles (5 each). Old chunk-level code computed
        # AP@10 = sum(i/i for 10 positions) / min(10, 2) = 10/2 = 5.0.
        gold = frozenset({_a("A"), _a("B")})
        retrieved = _chunks(
            *([[_a("A")]] * 5 + [[_a("B")]] * 5)
        )
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(1, 5, 10))
        for k, v in out["map_at_k"].items():
            self.assertGreaterEqual(v, 0.0, k)
            self.assertLessEqual(v, 1.0, k)
        # Deduped doc ranking is [A, B] (both relevant): AP = 1.0.
        self.assertAlmostEqual(out["map_at_k"][10], 1.0)
        self.assertEqual(out["n_docs_ranked"], 2)
        self.assertEqual(out["n_relevant_retrieved"], 2)


class TestTextbookAP(unittest.TestCase):
    def test_known_average_precision(self) -> None:
        # Document ranking R N R N R, gold = {d0, d2, d4} (3 relevant).
        gold = frozenset({_a("d0"), _a("d2"), _a("d4")})
        retrieved = _chunks(
            [_a("d0")], [_a("d1")], [_a("d2")], [_a("d3")], [_a("d4")]
        )
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(3, 5))
        # AP@5 = (1/3)(1/1 + 2/3 + 3/5) = 2.26667/3 = 0.75556
        self.assertAlmostEqual(out["map_at_k"][5], (1 + 2 / 3 + 3 / 5) / 3, places=6)
        # AP@3 over [R,N,R]: (1/1 + 2/3) / min(3,3) = 1.66667/3 = 0.55556
        self.assertAlmostEqual(out["map_at_k"][3], (1 + 2 / 3) / 3, places=6)
        self.assertEqual(out["hit_at_k"][3], 1.0)
        self.assertAlmostEqual(out["mrr"], 1.0)  # first relevant at doc-rank 1

    def test_perfect_ranking_is_one(self) -> None:
        gold = frozenset({_a("d0"), _a("d1"), _a("d2")})
        retrieved = _chunks([_a("d0")], [_a("d1")], [_a("d2")], [_a("x")])
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(3, 10))
        self.assertAlmostEqual(out["map_at_k"][3], 1.0)
        self.assertAlmostEqual(out["map_at_k"][10], 1.0)


class TestMrrAndHit(unittest.TestCase):
    def test_mrr_document_rank_first_relevant(self) -> None:
        # Two non-gold chunks of the same article X, then gold Y. The
        # deduped doc ranking is [X, Y] -> first relevant at doc-rank 2.
        gold = frozenset({_a("Y")})
        retrieved = _chunks([_a("X")], [_a("X")], [_a("Y")])
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(1, 5))
        self.assertAlmostEqual(out["mrr"], 0.5)
        self.assertEqual(out["hit_at_k"][1], 0.0)  # only X in top-1 doc
        self.assertEqual(out["hit_at_k"][5], 1.0)

    def test_no_relevant_all_zero(self) -> None:
        gold = frozenset({_a("Z")})
        retrieved = _chunks([_a("A")], [_a("B")], [_a("C")])
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(1, 5, 10))
        self.assertEqual(out["mrr"], 0.0)
        self.assertEqual(out["n_relevant_retrieved"], 0)
        for v in out["map_at_k"].values():
            self.assertEqual(v, 0.0)
        for v in out["hit_at_k"].values():
            self.assertEqual(v, 0.0)


class TestEdgeCases(unittest.TestCase):
    def test_empty_gold_skipped(self) -> None:
        out = score_retrieval_rank_aware(_chunks([_a("A")]), frozenset())
        self.assertEqual(out, {"skipped": True})

    def test_empty_retrieval(self) -> None:
        out = score_retrieval_rank_aware([], frozenset({_a("A")}), k_values=(1, 10))
        self.assertEqual(out["mrr"], 0.0)
        self.assertEqual(out["n_docs_ranked"], 0)
        self.assertEqual(out["map_at_k"][10], 0.0)

    def test_dedup_first_occurrence(self) -> None:
        # A appears at chunk 0 and 2; its document rank is 1 (first).
        gold = frozenset({_a("A")})
        retrieved = _chunks([_a("A")], [_a("B")], [_a("A")])
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(1,))
        self.assertEqual(out["n_docs_ranked"], 2)  # A, B
        self.assertAlmostEqual(out["mrr"], 1.0)


class TestMapBoundedProperty(unittest.TestCase):
    def test_map_in_unit_interval_randomized(self) -> None:
        # Deterministic LCG (no Math.random / external seed needed) over
        # many random rankings with heavy chunk multiplicity: MAP@K must
        # never escape [0, 1], the invariant the bug violated.
        state = 12345

        def rnd() -> int:
            nonlocal state
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            return state

        articles = [f"art{i}" for i in range(6)]
        for _ in range(500):
            n_gold = 1 + rnd() % 5
            gold = frozenset(_a(articles[i]) for i in range(n_gold))
            n_chunks = 1 + rnd() % 25
            atom_lists = [[_a(articles[rnd() % len(articles)])] for _ in range(n_chunks)]
            out = score_retrieval_rank_aware(_chunks(*atom_lists), gold, k_values=(1, 5, 10))
            for k, v in out["map_at_k"].items():
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0, f"MAP@{k}={v} escaped [0,1]")
            self.assertGreaterEqual(out["mrr"], 0.0)
            self.assertLessEqual(out["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
