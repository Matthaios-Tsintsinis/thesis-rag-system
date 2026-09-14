"""Tests for the document-level MAP@K / Hit@K / MRR scorer.

Runs without a GPU, an API key or the network.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.eval.alignment import score_retrieval_rank_aware


def _chunks(*atom_lists):
    """Build stand-in RetrievedChunks, one per list of (parent, span) atoms."""
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
        """Repeat chunks of a gold article count once; MAP stays in [0, 1]."""
        # harness choice: collapse to document ranking by first occurrence (METHODS §C.5)
        # gold = {A, B}; ten chunks, five from each gold article.
        gold = frozenset({_a("A"), _a("B")})
        retrieved = _chunks(
            *([[_a("A")]] * 5 + [[_a("B")]] * 5)
        )
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(1, 5, 10))
        for k, v in out["map_at_k"].items():
            self.assertGreaterEqual(v, 0.0, k)
            self.assertLessEqual(v, 1.0, k)
        # Document ranking is [A, B], both relevant: AP = 1.0.
        self.assertAlmostEqual(out["map_at_k"][10], 1.0)
        self.assertEqual(out["n_docs_ranked"], 2)
        self.assertEqual(out["n_relevant_retrieved"], 2)


class TestTextbookAP(unittest.TestCase):
    def test_known_average_precision(self) -> None:
        """AP@K over R N R N R matches the hand-computed value."""
        # official: retrieval_evaluate.py @ cde8e844 (denominator only)
        # Document ranking R N R N R, gold = {d0, d2, d4}.
        gold = frozenset({_a("d0"), _a("d2"), _a("d4")})
        retrieved = _chunks(
            [_a("d0")], [_a("d1")], [_a("d2")], [_a("d3")], [_a("d4")]
        )
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(3, 5))
        # AP@5 = (1/1 + 2/3 + 3/5) / min(5, 3)
        self.assertAlmostEqual(out["map_at_k"][5], (1 + 2 / 3 + 3 / 5) / 3, places=6)
        # AP@3 over [R, N, R] = (1/1 + 2/3) / min(3, 3)
        self.assertAlmostEqual(out["map_at_k"][3], (1 + 2 / 3) / 3, places=6)
        self.assertEqual(out["hit_at_k"][3], 1.0)
        self.assertAlmostEqual(out["mrr"], 1.0)  # first relevant at doc-rank 1

    def test_perfect_ranking_is_one(self) -> None:
        """All gold documents ranked first gives MAP 1.0 at every K."""
        gold = frozenset({_a("d0"), _a("d1"), _a("d2")})
        retrieved = _chunks([_a("d0")], [_a("d1")], [_a("d2")], [_a("x")])
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(3, 10))
        self.assertAlmostEqual(out["map_at_k"][3], 1.0)
        self.assertAlmostEqual(out["map_at_k"][10], 1.0)


class TestMrrAndHit(unittest.TestCase):
    def test_mrr_document_rank_first_relevant(self) -> None:
        """MRR uses the document rank of the first hit, not the chunk rank."""
        # Two chunks of non-gold X, then gold Y: document ranking [X, Y].
        gold = frozenset({_a("Y")})
        retrieved = _chunks([_a("X")], [_a("X")], [_a("Y")])
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(1, 5))
        self.assertAlmostEqual(out["mrr"], 0.5)
        self.assertEqual(out["hit_at_k"][1], 0.0)  # only X in top-1 doc
        self.assertEqual(out["hit_at_k"][5], 1.0)

    def test_no_relevant_all_zero(self) -> None:
        """No gold retrieved gives zero MRR, MAP and Hit at every K."""
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
        """Empty gold returns the skipped marker instead of a score."""
        out = score_retrieval_rank_aware(_chunks([_a("A")]), frozenset())
        self.assertEqual(out, {"skipped": True})

    def test_empty_retrieval(self) -> None:
        """Empty retrieval scores zero with no documents ranked."""
        out = score_retrieval_rank_aware([], frozenset({_a("A")}), k_values=(1, 10))
        self.assertEqual(out["mrr"], 0.0)
        self.assertEqual(out["n_docs_ranked"], 0)
        self.assertEqual(out["map_at_k"][10], 0.0)

    def test_dedup_first_occurrence(self) -> None:
        """A repeated document keeps its first-occurrence rank."""
        # A appears at chunks 0 and 2; its document rank is 1.
        gold = frozenset({_a("A")})
        retrieved = _chunks([_a("A")], [_a("B")], [_a("A")])
        out = score_retrieval_rank_aware(retrieved, gold, k_values=(1,))
        self.assertEqual(out["n_docs_ranked"], 2)  # A, B
        self.assertAlmostEqual(out["mrr"], 1.0)


class TestMapBoundedProperty(unittest.TestCase):
    def test_map_in_unit_interval_randomized(self) -> None:
        """MAP@K and MRR stay in [0, 1] over random rankings with repeats."""
        # A small LCG keeps the rankings reproducible without a seed.
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
