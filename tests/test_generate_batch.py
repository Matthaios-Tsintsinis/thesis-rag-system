"""Tests for batched generation.

SCOPE, stated honestly: the torch-dependent path cannot be exercised on
a host without torch and a GPU, so what is covered here is the ordering
algebra, the routing, and the input contract. The left-padding /
pad-token / attention-mask / slice-offset behaviour is first exercised
for real in the measurement round on Colab — that is a known gap, not an
oversight, and it is why those four requirements are enforced inside
`generate_batch` rather than left to call sites.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.config import GenerationConfig
from src.models import deterministic_batch_order, generate_batch


class TestDeterministicBatchOrder(unittest.TestCase):
    def test_sorts_ascending_by_length(self):
        order, _ = deterministic_batch_order([400, 100, 300, 200])
        self.assertEqual(order, [1, 3, 2, 0])

    def test_ties_break_on_original_index(self):
        """Stability is the whole point — a tie must never resolve on
        dict order, score, or timing, or the run stops being
        reproducible."""
        order, _ = deterministic_batch_order([50, 50, 50, 50])
        self.assertEqual(order, [0, 1, 2, 3])

    def test_inverse_restores_input_order(self):
        lengths = [9, 1, 7, 3, 3, 12]
        order, inverse = deterministic_batch_order(lengths)
        shuffled = [lengths[i] for i in order]
        restored = [shuffled[i] for i in inverse]
        self.assertEqual(restored, lengths)

    def test_pure_function_of_lengths(self):
        a, _ = deterministic_batch_order([5, 2, 9, 2])
        b, _ = deterministic_batch_order([5, 2, 9, 2])
        self.assertEqual(a, b)

    def test_empty_and_single(self):
        self.assertEqual(deterministic_batch_order([]), ([], []))
        self.assertEqual(deterministic_batch_order([7]), ([0], [0]))


class TestRoutingAndContract(unittest.TestCase):
    def test_openai_ids_use_the_sequential_api_path(self):
        """Batching is a local-model concern; the API parallelises
        server-side. Also keeps this path torch-free."""
        cfg = GenerationConfig(model="gpt-4o-mini")
        with mock.patch(
            "src.models._generate_openai", side_effect=lambda s, u, c: f"A:{u}"
        ) as m:
            out = generate_batch(["sys"] * 3, ["q1", "q2", "q3"], cfg)
        self.assertEqual(out, ["A:q1", "A:q2", "A:q3"])
        self.assertEqual(m.call_count, 3)

    def test_results_align_to_input_order_not_sorted_order(self):
        cfg = GenerationConfig(model="gpt-4o-mini")
        with mock.patch(
            "src.models._generate_openai", side_effect=lambda s, u, c: u.upper()
        ):
            out = generate_batch(
                ["s"] * 3, ["short", "a much longer prompt here", "mid one"], cfg
            )
        self.assertEqual(out, ["SHORT", "A MUCH LONGER PROMPT HERE", "MID ONE"])

    def test_empty_input_returns_empty(self):
        self.assertEqual(generate_batch([], [], GenerationConfig()), [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            generate_batch(["a", "b"], ["only one"], GenerationConfig())

    def test_bad_batch_size_raises(self):
        with self.assertRaises(ValueError):
            generate_batch(
                ["s"], ["u"], GenerationConfig(model="gpt-4o-mini"), batch_size=0
            )


if __name__ == "__main__":
    unittest.main()
