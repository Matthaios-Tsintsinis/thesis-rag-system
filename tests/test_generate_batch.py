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
from src.models import (
    configure_cuda_allocator,
    deterministic_batch_order,
    generate_batch,
    token_budget_batches,
)


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



class TestTokenBudgetBatches(unittest.TestCase):
    """Bound n * longest instead of n.

    A fixed count must be sized for the worst-case batch, because a
    batch pads to its longest member. Measured: uniform 4k prompts
    survived batch 8 at 21.7GB; real ragged MultiHop prompts OOM'd at
    the same count.
    """

    def test_narrow_batches_for_long_prompts_wide_for_short(self):
        order, lengths = [0, 1, 2, 3, 4, 5], [100, 100, 100, 4000, 4000, 4000]
        groups = token_budget_batches(
            order, lengths, max_padded_tokens=8000, max_batch_size=8
        )
        self.assertEqual(groups, [[0, 1, 2], [3, 4], [5]])

    def test_never_exceeds_the_padded_budget(self):
        import random

        rng = random.Random(0)
        lengths = [rng.randint(200, 4200) for _ in range(200)]
        order, _ = deterministic_batch_order(lengths)
        groups = token_budget_batches(
            order, lengths, max_padded_tokens=20_000, max_batch_size=64
        )
        for g in groups:
            padded = len(g) * max(lengths[i] for i in g)
            if len(g) > 1:
                self.assertLessEqual(padded, 20_000)

    def test_every_item_appears_exactly_once(self):
        lengths = [i * 37 % 900 + 50 for i in range(97)]
        order, _ = deterministic_batch_order(lengths)
        groups = token_budget_batches(
            order, lengths, max_padded_tokens=5000
        )
        flat = [i for g in groups for i in g]
        self.assertEqual(sorted(flat), sorted(range(97)))

    def test_oversize_item_gets_its_own_batch_not_dropped(self):
        """Same include-it-anyway policy as the context packer: one
        over-budget item beats zero output."""
        groups = token_budget_batches([0], [99_999], max_padded_tokens=1000)
        self.assertEqual(groups, [[0]])

    def test_count_cap_still_applies(self):
        lengths = [10] * 50
        order, _ = deterministic_batch_order(lengths)
        groups = token_budget_batches(
            order, lengths, max_padded_tokens=10_000_000, max_batch_size=8
        )
        self.assertTrue(all(len(g) <= 8 for g in groups))

    def test_deterministic(self):
        lengths = [i * 13 % 700 + 40 for i in range(60)]
        order, _ = deterministic_batch_order(lengths)
        a = token_budget_batches(order, lengths, max_padded_tokens=6000)
        b = token_budget_batches(order, lengths, max_padded_tokens=6000)
        self.assertEqual(a, b)

    def test_rejects_nonpositive_budget(self):
        with self.assertRaises(ValueError):
            token_budget_batches([0], [10], max_padded_tokens=0)


class TestAllocatorConfig(unittest.TestCase):
    def test_sets_expandable_segments_when_unset(self):
        import os

        prev = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        try:
            self.assertEqual(
                configure_cuda_allocator(), "expandable_segments:True"
            )
        finally:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            if prev is not None:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = prev

    def test_respects_an_explicit_user_setting(self):
        import os

        prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        try:
            self.assertEqual(
                configure_cuda_allocator(), "max_split_size_mb:128"
            )
        finally:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            if prev is not None:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = prev

if __name__ == "__main__":
    unittest.main()
