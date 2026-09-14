"""Tests for the torch-free parts of generate_batch: batch ordering,
token-budget batching, the input contract and the allocator setting."""

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
        """Items are ordered by ascending prompt length."""
        order, _ = deterministic_batch_order([400, 100, 300, 200])
        self.assertEqual(order, [1, 3, 2, 0])

    def test_ties_break_on_original_index(self):
        """Equal lengths keep their original index order."""
        order, _ = deterministic_batch_order([50, 50, 50, 50])
        self.assertEqual(order, [0, 1, 2, 3])

    def test_inverse_restores_input_order(self):
        """The inverse permutation maps the sorted order back to the input."""
        lengths = [9, 1, 7, 3, 3, 12]
        order, inverse = deterministic_batch_order(lengths)
        shuffled = [lengths[i] for i in order]
        restored = [shuffled[i] for i in inverse]
        self.assertEqual(restored, lengths)

    def test_pure_function_of_lengths(self):
        """The same lengths always give the same order."""
        a, _ = deterministic_batch_order([5, 2, 9, 2])
        b, _ = deterministic_batch_order([5, 2, 9, 2])
        self.assertEqual(a, b)

    def test_empty_and_single(self):
        """Empty and single-item inputs are handled."""
        self.assertEqual(deterministic_batch_order([]), ([], []))
        self.assertEqual(deterministic_batch_order([7]), ([0], [0]))


class TestRoutingAndContract(unittest.TestCase):
    def test_openai_ids_are_refused_before_any_load(self):
        """An OpenAI model id raises ValueError before anything loads."""
        cfg = GenerationConfig(model="gpt-4o-mini")
        with self.assertRaises(ValueError) as ctx:
            generate_batch(["sys"] * 3, ["q1", "q2", "q3"], cfg)
        self.assertIn("gpt-4o-mini", str(ctx.exception))

    def test_empty_input_returns_empty(self):
        """No prompts means no outputs and no model load."""
        self.assertEqual(generate_batch([], [], GenerationConfig()), [])

    def test_length_mismatch_raises(self):
        """System and user prompt lists must be the same length."""
        with self.assertRaises(ValueError):
            generate_batch(["a", "b"], ["only one"], GenerationConfig())

    def test_bad_batch_size_raises(self):
        """A non-positive batch size is rejected."""
        with self.assertRaises(ValueError):
            generate_batch(
                ["s"], ["u"], GenerationConfig(model="gpt-4o-mini"), batch_size=0
            )



class TestTokenBudgetBatches(unittest.TestCase):
    """A batch pads to its longest member, so n * longest bounds its cost."""

    def test_narrow_batches_for_long_prompts_wide_for_short(self):
        """Short prompts pack wide; long prompts pack narrow."""
        order, lengths = [0, 1, 2, 3, 4, 5], [100, 100, 100, 4000, 4000, 4000]
        groups = token_budget_batches(
            order, lengths, max_padded_tokens=8000, max_batch_size=8
        )
        self.assertEqual(groups, [[0, 1, 2], [3, 4], [5]])

    def test_never_exceeds_the_padded_budget(self):
        """No multi-item batch exceeds the padded token budget."""
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
        """Batching is a partition of the input indices."""
        lengths = [i * 37 % 900 + 50 for i in range(97)]
        order, _ = deterministic_batch_order(lengths)
        groups = token_budget_batches(
            order, lengths, max_padded_tokens=5000
        )
        flat = [i for g in groups for i in g]
        self.assertEqual(sorted(flat), sorted(range(97)))

    def test_oversize_item_gets_its_own_batch_not_dropped(self):
        """An over-budget item gets its own batch instead of being dropped."""
        groups = token_budget_batches([0], [99_999], max_padded_tokens=1000)
        self.assertEqual(groups, [[0]])

    def test_count_cap_still_applies(self):
        """max_batch_size caps a batch even under a huge token budget."""
        lengths = [10] * 50
        order, _ = deterministic_batch_order(lengths)
        groups = token_budget_batches(
            order, lengths, max_padded_tokens=10_000_000, max_batch_size=8
        )
        self.assertTrue(all(len(g) <= 8 for g in groups))

    def test_deterministic(self):
        """The same inputs always give the same batches."""
        lengths = [i * 13 % 700 + 40 for i in range(60)]
        order, _ = deterministic_batch_order(lengths)
        a = token_budget_batches(order, lengths, max_padded_tokens=6000)
        b = token_budget_batches(order, lengths, max_padded_tokens=6000)
        self.assertEqual(a, b)

    def test_rejects_nonpositive_budget(self):
        """A non-positive token budget is rejected."""
        with self.assertRaises(ValueError):
            token_budget_batches([0], [10], max_padded_tokens=0)


class TestAllocatorConfig(unittest.TestCase):
    def test_sets_expandable_segments_when_unset(self):
        """With no allocator setting in the env, expandable segments is set."""
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
        """An allocator setting already in the env is left alone."""
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

class TestGeneratedTokensAreReserved(unittest.TestCase):
    """reserve_tokens_per_seq charges the tokens still to decode to the budget."""

    def test_short_prompts_long_outputs_are_bounded(self):
        """Reserving output tokens narrows batches of short prompts."""
        order, lengths = list(range(400)), [45] * 400
        loose = token_budget_batches(order, lengths, max_padded_tokens=20000)
        tight = token_budget_batches(
            order, lengths, max_padded_tokens=20000,
            reserve_tokens_per_seq=512,
        )
        self.assertEqual(max(len(b) for b in loose), 400)
        self.assertLess(max(len(b) for b in tight), 50)

    def test_default_is_zero_so_the_summary_path_is_unchanged(self):
        """The reserve defaults to 0, so summary batches keep their shape."""
        order, lengths = list(range(20)), [100] * 20
        self.assertEqual(
            token_budget_batches(order, lengths, max_padded_tokens=1000),
            token_budget_batches(order, lengths, max_padded_tokens=1000,
                                 reserve_tokens_per_seq=0),
        )

    def test_every_item_survives_reserving(self):
        """Reserving never drops an item."""
        order, lengths = list(range(37)), [10 + i for i in range(37)]
        batches = token_budget_batches(
            order, lengths, max_padded_tokens=500, reserve_tokens_per_seq=100
        )
        self.assertEqual(sorted(i for b in batches for i in b), sorted(order))

    def test_a_negative_reserve_is_rejected(self):
        """A negative reserve is rejected."""
        with self.assertRaises(ValueError):
            token_budget_batches([0], [10], max_padded_tokens=100,
                                 reserve_tokens_per_seq=-1)


if __name__ == "__main__":
    unittest.main()
