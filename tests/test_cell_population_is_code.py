"""Pins that a NarrativeQA cell draws its seeded 40 stories by default,
and that every benchmark declares the unit count a cell resolves to.
"""

from __future__ import annotations

import unittest

from src.eval.narrativeqa import CELL_UNITS, select_units
from src.eval.sampling import subsample_indices

# dataset: deepmind/narrativeqa, validation (115 stories), full-story setting
FULL_SPLIT = [f"story{i:03d}" for i in range(115)]


class TestDefaultDrawsTheCell(unittest.TestCase):
    """select_units with no cap returns the seeded cell draw."""

    def test_no_max_units_yields_exactly_the_cell_size(self):
        """A None cap resolves to the cell size, not the whole split."""
        picked = select_units(FULL_SPLIT, None)
        self.assertEqual(len(picked), CELL_UNITS)

    def test_no_max_units_yields_the_SEEDED_draw_not_a_prefix(self):
        """A None cap returns the seeded draw, not the first 40 stories."""
        picked = select_units(FULL_SPLIT, None)
        expected = [FULL_SPLIT[i]
                    for i in subsample_indices(len(FULL_SPLIT), CELL_UNITS)]
        self.assertEqual(picked, expected)
        self.assertNotEqual(picked, FULL_SPLIT[:CELL_UNITS])

    def test_the_draw_is_identical_across_invocations(self):
        """The default draw is the same on every call."""
        self.assertEqual(select_units(FULL_SPLIT, None),
                         select_units(FULL_SPLIT, None))

    def test_an_explicit_full_split_is_still_reachable(self):
        """An explicit cap of 115 returns the whole split."""
        self.assertEqual(select_units(FULL_SPLIT, 115), FULL_SPLIT)

    def test_an_explicit_smaller_draw_is_honoured_and_is_a_different_set(self):
        """An explicit cap of 10 draws 10 stories, not a prefix of the 40."""
        ten = select_units(FULL_SPLIT, 10)
        self.assertEqual(len(ten), 10)
        self.assertNotEqual(ten, select_units(FULL_SPLIT, None)[:10])

    def test_a_short_split_is_returned_whole_rather_than_padded(self):
        """A split shorter than the cell comes back whole."""
        short = FULL_SPLIT[:12]
        self.assertEqual(select_units(short, None), short)


class TestDeclaredCounts(unittest.TestCase):
    """Every benchmark declares the unit count a cell resolves to."""

    def test_narrativeqa_declares_the_cell_draw(self):
        """NarrativeQA declares the 40-story cell."""
        from src.eval.narrativeqa import NarrativeQABenchmark

        self.assertEqual(NarrativeQABenchmark.cell_units, CELL_UNITS)

    def test_multihop_declares_one_shared_corpus(self):
        """MultiHop declares one unit: the shared corpus."""
        from src.eval.multihop import MultiHopBenchmark

        self.assertEqual(MultiHopBenchmark.cell_units, 1)

    def test_hotpotqa_declares_one_unit_per_question(self):
        """HotpotQA declares one unit per seeded question."""
        from src.eval.hotpotqa import PREREGISTERED_Q, HotpotQABenchmark

        self.assertEqual(HotpotQABenchmark().cell_units, PREREGISTERED_Q)

    def test_pooled_declares_one_unit_per_shard(self):
        """Pooled HotpotQA declares one unit per shard of questions."""
        from src.eval.hotpotqa import (
            PREREGISTERED_Q,
            SHARD_QUESTIONS,
            HotpotQAPooledBenchmark,
        )

        expected = -(-PREREGISTERED_Q // SHARD_QUESTIONS)  # ceil
        self.assertEqual(HotpotQAPooledBenchmark().cell_units, expected)

    def test_hotpotqa_population_is_already_a_constructor_default(self):
        """HotpotQA's seeded subsample size is a constructor default."""
        import inspect

        from src.eval.hotpotqa import PREREGISTERED_Q, HotpotQABenchmark

        sig = inspect.signature(HotpotQABenchmark.__init__)
        self.assertEqual(sig.parameters["max_questions"].default,
                         PREREGISTERED_Q)
        self.assertIsNotNone(PREREGISTERED_Q)


if __name__ == "__main__":
    unittest.main()
