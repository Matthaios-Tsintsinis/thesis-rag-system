"""A cell's population must be a property of the CODE, not of a flag.

THE SEVENTH INSTANCE. P7 declared a seeded 40-story NarrativeQA sample.
The sampler was written, seeded, and tested — `subsample_indices(115, 40)`
is stable and is not `range(40)`. But `iter_eval_units` applied it only
when `max_units` was passed, and `--max-units` defaults to None, so a
cell launched without the flag ran all 115 stories and 3,461 questions.
It would have succeeded. P7 shipped a sampler, not a sample.

The distinction this file exists to enforce: "the sampler works" and
"a cell draws 40" are different claims, and only the second kind of test
catches the defect. The pre-existing tests asserted the first.

WHY THESE DRIVE THE SELECTION HELPER AND NOT THE FULL LOADER. Driving
`iter_eval_units` end to end needs the NarrativeQA corpus — a multi-
gigabyte download the agent host does not have. A test that cannot run
has not passed, so the selection step is factored out and tested
directly. It is the same call the loader makes, with the same default,
not a re-implementation of it.
"""

from __future__ import annotations

import unittest

from src.eval.narrativeqa import CELL_UNITS, select_units
from src.eval.sampling import subsample_indices

FULL_SPLIT = [f"story{i:03d}" for i in range(115)]


class TestDefaultDrawsTheCell(unittest.TestCase):
    def test_no_max_units_yields_exactly_the_cell_size(self):
        """THE DEFECT, inverted. None used to mean 'all 115'."""
        picked = select_units(FULL_SPLIT, None)
        self.assertEqual(len(picked), CELL_UNITS)

    def test_no_max_units_yields_the_SEEDED_draw_not_a_prefix(self):
        picked = select_units(FULL_SPLIT, None)
        expected = [FULL_SPLIT[i]
                    for i in subsample_indices(len(FULL_SPLIT), CELL_UNITS)]
        self.assertEqual(picked, expected)
        self.assertNotEqual(picked, FULL_SPLIT[:CELL_UNITS])

    def test_the_draw_is_identical_across_invocations(self):
        self.assertEqual(select_units(FULL_SPLIT, None),
                         select_units(FULL_SPLIT, None))

    def test_an_explicit_full_split_is_still_reachable(self):
        """Explicit 115 stays possible; silent 115 does not."""
        self.assertEqual(select_units(FULL_SPLIT, 115), FULL_SPLIT)

    def test_an_explicit_smaller_draw_is_honoured_and_is_a_different_set(self):
        ten = select_units(FULL_SPLIT, 10)
        self.assertEqual(len(ten), 10)
        self.assertNotEqual(ten, select_units(FULL_SPLIT, None)[:10])

    def test_a_short_split_is_returned_whole_rather_than_padded(self):
        short = FULL_SPLIT[:12]
        self.assertEqual(select_units(short, None), short)


class TestDeclaredCounts(unittest.TestCase):
    """Every benchmark states the unit count a cell should resolve to, so
    the runner can check the resolved population against a declaration
    rather than against nothing."""

    def test_narrativeqa_declares_the_cell_draw(self):
        from src.eval.narrativeqa import NarrativeQABenchmark

        self.assertEqual(NarrativeQABenchmark.cell_units, CELL_UNITS)

    def test_multihop_declares_one_shared_corpus(self):
        from src.eval.multihop import MultiHopBenchmark

        self.assertEqual(MultiHopBenchmark.cell_units, 1)

    def test_hotpotqa_declares_one_unit_per_question(self):
        from src.eval.hotpotqa import PREREGISTERED_Q, HotpotQABenchmark

        self.assertEqual(HotpotQABenchmark().cell_units, PREREGISTERED_Q)

    def test_pooled_declares_one_unit_per_shard(self):
        from src.eval.hotpotqa import (
            PREREGISTERED_Q,
            SHARD_QUESTIONS,
            HotpotQAPooledBenchmark,
        )

        expected = -(-PREREGISTERED_Q // SHARD_QUESTIONS)  # ceil
        self.assertEqual(HotpotQAPooledBenchmark().cell_units, expected)

    def test_hotpotqa_population_is_already_a_constructor_default(self):
        """CONFIRMATION, requested explicitly: HotpotQA cannot resolve to
        a different population the way NarrativeQA could. Its seeded
        subsample keys on `max_questions`, a CONSTRUCTOR default, and its
        `max_units` is an ordinary cap that selects no set. The correct
        pattern already existed one file over."""
        import inspect

        from src.eval.hotpotqa import PREREGISTERED_Q, HotpotQABenchmark

        sig = inspect.signature(HotpotQABenchmark.__init__)
        self.assertEqual(sig.parameters["max_questions"].default,
                         PREREGISTERED_Q)
        self.assertIsNotNone(PREREGISTERED_Q)


if __name__ == "__main__":
    unittest.main()
