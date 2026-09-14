"""NarrativeQA draws its 40-story cell through the shared seeded sampler.

Pins the seed, the draw, and that the drawn ids reach the run summary.
"""

from __future__ import annotations

import inspect
import unittest

from src.eval import narrativeqa
from src.eval.sampling import SUBSAMPLE_SEED, subsample_indices


class TestTheSharedSampler(unittest.TestCase):
    def test_the_seed_is_the_one_dated_constant(self):
        # harness choice: preregistered seed (METHODS §B)
        self.assertEqual(SUBSAMPLE_SEED, 20260805)

    def test_the_draw_is_reproducible(self):
        self.assertEqual(subsample_indices(115, 40), subsample_indices(115, 40))

    def test_the_draw_is_not_a_head_slice(self):
        self.assertNotEqual(subsample_indices(115, 40), list(range(40)))

    def test_indices_keep_dataset_order(self):
        picked = subsample_indices(115, 40)
        self.assertEqual(picked, sorted(picked))
        self.assertEqual(len(set(picked)), 40)

    def test_asking_for_more_than_exists_returns_everything(self):
        self.assertEqual(subsample_indices(10, 40), list(range(10)))

    def test_hotpotqa_and_narrativeqa_share_one_sampler(self):
        from src.eval import hotpotqa

        self.assertIs(hotpotqa.subsample_indices, subsample_indices)
        self.assertIs(narrativeqa.subsample_indices, subsample_indices)
        self.assertEqual(hotpotqa.SUBSAMPLE_SEED, SUBSAMPLE_SEED)


class TestTheLoaderNoLongerTakesAPrefix(unittest.TestCase):
    def test_no_break_on_first_n_pattern_remains(self):
        """The loader has no first-n prefix loop."""
        src = inspect.getsource(narrativeqa.NarrativeQABenchmark.iter_eval_units)
        self.assertNotIn("n_done", src)
        self.assertNotIn("break", src)

    def test_the_loader_samples_through_the_shared_helper(self):
        """The loader draws through select_units and the shared sampler."""
        src = inspect.getsource(narrativeqa.NarrativeQABenchmark.iter_eval_units)
        self.assertIn("select_units", src)
        helper = inspect.getsource(narrativeqa.select_units)
        self.assertIn("subsample_indices", helper)

    def test_the_default_draw_is_the_cell_not_the_split(self):
        """select_units with no limit returns the 40-story cell."""
        order = [f"s{i}" for i in range(115)]
        self.assertEqual(len(narrativeqa.select_units(order, None)),
                         narrativeqa.CELL_UNITS)

    def test_stats_declare_the_seed_and_the_drawn_ids(self):
        bench = narrativeqa.NarrativeQABenchmark()
        self.assertIn("subsample_seed", bench.stats)
        self.assertIn("sampled_story_ids", bench.stats)


class TestDrawnIdsReachTheRunSummary(unittest.TestCase):
    def test_the_runner_copies_benchmark_stats_verbatim(self):
        """The runner writes benchmark.stats whole into the summary."""
        from src.eval import runner

        self.assertIn('"benchmark_stats": getattr(benchmark, "stats", {})',
                      inspect.getsource(runner.main))


if __name__ == "__main__":
    unittest.main()
