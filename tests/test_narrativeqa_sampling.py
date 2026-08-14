"""P7: NarrativeQA draws a seeded sample, not the head of the split.

The 40-story subsample used to be `order[:40]` — a head slice of the 115
validation stories, under exactly the objection HotpotQA's seeded draw
exists to answer: dataset order is not guaranteed random, so a prefix can
be skewed on any dimension the ordering carries. Two benchmarks, two
sampling standards, no stated reason.

Both now draw through `src/eval/sampling.py`, so they cannot drift into
two conventions.
"""

from __future__ import annotations

import inspect
import unittest

from src.eval import narrativeqa
from src.eval.sampling import SUBSAMPLE_SEED, subsample_indices


class TestTheSharedSampler(unittest.TestCase):
    def test_the_seed_is_the_one_dated_constant(self):
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
        """The Done-when line, asserted rather than eyeballed."""
        src = inspect.getsource(narrativeqa.NarrativeQABenchmark.iter_eval_units)
        self.assertNotIn("n_done", src)
        self.assertNotIn("break", src)

    def test_the_loader_samples_through_the_shared_helper(self):
        src = inspect.getsource(narrativeqa.NarrativeQABenchmark.iter_eval_units)
        self.assertIn("subsample_indices", src)

    def test_stats_declare_the_seed_and_the_drawn_ids(self):
        bench = narrativeqa.NarrativeQABenchmark()
        self.assertIn("subsample_seed", bench.stats)
        self.assertIn("sampled_story_ids", bench.stats)


class TestDrawnIdsReachTheRunSummary(unittest.TestCase):
    def test_the_runner_copies_benchmark_stats_verbatim(self):
        """`sampled_story_ids` is only reproducible-and-inspectable if it
        actually lands in the summary; the runner writes benchmark.stats
        wholesale, which is the mechanism."""
        from src.eval import runner

        self.assertIn('"benchmark_stats": getattr(benchmark, "stats", {})',
                      inspect.getsource(runner.main))


if __name__ == "__main__":
    unittest.main()
