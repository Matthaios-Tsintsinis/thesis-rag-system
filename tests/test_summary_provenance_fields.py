"""Summary provenance must describe the run that produced it.

Three defects, found by reading a real cell summary rather than the code
that writes it — which is the only way this class surfaces.

1. `expected_n_queries: null` on every HotpotQA cell. The field reads
   `benchmark.stats["n_queries"]`, but HotpotQA's loader records
   `n_questions`. MultiHop and NarrativeQA use `n_queries`. So P8's
   short-cell guard — the assertion that a truncated cell fails loudly
   rather than reporting a partial mean — had nothing to compare against
   on 10 of the 20 cells. Not a `--max-units` artifact: the key was
   simply never there.

2. `chunking_strategy: "word_window"` on an M4 cell whose own components
   line said `raptor_100tok`. The field read the HARNESS default
   (`system.config.chunking.strategy`) instead of the chunker the system
   RESOLVED (`resolved_components.chunker_config.strategy`). Every M4 row
   in the final table would have named the wrong chunker.

3. `evidence_budget: null` and `max_new_tokens: 512` on a cell where M4
   runs a 2,000-token budget and summarises at 100. Those two are the
   ANSWER-path values and are correct as such — but a reader cannot tell
   that from the names, and the per-system values were recorded nowhere.
   Not a mis-record; an incomplete one.
"""

from __future__ import annotations

import unittest


class TestLoaderStatsAgreeOnTheKey(unittest.TestCase):
    """One key name across loaders, so a consumer needs no fallback."""

    def test_every_live_loader_exposes_n_queries(self):
        from src.eval.hotpotqa import HotpotQABenchmark, HotpotQAPooledBenchmark
        from src.eval.multihop import MultiHopBenchmark
        from src.eval.narrativeqa import NarrativeQABenchmark

        for cls in (MultiHopBenchmark, NarrativeQABenchmark,
                    HotpotQABenchmark, HotpotQAPooledBenchmark):
            with self.subTest(loader=cls.__name__):
                self.assertIn("n_queries", cls().stats,
                              f"{cls.__name__} would write "
                              "expected_n_queries: null")

    def test_hotpotqa_keeps_n_questions_too(self):
        """The old key is read elsewhere; adding the new one must not
        remove it."""
        from src.eval.hotpotqa import HotpotQABenchmark

        self.assertIn("n_questions", HotpotQABenchmark().stats)


class TestExpectedNQueriesResolution(unittest.TestCase):
    def test_it_reads_n_queries_from_stats(self):
        from src.eval.runner import resolve_expected_n_queries

        class B:
            stats = {"n_queries": 2556}

        self.assertEqual(resolve_expected_n_queries(B()), 2556)

    def test_a_loader_without_the_key_returns_none_rather_than_guessing(self):
        from src.eval.runner import resolve_expected_n_queries

        class B:
            stats = {"something_else": 1}

        self.assertIsNone(resolve_expected_n_queries(B()))

    def test_an_uncapped_run_with_no_count_is_an_error(self):
        """P8's guard exists so a short cell fails loudly. A null here
        removes the guard silently, which is worse than a short cell."""
        from src.eval.runner import assert_expected_n_queries_usable

        with self.assertRaises(SystemExit):
            assert_expected_n_queries_usable(
                None, max_units=None, max_queries=None
            )

    def test_a_capped_run_may_legitimately_have_none(self):
        from src.eval.runner import assert_expected_n_queries_usable

        assert_expected_n_queries_usable(None, max_units=1, max_queries=5)

    def test_a_populated_count_passes_either_way(self):
        from src.eval.runner import assert_expected_n_queries_usable

        assert_expected_n_queries_usable(1000, max_units=None, max_queries=None)


class TestChunkerIsTheResolvedOne(unittest.TestCase):
    def test_it_prefers_the_systems_resolved_chunker(self):
        from src.eval.runner import resolve_chunking_strategy

        class R:
            class chunker_config:
                strategy = "raptor_100tok"

        class S:
            resolved_components = R()

            class config:
                class chunking:
                    strategy = "word_window"

        self.assertEqual(resolve_chunking_strategy(S()), "raptor_100tok")

    def test_it_falls_back_to_the_harness_default(self):
        from src.eval.runner import resolve_chunking_strategy

        class S:
            resolved_components = None

            class config:
                class chunking:
                    strategy = "word_window"

        self.assertEqual(resolve_chunking_strategy(S()), "word_window")


# The retrieval-denominator test lives in `test_cli_entrypoints.py`,
# where a real summary is written by main(). The first draft of it was
# `assertIn('"n_retrieval_scored"', inspect.getsource(runner.main))`
# plus an assertion that 2556 - 301 == 2255 — a source grep and a
# statement about two literals, written ONE COMMIT after the sweep that
# converted six of exactly that shape. Caught in review of its own diff.
#
# It is the thirteenth instance and the sharpest evidence for the
# sweep's own conclusion: the reflex is structural, not inattentive, and
# the remedy is the periodic grep rather than the intention to do better.


if __name__ == "__main__":
    unittest.main()
