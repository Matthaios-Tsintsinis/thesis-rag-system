"""Pins the summary provenance fields: the query count, the chunker and
the budget the run summary records must describe the system that ran.
"""

from __future__ import annotations

import unittest


class TestLoaderStatsAgreeOnTheKey(unittest.TestCase):
    """Every live loader exposes n_queries under one key name."""

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
        """HotpotQA keeps n_questions beside n_queries."""
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
        """An uncapped run with no expected count exits instead of guessing."""
        from src.eval.runner import assert_expected_n_queries_usable

        with self.assertRaises(SystemExit):
            assert_expected_n_queries_usable(None)

    def test_a_populated_count_passes(self):
        from src.eval.runner import assert_expected_n_queries_usable

        assert_expected_n_queries_usable(1000)


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


# The retrieval-denominator check lives in test_cli_entrypoints.py, where
# main() writes a real summary and the test reads the value back.


if __name__ == "__main__":
    unittest.main()
