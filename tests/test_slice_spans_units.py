"""A timed slice must span as many units as the sample size requires.

THE DEFECT. `probe_query_slice` took `units[0]` and aborted if that one
unit held fewer than `n` queries. Queries per EvalUnit vary by three
orders of magnitude across this matrix:

    MultiHop              2,556 per unit (one shared corpus)
    HotpotQA-pooled         100 per unit (a shard)
    NarrativeQA              ~30 per unit (a story)
    HotpotQA-distractor        1 per unit (one corpus per question)

So a 50-query slice was impossible on two of the four benchmarks — the
probe aborted rather than measuring, which is why 15 of 20 cells were
still carrying a rate extrapolated from MultiHop.

The unit count also determines what a slice COSTS: on HotpotQA-distractor
50 queries means 50 tree builds, on pooled it means one. That is the
difference between a quarter-hour probe and a session, and the caller has
to be told which before starting.
"""

from __future__ import annotations

import unittest

from scripts.probe_cell_costs import units_covering


class _Unit:
    def __init__(self, n_queries: int, name: str = ""):
        self.queries = [object()] * n_queries
        self.name = name


class TestUnitsCovering(unittest.TestCase):
    def test_one_query_per_unit_needs_n_units(self):
        """HotpotQA-distractor. 50 queries = 50 corpora = 50 trees."""
        units = [_Unit(1) for _ in range(1000)]
        self.assertEqual(len(units_covering(units, 50)), 50)

    def test_thirty_per_unit_needs_two(self):
        """NarrativeQA: ~30 questions per story."""
        units = [_Unit(30) for _ in range(40)]
        self.assertEqual(len(units_covering(units, 50)), 2)

    def test_a_hundred_per_unit_needs_one(self):
        """HotpotQA-pooled: one shard covers the slice."""
        units = [_Unit(100) for _ in range(10)]
        self.assertEqual(len(units_covering(units, 50)), 1)

    def test_a_single_huge_unit_needs_one(self):
        """MultiHop: one shared corpus, 2,556 queries."""
        self.assertEqual(len(units_covering([_Unit(2556)], 50)), 1)

    def test_it_stops_as_soon_as_n_is_reached(self):
        """Indexing one more unit than needed is a whole extra tree."""
        units = [_Unit(40) for _ in range(10)]
        self.assertEqual(len(units_covering(units, 40)), 1)
        self.assertEqual(len(units_covering(units, 41)), 2)

    def test_it_returns_loader_order(self):
        units = [_Unit(10, f"u{i}") for i in range(10)]
        picked = units_covering(units, 25)
        self.assertEqual([u.name for u in picked], ["u0", "u1", "u2"])

    def test_too_few_queries_returns_everything_for_the_caller_to_reject(self):
        """The helper does not raise — the caller reports how many were
        actually available, which is the more useful error."""
        units = [_Unit(1) for _ in range(5)]
        self.assertEqual(len(units_covering(units, 50)), 5)

    def test_zero_requested_indexes_nothing(self):
        self.assertEqual(units_covering([_Unit(10)], 0), [])


if __name__ == "__main__":
    unittest.main()
