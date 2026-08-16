"""20-cell wall-time projection: arithmetic, not optimism.

Every input is MEASURED and passed in. The projector computes and flags;
it does not guess. A missing measurement raises rather than defaulting,
because a default here silently becomes a session plan.
"""

from __future__ import annotations

import unittest

from scripts.project_matrix_cost import (
    SESSION_GUARD_S,
    WARN_FRACTION,
    project_cell,
    project_matrix,
)


class TestCellArithmetic(unittest.TestCase):
    def test_query_only_cell(self):
        cell = project_cell(
            system="M2", benchmark="multihop_rag",
            n_queries=2556, s_per_query=3.718,
        )
        self.assertAlmostEqual(cell["query_s"], 2556 * 3.718)
        self.assertEqual(cell["build_s"], 0.0)
        self.assertAlmostEqual(cell["total_s"], 2556 * 3.718)

    def test_build_term_is_added_for_tree_systems(self):
        cell = project_cell(
            system="M4", benchmark="narrativeqa",
            n_queries=1208, s_per_query=5.0,
            build_s_per_unit=[82.4] * 40,
        )
        self.assertAlmostEqual(cell["build_s"], 82.4 * 40)
        self.assertAlmostEqual(cell["total_s"], 82.4 * 40 + 1208 * 5.0)

    def test_per_unit_builds_are_summed_not_averaged(self):
        """Story sizes span 37x; an average would understate the cell and
        hide exactly the unit that overruns a session."""
        cell = project_cell(
            system="M4", benchmark="narrativeqa", n_queries=10,
            s_per_query=1.0, build_s_per_unit=[10.0, 900.0],
        )
        self.assertAlmostEqual(cell["build_s"], 910.0)
        self.assertAlmostEqual(cell["max_unit_build_s"], 900.0)

    def test_missing_s_per_query_raises(self):
        with self.assertRaises(ValueError):
            project_cell(system="M2", benchmark="multihop_rag",
                         n_queries=2556, s_per_query=None)

    def test_zero_queries_raises(self):
        """A cell with no queries is a configuration error, not a free
        cell that quietly projects to zero."""
        with self.assertRaises(ValueError):
            project_cell(system="M2", benchmark="multihop_rag",
                         n_queries=0, s_per_query=1.0)


class TestSessionGuardFlagging(unittest.TestCase):
    def test_cell_over_the_warn_fraction_is_flagged(self):
        over = SESSION_GUARD_S * (WARN_FRACTION + 0.05)
        cell = project_cell(system="M4", benchmark="narrativeqa",
                            n_queries=1, s_per_query=over)
        self.assertTrue(cell["over_warn_fraction"])
        self.assertGreater(cell["fraction_of_session"], WARN_FRACTION)

    def test_cell_under_the_fraction_is_not_flagged(self):
        under = SESSION_GUARD_S * (WARN_FRACTION - 0.1)
        cell = project_cell(system="M4", benchmark="narrativeqa",
                            n_queries=1, s_per_query=under)
        self.assertFalse(cell["over_warn_fraction"])

    def test_a_cell_longer_than_a_whole_session_is_called_out_separately(self):
        """Over the guard is categorically worse than over 60% of it: the
        cell cannot complete in one session at all and needs --resume
        planning, not just a warning."""
        cell = project_cell(system="M4", benchmark="narrativeqa",
                            n_queries=1, s_per_query=SESSION_GUARD_S * 1.2)
        self.assertTrue(cell["exceeds_session"])


class TestMatrixRollup(unittest.TestCase):
    MEASURED = {
        "s_per_query": {
            "M1": 0.5, "M2": 3.718, "M3": 4.0, "M4": 5.0, "M9": 5.799,
        },
        "n_queries": {
            "multihop_rag": 2556, "narrativeqa": 1208,
            "hotpotqa": 1000, "hotpotqa_pooled": 1000,
        },
    }

    def test_all_twenty_cells_are_projected(self):
        m = project_matrix(**self.MEASURED)
        self.assertEqual(len(m["cells"]), 20)

    def test_missing_system_measurement_names_the_gap(self):
        measured = {**self.MEASURED,
                    "s_per_query": {k: v
                                    for k, v in self.MEASURED["s_per_query"].items()
                                    if k != "M9"}}
        with self.assertRaises(ValueError) as ctx:
            project_matrix(**measured)
        self.assertIn("M9", str(ctx.exception))

    def test_total_is_the_sum_of_cells(self):
        m = project_matrix(**self.MEASURED)
        self.assertAlmostEqual(
            m["total_s"], sum(c["total_s"] for c in m["cells"])
        )

    def test_flagged_cells_are_listed_for_the_reader(self):
        m = project_matrix(
            **{**self.MEASURED,
               "s_per_query": {**self.MEASURED["s_per_query"], "M4": 20.0}}
        )
        flagged = {(c["system"], c["benchmark"]) for c in m["flagged"]}
        for benchmark in ("multihop_rag", "narrativeqa",
                          "hotpotqa", "hotpotqa_pooled"):
            self.assertIn(("M4", benchmark), flagged)

    def test_m9_on_multihop_is_heavy_at_its_measured_rate(self):
        """NOT a contrived case. M9 measured 5.799 s/query in the probe
        era and MultiHop is 2,556 queries, so the cell projects to ~4.1 h
        — 82% of a 5 h session on query time ALONE, before any variance.
        This pins the arithmetic that makes it a session-planning problem
        rather than a footnote, and it is why the flagged list must not be
        assumed to contain only M4."""
        m = project_matrix(**self.MEASURED)
        by = {(c["system"], c["benchmark"]): c for c in m["cells"]}
        cell = by[("M9", "multihop_rag")]
        self.assertTrue(cell["over_warn_fraction"])
        self.assertFalse(cell["exceeds_session"])
        self.assertGreater(cell["fraction_of_session"], 0.8)

    def test_m4_build_terms_are_applied_only_to_m4(self):
        m = project_matrix(
            **self.MEASURED,
            m4_build_s_per_unit={"narrativeqa": [82.4] * 40},
        )
        by = {(c["system"], c["benchmark"]): c for c in m["cells"]}
        self.assertAlmostEqual(by[("M4", "narrativeqa")]["build_s"], 82.4 * 40)
        self.assertEqual(by[("M2", "narrativeqa")]["build_s"], 0.0)

    def test_unmeasured_m4_build_is_reported_as_unmeasured(self):
        """M4 cells without a measured build carry the flag rather than a
        zero that reads as 'no build needed'."""
        m = project_matrix(**self.MEASURED)
        by = {(c["system"], c["benchmark"]): c for c in m["cells"]}
        self.assertTrue(by[("M4", "multihop_rag")]["build_unmeasured"])
        self.assertFalse(by[("M2", "multihop_rag")]["build_unmeasured"])


if __name__ == "__main__":
    unittest.main()
