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


class TestCheckpointRisk(unittest.TestCase):
    """Per-unit tree caching changes what an overrun COSTS, and that
    drives packing more than the raw percentage.

    `index_items` runs once per EvalUnit and each story's tree is flushed
    to its own cache dir, manifest last, before any query for it is
    answered. So a multi-unit M4 cell that dies mid-build loses ONE story.
    A single-unit cell — MultiHop is one shared corpus — loses the whole
    build. A cell at 90% of a session with per-unit checkpointing is safer
    than one at 70% without it.
    """

    def test_multi_unit_tree_cell_is_checkpointed(self):
        cell = project_cell(
            system="M4", benchmark="narrativeqa", n_queries=1208,
            s_per_query=5.0, build_s_per_unit=[80.0] * 40,
        )
        self.assertTrue(cell["build_checkpointed"])

    def test_single_unit_tree_cell_is_not_checkpointed(self):
        """MultiHop is ONE EvalUnit, so there is no per-unit granularity
        to fall back on — an interrupted build loses all of it."""
        cell = project_cell(
            system="M4", benchmark="multihop_rag", n_queries=2556,
            s_per_query=5.0, build_s_per_unit=[3000.0],
        )
        self.assertFalse(cell["build_checkpointed"])

    def test_non_tree_systems_have_no_build_to_checkpoint(self):
        cell = project_cell(system="M2", benchmark="narrativeqa",
                            n_queries=1208, s_per_query=3.7)
        self.assertFalse(cell["build_checkpointed"])

    def test_a_cell_with_no_build_carries_no_build_risk(self):
        """A query-only cell has nothing to lose to an interrupt beyond
        one batch: answers are flushed per batch and `--resume` skips
        what is banked. Charging it the unprotected-build penalty would
        rank M9/MultiHop — which builds no tree at all — above the M4
        cell whose whole hour of tree work is genuinely at risk."""
        query_only = project_cell(
            system="M9", benchmark="multihop_rag", n_queries=2556,
            s_per_query=5.793,
        )
        self.assertEqual(query_only["build_s"], 0.0)
        self.assertEqual(
            query_only["risk_rank"], query_only["fraction_of_session"]
        )
        self.assertEqual(query_only["build_loss_on_interrupt_s"], 0.0)

    def test_risk_ranks_an_unprotected_cell_above_a_larger_protected_one(self):
        """The ordering that makes the column worth having."""
        protected = project_cell(
            system="M4", benchmark="narrativeqa", n_queries=1,
            s_per_query=SESSION_GUARD_S * 0.9,
            build_s_per_unit=[1.0] * 40,
        )
        exposed = project_cell(
            system="M4", benchmark="multihop_rag", n_queries=1,
            s_per_query=SESSION_GUARD_S * 0.7, build_s_per_unit=[1.0],
        )
        self.assertGreater(exposed["risk_rank"], protected["risk_rank"])


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


class TestPerBenchmarkRates(unittest.TestCase):
    """s_per_query does NOT transfer across benchmarks, and the projector
    must not pretend it does.

    M4 measured 1.920 s/query on MultiHop — the FASTEST of the five,
    because its 2,000-token budget means less to read. That ratio cannot
    carry to NarrativeQA, where M4 builds 40 separate per-unit trees
    against MultiHop's single shared one. A projector that reused one
    rate everywhere would produce a confident and wrong plan.
    """

    NESTED = {
        "multihop_rag": {"M1": 4.495, "M2": 4.060, "M3": 3.607,
                         "M4": 1.920, "M9": 5.793},
    }
    N = {"multihop_rag": 2556, "narrativeqa": 1208,
         "hotpotqa": 1000, "hotpotqa_pooled": 1000}

    def test_nested_rates_are_used_per_benchmark(self):
        m = project_matrix(
            s_per_query={**self.NESTED,
                         "narrativeqa": {"M1": 9.0, "M2": 9.0, "M3": 9.0,
                                         "M4": 9.0, "M9": 9.0},
                         "hotpotqa": {k: 1.0 for k in
                                      ("M1", "M2", "M3", "M4", "M9")},
                         "hotpotqa_pooled": {k: 1.0 for k in
                                             ("M1", "M2", "M3", "M4", "M9")}},
            n_queries=self.N,
        )
        by = {(c["system"], c["benchmark"]): c for c in m["cells"]}
        self.assertAlmostEqual(by[("M4", "multihop_rag")]["s_per_query"], 1.920)
        self.assertAlmostEqual(by[("M4", "narrativeqa")]["s_per_query"], 9.0)

    def test_missing_benchmark_raises_without_an_explicit_source(self):
        with self.assertRaises(ValueError) as ctx:
            project_matrix(s_per_query=self.NESTED, n_queries=self.N)
        msg = str(ctx.exception)
        self.assertIn("narrativeqa", msg)

    def test_extrapolation_is_opt_in_and_marks_every_cell_it_touches(self):
        m = project_matrix(
            s_per_query=self.NESTED, n_queries=self.N,
            extrapolate_from="multihop_rag",
        )
        by = {(c["system"], c["benchmark"]): c for c in m["cells"]}
        measured = by[("M4", "multihop_rag")]
        borrowed = by[("M4", "narrativeqa")]
        self.assertFalse(measured["s_per_query_extrapolated"])
        self.assertTrue(borrowed["s_per_query_extrapolated"])
        self.assertEqual(borrowed["s_per_query_source"], "multihop_rag")

    def test_extrapolated_cells_are_listed_in_the_rollup(self):
        m = project_matrix(
            s_per_query=self.NESTED, n_queries=self.N,
            extrapolate_from="multihop_rag",
        )
        self.assertEqual(len(m["extrapolated"]), 15)


class TestMeasuredZeroIsNotUnmeasured(unittest.TestCase):
    """M4 on HotpotQA-distractor has NO TREE — ~10 leaves per question
    falls below the layer stop condition, so it degenerates to flat.
    A zero build term there is a MEASURED FACT, and reporting it as
    BUILD-UNMEASURED would send someone to measure a thing that does not
    exist."""

    def test_none_means_unmeasured(self):
        cell = project_cell(system="M4", benchmark="hotpotqa",
                            n_queries=1000, s_per_query=1.0,
                            build_s_per_unit=None)
        self.assertTrue(cell["build_unmeasured"])

    def test_empty_list_means_measured_zero(self):
        cell = project_cell(system="M4", benchmark="hotpotqa",
                            n_queries=1000, s_per_query=1.0,
                            build_s_per_unit=[])
        self.assertFalse(cell["build_unmeasured"])
        self.assertEqual(cell["build_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
