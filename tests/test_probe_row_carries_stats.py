"""The probe row must carry everything the build reported.

WHAT THIS COSTS WHEN IT IS WRONG: two cold tree builds, one of them ~17
minutes and one ~5h45m. `phase_seconds` and `generate_calls` were written
correctly by `raptor_paper`, surfaced correctly by
`RaptorSystem._collect_index_stats`, and then discarded by the probe,
which assembled its row by naming fields one at a time. Everything
upstream was right and the JSON still arrived empty.

A hand-picked subset silently drops every field added after it was
written. So the row now carries the WHOLE stats dict under `index_stats`,
and these tests fail if that stops being true — including for a key that
does not exist yet, which is the case a fixed schema cannot cover.

No GPU: the stats dict is stubbed.
"""

from __future__ import annotations

import unittest

from scripts.probe_cell_costs import build_row_diagnostics


# What a real build reports, trimmed to the keys under test.
STATS = {
    "n_leaves": 132,
    "flat_n_summaries": 24,
    "n_summary_calls_at_index": 24,
    "layer_sizes": {0: 132, 1: 20, 2: 4},
    "phase_seconds": {"summarize": 900.1, "umap": 61.2, "gmm_bic_sweep": 8.0},
    "phase_calls": {"summarize": 3, "umap": 6, "gmm_bic_sweep": 6},
    "phase_share": {"summarize": 0.928, "umap": 0.063},
    "phase_measured_total_s": 969.3,
    "generate_calls": {"n_calls": 3, "mean_width": 8.0,
                       "max_width": 10, "min_width": 4},
    "degenerate_no_tree": False,
    "gate_children_per_parent": 6.6,
}


class TestNothingIsDropped(unittest.TestCase):
    def test_every_stats_key_survives_into_the_row(self):
        row = build_row_diagnostics(STATS)
        for key in STATS:
            self.assertIn(key, row["index_stats"], key)
            self.assertEqual(row["index_stats"][key], STATS[key], key)

    def test_a_key_that_does_not_exist_yet_also_survives(self):
        """The case a fixed schema cannot cover, and the reason
        `index_stats` carries the whole dict rather than a list of
        names. The next diagnostic added upstream must arrive here
        without this file being edited."""
        stats = dict(STATS, some_future_diagnostic={"invented": 1})
        row = build_row_diagnostics(stats)
        self.assertEqual(row["index_stats"]["some_future_diagnostic"],
                         {"invented": 1})


class TestThePhaseBlockIsPromoted(unittest.TestCase):
    """The two fields whose absence cost the builds are also lifted to the
    top level, so they are readable without opening `index_stats`."""

    def test_phase_fields_are_present_at_top_level(self):
        row = build_row_diagnostics(STATS)
        for key in ("phase_seconds", "phase_calls", "phase_share",
                    "phase_measured_total_s"):
            self.assertEqual(row[key], STATS[key], key)

    def test_generate_calls_is_present_at_top_level(self):
        row = build_row_diagnostics(STATS)
        self.assertEqual(row["generate_calls"], STATS["generate_calls"])
        self.assertEqual(row["generate_calls"]["n_calls"], 3)

    def test_the_convenience_view_is_a_view_not_the_record(self):
        """Losing a promoted field to a rename must not lose the
        underlying number, so both copies are asserted."""
        row = build_row_diagnostics(STATS)
        self.assertEqual(row["phase_seconds"],
                         row["index_stats"]["phase_seconds"])


class TestMissingFieldsDegradeQuietly(unittest.TestCase):
    """A build predating the instrumentation must not crash the probe —
    it reports None, which reads as 'not measured' rather than 'zero'."""

    def test_absent_phase_block_yields_none_not_zero(self):
        row = build_row_diagnostics({"n_leaves": 10, "flat_n_summaries": 2})
        self.assertIsNone(row["phase_seconds"])
        self.assertIsNone(row["generate_calls"])
        self.assertEqual(row["n_leaves"], 10)

    def test_an_empty_stats_dict_does_not_raise(self):
        row = build_row_diagnostics({})
        self.assertEqual(row["n_leaves"], 0)
        self.assertEqual(row["index_stats"], {})


class TestTheRowIsDecoupledFromTheSource(unittest.TestCase):
    def test_mutating_the_row_does_not_mutate_the_build_stats(self):
        stats = dict(STATS)
        row = build_row_diagnostics(stats)
        row["index_stats"]["n_leaves"] = -1
        self.assertEqual(stats["n_leaves"], 132)


if __name__ == "__main__":
    unittest.main()
