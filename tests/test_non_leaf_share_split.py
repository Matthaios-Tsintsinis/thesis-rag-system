"""The App. I non-leaf gate, split by whether the unit actually built a tree.

WHY THE SPLIT EXISTS. The gate failed on M4/hotpotqa at 16.4% micro against
the paper's 18.5-57.0% band. A unit at or below the stop condition has NO
summary nodes by construction, so it contributes leaves to the denominator
and nothing to the numerator: pooling it with tree-building units measures a
MIXTURE, while the paper's band describes RAPTOR trees. Both figures are
reported so a reader can tell a mixing artifact from a real finding.

WHAT THESE TESTS PIN, and it is deliberately both directions:
  * the arithmetic of each figure, against hand-computed values;
  * that the degenerate rows are excluded from the tree-building figure by
    READING `metadata.m4_tree_degenerate` -- never by recomputing it (that
    recomputation was instance 13 of this project's recurring defect);
  * that an in-band tree-building figure under an out-of-band all-rows
    figure prints the ARTIFACT verdict, and an out-of-band one prints the
    FINDING verdict. A split that could only ever exonerate would be an
    escape hatch rather than a diagnostic.
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.analyse import _aggregate, _print_text


def _row(qid, *, chunks, summaries, degenerate):
    """One M4 row. `retrieved_unit_types` is what the gate reads."""
    total = chunks + summaries
    unit_types = {"chunk": chunks}
    if summaries:
        unit_types["summary_low"] = summaries
    return {
        "query_id": qid,
        "system_id": "M4",
        "benchmark": "hotpotqa",
        "question_type": "bridge",
        "n_retrieved": total,
        "n_packed": total,
        "evidence_tokens": 100,
        "n_input_tokens": 200,
        "latency_s": 1.0,
        "retrieved_unit_types": unit_types,
        "packed_unit_types": unit_types,
        "retrieval": {"skipped": False, "f1": 0.1, "recall": 0.1, "precision": 0.1},
        "answer": {"value": 0.2, "method": "token_f1", "metadata": {}},
        "metadata": {
            "m4_non_leaf_share": summaries / total if total else 0.0,
            "m4_tree_degenerate": degenerate,
        },
    }


class TestSplitArithmetic(unittest.TestCase):
    def setUp(self):
        # 2 tree-building rows: 8 chunks + 2 summaries each
        #   -> tree-building micro = 4 / 20 = 20.0%  (IN band)
        # 2 degenerate rows: 10 chunks, 0 summaries each
        #   -> all-rows micro     = 4 / 40 = 10.0%  (OUT of band)
        rows = [
            _row("t1", chunks=8, summaries=2, degenerate=False),
            _row("t2", chunks=8, summaries=2, degenerate=False),
            _row("d1", chunks=10, summaries=0, degenerate=True),
            _row("d2", chunks=10, summaries=0, degenerate=True),
        ]
        self.nl = _aggregate(rows)["systems"]["M4"]["non_leaf"]

    def test_all_rows_micro_is_the_mixed_figure(self):
        self.assertAlmostEqual(self.nl["micro"], 4 / 40)
        self.assertEqual(self.nl["n_units"], 40)
        self.assertEqual(self.nl["n_non_leaf"], 4)
        self.assertFalse(self.nl["in_band"])

    def test_tree_building_micro_excludes_degenerate_units(self):
        self.assertAlmostEqual(self.nl["micro_treebuilding"], 4 / 20)
        self.assertEqual(self.nl["n_units_treebuilding"], 20)
        self.assertEqual(self.nl["n_non_leaf_treebuilding"], 4)
        self.assertTrue(self.nl["in_band_treebuilding"])

    def test_the_two_figures_actually_differ(self):
        """Guards against a split that silently aggregates the same rows."""
        self.assertNotAlmostEqual(self.nl["micro"], self.nl["micro_treebuilding"])

    def test_macro_is_split_too_and_counts_its_rows(self):
        self.assertAlmostEqual(self.nl["macro"], (0.2 + 0.2 + 0.0 + 0.0) / 4)
        self.assertAlmostEqual(self.nl["macro_treebuilding"], 0.2)
        self.assertEqual(self.nl["n_queries_treebuilding"], 2)

    def test_degenerate_rows_are_still_counted(self):
        self.assertEqual(self.nl["degenerate_rows"], 2)


class TestExclusionReadsTheRecordedFlag(unittest.TestCase):
    """The flag is READ, never re-derived.

    A row whose unit types show zero summaries but whose recorded flag is
    False must still count as tree-building: the run-time flag is the
    record of what the builder did, and a query can legitimately retrieve
    no summary node from a tree that exists.
    """

    def test_a_zero_summary_row_flagged_non_degenerate_stays_included(self):
        rows = [
            _row("t1", chunks=10, summaries=2, degenerate=False),
            _row("t2", chunks=10, summaries=0, degenerate=False),
        ]
        nl = _aggregate(rows)["systems"]["M4"]["non_leaf"]
        self.assertEqual(nl["degenerate_rows"], 0)
        self.assertEqual(nl["n_units_treebuilding"], 22)
        self.assertAlmostEqual(nl["micro_treebuilding"], nl["micro"])

    def test_a_summary_bearing_row_flagged_degenerate_is_excluded(self):
        """The converse, and it is the one that proves the direction: if the
        exclusion were derived from the unit types it would keep this row."""
        rows = [
            _row("t1", chunks=10, summaries=2, degenerate=False),
            _row("d1", chunks=10, summaries=5, degenerate=True),
        ]
        nl = _aggregate(rows)["systems"]["M4"]["non_leaf"]
        self.assertEqual(nl["n_non_leaf"], 7)
        self.assertEqual(nl["n_non_leaf_treebuilding"], 2)


class TestVerdictsPrintBothWays(unittest.TestCase):
    def _report(self, rows):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_text(_aggregate(rows), by_type=False)
        return buf.getvalue()

    def test_in_band_tree_building_prints_the_artifact_verdict(self):
        out = self._report([
            _row("t1", chunks=8, summaries=2, degenerate=False),
            _row("d1", chunks=10, summaries=0, degenerate=True),
            _row("d2", chunks=10, summaries=0, degenerate=True),
        ])
        self.assertIn("TREE-BUILDING UNITS ONLY", out)
        self.assertIn("REPORTING", out)
        self.assertIn("ARTIFACT", out)

    def test_out_of_band_tree_building_prints_the_finding_verdict(self):
        """A share below the band on tree-building units too. This is the
        assertion that stops the split being an escape hatch."""
        out = self._report([
            _row("t1", chunks=99, summaries=1, degenerate=False),
            _row("d1", chunks=10, summaries=0, degenerate=True),
        ])
        self.assertIn("OUT OF BAND ON TREE-BUILDING UNITS TOO", out)
        self.assertIn("do not explain it away", out)
        self.assertNotIn("REPORTING ARTIFACT", out)

    def test_a_cell_with_no_degenerate_rows_prints_one_figure_only(self):
        out = self._report([
            _row("t1", chunks=8, summaries=2, degenerate=False),
            _row("t2", chunks=8, summaries=2, degenerate=False),
        ])
        self.assertNotIn("TREE-BUILDING UNITS ONLY", out)


if __name__ == "__main__":
    unittest.main()
