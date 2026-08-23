"""Pure-function tests for the rebuild differ (scripts/rebuild_unit_diff).

The build path itself is GPU-bound and operator-executed; what is testable
here is the arithmetic and the refusal logic — the parts where a silent
wrong answer would masquerade as a verdict. The GPU path's own guards
(interpreter assert, cache-root refusal, cache-hit refusal) are asserted by
the script at run time; these tests pin the comparison semantics.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rebuild_unit_diff import _env_package_tokens, compare_stats


BASE = {
    "tree_n_nodes": 21,
    "tree_depth_counts": {"0": 18, "1": 3},
    "n_summary_nodes_at_index": 3,
    "flat_n_chunks": 18,
}


class TestCompareStats(unittest.TestCase):
    def test_identical_trees_produce_no_diffs(self):
        self.assertEqual(compare_stats(BASE, dict(BASE)), [])

    def test_layer_sizes_compare_across_json_key_types(self):
        """A banked manifest round-trips through JSON, so its layer keys
        are strings; a fresh in-process build carries ints. The comparison
        must treat {"0": 18} and {0: 18} as the same tree."""
        rebuilt = dict(BASE)
        rebuilt["tree_depth_counts"] = {0: 18, 1: 3}
        self.assertEqual(compare_stats(BASE, rebuilt), [])

    def test_a_moved_layer_size_is_named(self):
        rebuilt = dict(BASE)
        rebuilt["tree_depth_counts"] = {"0": 18, "1": 4}
        diffs = compare_stats(BASE, rebuilt)
        self.assertEqual(len(diffs), 1)
        self.assertIn("layer_sizes", diffs[0])

    def test_every_compared_field_is_load_bearing(self):
        """Perturb each field alone; each must produce exactly one diff.
        A field that cannot fail is not being compared."""
        perturbed = {
            "tree_n_nodes": 22,
            "tree_depth_counts": {"0": 17, "1": 3},
            "n_summary_nodes_at_index": 4,
            "flat_n_chunks": 17,
        }
        for key, bad in perturbed.items():
            rebuilt = dict(BASE)
            rebuilt[key] = bad
            diffs = compare_stats(BASE, rebuilt)
            self.assertEqual(len(diffs), 1, f"{key} not compared")

    def test_a_missing_rebuilt_field_reads_as_a_difference(self):
        rebuilt = dict(BASE)
        del rebuilt["n_summary_nodes_at_index"]
        diffs = compare_stats(BASE, rebuilt)
        self.assertEqual(len(diffs), 1)
        self.assertIn("n_summary_nodes", diffs[0])


class TestEnvPackageTokens(unittest.TestCase):
    def test_python_token_is_excluded_from_the_stack_comparison(self):
        """The banked cell-6 env predates the python token; the current
        env carries it. The interpreter is the variable under test, so the
        package comparison must see the two stacks as EQUAL."""
        banked = "umap-learn=0.5.12; scikit-learn=1.6.1; numpy=2.5.2"
        current = ("umap-learn=0.5.12; scikit-learn=1.6.1; numpy=2.5.2; "
                   "python=3.12")
        self.assertEqual(_env_package_tokens(banked),
                         _env_package_tokens(current))

    def test_a_real_package_drift_still_differs(self):
        a = "umap-learn=0.5.12; scikit-learn=1.6.1; numpy=2.5.2"
        b = "umap-learn=0.5.12; scikit-learn=1.6.1; numpy=2.5.3"
        self.assertNotEqual(_env_package_tokens(a), _env_package_tokens(b))

    def test_token_order_is_irrelevant(self):
        a = "numpy=2.5.2; umap-learn=0.5.12"
        b = "umap-learn=0.5.12; numpy=2.5.2"
        self.assertEqual(_env_package_tokens(a), _env_package_tokens(b))

    def test_empty_env_yields_empty_tokens(self):
        self.assertEqual(_env_package_tokens(None), ())
        self.assertEqual(_env_package_tokens(""), ())


if __name__ == "__main__":
    unittest.main()
