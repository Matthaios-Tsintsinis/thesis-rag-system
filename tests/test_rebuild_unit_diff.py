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

from scripts.rebuild_unit_diff import (
    _assert_locked_environment,
    _env_package_tokens,
    compare_stats,
)


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


class TestRootRefusal(unittest.TestCase):
    """The fifth refusal, and the one the others depend on: the FULL pin
    check runs before any dataset or model import. An interpreter-only
    check once approved an environment with `datasets` entirely absent;
    the failure arrived 1,206 manifests later as a ModuleNotFoundError
    instead of a refusal. These tests drive the real `check_lockfile`
    against real lockfiles on disk."""

    def test_a_drifted_lockfile_refuses_with_systemexit(self):
        """A pin the running environment cannot satisfy, under the CORRECT
        interpreter line — so the refusal is about the PACKAGES, proving
        the check is wider than the interpreter."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            lines = [
                "# python=" + sys.version.split()[0],
                "numpy==0.0.0.impossible",
                "",
            ]
            lock.write_text("\n".join(lines), encoding="utf-8")
            with self.assertRaises(SystemExit):
                _assert_locked_environment(lock)

    def test_a_snapshot_of_this_environment_passes(self):
        """`write_lockfile` snapshots the running environment, so checking
        it back must succeed — the refusal fails for the right reason and
        only that reason."""
        import tempfile
        from scripts.pin_environment import write_lockfile
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            write_lockfile(lock)
            self.assertEqual(
                _assert_locked_environment(lock),
                __import__("sys").version.split()[0],
            )

    def test_a_missing_lockfile_refuses(self):
        with self.assertRaises(SystemExit):
            _assert_locked_environment(Path("no-such-lockfile-anywhere"))


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
