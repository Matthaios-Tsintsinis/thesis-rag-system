"""Tests for the environment pin: the lockfile round trip, its hash, and
the provenance block each cell summary carries.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.pin_environment import (
    TOPOLOGY_CRITICAL,
    check_lockfile,
    environment_provenance,
    gpu_model,
    lockfile_hash,
    write_lockfile,
)


class TestLockfileRoundTrip(unittest.TestCase):
    def test_a_freshly_written_lockfile_checks_clean(self):
        """A lockfile written from this environment checks clean."""
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            self.assertEqual(write_lockfile(lock), 0)
            self.assertEqual(check_lockfile(lock), 0)

    def test_a_drifted_version_fails_the_check(self):
        """A version that differs from the installed one fails the check."""
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            write_lockfile(lock)
            text = lock.read_text(encoding="utf-8")
            drifted = []
            for line in text.splitlines():
                if line.startswith("numpy=="):
                    drifted.append("numpy==0.0.1")
                else:
                    drifted.append(line)
            lock.write_text("\n".join(drifted) + "\n", encoding="utf-8")
            self.assertEqual(check_lockfile(lock), 1)

    def test_an_absent_package_fails_the_check(self):
        """A pinned package that is not installed fails the check."""
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            lock.write_text("definitely-not-installed==1.0.0\n", encoding="utf-8")
            self.assertEqual(check_lockfile(lock), 1)


class TestLockfileHash(unittest.TestCase):
    def test_comments_and_ordering_do_not_change_the_hash(self):
        """Comments, blank lines and ordering do not change the hash."""
        a = "# written monday\nnumpy==2.2.6\nnumba==0.66.0\n"
        b = "# written tuesday\nnumba==0.66.0\n\nnumpy==2.2.6\n"
        self.assertEqual(lockfile_hash(a), lockfile_hash(b))

    def test_a_changed_version_changes_the_hash(self):
        """A changed version string changes the hash."""
        a = "numpy==2.2.6\n"
        b = "numpy==2.2.7\n"
        self.assertNotEqual(lockfile_hash(a), lockfile_hash(b))


class TestProvenanceBlock(unittest.TestCase):
    def test_it_carries_the_gpu_string(self):
        """The provenance block records the GPU model string."""
        prov = environment_provenance(None)
        self.assertIn("gpu", prov)
        self.assertEqual(prov["gpu"], gpu_model())

    def test_it_carries_the_topology_libraries_that_are_installed(self):
        """The provenance block records the three topology libraries."""
        prov = environment_provenance(None)
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(pkg, prov["versions"], pkg)

    def test_the_topology_critical_set_covers_the_cold_tree_lever(self):
        """TOPOLOGY_CRITICAL covers the three libraries in the M4 tree key."""
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(pkg, TOPOLOGY_CRITICAL)


class TestTheRunnerRecordsIt(unittest.TestCase):
    def test_summary_carries_environment_and_model_revisions(self):
        """The runner writes environment and model revisions to the summary."""
        import inspect

        from src.eval import runner

        src = inspect.getsource(runner.main)
        self.assertIn('"environment": _environment_provenance(args.lockfile)', src)
        self.assertIn('"model_revisions": _model_revisions(system)', src)

    def test_provenance_never_kills_a_run(self):
        """A missing lockfile or a broken hub lookup is recorded, not raised."""
        from src.eval.runner import _environment_provenance, _model_revisions

        with tempfile.TemporaryDirectory() as td:
            prov = _environment_provenance(Path(td) / "absent.lock")
        self.assertIsInstance(prov, dict)
        self.assertIsNone(prov["lockfile_hash"])

        class _Broken:
            @property
            def resolved_components(self):
                raise RuntimeError("boom")

        self.assertIsInstance(_model_revisions(_Broken()), dict)


if __name__ == "__main__":
    unittest.main()
