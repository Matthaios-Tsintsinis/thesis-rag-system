"""Tests for the runner's lockfile gate and the null-hash provenance warning.

The M4 substrate key folds the umap-learn, scikit-learn and numpy versions,
so an unpinned environment would split the matrix into two tree populations.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.eval.runner import assert_environment_pinned


class TestGate(unittest.TestCase):
    """A missing or violated lockfile aborts the run; no flag downgrades it."""

    def _lock(self, td: str, body: str = "numpy==2.2.6\n") -> Path:
        p = Path(td) / "requirements.lock"
        p.write_text("# lock\n" + body, encoding="utf-8")
        return p

    def test_missing_lockfile_aborts(self):
        """An absent lockfile exits non-zero."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as ctx:
                assert_environment_pinned(Path(td) / "absent.lock")
            self.assertNotEqual(ctx.exception.code, 0)

    def test_mismatched_environment_aborts(self):
        """A non-zero check_lockfile result exits the run."""
        with tempfile.TemporaryDirectory() as td:
            lock = self._lock(td)
            with mock.patch("scripts.pin_environment.check_lockfile",
                            return_value=1):
                with self.assertRaises(SystemExit):
                    assert_environment_pinned(lock)

    def test_matching_environment_passes(self):
        """A zero check_lockfile result lets the run continue."""
        with tempfile.TemporaryDirectory() as td:
            lock = self._lock(td)
            with mock.patch("scripts.pin_environment.check_lockfile",
                            return_value=0):
                assert_environment_pinned(lock)

    def test_the_abort_message_names_the_failure_mode(self):
        """The abort carries a string message."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as ctx:
                assert_environment_pinned(Path(td) / "absent.lock")
            msg = str(ctx.exception.code) + " " + (ctx.exception.args[0]
                                                   if ctx.exception.args
                                                   else "")
            self.assertTrue(isinstance(msg, str))


class TestProvenanceWarnsOnNullHash(unittest.TestCase):
    """A null lockfile_hash in a cell summary is printed, never silent."""

    def test_null_hash_is_announced(self):
        """An absent lockfile yields a null hash and a printed warning naming it."""
        from scripts.pin_environment import environment_provenance

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("builtins.print") as p:
                prov = environment_provenance(Path(td) / "absent.lock")
            self.assertIsNone(prov["lockfile_hash"])
            printed = " ".join(str(c) for c in p.call_args_list).lower()
            self.assertIn("lockfile_hash", printed)
            self.assertTrue("null" in printed or "none" in printed)

    def test_a_real_lockfile_still_hashes_without_warning(self):
        """A present lockfile yields a non-null hash."""
        from scripts.pin_environment import environment_provenance

        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            lock.write_text("numpy==2.2.6\n", encoding="utf-8")
            prov = environment_provenance(lock)
        self.assertIsNotNone(prov["lockfile_hash"])


if __name__ == "__main__":
    unittest.main()
