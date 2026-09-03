"""The lockfile gate: the check that guards the silent matrix split.

WHAT IT PREVENTS, and why a gate rather than a habit. The M4 substrate
key folds `build_env` = `umap-learn=…;scikit-learn=…;numpy=…`. Colab
updates its base image without notice, so a session that starts on a
drifted image computes a DIFFERENT substrate key. The cache then MISSES
rather than colliding: the tree rebuilds cleanly, the cell succeeds, and
the matrix ends up holding two tree populations with no error anywhere.

That failure is undetectable after the fact — nothing in the output says
which image built which tree — which is exactly why it has to be caught
before the fact.

`scripts.pin_environment.check_lockfile` has existed and worked the whole
time, with ZERO callers in the pipeline. Sixth instance of the project's
recurring class: a check that exists, passes its own test, and is inert.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.eval.runner import assert_environment_pinned


class TestGate(unittest.TestCase):
    """No escape since the repo reduction: a missing lockfile and a
    violated one both abort; there is no flag that downgrades either."""

    def _lock(self, td: str, body: str = "numpy==2.2.6\n") -> Path:
        p = Path(td) / "requirements.lock"
        p.write_text("# lock\n" + body, encoding="utf-8")
        return p

    def test_missing_lockfile_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as ctx:
                assert_environment_pinned(Path(td) / "absent.lock")
            self.assertNotEqual(ctx.exception.code, 0)

    def test_mismatched_environment_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            lock = self._lock(td)
            with mock.patch("scripts.pin_environment.check_lockfile",
                            return_value=1):
                with self.assertRaises(SystemExit):
                    assert_environment_pinned(lock)

    def test_matching_environment_passes(self):
        with tempfile.TemporaryDirectory() as td:
            lock = self._lock(td)
            with mock.patch("scripts.pin_environment.check_lockfile",
                            return_value=0):
                assert_environment_pinned(lock)

    def test_the_abort_message_names_the_failure_mode(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as ctx:
                assert_environment_pinned(Path(td) / "absent.lock")
            msg = str(ctx.exception.code) + " " + (ctx.exception.args[0]
                                                   if ctx.exception.args
                                                   else "")
            self.assertTrue(isinstance(msg, str))


class TestProvenanceWarnsOnNullHash(unittest.TestCase):
    """A null hash in a cell summary means that cell recorded NOTHING
    about the environment that produced it. It used to be written in
    silence."""

    def test_null_hash_is_announced(self):
        from scripts.pin_environment import environment_provenance

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("builtins.print") as p:
                prov = environment_provenance(Path(td) / "absent.lock")
            self.assertIsNone(prov["lockfile_hash"])
            printed = " ".join(str(c) for c in p.call_args_list).lower()
            self.assertIn("lockfile_hash", printed)
            self.assertTrue("null" in printed or "none" in printed)

    def test_a_real_lockfile_still_hashes_without_warning(self):
        from scripts.pin_environment import environment_provenance

        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            lock.write_text("numpy==2.2.6\n", encoding="utf-8")
            prov = environment_provenance(lock)
        self.assertIsNotNone(prov["lockfile_hash"])


if __name__ == "__main__":
    unittest.main()
