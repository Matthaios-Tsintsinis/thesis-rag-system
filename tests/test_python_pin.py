"""Pins the interpreter check: the lockfile's python comment is read by
the gate and by the M4 substrate key, and never enters the lockfile hash.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from scripts.pin_environment import check_lockfile, lockfile_hash, locked_python


def _lock(python: str | None, packages: str = "numpy==2.5.2") -> str:
    head = "# Environment lock for the thesis RAG matrix.\n# gpu=NVIDIA L4\n"
    if python is not None:
        head += f"# python={python}\n"
    return head + "\n" + packages + "\n"


def _write(text: str) -> Path:
    p = Path(tempfile.mkdtemp()) / "requirements.lock"
    p.write_text(text, encoding="utf-8")
    return p


class TestTheHashDoesNotMove(unittest.TestCase):
    """The python comment stays out of the lockfile hash."""

    def test_adding_or_changing_the_python_comment_leaves_the_hash_alone(self):
        """Adding or changing the python comment leaves the hash unchanged."""
        base = lockfile_hash(_lock(None))
        self.assertEqual(lockfile_hash(_lock("3.12.13")), base)
        self.assertEqual(lockfile_hash(_lock("3.13.15")), base)

    def test_a_requirement_line_WOULD_move_it(self):
        """A python requirement line moves the hash; a comment does not."""
        self.assertNotEqual(
            lockfile_hash(_lock("3.12.13", "numpy==2.5.2\npython==3.12.13")),
            lockfile_hash(_lock("3.12.13")))


class TestLockedPython(unittest.TestCase):
    """locked_python reads the python comment the lockfile writer emits."""

    def test_it_reads_the_comment_write_lockfile_emits(self):
        """The python comment is parsed back to its version string."""
        self.assertEqual(locked_python(_lock("3.12.13")), "3.12.13")

    def test_a_lockfile_without_the_line_returns_None(self):
        """A lockfile with no python comment yields None."""
        self.assertIsNone(locked_python(_lock(None)))


class TestTheCheckFires(unittest.TestCase):
    """check_lockfile compares the locked python to the running one."""

    def test_a_mismatched_interpreter_FAILS(self):
        """A locked python that differs from the running one returns 1."""
        wrong = "9.9.9" if sys.version.split()[0] != "9.9.9" else "8.8.8"
        rc = check_lockfile(_write(_lock(wrong, packages="")))
        self.assertEqual(rc, 1)

    def test_the_running_interpreter_PASSES(self):
        """A locked python equal to the running one returns 0."""
        rc = check_lockfile(_write(_lock(sys.version.split()[0], packages="")))
        self.assertEqual(rc, 0)

    def test_a_lockfile_predating_the_check_warns_but_does_not_fail(self):
        """A lockfile with no python comment warns and returns 0."""
        rc = check_lockfile(_write(_lock(None, packages="")))
        self.assertEqual(rc, 0)


class TestTheSubstrateKeyNamesTheInterpreter(unittest.TestCase):
    """The M4 topology key names python at major.minor granularity."""

    def test_build_env_carries_python_major_minor(self):
        """The topology env id contains python=MAJOR.MINOR."""
        from src.raptor_paper import _topology_env_id

        env = _topology_env_id()
        want = f"python={sys.version_info.major}.{sys.version_info.minor}"
        self.assertIn(want, env)

    def test_the_patch_level_is_NOT_keyed(self):
        """The full patch version is absent from the topology env id."""
        from src.raptor_paper import _topology_env_id

        self.assertNotIn(sys.version.split()[0], _topology_env_id())

    def test_the_three_topology_packages_are_still_named(self):
        """umap-learn, scikit-learn and numpy are named in the env id."""
        from src.raptor_paper import _topology_env_id

        env = _topology_env_id()
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(f"{pkg}=", env)


if __name__ == "__main__":
    unittest.main()
