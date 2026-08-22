"""The interpreter is a pinned input, and the lockfile hash must not move.

THE DEFECT THIS CLOSES. `write_lockfile` has always emitted
`# python=3.12.13` into the lockfile. `lockfile_hash` strips comments by
design and `check_lockfile` skipped every `#` line, so the value was
written correctly and read by nothing — cells 1-5 ran on CPython 3.12.13
and cell 6 on 3.13.15 with the pin reporting OK both times. Fourteenth
instance of a correct value nothing consumed.
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
    """THE CONSTRAINT THAT SHAPED THE FIX. Promoting python to a
    requirement line would enter `lockfile_hash`, move it off
    `17878bc8740173be`, fire the environment gate on every remaining
    cell and orphan the hash recorded in six banked summaries. The data
    was already on disk; only the reader was missing."""

    def test_adding_or_changing_the_python_comment_leaves_the_hash_alone(self):
        base = lockfile_hash(_lock(None))
        self.assertEqual(lockfile_hash(_lock("3.12.13")), base)
        self.assertEqual(lockfile_hash(_lock("3.13.15")), base)

    def test_a_requirement_line_WOULD_move_it(self):
        """Shown, so the reason for parsing a comment is evidenced rather
        than asserted."""
        self.assertNotEqual(
            lockfile_hash(_lock("3.12.13", "numpy==2.5.2\npython==3.12.13")),
            lockfile_hash(_lock("3.12.13")))


class TestLockedPython(unittest.TestCase):
    def test_it_reads_the_comment_write_lockfile_emits(self):
        self.assertEqual(locked_python(_lock("3.12.13")), "3.12.13")

    def test_a_lockfile_without_the_line_returns_None(self):
        self.assertIsNone(locked_python(_lock(None)))


class TestTheCheckFires(unittest.TestCase):
    def test_a_mismatched_interpreter_FAILS(self):
        """The whole point: this exact condition passed on cell 6."""
        wrong = "9.9.9" if sys.version.split()[0] != "9.9.9" else "8.8.8"
        rc = check_lockfile(_write(_lock(wrong, packages="")))
        self.assertEqual(rc, 1)

    def test_the_running_interpreter_PASSES(self):
        rc = check_lockfile(_write(_lock(sys.version.split()[0], packages="")))
        self.assertEqual(rc, 0)

    def test_a_lockfile_predating_the_check_warns_but_does_not_fail(self):
        """Refusing here would brick every existing lockfile."""
        rc = check_lockfile(_write(_lock(None, packages="")))
        self.assertEqual(rc, 0)


class TestTheSubstrateKeyNamesTheInterpreter(unittest.TestCase):
    def test_build_env_carries_python_major_minor(self):
        from src.raptor_paper import _topology_env_id

        env = _topology_env_id()
        want = f"python={sys.version_info.major}.{sys.version_info.minor}"
        self.assertIn(want, env)

    def test_the_patch_level_is_NOT_keyed(self):
        """3.12.13 -> 3.12.14 changes no ABI and no wheel tag, so keying
        it would force cold rebuilds for a change that cannot move
        topology. Checked by pin_environment, coarser in the key."""
        from src.raptor_paper import _topology_env_id

        self.assertNotIn(sys.version.split()[0], _topology_env_id())

    def test_the_three_topology_packages_are_still_named(self):
        from src.raptor_paper import _topology_env_id

        env = _topology_env_id()
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(f"{pkg}=", env)


if __name__ == "__main__":
    unittest.main()
