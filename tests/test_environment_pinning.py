"""P9: the environment a cell was produced under is recorded with the cell.

M4 tree topology is version-sensitive even when seeded, so "reproducible"
is a claim about a PINNED stack on a given GPU class. A row that does not
say which stack and which accelerator produced it cannot support that
claim, and a lockfile nobody checks is a comment.

THE ACCEPTANCE TEST IS OPERATOR-EXECUTED, deliberately: building a fresh
environment from the lockfile and reproducing one M4 unit's tree node
count needs the GPU the matrix runs on. It is recorded as an operator
line in docs/EVAL_CORRECTION_PLAN.md, in the same class as the tree-cache
preflight, and is NOT claimed to be verified here.
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
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            self.assertEqual(write_lockfile(lock), 0)
            self.assertEqual(check_lockfile(lock), 0)

    def test_a_drifted_version_fails_the_check(self):
        """The check has to be able to FAIL, or it is decoration."""
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
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            lock.write_text("definitely-not-installed==1.0.0\n", encoding="utf-8")
            self.assertEqual(check_lockfile(lock), 1)


class TestLockfileHash(unittest.TestCase):
    def test_comments_and_ordering_do_not_change_the_hash(self):
        """The hash names the ENVIRONMENT, not the file's formatting, so a
        reordered or re-commented lockfile must not read as a changed
        stack."""
        a = "# written monday\nnumpy==2.2.6\nnumba==0.66.0\n"
        b = "# written tuesday\nnumba==0.66.0\n\nnumpy==2.2.6\n"
        self.assertEqual(lockfile_hash(a), lockfile_hash(b))

    def test_a_changed_version_changes_the_hash(self):
        a = "numpy==2.2.6\n"
        b = "numpy==2.2.7\n"
        self.assertNotEqual(lockfile_hash(a), lockfile_hash(b))


class TestProvenanceBlock(unittest.TestCase):
    def test_it_carries_the_gpu_string(self):
        prov = environment_provenance(None)
        self.assertIn("gpu", prov)
        self.assertEqual(prov["gpu"], gpu_model())

    def test_it_carries_the_topology_libraries_that_are_installed(self):
        prov = environment_provenance(None)
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(pkg, prov["versions"], pkg)

    def test_the_topology_critical_set_covers_the_cold_tree_lever(self):
        """The lever keys on three libraries; the lockfile must pin at
        least those, or a stack could drift in exactly the dimension the
        lever watches without the lockfile noticing."""
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(pkg, TOPOLOGY_CRITICAL)


class TestOneBatchSizeForEveryCell(unittest.TestCase):
    """`assertEqual(MATRIX_BATCH_SIZE, 16)` used to live here.

    That is a statement about a constant and proves nothing about the
    pipeline — the exact tautology I2 was created to replace, surviving
    in a second file while the replacement was written in a first. It
    also blocked a measured change to the value while asserting nothing
    about whether anything reads it.

    What matters is that the runner RESOLVES the constant into the batch
    shape it actually uses. `test_cli_entrypoints` asserts the summary's
    batch_size tracks the constant end to end; this asserts the meaning
    of the value the matrix now runs at.
    """

    def test_the_matrix_value_resolves_to_sequential_answering(self):
        from src.config import MATRIX_BATCH_SIZE

        # `batch_size = args.batch_size if args.batch_size else None`,
        # and None is the sequential path. Measured faster on the answer
        # path than any batched cap: 4.2558 s/query against 5.1654 at the
        # best one, because a batch runs until its longest member stops
        # and the 512-token answer cap makes that tail dominate.
        self.assertFalse(
            MATRIX_BATCH_SIZE,
            "MATRIX_BATCH_SIZE must be falsy to select the sequential "
            "path the answer-side measurement chose",
        )


class TestTheRunnerRecordsIt(unittest.TestCase):
    def test_summary_carries_environment_and_model_revisions(self):
        import inspect

        from src.eval import runner

        src = inspect.getsource(runner.main)
        self.assertIn('"environment": _environment_provenance()', src)
        self.assertIn('"model_revisions": _model_revisions(system)', src)

    def test_provenance_never_kills_a_run(self):
        """A missing lockfile or an offline hub must degrade to a
        recorded error, not abort a 20-cell pass."""
        from src.eval.runner import _environment_provenance, _model_revisions

        self.assertIsInstance(_environment_provenance(), dict)

        class _Broken:
            @property
            def resolved_components(self):
                raise RuntimeError("boom")

        self.assertIsInstance(_model_revisions(_Broken()), dict)


if __name__ == "__main__":
    unittest.main()
