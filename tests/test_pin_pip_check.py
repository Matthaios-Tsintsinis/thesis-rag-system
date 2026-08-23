"""The pip-check screen inside the pin gate (scripts/pin_environment).

WHY IT EXISTS. torchvision 0.28.0 was installed against torch 2.13.0 by
requirements.txt, torch was then downgraded to the locked 2.11.0+cu128
underneath it, and the broken C++ extension took down `PreTrainedModel`
via transformers' image_utils — after `[pin] OK` had printed. torchvision
is not in the lockfile, so the version loop could not see it; pip's own
resolver COULD, and had warned at install time with nothing enforcing.
`pip check` is one subprocess and would have refused before any model
load.

WHAT IS PINNED HERE. The classification is the load-bearing part: a
conflict naming a LOCKED package fails the pin (pip is reporting a
violated contract on something the lock vouches for), while a conflict
among packages the lock never mentions warns only — the run host's system
python routinely carries preinstalled-package conflicts that touch
nothing the matrix uses, and a gate failing on those blocks cells for
noise. Both directions are asserted; a classifier that could only ever
warn would have reproduced the original defect with extra steps.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pin_environment import _pip_check_conflicts, classify_conflicts

LOCKED = ["torch", "transformers", "numpy", "scikit-learn"]

# The real incident, verbatim shape of a pip-check line.
TORCHVISION_LINE = (
    "torchvision 0.28.0 has requirement torch==2.13.0, "
    "but you have torch 2.11.0+cu128."
)
NOISE_LINE = (
    "google-colab 1.0.0 has requirement pandas==2.2.2, "
    "but you have pandas 3.0.3."
)


class TestClassification(unittest.TestCase):
    def test_the_torchvision_incident_fails_the_pin(self):
        """The exact conflict that got past the version loop: torchvision
        is unlocked, but the line names torch, which IS locked."""
        failing, warn = classify_conflicts([TORCHVISION_LINE], LOCKED)
        self.assertEqual(failing, [TORCHVISION_LINE])
        self.assertEqual(warn, [])

    def test_a_conflict_among_unlocked_packages_warns_only(self):
        failing, warn = classify_conflicts([NOISE_LINE], LOCKED)
        self.assertEqual(failing, [])
        self.assertEqual(warn, [NOISE_LINE])

    def test_mixed_lines_split_correctly(self):
        failing, warn = classify_conflicts(
            [NOISE_LINE, TORCHVISION_LINE], LOCKED
        )
        self.assertEqual(failing, [TORCHVISION_LINE])
        self.assertEqual(warn, [NOISE_LINE])

    def test_matching_is_case_insensitive(self):
        line = "Torchvision 0.28.0 has requirement Torch==2.13.0, but ..."
        failing, _ = classify_conflicts([line], ["torch"])
        self.assertEqual(failing, [line])

    def test_no_conflicts_is_clean_both_ways(self):
        self.assertEqual(classify_conflicts([], LOCKED), ([], []))


class TestSubprocessSmoke(unittest.TestCase):
    def test_pip_check_runs_and_returns_a_list(self):
        """The helper must never crash the gate: whatever this
        environment's state, the call returns a list."""
        out = _pip_check_conflicts()
        self.assertIsInstance(out, list)
        for line in out:
            self.assertIsInstance(line, str)


class TestGateIntegration(unittest.TestCase):
    def test_a_snapshot_lockfile_still_passes_on_a_clean_env(self):
        """End to end through check_lockfile: a snapshot of THIS
        environment must still return 0 with the pip screen in place —
        unless pip itself reports a conflict naming a locked package
        here, in which case this host is genuinely broken and the test
        SHOULD fail."""
        import tempfile

        from scripts.pin_environment import check_lockfile, write_lockfile

        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            write_lockfile(lock)
            self.assertEqual(check_lockfile(lock), 0)


if __name__ == "__main__":
    unittest.main()
