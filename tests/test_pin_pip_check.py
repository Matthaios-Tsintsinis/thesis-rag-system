"""Tests for the pip-check screen in the pin gate (scripts/pin_environment).

A conflict naming a locked package fails the pin; a conflict among
packages the lock never mentions only warns.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pin_environment import _pip_check_conflicts, classify_conflicts

LOCKED = ["torch", "transformers", "numpy", "scikit-learn"]

# Two pip-check lines: one names a locked package, one does not.
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
        """A line naming a locked package fails, even from an unlocked one."""
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


class TestTokenMatching(unittest.TestCase):
    """Locked names match whole canonical package tokens, not substrings."""

    DOCLING_LINE = (
        "docling-ibm-models 3.14.0 requires torchvision, "
        "which is not installed."
    )

    def test_the_docling_incident_line_is_a_warning(self):
        """A line naming only unlocked packages warns."""
        failing, warn = classify_conflicts([self.DOCLING_LINE], LOCKED)
        self.assertEqual(failing, [])
        self.assertEqual(warn, [self.DOCLING_LINE])

    def test_torchvision_does_not_match_locked_torch(self):
        """torchvision does not match a lock that names only torch."""
        line = "somepkg 1.0 requires torchvision, which is not installed."
        failing, warn = classify_conflicts([line], ["torch"])
        self.assertEqual(failing, [])
        self.assertEqual(warn, [line])

    def test_sentence_transformers_does_not_match_locked_transformers(self):
        """A locked name as a prefix of a longer token does not match."""
        line = ("sentence-transformers 5.0.0 requires transformers-fork, "
                "which is not installed.")
        failing, warn = classify_conflicts([line], ["transformers"])
        self.assertEqual(failing, [])
        self.assertEqual(warn, [line])

    def test_a_locked_name_as_a_whole_token_still_fails(self):
        """A locked name appearing as a whole token still fails."""
        failing, _ = classify_conflicts([TORCHVISION_LINE], ["torch"])
        self.assertEqual(failing, [TORCHVISION_LINE])

    def test_hyphen_underscore_and_case_are_folded(self):
        line = "otherpkg 1.0 requires rank_bm25==0.2.2, but you have rank-bm25 0.1.0."
        failing, _ = classify_conflicts([line], ["Rank-BM25"])
        self.assertEqual(failing, [line])


class TestSubprocessSmoke(unittest.TestCase):
    def test_pip_check_runs_and_returns_a_list(self):
        """The helper returns a list of strings in any environment."""
        out = _pip_check_conflicts()
        self.assertIsInstance(out, list)
        for line in out:
            self.assertIsInstance(line, str)


class TestGateIntegration(unittest.TestCase):
    def test_a_snapshot_lockfile_still_passes_on_a_clean_env(self):
        """check_lockfile returns 0 on a snapshot of a clean environment."""
        import tempfile

        from scripts.pin_environment import check_lockfile, write_lockfile

        # Snapshot this environment, then run the full gate against it.
        with tempfile.TemporaryDirectory() as td:
            lock = Path(td) / "requirements.lock"
            write_lockfile(lock)
            self.assertEqual(check_lockfile(lock), 0)


if __name__ == "__main__":
    unittest.main()
