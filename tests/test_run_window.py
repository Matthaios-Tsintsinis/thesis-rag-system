"""The --from-summary run window: anchored at the run's START.

WHY THIS EXISTS. The first `_run_window` subtracted `elapsed_s` from the
summary timestamp, reading "written at the end of the run" as "marks the
end of the run". The runner actually captures `stamp` at ENTRY (for output
naming) and only writes it later, so the window landed one run-length too
early -- caught on cell 2's real manifests, which matched only because the
30-minute pad happened to cover a short run's worth of drift. These tests
pin the corrected semantics against a fixture whose numbers make the wrong
anchoring fail loudly:

  a manifest created at start + elapsed - epsilon (late in the run, where
  most of a cell's builds land) MUST match; under the old end-anchored
  window it sat ~elapsed past the upper edge and could not.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_children_per_parent import _created_in, _run_window

START = datetime(2026, 8, 18, 23, 9, 13, tzinfo=timezone.utc)
ELAPSED = 11109.6  # cell 2's real wall clock, ~3.09 h
PAD = timedelta(seconds=1800)


def _summary_file(tmpdir: str) -> Path:
    p = Path(tmpdir) / "cell.summary.json"
    p.write_text(json.dumps({
        "timestamp": START.strftime("%Y%m%d-%H%M%S"),
        "elapsed_s": ELAPSED,
    }))
    return p


def _manifest(created: datetime) -> dict:
    return {"created_at": created.isoformat()}


class TestWindowAnchoring(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.window = _run_window(_summary_file(self._td.name))

    def tearDown(self):
        self._td.cleanup()

    def test_window_spans_start_minus_pad_to_end_plus_pad(self):
        self.assertEqual(self.window[0], START - PAD)
        self.assertEqual(
            self.window[1], START + timedelta(seconds=ELAPSED) + PAD
        )

    def test_a_manifest_late_in_the_run_matches(self):
        """start + elapsed - epsilon: the case the end-anchored window
        rejected. Most of a cell's builds are created here."""
        late = START + timedelta(seconds=ELAPSED - 60)
        self.assertTrue(_created_in(_manifest(late), self.window))

    def test_a_manifest_at_the_first_build_matches(self):
        """Cell 2's first manifest landed ~4 minutes after the stamp --
        the observation that exposed the bug."""
        first = START + timedelta(minutes=4)
        self.assertTrue(_created_in(_manifest(first), self.window))

    def test_the_old_anchoring_would_have_failed_the_late_manifest(self):
        """Documents the defect: under end-anchored [ts - elapsed - pad,
        ts + pad], a build at ts + elapsed - 60s sits far past the edge."""
        old_upper = START + PAD
        late = START + timedelta(seconds=ELAPSED - 60)
        self.assertGreater(late, old_upper)

    def test_a_probe_build_days_earlier_is_excluded(self):
        probe = START - timedelta(days=3)
        self.assertFalse(_created_in(_manifest(probe), self.window))

    def test_a_build_just_past_the_pad_is_excluded_both_sides(self):
        before = START - PAD - timedelta(seconds=1)
        after = START + timedelta(seconds=ELAPSED) + PAD + timedelta(seconds=1)
        self.assertFalse(_created_in(_manifest(before), self.window))
        self.assertFalse(_created_in(_manifest(after), self.window))

    def test_naive_manifest_timestamps_are_read_as_utc(self):
        naive = (START + timedelta(minutes=10)).replace(tzinfo=None)
        self.assertTrue(_created_in({"created_at": naive.isoformat()},
                                    self.window))

    def test_missing_fields_refuse_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.summary.json"
            p.write_text(json.dumps({"timestamp": "20260818-230913"}))
            with self.assertRaises(SystemExit):
                _run_window(p)


if __name__ == "__main__":
    unittest.main()
