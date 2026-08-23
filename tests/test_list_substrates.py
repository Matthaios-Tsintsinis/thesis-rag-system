"""The cp313-window screen's filter (scripts/list_substrates_by_date).

Three-valued on purpose: in-window, out-of-window, and UNDATED — an
undated manifest cannot prove it predates the window, and collapsing it
into either bucket would let a suspect substrate pass as clean or condemn
a clean one silently.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.list_substrates_by_date import in_window, parse_when

AFTER = datetime(2026, 8, 19, 0, 0, tzinfo=timezone.utc)
BEFORE = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


class TestWindowFilter(unittest.TestCase):
    def test_inside_the_window_is_flagged(self):
        self.assertTrue(in_window("2026-08-20T05:30:00+00:00", AFTER, BEFORE))

    def test_before_and_after_are_clean(self):
        self.assertFalse(in_window("2026-08-18T23:59:59+00:00", AFTER, BEFORE))
        self.assertFalse(in_window("2026-08-23T12:00:01+00:00", AFTER, BEFORE))

    def test_boundaries_are_inclusive(self):
        """A build at the exact boundary is IN — the window edges are the
        operator's uncertainty margins, so the screen errs toward flagging."""
        self.assertTrue(in_window("2026-08-19T00:00:00+00:00", AFTER, BEFORE))
        self.assertTrue(in_window("2026-08-23T12:00:00+00:00", AFTER, BEFORE))

    def test_naive_timestamps_read_as_utc(self):
        self.assertTrue(in_window("2026-08-20T05:30:00", AFTER, BEFORE))

    def test_undated_is_neither_clean_nor_flagged(self):
        self.assertIsNone(in_window(None, AFTER, BEFORE))
        self.assertIsNone(in_window("not-a-date", AFTER, BEFORE))

    def test_parse_when_rejects_garbage(self):
        self.assertIsNone(parse_when("yesterday"))
        self.assertIsNone(parse_when(""))


if __name__ == "__main__":
    unittest.main()
