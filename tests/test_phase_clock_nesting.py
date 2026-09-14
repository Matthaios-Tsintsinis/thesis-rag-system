"""Pins that nested _timed phases partition the build: a parent phase
records only its own time, so the phase sum never exceeds the wall clock.
"""

from __future__ import annotations

import time
import unittest

from src.raptor_paper import _CLOCK, _timed


@_timed("child")
def _child(dt: float) -> None:
    time.sleep(dt)


@_timed("parent")
def _parent(child_dt: float, own_dt: float) -> None:
    _child(child_dt)
    time.sleep(own_dt)


@_timed("outer")
def _outer(dt: float) -> None:
    _parent(dt, dt)


class TestNestedPhasesDoNotDoubleCount(unittest.TestCase):
    def setUp(self):
        _CLOCK.reset()

    def tearDown(self):
        _CLOCK.reset()

    def test_parent_excludes_its_child(self):
        """A parent phase records its own time, not its child's."""
        _parent(0.05, 0.05)
        parent = _CLOCK.seconds["parent"]
        child = _CLOCK.seconds["child"]
        self.assertGreater(child, 0.03)
        # Parent keeps only its own ~0.05 s, not the full ~0.10 s.
        self.assertLess(parent, child * 1.8)

    def test_the_sum_does_not_exceed_the_wall_time(self):
        """The sum over phases stays within the measured wall time."""
        t0 = time.perf_counter()
        _parent(0.05, 0.05)
        wall = time.perf_counter() - t0
        total = sum(_CLOCK.seconds.values())
        self.assertLessEqual(total, wall * 1.05)

    def test_three_deep_nesting_still_partitions(self):
        """Three levels of nesting still sum to at most the wall time."""
        t0 = time.perf_counter()
        _outer(0.03)
        wall = time.perf_counter() - t0
        self.assertLessEqual(sum(_CLOCK.seconds.values()), wall * 1.05)

    def test_siblings_still_add_normally(self):
        """Sibling calls of one phase accumulate calls and seconds."""
        _child(0.02)
        _child(0.02)
        self.assertEqual(_CLOCK.calls["child"], 2)
        self.assertGreater(_CLOCK.seconds["child"], 0.03)

    def test_an_exception_still_closes_the_frame(self):
        """A phase that raises still pops its frame; later phases are clean."""

        @_timed("boom")
        def _boom():
            raise RuntimeError("x")

        with self.assertRaises(RuntimeError):
            _boom()
        _child(0.01)
        self.assertIn("child", _CLOCK.seconds)
        self.assertIn("boom", _CLOCK.seconds)

    def test_stats_report_the_partition_invariant(self):
        """as_stats reports phase_measured_total_s as the rounded phase sum."""
        _parent(0.03, 0.03)
        stats = _CLOCK.as_stats()
        self.assertIn("phase_measured_total_s", stats)
        self.assertAlmostEqual(
            stats["phase_measured_total_s"],
            round(sum(_CLOCK.seconds.values()), 2),
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
