"""Phase times must PARTITION the build, not overlap it.

THE DEFECT. `_gmm_cluster` is decorated `@_timed("gmm_final_fit")` and
calls `_get_optimal_clusters`, decorated `@_timed("gmm_bic_sweep")`. The
sweep's wall time therefore accrued to BOTH buckets, so the phases summed
to more than the build they were measuring: nqa_largest reported
777+254+248+84+2 = 1,366 s against a 1,150 s build (+19%), MultiHop
3,199 s against 2,852 s (+12%). The overshoot scaled with corpus size
because the BIC sweep does.

Wall clock was never affected — the tree took what it took. But phase
attribution is now used to make decisions (it is what retired UMAP as a
suspect and what surfaced GMM as the second cost), and a decision made on
double-counted time is a decision made on a wrong number.

THE INVARIANT: a parent phase records its own time only. Child time
belongs to the child. The sum over phases is then bounded by the build,
which is a property a probe can assert instead of a reader eyeballing.
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
        _parent(0.05, 0.05)
        parent = _CLOCK.seconds["parent"]
        child = _CLOCK.seconds["child"]
        self.assertGreater(child, 0.03)
        # Parent kept only its own ~0.05 s, not the full ~0.10 s.
        self.assertLess(parent, child * 1.8)

    def test_the_sum_does_not_exceed_the_wall_time(self):
        """The property the build violated by 19%."""
        t0 = time.perf_counter()
        _parent(0.05, 0.05)
        wall = time.perf_counter() - t0
        total = sum(_CLOCK.seconds.values())
        self.assertLessEqual(total, wall * 1.05)

    def test_three_deep_nesting_still_partitions(self):
        t0 = time.perf_counter()
        _outer(0.03)
        wall = time.perf_counter() - t0
        self.assertLessEqual(sum(_CLOCK.seconds.values()), wall * 1.05)

    def test_siblings_still_add_normally(self):
        _child(0.02)
        _child(0.02)
        self.assertEqual(_CLOCK.calls["child"], 2)
        self.assertGreater(_CLOCK.seconds["child"], 0.03)

    def test_an_exception_still_closes_the_frame(self):
        """A build that raises mid-phase must not leave the stack dirty
        and mis-attribute every later phase to it."""

        @_timed("boom")
        def _boom():
            raise RuntimeError("x")

        with self.assertRaises(RuntimeError):
            _boom()
        _child(0.01)
        self.assertIn("child", _CLOCK.seconds)
        self.assertIn("boom", _CLOCK.seconds)

    def test_stats_report_the_partition_invariant(self):
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
