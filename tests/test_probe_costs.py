"""The cost probe must refuse rather than report a contaminated number.

Four cost forecasts in this project have been wrong, and two of those
because a probe silently did not measure what it claimed:

  * a synthetic benchmark run at a batch size that OOMs on real prompts;
  * a 1-token probe whose cap never reached generation -- it printed the
    cap, generated full-length answers, and its timings additionally
    included a 15GB model download.

Both produced numbers rather than errors, and in both cases the user
caught it rather than the code. These tests pin the preconditions, so a
probe that cannot measure says so.

Stubbed at the subprocess boundary: the subject is the probe's refusal
logic, not the runner it shells out to (which has its own end-to-end
tests in test_cli_entrypoints.py).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import probe_costs as P


HEALTHY = {
    "s_per_query": 1.0,
    "prewarm_load_s": 10.0,
    "batch_size": 8,
    "mean_answer_score": 0.2,
    "n_queries_scored": 2,
}

TWO_ROWS = [
    {"query_id": "a", "predicted_answer": "x"},
    {"query_id": "b", "predicted_answer": "y"},
]


class _ProbeCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _write(self, tag, summary=None, rows=None):
        if rows is not None:
            (self.dir / f"{tag}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )
        if summary is not None:
            (self.dir / f"{tag}.summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

    @staticmethod
    def _subprocess(returncode=0):
        return mock.patch.object(
            P.subprocess, "run",
            lambda *a, **k: mock.Mock(returncode=returncode),
        )


class TestRunPreconditions(_ProbeCase):
    def test_a_crashed_runner_is_not_a_measurement(self):
        self._write("t", summary=HEALTHY)
        with self._subprocess(returncode=1):
            with self.assertRaises(P.ProbeFailure):
                P._run(self.dir, "t")

    def test_a_missing_summary_is_refused(self):
        with self._subprocess():
            with self.assertRaises(P.ProbeFailure):
                P._run(self.dir, "never_written")

    def test_timings_without_prewarm_are_refused(self):
        """Model load inside the timing is the exact contamination that
        made M1 look slower than M2 in the first 1-token probe."""
        self._write("t", summary={**HEALTHY, "prewarm_load_s": None})
        with self._subprocess():
            with self.assertRaises(P.ProbeFailure) as cm:
                P._run(self.dir, "t")
        self.assertIn("prewarm", str(cm.exception))

    def test_a_build_without_timing_fields_is_refused(self):
        """Guards against running the probe against an older checkout
        and hand-timing the result instead."""
        self._write("t", summary={**HEALTHY, "s_per_query": None})
        with self._subprocess():
            with self.assertRaises(P.ProbeFailure):
                P._run(self.dir, "t")

    def test_a_healthy_summary_is_accepted(self):
        self._write("t", summary=HEALTHY)
        with self._subprocess():
            self.assertEqual(P._run(self.dir, "t")["s_per_query"], 1.0)


class TestM1Comparison(_ProbeCase):
    def test_identical_timings_mean_the_batched_path_never_ran(self):
        """If supports_batched_answer were False the runner would fall
        back to sequential and both numbers would match — a 1.0x
        "result" that is really a measurement of nothing."""
        self._write("m1_sequential", rows=TWO_ROWS)
        self._write("m1_batched", rows=TWO_ROWS)
        with mock.patch.object(P, "_run", lambda *a, **k: dict(HEALTHY)):
            with self.assertRaises(P.ProbeFailure) as cm:
                P.probe_m1(self.dir, 2, 8, 20000)
        self.assertIn("did not run", str(cm.exception))

    def test_different_query_sets_make_the_ratio_meaningless(self):
        self._write("m1_sequential", rows=TWO_ROWS)
        self._write("m1_batched", rows=TWO_ROWS[:1])
        it = iter([{**HEALTHY, "s_per_query": 4.0},
                   {**HEALTHY, "s_per_query": 1.0}])
        with mock.patch.object(P, "_run", lambda *a, **k: next(it)):
            with self.assertRaises(P.ProbeFailure):
                P.probe_m1(self.dir, 2, 8, 20000)

    def test_a_short_pass_is_refused(self):
        self._write("m1_sequential", rows=TWO_ROWS)
        self._write("m1_batched", rows=TWO_ROWS)
        it = iter([{**HEALTHY, "s_per_query": 4.0},
                   {**HEALTHY, "s_per_query": 1.0}])
        with mock.patch.object(P, "_run", lambda *a, **k: next(it)):
            with self.assertRaises(P.ProbeFailure):
                P.probe_m1(self.dir, 99, 8, 20000)  # asked for 99, got 2

    def test_healthy_comparison_reports_speedup_and_answer_drift(self):
        self._write("m1_sequential", rows=TWO_ROWS)
        self._write("m1_batched", rows=[
            TWO_ROWS[0], {"query_id": "b", "predicted_answer": "CHANGED"},
        ])
        it = iter([{**HEALTHY, "s_per_query": 4.0},
                   {**HEALTHY, "s_per_query": 1.0}])
        with mock.patch.object(P, "_run", lambda *a, **k: next(it)):
            r = P.probe_m1(self.dir, 2, 8, 20000)
        self.assertEqual(r["speedup"], 4.0)
        # Reported, never asserted: batch composition can legitimately
        # move argmax on near-ties at temperature 0.
        self.assertEqual(r["answers_changed"], 1)
        self.assertEqual(r["answers_changed_pct"], 50.0)


class TestBuildProbe(_ProbeCase):
    def test_a_wrong_number_of_index_lines_is_refused(self):
        """Parsing fewer index_s values than units means the average is
        over something other than what the report claims."""
        stdout = "  index_s=1.00\n  index_s=2.00\n"
        with mock.patch.object(
            P.subprocess, "run",
            lambda *a, **k: mock.Mock(returncode=0, stdout=stdout, stderr=""),
        ):
            with self.assertRaises(P.ProbeFailure) as cm:
                P.probe_build(self.dir, units=20)
        self.assertIn("index_s", str(cm.exception))

    def test_healthy_build_probe_reports_the_distribution(self):
        stdout = "".join(f"  index_s={i}.00\n" for i in range(1, 6))
        with mock.patch.object(
            P.subprocess, "run",
            lambda *a, **k: mock.Mock(returncode=0, stdout=stdout, stderr=""),
        ):
            r = P.probe_build(self.dir, units=5)
        for system in ("M2", "M4"):
            self.assertEqual(r[system]["n_units"], 5)
            self.assertEqual(r[system]["mean_index_s"], 3.0)
            self.assertEqual(r[system]["max_index_s"], 5.0)


if __name__ == "__main__":
    unittest.main()
