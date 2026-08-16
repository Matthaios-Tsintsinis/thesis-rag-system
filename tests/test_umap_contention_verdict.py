"""Does running UMAP poison generation in the same process?

THE HYPOTHESIS THIS DECIDES. If the snapshot build returns healthy
`free_gb`, memory pressure dies with everything else and the remaining
candidate is CPU-side: numba spawns a worker pool for UMAP, `n_jobs=1`
under a seed constrains parallelism but need not tear down pools already
created, and contended CPU threads starve the Python-side loop between
CUDA kernel launches. That produces a flat per-call cost, independent of
batch shape, invisible to every GPU metric, and absent from a process
that never ran UMAP — which is the full observed signature including the
33x isolated-vs-build gap.

THE CONTROL THAT MAKES IT A MEASUREMENT. Two timed calls run BEFORE any
UMAP, not one. A first-call warmup effect would otherwise read as a UMAP
effect, and the difference between "the second call is always faster" and
"the post-UMAP call is slower" is the whole result. If the two baseline
calls disagree, the probe reports INCONCLUSIVE rather than a verdict off
an unstable floor.
"""

from __future__ import annotations

import unittest

from scripts.probe_umap_contention import (
    BASELINE_DRIFT_TOLERANCE,
    CONTENTION_FACTOR,
    classify_umap_contention,
)

FAST = 0.069        # the isolated healthy call
SLOW = 2.28         # what a build call costs


class TestVerdict(unittest.TestCase):
    def test_post_umap_slowdown_is_the_finding(self):
        v = classify_umap_contention(
            pre_umap=[FAST, FAST], post_umap=SLOW, post_teardown=None
        )
        self.assertEqual(v["verdict"], "UMAP_CONTENTION")
        self.assertGreater(v["slowdown_factor"], CONTENTION_FACTOR)

    def test_no_slowdown_clears_umap(self):
        v = classify_umap_contention(
            pre_umap=[FAST, FAST], post_umap=FAST * 1.05, post_teardown=None
        )
        self.assertEqual(v["verdict"], "NOT_UMAP")

    def test_teardown_that_restores_speed_names_the_fix(self):
        v = classify_umap_contention(
            pre_umap=[FAST, FAST], post_umap=SLOW, post_teardown=FAST * 1.1
        )
        self.assertEqual(v["verdict"], "UMAP_CONTENTION_FIXED_BY_TEARDOWN")

    def test_teardown_that_does_not_help_says_so(self):
        """Confirming the cause while the obvious remedy fails is a real
        outcome and must be reportable, not collapsed into success."""
        v = classify_umap_contention(
            pre_umap=[FAST, FAST], post_umap=SLOW, post_teardown=SLOW * 0.98
        )
        self.assertEqual(v["verdict"], "UMAP_CONTENTION_TEARDOWN_INEFFECTIVE")

    def test_unstable_baseline_refuses_to_conclude(self):
        """A floor that moves by more than the tolerance cannot support a
        ratio. Reported rather than averaged away."""
        v = classify_umap_contention(
            pre_umap=[FAST, FAST * 3], post_umap=SLOW, post_teardown=None
        )
        self.assertEqual(v["verdict"], "INCONCLUSIVE_UNSTABLE_BASELINE")

    def test_baseline_drift_within_tolerance_is_accepted(self):
        drift = 1 + (BASELINE_DRIFT_TOLERANCE / 2)
        v = classify_umap_contention(
            pre_umap=[FAST, FAST * drift], post_umap=SLOW, post_teardown=None
        )
        self.assertEqual(v["verdict"], "UMAP_CONTENTION")

    def test_the_slower_baseline_call_is_the_floor(self):
        """Comparing against the FASTER of two baselines would inflate
        the slowdown. The conservative choice is the slower one."""
        v = classify_umap_contention(
            pre_umap=[0.069, 0.072], post_umap=2.28, post_teardown=None
        )
        self.assertAlmostEqual(v["baseline_s_per_step"], 0.072)
        self.assertAlmostEqual(v["slowdown_factor"], round(2.28 / 0.072, 2))

    def test_missing_baseline_is_an_error(self):
        with self.assertRaises(ValueError):
            classify_umap_contention(
                pre_umap=[], post_umap=SLOW, post_teardown=None
            )

    def test_a_failed_post_umap_call_is_inconclusive_not_fast(self):
        v = classify_umap_contention(
            pre_umap=[FAST, FAST], post_umap=None, post_teardown=None
        )
        self.assertEqual(v["verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
