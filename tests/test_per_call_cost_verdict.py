"""The verdict table for the 230 s/call investigation.

Pure classification over measured per-step costs, so the decision rule is
testable on a host with no GPU and the probe script carries no judgement
of its own. The measurements themselves are operator-executed.

WHY A VERDICT FUNCTION AT ALL. Four variants produce four numbers, and
the mapping from those numbers to a cause is exactly the step where a
tired reader guesses. Writing it down means the probe states a conclusion
that can be argued with, rather than a table that can be misread.
"""

from __future__ import annotations

import unittest

from scripts.probe_generate_call_isolation import (
    HEALTHY_S_PER_STEP,
    classify_per_call_cost,
)

FAST = HEALTHY_S_PER_STEP * 1.2      # ~0.079 s/step
SLOW = 2.3                            # what the build measures


class TestVerdict(unittest.TestCase):
    def test_fast_default_means_the_build_context_is_the_trigger(self):
        """If an isolated call is healthy, the cost is not in the
        generation path — something the BUILD does to the process is."""
        v = classify_per_call_cost({"default": FAST})
        self.assertEqual(v["verdict"], "NOT_REPRODUCED")
        self.assertIn("build", v["explanation"].lower())

    def test_placement_named_when_forcing_all_layers_on_gpu_fixes_it(self):
        v = classify_per_call_cost(
            {"default": SLOW, "all_on_gpu": FAST,
             "use_cache_true": SLOW, "use_cache_false": SLOW}
        )
        self.assertEqual(v["verdict"], "PLACEMENT")

    def test_kv_cache_named_when_forcing_it_on_fixes_it(self):
        v = classify_per_call_cost(
            {"default": SLOW, "use_cache_true": FAST, "all_on_gpu": SLOW}
        )
        self.assertEqual(v["verdict"], "KV_CACHE_DEFAULT")

    def test_everything_slow_is_intrinsic_and_says_so(self):
        """The uncomfortable outcome must be reportable. A probe that can
        only confirm hypotheses is not a measurement."""
        v = classify_per_call_cost(
            {"default": SLOW, "all_on_gpu": SLOW,
             "use_cache_true": SLOW, "use_cache_false": SLOW}
        )
        self.assertEqual(v["verdict"], "INTRINSIC")

    def test_placement_wins_when_both_placement_and_cache_look_fixed(self):
        """Forcing all layers onto the GPU also reloads the model, so it
        can mask a cache effect. Placement is reported as the primary
        with the ambiguity NAMED rather than silently resolved."""
        v = classify_per_call_cost(
            {"default": SLOW, "all_on_gpu": FAST, "use_cache_true": FAST}
        )
        self.assertEqual(v["verdict"], "PLACEMENT")
        self.assertIn("ambiguous", v["explanation"].lower())

    def test_ratios_against_the_healthy_baseline_are_reported(self):
        """Rounded to 2dp: this is a report, and a ratio quoted to twelve
        decimal places implies a precision the timings do not have."""
        v = classify_per_call_cost({"default": 2.3})
        self.assertEqual(v["ratios"]["default"], 34.85)

    def test_missing_default_is_an_error_not_a_verdict(self):
        with self.assertRaises(ValueError):
            classify_per_call_cost({"all_on_gpu": FAST})

    def test_a_none_measurement_is_carried_not_silently_dropped(self):
        """A variant that OOMed or was skipped must not read as fast."""
        v = classify_per_call_cost({"default": SLOW, "all_on_gpu": None})
        self.assertIsNone(v["ratios"]["all_on_gpu"])
        self.assertEqual(v["verdict"], "INCONCLUSIVE")

    def test_no_cache_reference_is_reported_against_the_flatness_argument(self):
        """use_cache=False was already ruled out by the flatness of the
        build's own call table; measuring it here is a CONTROL, and the
        verdict must not name it as the cause on its own."""
        v = classify_per_call_cost(
            {"default": SLOW, "use_cache_false": SLOW * 1.01,
             "use_cache_true": SLOW, "all_on_gpu": SLOW}
        )
        self.assertEqual(v["verdict"], "INTRINSIC")


if __name__ == "__main__":
    unittest.main()
