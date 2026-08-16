"""Per-call breakdown of `model.generate()`, so 230 s becomes arithmetic.

WHAT THIS EXISTS TO SETTLE. A tree build spends 96% of its time in
`summarize`, at ~230 s per `generate()` call, and that figure is CONSTANT
across a threefold range of batch width (8.5 -> 2.92). Constant per call
means the cost is neither per-token nor per-sequence, and falling VRAM
with no speedup rules out allocator pressure. What is left is a fixed
per-call cost that the wall-clock total cannot distinguish between:
prefill of a very long padded prompt, a decode loop running far more
steps than the 100-token cap implies, or CPU-side tokenise/decode work
outside generation entirely.

One number cannot separate those. Three timed phases plus the shapes can:
`s_per_decode_step` against the healthy 66 ms baseline (6.6 s / 100 steps,
one ~2,000-token prompt, fp16 resident on an L4) says whether generation
itself is slow, and `new_tokens` says whether the loop is even running the
number of steps it claims.

SCOPE, stated as `test_generate_batch.py` states it: the torch path
cannot run on a host without torch, so what is tested here is the record
structure, the accumulation, and the anti-enumeration property. The
timings themselves are first exercised on Colab.
"""

from __future__ import annotations

import unittest

from src.models import (
    GENERATE_CALLS,
    generate_calls_summary,
    record_generate_call,
    reset_generate_calls,
)


class TestRecordGenerateCall(unittest.TestCase):
    def setUp(self):
        reset_generate_calls()

    def test_record_carries_every_phase_and_shape(self):
        record_generate_call(
            width=8,
            prompt_tokens_padded=3500,
            new_tokens=100,
            max_new_tokens=100,
            tokenise_s=0.4,
            generate_s=229.0,
            decode_s=0.6,
        )
        (call,) = GENERATE_CALLS["calls"]
        self.assertEqual(call["call_no"], 1)
        self.assertEqual(call["width"], 8)
        self.assertEqual(call["prompt_tokens_padded"], 3500)
        self.assertEqual(call["new_tokens"], 100)
        self.assertEqual(call["max_new_tokens"], 100)
        self.assertEqual(call["tokenise_s"], 0.4)
        self.assertEqual(call["generate_s"], 229.0)
        self.assertEqual(call["decode_s"], 0.6)

    def test_derived_figures_are_the_whole_point(self):
        """`s_per_decode_step` is what gets compared to the 66 ms
        baseline; `padded_input_cells` is what the cap actually bounds."""
        record_generate_call(
            width=8, prompt_tokens_padded=1000, new_tokens=100,
            max_new_tokens=100, tokenise_s=0.5, generate_s=230.0,
            decode_s=0.5,
        )
        (call,) = GENERATE_CALLS["calls"]
        self.assertAlmostEqual(call["s_per_decode_step"], 2.3, places=6)
        self.assertAlmostEqual(call["total_s"], 231.0, places=6)
        self.assertEqual(call["padded_input_cells"], 8000)

    def test_zero_new_tokens_does_not_divide_by_zero(self):
        """A call that emitted nothing is a real outcome, not a crash."""
        record_generate_call(
            width=2, prompt_tokens_padded=100, new_tokens=0,
            max_new_tokens=100, tokenise_s=0.1, generate_s=1.0, decode_s=0.1,
        )
        (call,) = GENERATE_CALLS["calls"]
        self.assertIsNone(call["s_per_decode_step"])

    def test_calls_accumulate_in_order(self):
        for w in (4, 8, 2):
            record_generate_call(
                width=w, prompt_tokens_padded=10, new_tokens=1,
                max_new_tokens=100, tokenise_s=0.0, generate_s=1.0,
                decode_s=0.0,
            )
        self.assertEqual([c["width"] for c in GENERATE_CALLS["calls"]],
                         [4, 8, 2])
        self.assertEqual([c["call_no"] for c in GENERATE_CALLS["calls"]],
                         [1, 2, 3])

    def test_legacy_counters_still_move(self):
        """`n_calls` and `widths` are read by existing consumers; the
        richer record is additive, not a replacement."""
        record_generate_call(
            width=5, prompt_tokens_padded=10, new_tokens=1,
            max_new_tokens=100, tokenise_s=0.0, generate_s=1.0, decode_s=0.0,
        )
        self.assertEqual(GENERATE_CALLS["n_calls"], 1)
        self.assertEqual(GENERATE_CALLS["widths"], [5])

    def test_reset_clears_the_records(self):
        record_generate_call(
            width=1, prompt_tokens_padded=1, new_tokens=1, max_new_tokens=1,
            tokenise_s=0.0, generate_s=0.0, decode_s=0.0,
        )
        reset_generate_calls()
        self.assertEqual(GENERATE_CALLS, {})


class TestSummaryCarriesWholeStructures(unittest.TestCase):
    """The bug this guards has already cost two cold builds.

    `phase_seconds` and `generate_calls` were written correctly, surfaced
    correctly, and dropped by a hand-picked field list three lines before
    the JSON. The tree-stats block enumerated four keys of GENERATE_CALLS
    by name, so anything added to it later would vanish the same way.
    """

    def setUp(self):
        reset_generate_calls()

    def test_summary_reports_the_aggregates_consumers_already_read(self):
        for w in (4, 8):
            record_generate_call(
                width=w, prompt_tokens_padded=10, new_tokens=1,
                max_new_tokens=100, tokenise_s=0.0, generate_s=1.0,
                decode_s=0.0,
            )
        s = generate_calls_summary()
        self.assertEqual(s["n_calls"], 2)
        self.assertEqual(s["mean_width"], 6.0)
        self.assertEqual(s["max_width"], 8)
        self.assertEqual(s["min_width"], 4)

    def test_summary_carries_the_per_call_records(self):
        record_generate_call(
            width=3, prompt_tokens_padded=99, new_tokens=7,
            max_new_tokens=100, tokenise_s=0.1, generate_s=2.0, decode_s=0.3,
        )
        s = generate_calls_summary()
        self.assertEqual(len(s["calls"]), 1)
        self.assertEqual(s["calls"][0]["prompt_tokens_padded"], 99)

    def test_a_field_added_later_survives_into_the_summary(self):
        """THE ANTI-ENUMERATION PROPERTY, asserted rather than trusted.
        A key nothing knows about must still arrive at the consumer."""
        record_generate_call(
            width=1, prompt_tokens_padded=1, new_tokens=1, max_new_tokens=1,
            tokenise_s=0.0, generate_s=0.0, decode_s=0.0,
        )
        GENERATE_CALLS["invented_later"] = {"nested": 1}
        self.assertEqual(
            generate_calls_summary()["invented_later"], {"nested": 1}
        )

    def test_empty_summary_is_reported_not_faked(self):
        s = generate_calls_summary()
        self.assertEqual(s["n_calls"], 0)
        self.assertIsNone(s["mean_width"])
        self.assertEqual(s["calls"], [])


if __name__ == "__main__":
    unittest.main()
