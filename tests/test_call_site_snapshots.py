"""Placement and VRAM captured AT THE MOMENT OF THE CALL, not around it.

WHY THIS EXISTS. An isolated `generate()` at the build's own call-3 shape
runs in 6.87 s; inside a build the same shape takes 228 s. Placement is
clean in isolation, the KV cache is on and healthy, and nothing in the
build path releases or reloads the generator — so the generation path is
exonerated and the BUILD CONTEXT is the trigger. The one measured
difference is headroom: 14.6 GB peak with 7.58 GB free in isolation
against 21.51 GB peak with ~0.5 GB free in a build.

Every reading so far was taken BEFORE or AFTER a build. These snapshots
fire immediately before `model.generate()`, inside the call that is slow.

SCOPE. The snapshot helpers are torch-dependent, and BOTH sides of that
contract are pinned by stubbing torch rather than by relying on the host:
without torch or CUDA the helper reports absence (None, never a raise
that would take the probe down); with CUDA it reports the measured
numbers. The first version asserted None against the REAL helper, which
held only on the torch-less agent host and failed on the run host with
`0.0 is not None` on an idle L4 (2026-09-04) — a test that encoded an
accident of where it ran.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from src.models import (
    GENERATE_CALLS,
    GENERATOR_LOADS,
    cuda_memory_snapshot,
    model_placement_snapshot,
    record_generate_call,
    reset_generate_calls,
)


def _fake_torch(*, cuda_available: bool) -> types.SimpleNamespace:
    """A torch stand-in exposing exactly what cuda_memory_snapshot reads."""
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        memory_allocated=lambda: 3 * 2**30,
        memory_reserved=lambda: 4 * 2**30,
        mem_get_info=lambda: (18 * 2**30, 24 * 2**30),
    )
    return types.SimpleNamespace(cuda=cuda)


class TestSnapshotsDegradeHonestly(unittest.TestCase):
    """Absence is reported, never fabricated; presence is measured.

    Simulated on every host: `sys.modules["torch"] = None` makes
    `import torch` raise (the agent host), a stub module with a CUDA
    device stands in for the run host. Neither case depends on what the
    machine running the suite actually has."""

    def test_memory_snapshot_reports_none_without_torch(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            snap = cuda_memory_snapshot()
        self.assertIsNone(snap["allocated_gb"])
        self.assertIsNone(snap["reserved_gb"])
        self.assertIsNone(snap["free_gb"])
        self.assertFalse(snap["cuda_available"])

    def test_memory_snapshot_reports_none_with_torch_but_no_cuda(self):
        with mock.patch.dict(sys.modules,
                             {"torch": _fake_torch(cuda_available=False)}):
            snap = cuda_memory_snapshot()
        self.assertIsNone(snap["allocated_gb"])
        self.assertFalse(snap["cuda_available"])

    def test_memory_snapshot_reports_the_measured_numbers_with_cuda(self):
        """The run-host case: an idle L4 reads 0.0 allocated, which is a
        measurement, not an absence — the case the old test refused."""
        with mock.patch.dict(sys.modules,
                             {"torch": _fake_torch(cuda_available=True)}):
            snap = cuda_memory_snapshot()
        self.assertTrue(snap["cuda_available"])
        self.assertEqual(snap["allocated_gb"], 3.0)
        self.assertEqual(snap["reserved_gb"], 4.0)
        self.assertEqual(snap["reserved_minus_allocated_gb"], 1.0)
        self.assertEqual(snap["free_gb"], 18.0)
        self.assertEqual(snap["total_gb"], 24.0)

    def test_memory_snapshot_still_has_every_key(self):
        """A missing key would read as a probe bug at 3 a.m. on Colab;
        an explicit None reads as what it is."""
        snap = cuda_memory_snapshot()
        for k in ("allocated_gb", "reserved_gb", "free_gb", "total_gb",
                  "reserved_minus_allocated_gb", "cuda_available"):
            self.assertIn(k, snap)

    def test_placement_snapshot_survives_a_model_without_parameters(self):
        class Bare:
            config = type("C", (), {"_attn_implementation": "sdpa"})()

            def named_parameters(self):
                return iter(())

        snap = model_placement_snapshot(Bare())
        self.assertIsNone(snap["fraction_params_off_gpu"])
        self.assertEqual(snap["param_tensors_by_device"], {})

    def test_placement_snapshot_counts_devices_and_computes_the_fraction(self):
        class P:
            def __init__(self, device, n):
                self.device = device
                self._n = n

            def numel(self):
                return self._n

        class Fake:
            config = type("C", (), {"_attn_implementation": "eager"})()

            def named_parameters(self):
                return iter([
                    ("a", P("cuda:0", 60)),
                    ("b", P("cuda:0", 20)),
                    ("c", P("cpu", 20)),
                ])

        snap = model_placement_snapshot(Fake())
        self.assertEqual(snap["param_tensors_by_device"],
                         {"cuda:0": 2, "cpu": 1})
        self.assertAlmostEqual(snap["fraction_params_off_gpu"], 0.2)
        self.assertEqual(snap["attn_implementation"], "eager")


class TestCallRecordCarriesTheSnapshots(unittest.TestCase):
    def setUp(self):
        reset_generate_calls()

    def test_snapshots_are_carried_wholesale_into_the_record(self):
        """Whole structures, not named fields — the enumeration bug has
        already cost this project two cold builds."""
        record_generate_call(
            width=2, prompt_tokens_padded=1092, new_tokens=100,
            max_new_tokens=100, tokenise_s=0.02, generate_s=228.0,
            decode_s=0.001,
            placement={"fraction_params_off_gpu": 0.0, "invented": 1},
            memory={"free_gb": 0.5, "also_invented": 2},
        )
        (call,) = GENERATE_CALLS["calls"]
        self.assertEqual(call["placement"]["invented"], 1)
        self.assertEqual(call["memory"]["also_invented"], 2)
        self.assertEqual(call["memory"]["free_gb"], 0.5)

    def test_absent_snapshots_are_none_not_empty_dicts(self):
        """An empty dict would read as 'measured, nothing there'."""
        record_generate_call(
            width=1, prompt_tokens_padded=10, new_tokens=1, max_new_tokens=1,
            tokenise_s=0.0, generate_s=0.0, decode_s=0.0,
        )
        (call,) = GENERATE_CALLS["calls"]
        self.assertIsNone(call["placement"])
        self.assertIsNone(call["memory"])


class TestGeneratorLoadAccounting(unittest.TestCase):
    """A second load mid-build would mean re-placement into whatever VRAM
    was left. `load_generator` is lru_cached so this should stay at one;
    the point is that the run RECORDS it rather than assuming it."""

    def test_load_events_is_a_list_the_probe_can_read(self):
        self.assertIsInstance(GENERATOR_LOADS, list)


if __name__ == "__main__":
    unittest.main()
