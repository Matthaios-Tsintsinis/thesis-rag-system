"""Tests for the placement and VRAM snapshots taken right before each
generate() call. Torch is stubbed both ways so no test depends on the host.
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
    """Build a torch stand-in exposing what cuda_memory_snapshot reads."""
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        memory_allocated=lambda: 3 * 2**30,
        memory_reserved=lambda: 4 * 2**30,
        mem_get_info=lambda: (18 * 2**30, 24 * 2**30),
    )
    return types.SimpleNamespace(cuda=cuda)


class TestSnapshotsDegradeHonestly(unittest.TestCase):
    """The snapshot reports absence as None and presence as numbers."""

    def test_memory_snapshot_reports_none_without_torch(self):
        """Without torch every memory field is None and cuda is False."""
        with mock.patch.dict(sys.modules, {"torch": None}):
            snap = cuda_memory_snapshot()
        self.assertIsNone(snap["allocated_gb"])
        self.assertIsNone(snap["reserved_gb"])
        self.assertIsNone(snap["free_gb"])
        self.assertFalse(snap["cuda_available"])

    def test_memory_snapshot_reports_none_with_torch_but_no_cuda(self):
        """With torch but no CUDA device the fields are still None."""
        with mock.patch.dict(sys.modules,
                             {"torch": _fake_torch(cuda_available=False)}):
            snap = cuda_memory_snapshot()
        self.assertIsNone(snap["allocated_gb"])
        self.assertFalse(snap["cuda_available"])

    def test_memory_snapshot_reports_the_measured_numbers_with_cuda(self):
        """With CUDA the snapshot carries the measured GB figures."""
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
        """Every snapshot key is present even when its value is None."""
        snap = cuda_memory_snapshot()
        for k in ("allocated_gb", "reserved_gb", "free_gb", "total_gb",
                  "reserved_minus_allocated_gb", "cuda_available"):
            self.assertIn(k, snap)

    def test_placement_snapshot_survives_a_model_without_parameters(self):
        """A model with no parameters yields None fraction and empty counts."""
        class Bare:
            config = type("C", (), {"_attn_implementation": "sdpa"})()

            def named_parameters(self):
                return iter(())

        snap = model_placement_snapshot(Bare())
        self.assertIsNone(snap["fraction_params_off_gpu"])
        self.assertEqual(snap["param_tensors_by_device"], {})

    def test_placement_snapshot_counts_devices_and_computes_the_fraction(self):
        """Tensors are counted per device; the off-GPU fraction is by numel."""
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
    """Each generate-call record carries the snapshots it is given."""

    def setUp(self):
        reset_generate_calls()

    def test_snapshots_are_carried_wholesale_into_the_record(self):
        """Snapshot dicts are stored whole, unknown keys included."""
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
        """A call recorded without snapshots stores None, not empty dicts."""
        record_generate_call(
            width=1, prompt_tokens_padded=10, new_tokens=1, max_new_tokens=1,
            tokenise_s=0.0, generate_s=0.0, decode_s=0.0,
        )
        (call,) = GENERATE_CALLS["calls"]
        self.assertIsNone(call["placement"])
        self.assertIsNone(call["memory"])


class TestGeneratorLoadAccounting(unittest.TestCase):
    """Generator loads are recorded in a list the run summary can read."""

    def test_load_events_is_a_list_the_probe_can_read(self):
        """GENERATOR_LOADS is a list."""
        self.assertIsInstance(GENERATOR_LOADS, list)


if __name__ == "__main__":
    unittest.main()
