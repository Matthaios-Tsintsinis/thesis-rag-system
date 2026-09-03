"""The two ADDENDUM-2 activation gates (src/eval/runner.py).

Both close gaps that were convention-only when the Llama column was ruled
GO: the bank separation existed as a directory-naming habit, and the gated
repo failed mid-session as a 401 inside from_pretrained rather than at
preflight. Both gates run before any model loads.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.runner import (
    assert_bank_generator_consistent,
    assert_bank_gpu_consistent,
    assert_generator_accessible,
)

QWEN = "Qwen/Qwen2.5-7B-Instruct"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


def _summary(d: Path, name: str, generator: str | None) -> None:
    body: dict = {"n_queries_scored": 1}
    if generator is not None:
        body["model_revisions"] = {"generator": generator}
    (d / name).write_text(json.dumps(body), encoding="utf-8")


class TestBankGeneratorGate(unittest.TestCase):
    def test_a_llama_cell_into_the_qwen_bank_refuses(self):
        """The exact mistake the gate exists for."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "multihop_rag_M4_validation.summary.json", QWEN)
            with self.assertRaises(SystemExit) as cm:
                assert_bank_generator_consistent(d, LLAMA)
            self.assertIn("p11", str(cm.exception))

    def test_the_reverse_mistake_also_refuses(self):
        """Data-driven means direction-free: a Qwen cell into a bank the
        Llama column owns refuses identically, with no name registry."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "multihop_rag_M4_validation.summary.json", LLAMA)
            with self.assertRaises(SystemExit):
                assert_bank_generator_consistent(d, QWEN)

    def test_a_matching_bank_passes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "a.summary.json", QWEN)
            _summary(d, "b.summary.json", QWEN)
            assert_bank_generator_consistent(d, QWEN)  # no raise

    def test_an_empty_or_absent_directory_passes(self):
        with tempfile.TemporaryDirectory() as td:
            assert_bank_generator_consistent(Path(td), LLAMA)
            assert_bank_generator_consistent(Path(td) / "not-yet", LLAMA)

    def test_a_generatorless_summary_warns_but_cannot_convict(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "old.summary.json", None)
            assert_bank_generator_consistent(d, LLAMA)  # no raise

    def test_the_refusal_names_the_offending_cells(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "hotpotqa_M2_validation.summary.json", QWEN)
            with self.assertRaises(SystemExit) as cm:
                assert_bank_generator_consistent(d, LLAMA)
            self.assertIn("hotpotqa_M2_validation.summary.json",
                          str(cm.exception))


class TestBankGpuGate(unittest.TestCase):
    """Instance 16: the GPU string was recorded per-cell since P9 and
    compared by nothing — a T4 silently replaced the L4 and only the OOM
    caught it. Same data-driven shape as the generator gate; the current
    GPU is injectable so the tests never depend on this host's hardware."""

    @staticmethod
    def _gpu_summary(d: Path, name: str, gpu: str | None) -> None:
        body: dict = {"n_queries_scored": 1}
        if gpu is not None:
            body["environment"] = {"gpu": gpu}
        (d / name).write_text(json.dumps(body), encoding="utf-8")

    def test_the_t4_incident_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gpu_summary(d, "a.summary.json", "NVIDIA L4")
            with self.assertRaises(SystemExit) as cm:
                assert_bank_gpu_consistent(d, current_gpu="Tesla T4")
            msg = str(cm.exception)
            self.assertIn("Tesla T4", msg)
            self.assertIn("NVIDIA L4", msg)
            self.assertIn("DIFFERENT bank", msg)

    def test_matching_hardware_passes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gpu_summary(d, "a.summary.json", "NVIDIA L4")
            assert_bank_gpu_consistent(d, current_gpu="NVIDIA L4")

    def test_the_first_cell_into_an_empty_bank_sets_the_hardware(self):
        with tempfile.TemporaryDirectory() as td:
            assert_bank_gpu_consistent(Path(td), current_gpu="Tesla T4")

    def test_an_unknown_current_gpu_warns_but_cannot_fail(self):
        """Absence of measurement is not a measured change — and the
        agent host (no CUDA) must be able to import and drive main()."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gpu_summary(d, "a.summary.json", "NVIDIA L4")
            assert_bank_gpu_consistent(d, current_gpu="unknown")

    def test_summaries_without_the_field_cannot_convict(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gpu_summary(d, "old.summary.json", None)
            assert_bank_gpu_consistent(d, current_gpu="Tesla T4")


class TestAccessGateRouting(unittest.TestCase):
    """The network call itself is operator-territory; what is testable is
    the ROUTING — which model ids the gate even attempts."""

    def test_local_and_api_models_are_skipped(self):
        # No "/" -> nothing to check; API prefixes -> the OpenAI router's
        # concern, not the hub's. Neither may touch the network.
        assert_generator_accessible("gpt-4o-mini")
        assert_generator_accessible("some-local-alias")


if __name__ == "__main__":
    unittest.main()
