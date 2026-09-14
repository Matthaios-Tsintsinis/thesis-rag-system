"""Tests for the preflight bank gates in src/eval/runner.py.

They pin the generator gate, the GPU gate and the access-gate routing,
all of which run before any model loads.
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
    """Write a minimal cell summary, with or without a generator field."""
    body: dict = {"n_queries_scored": 1}
    if generator is not None:
        body["model_revisions"] = {"generator": generator}
    (d / name).write_text(json.dumps(body), encoding="utf-8")


class TestBankGeneratorGate(unittest.TestCase):
    """The bank refuses a cell whose generator differs from the bank's."""

    def test_a_llama_cell_into_the_qwen_bank_refuses(self):
        """A Llama cell into a Qwen bank exits and names the other bank."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "multihop_rag_M4_validation.summary.json", QWEN)
            with self.assertRaises(SystemExit) as cm:
                assert_bank_generator_consistent(d, LLAMA)
            self.assertIn("p11", str(cm.exception))

    def test_the_reverse_mistake_also_refuses(self):
        """A Qwen cell into a Llama bank refuses the same way."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "multihop_rag_M4_validation.summary.json", LLAMA)
            with self.assertRaises(SystemExit):
                assert_bank_generator_consistent(d, QWEN)

    def test_a_matching_bank_passes(self):
        """A bank whose cells all share the generator passes."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "a.summary.json", QWEN)
            _summary(d, "b.summary.json", QWEN)
            assert_bank_generator_consistent(d, QWEN)  # no raise

    def test_an_empty_or_absent_directory_passes(self):
        """An empty or missing bank directory passes."""
        with tempfile.TemporaryDirectory() as td:
            assert_bank_generator_consistent(Path(td), LLAMA)
            assert_bank_generator_consistent(Path(td) / "not-yet", LLAMA)

    def test_a_generatorless_summary_warns_but_cannot_convict(self):
        """A summary with no generator field never causes a refusal."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "old.summary.json", None)
            assert_bank_generator_consistent(d, LLAMA)  # no raise

    def test_the_refusal_names_the_offending_cells(self):
        """The refusal message lists the mismatching summary files."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _summary(d, "hotpotqa_M2_validation.summary.json", QWEN)
            with self.assertRaises(SystemExit) as cm:
                assert_bank_generator_consistent(d, LLAMA)
            self.assertIn("hotpotqa_M2_validation.summary.json",
                          str(cm.exception))


class TestBankGpuGate(unittest.TestCase):
    """The bank refuses a cell whose GPU differs from the bank's."""

    @staticmethod
    def _gpu_summary(d: Path, name: str, gpu: str | None) -> None:
        """Write a minimal cell summary, with or without a gpu field."""
        body: dict = {"n_queries_scored": 1}
        if gpu is not None:
            body["environment"] = {"gpu": gpu}
        (d / name).write_text(json.dumps(body), encoding="utf-8")

    def test_the_t4_incident_refuses(self):
        """A T4 cell into an L4 bank exits and names both GPUs."""
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
        """A bank on the same GPU passes."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gpu_summary(d, "a.summary.json", "NVIDIA L4")
            assert_bank_gpu_consistent(d, current_gpu="NVIDIA L4")

    def test_the_first_cell_into_an_empty_bank_sets_the_hardware(self):
        """An empty bank accepts any GPU."""
        with tempfile.TemporaryDirectory() as td:
            assert_bank_gpu_consistent(Path(td), current_gpu="Tesla T4")

    def test_an_unknown_current_gpu_warns_but_cannot_fail(self):
        """An unknown current GPU passes, so a CUDA-less host can run."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gpu_summary(d, "a.summary.json", "NVIDIA L4")
            assert_bank_gpu_consistent(d, current_gpu="unknown")

    def test_summaries_without_the_field_cannot_convict(self):
        """A summary with no gpu field never causes a refusal."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._gpu_summary(d, "old.summary.json", None)
            assert_bank_gpu_consistent(d, current_gpu="Tesla T4")


class TestAccessGateRouting(unittest.TestCase):
    """The access gate only attempts hub-hosted model ids."""

    def test_local_and_api_models_are_skipped(self):
        """API-prefixed and slash-less ids skip the hub check."""
        # Neither id may touch the network.
        assert_generator_accessible("gpt-4o-mini")
        assert_generator_accessible("some-local-alias")


if __name__ == "__main__":
    unittest.main()
