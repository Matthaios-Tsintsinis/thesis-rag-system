"""Pins the generator guard: the loaded reader must match the configured one
in checkpoint, quantization and dtype, or loading aborts.
"""

from __future__ import annotations

import unittest

from src.models import assert_loaded_generator_matches


# A load that matches the request in every field the guard compares.
OK = dict(
    requested_name="Qwen/Qwen2.5-7B-Instruct",
    loaded_name="Qwen/Qwen2.5-7B-Instruct",
    is_quantized=False,
    dtype_str="torch.float16",
)


class TestGeneratorGuard(unittest.TestCase):
    """Pins what assert_loaded_generator_matches accepts and rejects."""

    def test_matching_load_passes(self):
        """A load identical to the request passes."""
        assert_loaded_generator_matches(**OK)

    def test_bf16_also_passes(self):
        """bf16 counts as a half-precision match."""
        assert_loaded_generator_matches(**{**OK, "dtype_str": "torch.bfloat16"})

    def test_different_checkpoint_aborts(self):
        """A different checkpoint name aborts."""
        with self.assertRaises(RuntimeError) as cm:
            assert_loaded_generator_matches(
                **{**OK, "loaded_name": "Qwen/Qwen2.5-3B-Instruct"}
            )
        self.assertIn("b6e35c6", str(cm.exception))

    def test_silent_quantization_aborts(self):
        """A 4-bit load under an fp16 request aborts."""
        with self.assertRaises(RuntimeError) as cm:
            assert_loaded_generator_matches(**{**OK, "is_quantized": True})
        self.assertIn("quantization mismatch", str(cm.exception))

    def test_fp32_aborts(self):
        """An fp32 load aborts on dtype."""
        with self.assertRaises(RuntimeError) as cm:
            assert_loaded_generator_matches(**{**OK, "dtype_str": "torch.float32"})
        self.assertIn("dtype", str(cm.exception))

    def test_absent_loaded_name_does_not_crash(self):
        """A missing loaded name skips the checkpoint check."""
        assert_loaded_generator_matches(**{**OK, "loaded_name": None})


if __name__ == "__main__":
    unittest.main()
