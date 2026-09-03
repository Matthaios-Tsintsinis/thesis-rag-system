"""The b6e35c6 guard: a generator that isn't the one you named must abort.

b6e35c6 ran Qwen2.5-3B locally while every report said gpt-4o-mini. It
survived because nothing compared configured against loaded. These tests
pin the policy that makes that impossible to repeat silently.
"""

from __future__ import annotations

import unittest

from src.models import assert_loaded_generator_matches


OK = dict(
    requested_name="Qwen/Qwen2.5-7B-Instruct",
    loaded_name="Qwen/Qwen2.5-7B-Instruct",
    is_quantized=False,
    dtype_str="torch.float16",
)


class TestGeneratorGuard(unittest.TestCase):
    def test_matching_load_passes(self):
        assert_loaded_generator_matches(**OK)

    def test_bf16_also_passes(self):
        assert_loaded_generator_matches(**{**OK, "dtype_str": "torch.bfloat16"})

    def test_different_checkpoint_aborts(self):
        with self.assertRaises(RuntimeError) as cm:
            assert_loaded_generator_matches(
                **{**OK, "loaded_name": "Qwen/Qwen2.5-3B-Instruct"}
            )
        self.assertIn("b6e35c6", str(cm.exception))

    def test_silent_quantization_aborts(self):
        """The exact shape of the old bug: asked fp16, got 4-bit."""
        with self.assertRaises(RuntimeError) as cm:
            assert_loaded_generator_matches(**{**OK, "is_quantized": True})
        self.assertIn("quantization mismatch", str(cm.exception))

    def test_fp32_aborts(self):
        with self.assertRaises(RuntimeError) as cm:
            assert_loaded_generator_matches(**{**OK, "dtype_str": "torch.float32"})
        self.assertIn("dtype", str(cm.exception))

    def test_absent_loaded_name_does_not_crash(self):
        assert_loaded_generator_matches(**{**OK, "loaded_name": None})


if __name__ == "__main__":
    unittest.main()
