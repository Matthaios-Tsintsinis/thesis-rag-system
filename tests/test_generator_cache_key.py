"""Tests that one model name maps to one generator load, and that the
load asserts its own GPU placement."""

from __future__ import annotations

import unittest
from unittest import mock

from src.models import (
    GENERATOR_LOADS,
    assert_generator_fully_resident,
    load_generator,
)


def _stub_impl(model_name):  # noqa: ANN001
    """Stand-in loader that returns cheap placeholders instead of a model."""
    return (f"tok:{model_name}", f"model:{model_name}")


class TestOneKeyPerGenerator(unittest.TestCase):
    """Pins that load_generator caches on the model name alone."""

    def setUp(self):
        load_generator.cache_clear()
        GENERATOR_LOADS.clear()

    def tearDown(self):
        load_generator.cache_clear()
        GENERATOR_LOADS.clear()

    def test_positional_and_keyword_forms_are_the_same_load(self):
        """Positional and keyword spellings of one model share one load."""
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            a = load_generator("some/model")
            b = load_generator(model_name="some/model")
        self.assertEqual(impl.call_count, 1)
        self.assertIs(a, b)
        self.assertEqual(len(GENERATOR_LOADS), 1)

    def test_the_real_call_sites_agree(self):
        """The call form generate and generate_batch use loads only once."""
        from src.config import DEFAULT_CONFIG

        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator(DEFAULT_CONFIG.generation.model)
            load_generator(DEFAULT_CONFIG.generation.model)
        self.assertEqual(impl.call_count, 1, "the two call sites disagree")
        self.assertEqual(len(GENERATOR_LOADS), 1)

    def test_a_genuinely_different_model_still_loads_separately(self):
        """Two different model names load twice."""
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator("model/a")
            load_generator("model/b")
        self.assertEqual(impl.call_count, 2)

    def test_cache_clear_is_still_exposed(self):
        """load_generator.cache_clear exists and forces a reload."""
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator("some/model")
            load_generator.cache_clear()
            load_generator("some/model")
        self.assertEqual(impl.call_count, 2)


class TestResidencyAssertion(unittest.TestCase):
    """Pins how assert_generator_fully_resident reads a placement snapshot."""

    CLEAN = {"fraction_params_off_gpu": 0.0,
             "param_tensors_by_device": {"cuda:0": 339}}
    SPILLED = {"fraction_params_off_gpu": 0.6224,
               "param_tensors_by_device": {"cuda:0": 121, "meta": 218}}

    def test_clean_placement_passes(self):
        """A fully on-GPU placement passes."""
        assert_generator_fully_resident(
            self.CLEAN, model_name="m", cuda_available=True
        )

    def test_spilled_placement_raises(self):
        """A spilled placement raises, naming the share and the device."""
        with self.assertRaises(RuntimeError) as ctx:
            assert_generator_fully_resident(
                self.SPILLED, model_name="m", cuda_available=True
            )
        msg = str(ctx.exception)
        self.assertIn("62", msg)
        self.assertIn("meta", msg)

    def test_the_error_names_the_cause_not_just_the_symptom(self):
        """The error message names the PCIe cause and a prior load."""
        with self.assertRaises(RuntimeError) as ctx:
            assert_generator_fully_resident(
                self.SPILLED, model_name="m", cuda_available=True
            )
        msg = str(ctx.exception).lower()
        self.assertIn("pcie", msg)
        self.assertTrue("already" in msg or "second" in msg)

    def test_offload_is_permitted_when_explicitly_requested(self):
        """allow_offload=True accepts a spilled placement."""
        assert_generator_fully_resident(
            self.SPILLED, model_name="m", cuda_available=True,
            allow_offload=True,
        )

    def test_cpu_only_host_is_not_a_failure(self):
        """Without CUDA, an all-CPU placement passes."""
        assert_generator_fully_resident(
            {"fraction_params_off_gpu": 1.0,
             "param_tensors_by_device": {"cpu": 339}},
            model_name="m", cuda_available=False,
        )

    def test_unmeasured_placement_does_not_raise(self):
        """A snapshot with no measured share passes."""
        assert_generator_fully_resident(
            {"fraction_params_off_gpu": None, "param_tensors_by_device": {}},
            model_name="m", cuda_available=True,
        )


if __name__ == "__main__":
    unittest.main()
