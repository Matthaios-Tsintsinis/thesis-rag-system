"""One generator, one cache key, one load — and a load that checks itself.

THE DEFECT THIS PINS, measured 2026-08-16 and worth stating exactly.
`load_generator` is `functools.lru_cache`d, which keys on the ARGUMENT
TUPLE. The prewarm path called `load_generator(model)`; `generate_batch`
called `load_generator(cfg.model, cfg.load_in_4bit)`. Passing the default
explicitly makes `("Qwen/...",)` and `("Qwen/...", False)` two different
keys, so the second call was a cache MISS and loaded a SECOND ~15 GB copy
of the same model. `maxsize=2` was exactly large enough to hold both, so
nothing was ever evicted and nothing complained.

Load #2 landed in VRAM that already held load #1 plus the embedder, so
`device_map="auto"` did what it is designed to do and spilled: 218 of 339
tensors to `meta`/CPU, `fraction_params_off_gpu: 0.6224`, free VRAM down
to 0.13–1.84 GB. Every decode step then streamed 4.74 B parameters over
PCIe — the flat ~230 s per call, independent of batch width and prompt
length, a 33x tax on a 20-hour build.

No release was involved. The earlier code read was right that
`release_generator` is never called in the indexing path; a cache MISS on
a different argument tuple needs no cache clear.

THE OTHER HALF OF THE LESSON. `placement_at_load` already captured
`fraction_params_off_gpu: 0.6224` on load #2. It was recorded correctly,
surfaced correctly, and NOTHING CONSUMED IT — the project's recurring
defect class, for the fifth time. A value that is measured and not
asserted is not a check.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.models import (
    GENERATOR_LOADS,
    assert_generator_fully_resident,
    load_generator,
)


def _stub_impl(model_name, load_in_4bit):  # noqa: ANN001
    return (f"tok:{model_name}", f"model:{model_name}:{load_in_4bit}")


class TestOneKeyPerGenerator(unittest.TestCase):
    def setUp(self):
        load_generator.cache_clear()
        GENERATOR_LOADS.clear()

    def tearDown(self):
        load_generator.cache_clear()
        GENERATOR_LOADS.clear()

    def test_omitted_and_explicit_default_are_the_same_load(self):
        """THE BUG. Two call styles, one model, one load."""
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            a = load_generator("some/model")
            b = load_generator("some/model", False)
        self.assertEqual(impl.call_count, 1)
        self.assertIs(a, b)
        self.assertEqual(len(GENERATOR_LOADS), 1)

    def test_keyword_form_is_also_the_same_load(self):
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator("some/model")
            load_generator("some/model", load_in_4bit=False)
            load_generator(model_name="some/model")
        self.assertEqual(impl.call_count, 1)
        self.assertEqual(len(GENERATOR_LOADS), 1)

    def test_the_real_call_sites_agree(self):
        """Drives the two forms the pipeline actually uses, rather than
        asserting a normaliser in the abstract: the probe prewarm passes
        one positional arg, generate_batch passes two."""
        from src.config import DEFAULT_CONFIG

        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator(DEFAULT_CONFIG.generation.model)              # prewarm
            load_generator(DEFAULT_CONFIG.generation.model,              # generate_batch
                           DEFAULT_CONFIG.generation.load_in_4bit)
        self.assertEqual(impl.call_count, 1, "the two call sites disagree")
        self.assertEqual(len(GENERATOR_LOADS), 1)

    def test_a_genuinely_different_model_still_loads_separately(self):
        """Normalisation must not collapse distinct requests."""
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator("model/a")
            load_generator("model/b")
        self.assertEqual(impl.call_count, 2)

    def test_four_bit_is_a_different_load_because_it_is(self):
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator("some/model", False)
            load_generator("some/model", True)
        self.assertEqual(impl.call_count, 2)

    def test_cache_clear_is_still_exposed(self):
        """`release_generator` calls it; losing the attribute would break
        the probes rather than the run, which is worse than obvious."""
        with mock.patch("src.models._load_generator_impl",
                        side_effect=_stub_impl) as impl:
            load_generator("some/model")
            load_generator.cache_clear()
            load_generator("some/model")
        self.assertEqual(impl.call_count, 2)


class TestResidencyAssertion(unittest.TestCase):
    """The assertion that would have caught this on load #2."""

    CLEAN = {"fraction_params_off_gpu": 0.0,
             "param_tensors_by_device": {"cuda:0": 339}}
    SPILLED = {"fraction_params_off_gpu": 0.6224,
               "param_tensors_by_device": {"cuda:0": 121, "meta": 218}}

    def test_clean_placement_passes(self):
        assert_generator_fully_resident(
            self.CLEAN, model_name="m", cuda_available=True
        )

    def test_spilled_placement_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            assert_generator_fully_resident(
                self.SPILLED, model_name="m", cuda_available=True
            )
        msg = str(ctx.exception)
        self.assertIn("62", msg)
        self.assertIn("meta", msg)

    def test_the_error_names_the_cause_not_just_the_symptom(self):
        """An operator reading this at hour three of a build needs the
        remedy, not a percentage."""
        with self.assertRaises(RuntimeError) as ctx:
            assert_generator_fully_resident(
                self.SPILLED, model_name="m", cuda_available=True
            )
        msg = str(ctx.exception).lower()
        self.assertIn("pcie", msg)
        self.assertTrue("already" in msg or "second" in msg)

    def test_offload_is_permitted_when_explicitly_requested(self):
        assert_generator_fully_resident(
            self.SPILLED, model_name="m", cuda_available=True,
            allow_offload=True,
        )

    def test_cpu_only_host_is_not_a_failure(self):
        """The agent host has no GPU and legitimately holds everything on
        CPU; raising there would break every CPU smoke path."""
        assert_generator_fully_resident(
            {"fraction_params_off_gpu": 1.0,
             "param_tensors_by_device": {"cpu": 339}},
            model_name="m", cuda_available=False,
        )

    def test_unmeasured_placement_does_not_raise(self):
        assert_generator_fully_resident(
            {"fraction_params_off_gpu": None, "param_tensors_by_device": {}},
            model_name="m", cuda_available=True,
        )


if __name__ == "__main__":
    unittest.main()
