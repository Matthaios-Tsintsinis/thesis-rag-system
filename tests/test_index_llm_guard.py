"""Pins that every default model id routes locally and that M4 refuses to
build a tree with an API summariser unless explicitly allowed."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from src.config import (
    DEFAULT_CONFIG,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    GenerationConfig,
)
from src.models import _is_openai_model
from src.retrievers.m4_raptor import RaptorSystem


class TestLocalGeneratorIsWired(unittest.TestCase):
    """Every model id in the default config resolves to a local model."""

    def test_generator_is_local(self):
        """The shared reader is a local Qwen model, not an API model."""
        self.assertFalse(_is_openai_model(GENERATOR_MODEL))
        self.assertIn("Qwen", GENERATOR_MODEL)

    def test_index_time_llm_is_local(self):
        """The index-time model id routes locally."""
        self.assertFalse(_is_openai_model(JUDGE_MODEL))

    def test_default_generation_config_routes_local(self):
        """A default GenerationConfig routes locally."""
        self.assertFalse(_is_openai_model(GenerationConfig().model))

    def test_m4_summariser_is_local(self):
        """The M4 summariser routes locally."""
        self.assertFalse(_is_openai_model(DEFAULT_CONFIG.m4.summary_model))

    def test_no_openai_model_left_in_the_default_config(self):
        """No model field in DEFAULT_CONFIG routes to the OpenAI API."""
        for label, model in (
            ("generation", DEFAULT_CONFIG.generation.model),
            ("m4.summary_model", DEFAULT_CONFIG.m4.summary_model),
        ):
            with self.subTest(field=label):
                self.assertFalse(
                    _is_openai_model(model),
                    f"{label} still routes to the OpenAI API: {model!r}",
                )


class TestDoomedBuildGuard(unittest.TestCase):
    """The M4 index-LLM guard blocks API summarisers and honours overrides."""

    def _system(self, model, allow=False):
        """Build a RaptorSystem whose summariser is model."""
        cfg = replace(
            DEFAULT_CONFIG,
            m4=replace(
                DEFAULT_CONFIG.m4,
                summary_model=model,
                allow_api_index_llm=allow,
            ),
        )
        return RaptorSystem(config=cfg)

    def setUp(self):
        """Clear the env override so each test starts without it."""
        self._env = os.environ.pop("M4_ALLOW_API_INDEX_LLM", None)

    def tearDown(self):
        """Restore the env override to what it was before the test."""
        if self._env is not None:
            os.environ["M4_ALLOW_API_INDEX_LLM"] = self._env
        else:
            os.environ.pop("M4_ALLOW_API_INDEX_LLM", None)

    def test_api_summariser_refuses_to_build(self):
        """An API summariser raises; the message names key and override."""
        with self.assertRaises(RuntimeError) as cm:
            self._system("gpt-4o-mini")._guard_index_llm()
        msg = str(cm.exception)
        self.assertIn("substrate cache key", msg)
        self.assertIn("M4_ALLOW_API_INDEX_LLM", msg)

    def test_local_summariser_passes(self):
        """A local summariser passes the guard."""
        self._system("Qwen/Qwen2.5-7B-Instruct")._guard_index_llm()

    def test_default_config_passes(self):
        """DEFAULT_CONFIG passes the guard."""
        RaptorSystem(config=DEFAULT_CONFIG)._guard_index_llm()

    def test_explicit_config_override_allows_it(self):
        """allow_api_index_llm=True lets an API summariser through."""
        self._system("gpt-4o-mini", allow=True)._guard_index_llm()

    def test_env_override_allows_it(self):
        """The M4_ALLOW_API_INDEX_LLM env var lets an API summariser pass."""
        os.environ["M4_ALLOW_API_INDEX_LLM"] = "1"
        self._system("gpt-4o-mini")._guard_index_llm()

    def test_guard_covers_every_openai_prefix(self):
        """Every OpenAI model prefix trips the guard."""
        for model in ("gpt-4o-mini", "gpt-4", "o1-preview", "chatgpt-4o"):
            with self.subTest(model=model):
                with self.assertRaises(RuntimeError):
                    self._system(model)._guard_index_llm()

    def test_guard_runs_before_any_expensive_work(self):
        """The guard raises before any chunking or embedding runs."""
        sysm = self._system("gpt-4o-mini")
        with mock.patch(
            "src.retrievers.m4_raptor.chunk_corpus"
        ) as chunker, mock.patch(
            "src.retrievers.m4_raptor.embed_texts"
        ) as embedder, mock.patch(
            "src.retrievers.m4_raptor.walk_corpus", return_value=[]
        ):
            with self.assertRaises(RuntimeError):
                sysm._guard_index_llm()
        chunker.assert_not_called()
        embedder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
