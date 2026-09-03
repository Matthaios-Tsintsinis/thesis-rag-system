"""The doomed-build guard, and that the local generator is actually wired.

Both exist because of the same near-miss: a run began building
`M4_RAPTOR/34b630d8...` with gpt-4o-mini summaries and was stopped only
by a missing OPENAI_API_KEY. An absent credential is not a safety
mechanism — it fails for the right reason by accident and stops working
the moment a key is exported for something unrelated.
"""

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
    """Config, not just routing code — the routing was always correct;
    the model ids were what still pointed at the API."""

    def test_generator_is_local(self):
        self.assertFalse(_is_openai_model(GENERATOR_MODEL))
        self.assertIn("Qwen", GENERATOR_MODEL)

    def test_index_time_llm_is_local(self):
        self.assertFalse(_is_openai_model(JUDGE_MODEL))

    def test_default_generation_config_routes_local(self):
        self.assertFalse(_is_openai_model(GenerationConfig().model))

    def test_m4_summariser_is_local(self):
        self.assertFalse(_is_openai_model(DEFAULT_CONFIG.m4.summary_model))

    def test_no_openai_model_left_in_the_default_config(self):
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
    def _system(self, model, allow=False):
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
        self._env = os.environ.pop("M4_ALLOW_API_INDEX_LLM", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["M4_ALLOW_API_INDEX_LLM"] = self._env
        else:
            os.environ.pop("M4_ALLOW_API_INDEX_LLM", None)

    def test_api_summariser_refuses_to_build(self):
        with self.assertRaises(RuntimeError) as cm:
            self._system("gpt-4o-mini")._guard_index_llm()
        msg = str(cm.exception)
        self.assertIn("substrate cache key", msg)
        self.assertIn("M4_ALLOW_API_INDEX_LLM", msg)

    def test_local_summariser_passes(self):
        self._system("Qwen/Qwen2.5-7B-Instruct")._guard_index_llm()

    def test_default_config_passes(self):
        RaptorSystem(config=DEFAULT_CONFIG)._guard_index_llm()

    def test_explicit_config_override_allows_it(self):
        self._system("gpt-4o-mini", allow=True)._guard_index_llm()

    def test_env_override_allows_it(self):
        """The runner builds systems from DEFAULT_CONFIG and has no CLI
        path to the config field, so the env var is the usable escape."""
        os.environ["M4_ALLOW_API_INDEX_LLM"] = "1"
        self._system("gpt-4o-mini")._guard_index_llm()

    def test_guard_covers_every_openai_prefix(self):
        for model in ("gpt-4o-mini", "gpt-4", "o1-preview", "chatgpt-4o"):
            with self.subTest(model=model):
                with self.assertRaises(RuntimeError):
                    self._system(model)._guard_index_llm()

    def test_guard_runs_before_any_expensive_work(self):
        """It must fire on the cache-MISS path before chunking or
        embedding — the point is to spend nothing, not to fail late."""
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
