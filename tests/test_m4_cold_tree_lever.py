"""Pins that the M4 substrate cache key includes the topology stack
(umap-learn, scikit-learn, numpy), so a tree built under another stack
never serves warm.
"""

from __future__ import annotations

import unittest

from src.cache import compute_cache_key
from src.config import DEFAULT_CONFIG
from src.raptor_paper import PAPER_TREE_BUILD_ENV, paper_substrate_extra

M4 = DEFAULT_CONFIG.m4


def _key(**overrides) -> str:
    """Build an M4 substrate key with stub embedder and corpus."""
    extra = paper_substrate_extra(
        params=M4.paper,
        summary_model=M4.summary_model,
        summary_prompt_version=M4.summary_prompt_version,
        summary_max_tokens=M4.summary_max_tokens,
        summary_batch_size=M4.summary_batch_size,
        summary_max_padded_tokens=M4.summary_max_padded_tokens,
        rrf_k=M4.rrf_k,
        include_root=M4.include_root_in_flat_index,
        **overrides,
    )
    return compute_cache_key(
        chunking_config=M4.chunker,
        embedder_model="stub-embedder",
        corpus_hash="stub-corpus",
        extra=extra,
        parsing_identity={},
    )


class TestTheLeverTakes(unittest.TestCase):
    def test_a_different_stack_produces_a_different_key(self):
        """A different topology stack gives a different substrate key."""
        self.assertNotEqual(_key(), _key(build_env="some-other-stack"))

    def test_the_pre_lever_schema_produces_a_different_key(self):
        """Extras without a build_env entry give a different key."""
        self.assertNotEqual(_key(), _key(build_env="__LEGACY_NO_BUILD_ENV__"))

    def test_the_key_is_stable_within_one_stack(self):
        """The key is deterministic for one stack."""
        self.assertEqual(_key(), _key())


class TestTheEnvIdIsMeaningful(unittest.TestCase):
    def test_it_names_the_three_topology_libraries(self):
        """The env id names umap-learn, scikit-learn and numpy."""
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(pkg, PAPER_TREE_BUILD_ENV)

    def test_it_is_not_empty_or_a_placeholder(self):
        """The env id carries at least one name=version pair."""
        self.assertTrue(PAPER_TREE_BUILD_ENV.strip())
        self.assertIn("=", PAPER_TREE_BUILD_ENV)

    def test_it_reaches_the_substrate_extras(self):
        """paper_substrate_extra writes the env id under build_env."""
        extra = paper_substrate_extra(
            params=M4.paper,
            summary_model=M4.summary_model,
            summary_prompt_version=M4.summary_prompt_version,
        )
        self.assertEqual(extra["build_env"], PAPER_TREE_BUILD_ENV)


class TestCacheHitIsObservable(unittest.TestCase):
    def test_a_fresh_system_reports_no_verdict_yet(self):
        """tree_cache_hit is None until index() runs."""
        from src.retrievers.m4_raptor import RaptorSystem

        self.assertIsNone(RaptorSystem(DEFAULT_CONFIG).tree_cache_hit)


if __name__ == "__main__":
    unittest.main()
