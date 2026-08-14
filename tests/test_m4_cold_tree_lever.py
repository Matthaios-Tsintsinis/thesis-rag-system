"""The cold-tree lever: M4 substrate keys on the topology-relevant stack.

WHY. UMAP + GMM output is version-sensitive even when seeded. Every tree
in the old bank was built under an UNPINNED stack, so serving one warm
under the P9-pinned environment would put M4 cells on artifacts the
reproducibility control declares unreproducible. The lever makes those
trees unreachable by key rather than by discipline.

The M7 invariance property still holds by construction: the lever lives
in `paper_substrate_extra`, which is M4-local, and `src/raptor.py` is
never opened. M7 is withdrawn from the matrix, but the discipline stays —
a lever that moves a key it was not meant to move is the failure this
whole class of test exists to catch.
"""

from __future__ import annotations

import unittest

from src.cache import compute_cache_key
from src.config import DEFAULT_CONFIG
from src.raptor_paper import PAPER_TREE_BUILD_ENV, paper_substrate_extra

M4 = DEFAULT_CONFIG.m4


def _key(**overrides) -> str:
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
        """The whole point: a tree built under another stack cannot
        satisfy this key."""
        self.assertNotEqual(_key(), _key(build_env="some-other-stack"))

    def test_the_pre_lever_schema_produces_a_different_key(self):
        """Stands in for the old bank, whose extras carried no build_env
        at all."""
        self.assertNotEqual(_key(), _key(build_env="__LEGACY_NO_BUILD_ENV__"))

    def test_the_key_is_stable_within_one_stack(self):
        self.assertEqual(_key(), _key())


class TestTheEnvIdIsMeaningful(unittest.TestCase):
    def test_it_names_the_three_topology_libraries(self):
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            self.assertIn(pkg, PAPER_TREE_BUILD_ENV)

    def test_it_is_not_empty_or_a_placeholder(self):
        self.assertTrue(PAPER_TREE_BUILD_ENV.strip())
        self.assertIn("=", PAPER_TREE_BUILD_ENV)

    def test_it_reaches_the_substrate_extras(self):
        extra = paper_substrate_extra(
            params=M4.paper,
            summary_model=M4.summary_model,
            summary_prompt_version=M4.summary_prompt_version,
        )
        self.assertEqual(extra["build_env"], PAPER_TREE_BUILD_ENV)


class TestCacheHitIsObservable(unittest.TestCase):
    def test_a_fresh_system_reports_no_verdict_yet(self):
        """P10's preflight reads this. None means index() has not run;
        False means the lever took and the tree was rebuilt."""
        from src.retrievers.m4_raptor import RaptorSystem

        self.assertIsNone(RaptorSystem(DEFAULT_CONFIG).tree_cache_hit)


if __name__ == "__main__":
    unittest.main()
