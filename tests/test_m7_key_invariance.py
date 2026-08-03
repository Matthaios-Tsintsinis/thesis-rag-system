"""The frozen-M7 guard: no M4-local change may move M7's substrate key.

M7 is frozen and its substrate cache is expensive to rebuild. M4 and M7
were byte-identical in every cache-key input except the embedder, so the
M4 fidelity rebuild ran a real risk of invalidating M7 by accident. Two
near-misses were caught before the build (a new field on
RaptorBuildParams, then a new field on ChunkingConfig); this file exists
so a third is caught by CI instead of by a wasted rebuild.

TWO KINDS OF ASSERTION, deliberately:

  * THE PROPERTY (permanent). Mutating M4's config in any way that the
    fidelity work actually does — chunker, tree params, summariser,
    prompt id, summary cap, namespace — must leave M7's key bit-identical.
    This is what protects M7, and it stays true no matter what else in
    the project changes.

  * THE CONSTANT (expiring tripwire). A hardcoded key for fixed inputs.
    It catches accidental edits to the SHARED derivation that the
    property test cannot see — because such an edit would move M4 and M7
    together and the property would still hold.

    !! THE CONSTANT HAS A KNOWN EXPIRY. It is computed with
    summary_model = the current JUDGE_MODEL. The project is moving to a
    local summariser, and that change legitimately moves M7's key. When
    it lands, this test SHOULD fail. Recompute the constant, confirm the
    move was intended, and update it — do not weaken the test. The
    property assertions above must keep passing throughout.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.cache import compute_cache_key
from src.components import resolve_components
from src.config import (
    DEFAULT_CONFIG,
    RERANKER_MODEL,
    ChunkingConfig,
    PaperTreeParams,
)
from src.raptor import raptor_substrate_extra
from src.raptor_paper import paper_substrate_extra
from src.summarization import SUMMARY_PROMPT_VERSION


# Fixed inputs so the test is independent of the host's installed
# parsers (parsing_identity() probes them) and of any real corpus.
PARSING: dict = {}
CORPUS = "CORPUS_HASH_PLACEHOLDER"

# See the expiry warning in the module docstring before touching this.
#
# HISTORY of intended moves — every entry is a SHARED change that the
# property tests below cannot see, which is precisely why the constant
# exists:
#   3ff43a21bf448446740badd2bd25573f  gpt-4o-mini era (through 088ba1c)
#   a3fdf918a4197d4157bcf5cb44ec42bc  JUDGE_MODEL -> Qwen/Qwen2.5-7B-Instruct
#
# The 2026-08-02 move was expected: gpt-4o-mini was removed from the
# project and JUDGE_MODEL is read by BOTH M4 and the frozen M7, so M7's
# substrate key moves with it. When it fired, all ten property tests
# still passed — the correct signature of a shared change rather than an
# M4 leak. M7's preserved substrate (RAPTOR/78fb239...) is consequently
# dead and M7 rebuilds when it joins at test scale; that cost was
# accepted when the generator swap was approved.
EXPECTED_M7_KEY = "a3fdf918a4197d4157bcf5cb44ec42bc"


def m7_key(cfg) -> str:
    """M7's substrate key exactly as m7_three_axis.index() derives it."""
    r = resolve_components(cfg.m7, cfg, default_reranker=RERANKER_MODEL)
    extra = raptor_substrate_extra(
        build=cfg.m7.build,
        summary_model=cfg.m7.summary_model,
        summary_prompt_version=SUMMARY_PROMPT_VERSION,
        include_root=cfg.m7.include_root_in_flat_index,
        rrf_k=cfg.m7.rrf_k,
    )
    return compute_cache_key(
        chunking_config=r.chunker_config,
        embedder_model=r.embedder_id,
        corpus_hash=CORPUS,
        extra=extra,
        parsing_identity=PARSING,
    )


def m4_key(cfg) -> str:
    """M4's substrate key exactly as m4_raptor._cache_dir() derives it."""
    r = resolve_components(cfg.m4, cfg, default_reranker=None)
    extra = paper_substrate_extra(
        params=cfg.m4.paper,
        summary_model=cfg.m4.summary_model,
        summary_prompt_version=cfg.m4.summary_prompt_version,
        summary_max_tokens=cfg.m4.summary_max_tokens,
        summary_batch_size=cfg.m4.summary_batch_size,
        summary_max_padded_tokens=cfg.m4.summary_max_padded_tokens,
        rrf_k=cfg.m4.rrf_k,
        include_root=cfg.m4.include_root_in_flat_index,
    )
    return compute_cache_key(
        chunking_config=r.chunker_config,
        embedder_model=r.embedder_id,
        corpus_hash=CORPUS,
        extra=extra,
        parsing_identity=PARSING,
    )


class TestM7KeyInvariance(unittest.TestCase):
    """THE PROPERTY. Permanent; must never be weakened."""

    def setUp(self):
        self.base = DEFAULT_CONFIG
        self.baseline = m7_key(self.base)

    def _assert_m7_unmoved(self, m4_override, label: str):
        mutated = replace(self.base, m4=m4_override)
        self.assertEqual(
            m7_key(mutated), self.baseline,
            f"{label} moved M7's substrate key — the frozen M7 cache would "
            "be invalidated. Use an M4-local lever instead.",
        )

    def test_m4_chunker_override_does_not_move_m7(self):
        self._assert_m7_unmoved(
            replace(self.base.m4, chunker=ChunkingConfig(
                strategy="raptor_100tok", chunk_words=100, overlap_words=0)),
            "M4 chunker override",
        )

    def test_m4_tree_params_do_not_move_m7(self):
        self._assert_m7_unmoved(
            replace(self.base.m4, paper=PaperTreeParams(gmm_threshold=0.25)),
            "M4 PaperTreeParams change",
        )

    def test_m4_summariser_change_does_not_move_m7(self):
        self._assert_m7_unmoved(
            replace(self.base.m4, summary_model="Qwen/Qwen2.5-7B-Instruct"),
            "M4 summariser change",
        )

    def test_m4_local_prompt_id_does_not_move_m7(self):
        """The whole point of an M4-LOCAL prompt version.

        Bumping summarization.SUMMARY_PROMPT_VERSION instead would move
        M7 — it is a module global both systems read.
        """
        self._assert_m7_unmoved(
            replace(self.base.m4, summary_prompt_version="raptor_paper_v2"),
            "M4-local summary prompt id",
        )

    def test_m4_summary_cap_does_not_move_m7(self):
        self._assert_m7_unmoved(
            replace(self.base.m4, summary_max_tokens=150),
            "M4 summary token cap",
        )

    def test_m4_summary_batch_shape_does_not_move_m7(self):
        """The batched-summarisation lever, proven M4-local.

        Batch composition can change generated text at temperature 0 and
        summaries are cached, so the batch shape had to enter M4's
        substrate key. The safe lever for that is M4Config fields emitted
        by raptor_paper.paper_substrate_extra — nothing on M7's
        derivation reads either. The unsafe alternatives, for the record:
        a field on RaptorBuildParams or ChunkingConfig, or a bump of the
        global SUMMARY_PROMPT_VERSION, all of which M7 reads.
        """
        self._assert_m7_unmoved(
            replace(
                self.base.m4,
                summary_batch_size=4,
                summary_max_padded_tokens=8000,
            ),
            "M4 summary batch shape",
        )

    def test_m4_batch_shape_does_move_m4(self):
        """The other half: it must actually invalidate M4's own cache.

        A key that does not move is not a safe lever, it is a silent
        one — the original clustering landmine in reverse. If batch shape
        can change summary text, a warm cache built at a different shape
        must not be served.
        """
        base = m4_key(self.base)
        moved = m4_key(
            replace(self.base, m4=replace(self.base.m4, summary_batch_size=4))
        )
        self.assertNotEqual(base, moved)

    def test_m4_include_root_does_not_move_m7(self):
        self._assert_m7_unmoved(
            replace(self.base.m4, include_root_in_flat_index=False),
            "M4 include_root flag",
        )

    def test_all_m4_mutations_together_do_not_move_m7(self):
        self._assert_m7_unmoved(
            replace(
                self.base.m4,
                chunker=ChunkingConfig(strategy="raptor_100tok",
                                       chunk_words=100, overlap_words=0),
                paper=PaperTreeParams(reduction_dimension=7),
                summary_model="Qwen/Qwen2.5-7B-Instruct",
                summary_prompt_version="raptor_paper_v9",
                summary_max_tokens=42,
                summary_batch_size=3,
                summary_max_padded_tokens=1234,
            ),
            "every M4 fidelity change at once",
        )

    def test_m4_uses_its_own_extras_not_the_shared_ones(self):
        """Lever B, strict reading: src/raptor.py is never opened.

        The two extras functions must be distinguishable, otherwise M4
        would be riding the shared derivation and any edit to it would
        hit M7.
        """
        m4_extra = paper_substrate_extra(
            params=PaperTreeParams(), summary_model="x",
            summary_prompt_version="y",
        )
        m7_extra = raptor_substrate_extra(
            build=self.base.m7.build, summary_model="x",
            summary_prompt_version="y", include_root=False, rrf_k=60,
        )
        self.assertNotEqual(m4_extra, m7_extra)
        self.assertTrue(set(m7_extra) <= set(m4_extra))
        self.assertIn("clustering", m4_extra)
        self.assertNotIn("clustering", m7_extra)


class TestM4KeyMoved(unittest.TestCase):
    """The rebuild must NOT silently reuse the KMeans-era substrate.

    The original landmine: the shared extras folded tree PARAMETERS but
    never the clustering ALGORITHM, so swapping the algorithm would have
    changed the artifacts without changing the key, and every warm cache
    would have served the old tree while the change appeared to do
    nothing.
    """

    def test_m4_key_differs_from_the_legacy_shared_derivation(self):
        cfg = DEFAULT_CONFIG
        r = resolve_components(cfg.m4, cfg, default_reranker=None)
        legacy = compute_cache_key(
            chunking_config=ChunkingConfig(),  # the pre-rebuild chunker
            embedder_model=r.embedder_id,
            corpus_hash=CORPUS,
            extra=raptor_substrate_extra(
                build=cfg.m4.build, summary_model=cfg.m4.summary_model,
                summary_prompt_version=SUMMARY_PROMPT_VERSION,
                include_root=False, rrf_k=cfg.m4.rrf_k,
            ),
            parsing_identity=PARSING,
        )
        self.assertNotEqual(m4_key(cfg), legacy)

    def test_m4_and_m7_land_on_different_keys(self):
        cfg = DEFAULT_CONFIG
        self.assertNotEqual(m4_key(cfg), m7_key(cfg))


class TestExpiringConstant(unittest.TestCase):
    """THE TRIPWIRE. Expected to fail when the summariser changes."""

    def test_m7_key_matches_the_recorded_constant(self):
        self.assertEqual(
            m7_key(DEFAULT_CONFIG), EXPECTED_M7_KEY,
            "M7's substrate key changed. If you changed JUDGE_MODEL or "
            "anything else SHARED, this is expected — confirm the move was "
            "intended, recompute EXPECTED_M7_KEY, and update it. If you "
            "changed only M4, this is a BUG: the property tests above "
            "should have caught it first.",
        )


if __name__ == "__main__":
    unittest.main()
