"""The leaf-expanded diagnostic twin, and the App. I non-leaf-share gate.

Retrieved SUMMARY nodes carry an empty gold_provenance by design — a
summary is abstractive text with no gold span. The consequence is that a
large minority of M4's retrieved units cannot contribute to CK-2 at all,
so M4's retrieval F1 is not directly comparable to a leaf-only system's.
`expand_summary_nodes` produces the twin that measures that gap, and
these tests pin the three properties that make the measurement mean
something:

  * expansion is POST-SELECTION, so the twin measures the coverage of the
    paper's retrieval rather than of a different retriever;
  * the gate is counted PRE-EXPANSION, so turning the twin on does not
    move the number the gate reports;
  * a twin run is identifiable from any single row of its JSONL.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from src.chunking import Chunk
from src.config import DEFAULT_CONFIG
from src.eval.analyse import _aggregate, _non_leaf_summary
from src.raptor_paper import (
    PAPER_NON_LEAF_SHARE_BAND,
    PaperCollapsedIndex,
    PaperNode,
    PaperTree,
    PaperTreeParams,
)
from src.retrievers.m4_raptor import RaptorSystem


EMB_DIM = 4


class _FakeFaiss:
    """Returns every ref in a fixed order, best score first."""

    def __init__(self, n: int) -> None:
        self.n = n

    def search(self, q, k):  # noqa: D102
        k = min(k, self.n)
        scores = np.array([[1.0 - 0.01 * i for i in range(k)]], dtype=np.float32)
        idx = np.array([[i for i in range(k)]], dtype=np.int64)
        return scores, idx


def _system(expand: bool, n_leaves: int = 2) -> RaptorSystem:
    """Two leaves and one summary over them, ranked summary-first.

    Hand-built rather than driven through index(): the tree here is a
    fixture for the RETRIEVAL path, and building a real one would cost a
    UMAP fit and a summariser for no extra coverage.
    """
    cfg = replace(
        DEFAULT_CONFIG,
        m4=replace(
            DEFAULT_CONFIG.m4,
            expand_summary_nodes=expand,
            summary_expansion_leaves=n_leaves,
            retrieval_budget_tokens=None,
            top_k_final=3,
        ),
    )
    sysm = RaptorSystem(config=cfg)

    leaves = [
        PaperNode(node_id="L0_000000", layer=0, text="alpha leaf", leaf_indices=[0]),
        PaperNode(node_id="L0_000001", layer=0, text="beta leaf", leaf_indices=[1]),
    ]
    summary = PaperNode(
        node_id="L1_000000", layer=1, text="a summary of both",
        children=[n.node_id for n in leaves], leaf_indices=[0, 1],
    )
    nodes = {n.node_id: n for n in [*leaves, summary]}
    sysm._tree = PaperTree(
        nodes=nodes,
        layer_to_nodes={0: [n.node_id for n in leaves], 1: [summary.node_id]},
        n_layers=2, params=PaperTreeParams(),
    )
    # Summary first, so a post-selection expansion is observable.
    refs = [
        {"node_id": "L1_000000", "layer": 1, "is_leaf": False},
        {"node_id": "L0_000000", "layer": 0, "is_leaf": True},
        {"node_id": "L0_000001", "layer": 0, "is_leaf": True},
    ]
    sysm._flat = PaperCollapsedIndex(
        faiss_index=_FakeFaiss(len(refs)), refs=refs, dim=EMB_DIM
    )
    sysm.chunks = [
        Chunk(chunk_id="c0", doc_id="d", text="alpha leaf", n_words=2, position=0,
              gold_provenance=(("d", "a"),)),
        Chunk(chunk_id="c1", doc_id="d", text="beta leaf", n_words=2, position=1,
              gold_provenance=(("d", "b"),)),
    ]
    # Leaf 1 is the better match for the query embedding below, so the
    # ordering under expansion is observable rather than incidental.
    sysm.chunk_embeddings = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
    )
    sysm._resolved = None
    sysm._indexed = True
    return sysm


def _patch_embed(sysm: RaptorSystem, vec):
    """Stub the query embedder; retrieval here is about routing, not vectors."""
    import src.retrievers.m4_raptor as mod

    sysm.__dict__["_orig_embed"] = mod.embed_texts
    mod.embed_texts = lambda texts, model_name=None: np.array([vec], dtype=np.float32)


def _unpatch(sysm: RaptorSystem):
    import src.retrievers.m4_raptor as mod

    mod.embed_texts = sysm.__dict__["_orig_embed"]


class _ExpansionCase(unittest.TestCase):
    def _retrieve(self, expand: bool, n_leaves: int = 2, query_vec=None):
        sysm = _system(expand, n_leaves)
        # Resolve the embedder id lookup that retrieve() performs.
        class _R:
            embedder_id = "fake"
        sysm._resolved = _R()
        _patch_embed(sysm, query_vec or [0.0, 1.0, 0.0, 0.0])
        try:
            return sysm, sysm.retrieve("q")
        finally:
            _unpatch(sysm)


class TestExpansionOff(_ExpansionCase):
    def test_summary_is_returned_as_itself(self):
        _, out = self._retrieve(expand=False)
        self.assertEqual(out[0].source_unit_type, "summary_low")
        self.assertEqual(out[0].chunk.text, "a summary of both")

    def test_summary_carries_no_gold_provenance(self):
        """The fact the twin exists to quantify: unscoreable by CK-2."""
        _, out = self._retrieve(expand=False)
        self.assertEqual(out[0].chunk.gold_provenance, ())


class TestExpansionOn(_ExpansionCase):
    def test_summary_is_replaced_by_scoreable_leaves(self):
        _, out = self._retrieve(expand=True)
        for item in out:
            self.assertTrue(
                item.chunk.gold_provenance,
                "every unit of the twin must be CK-2 scoreable",
            )

    def test_originating_tier_is_preserved_not_relabelled(self):
        """The App. I gate must still see that a unit came from a summary,
        even though the text it now carries is a leaf's.

        n_leaves=1 so both tiers survive: with 2 the summary expands to
        BOTH leaves first and the directly-retrieved copies dedupe away —
        correct behaviour, but it leaves nothing labelled "chunk".
        """
        _, out = self._retrieve(expand=True, n_leaves=1)
        self.assertEqual(out[0].source_unit_type, "summary_low")
        self.assertTrue(any(i.source_unit_type == "chunk" for i in out))

    def test_summary_leaves_outrank_their_directly_retrieved_copies(self):
        """Dedup keeps the FIRST occurrence, so a leaf reached through a
        higher-ranked summary keeps that better rank rather than being
        demoted to where it was retrieved directly."""
        _, out = self._retrieve(expand=True, n_leaves=2)
        self.assertEqual([i.chunk.chunk_id for i in out], ["c1", "c0"])
        self.assertTrue(all(i.source_unit_type == "summary_low" for i in out))

    def test_leaves_are_ordered_by_query_similarity(self):
        _, out = self._retrieve(expand=True, query_vec=[0.0, 1.0, 0.0, 0.0])
        self.assertEqual(out[0].chunk.chunk_id, "c1")

    def test_duplicates_are_collapsed_keeping_the_best_rank(self):
        """A leaf reachable directly AND through a summary must appear
        once; duplicates would deflate precision."""
        _, out = self._retrieve(expand=True)
        ids = [i.chunk.chunk_id for i in out]
        self.assertEqual(len(ids), len(set(ids)))

    def test_ranks_are_contiguous_and_scores_non_increasing(self):
        _, out = self._retrieve(expand=True)
        self.assertEqual([i.rank for i in out], list(range(len(out))))
        scores = [i.score for i in out]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_leaf_budget_is_respected(self):
        _, out = self._retrieve(expand=True, n_leaves=1)
        from_summary = [i for i in out if i.source_unit_type != "chunk"]
        self.assertEqual(len(from_summary), 1)


class TestGateIsCountedPreExpansion(_ExpansionCase):
    def test_non_leaf_share_is_identical_with_and_without_expansion(self):
        """The gate is a property of the PAPER'S retrieval. If expansion
        moved it, the twin would be measuring itself."""
        off, _ = self._retrieve(expand=False)
        on, _ = self._retrieve(expand=True)
        self.assertEqual(
            off.last_trace["non_leaf_share"], on.last_trace["non_leaf_share"]
        )
        self.assertAlmostEqual(off.last_trace["non_leaf_share"], 1 / 3)

    def test_trace_records_the_mode(self):
        off, _ = self._retrieve(expand=False)
        on, _ = self._retrieve(expand=True)
        self.assertFalse(off.last_trace["summary_expansion"])
        self.assertTrue(on.last_trace["summary_expansion"])


class TestNonLeafGateReporting(unittest.TestCase):
    @staticmethod
    def _rows(sid, unit_types, share=None, expansion=False, n=4):
        md = {}
        if share is not None:
            md["m4_non_leaf_share"] = share
        if expansion:
            md["m4_summary_expansion"] = True
        return [
            {
                "system_id": sid, "n_retrieved": sum(unit_types.values()),
                "retrieved_unit_types": dict(unit_types),
                "retrieval": {"skipped": False, "f1": 0.5},
                "answer": {"value": 0.5}, "predicted_answer": "x",
                "question_type": "t", "metadata": dict(md),
            }
            for _ in range(n)
        ]

    def test_micro_share_comes_from_unit_types_alone(self):
        """So the gate reads correctly on JSONLs banked before the m4_*
        metadata existed."""
        roll = _aggregate(self._rows("M4", {"chunk": 6, "summary_low": 4}))
        nl = roll["systems"]["M4"]["non_leaf"]
        self.assertAlmostEqual(nl["micro"], 0.4)
        self.assertIsNone(nl["macro"])
        self.assertTrue(nl["in_band"])

    def test_out_of_band_is_flagged(self):
        roll = _aggregate(self._rows("M4", {"chunk": 99, "summary_low": 1}))
        self.assertFalse(roll["systems"]["M4"]["non_leaf"]["in_band"])

    def test_a_leaf_only_system_is_out_of_scope_not_failing(self):
        """Printing FAIL against M2 would be a category error: a flat
        retriever is not subject to a RAPTOR gate."""
        roll = _aggregate(self._rows("M2", {"chunk": 10}))
        nl = roll["systems"]["M2"]["non_leaf"]
        self.assertIsNone(nl["in_band"])
        self.assertEqual(nl["n_non_leaf"], 0)

    def test_macro_is_reported_when_present(self):
        roll = _aggregate(
            self._rows("M4", {"chunk": 6, "summary_low": 4}, share=0.25)
        )
        self.assertAlmostEqual(roll["systems"]["M4"]["non_leaf"]["macro"], 0.25)

    def test_expansion_rows_are_counted_for_the_loud_warning(self):
        roll = _aggregate(
            self._rows("M4", {"chunk": 8, "summary_low": 2}, expansion=True, n=3)
        )
        self.assertEqual(roll["systems"]["M4"]["non_leaf"]["expansion_rows"], 3)

    def test_band_matches_the_paper_constant(self):
        nl = _non_leaf_summary({
            "retrieved_unit_types_agg": {"chunk": 1},
            "m4_non_leaf_share": [],
            "m4_expansion_rows": 0,
        })
        self.assertEqual(tuple(nl["band"]), PAPER_NON_LEAF_SHARE_BAND)


class TestDegenerateNoTreeIsLoud(unittest.TestCase):
    """A corpus below the layer stop condition yields NO tree.

    Measured boundary: <= 11 leaves breaks the layer loop on the first
    iteration, so M4 becomes flat dense retrieval over the leaves. It
    still retrieves, still answers, and still produces a plausible row.
    HotpotQA's standard distractor setting gives ~10 paragraphs per
    question, so this is about to happen thousands of times.
    """

    @staticmethod
    def _tiny_tree(n_leaves: int):
        from src.raptor_paper import build_paper_tree

        texts = [f"leaf {i} text here." for i in range(n_leaves)]
        embs = np.eye(n_leaves, EMB_DIM, dtype=np.float32)
        return build_paper_tree(
            texts, embs, params=PaperTreeParams(),
            summarize_batch_fn=lambda cs: ["s"] * len(cs),
            embed_fn=lambda ts: np.eye(len(ts), EMB_DIM, dtype=np.float32),
        )

    def test_flag_is_set_below_the_boundary(self):
        tree = self._tiny_tree(8)
        self.assertTrue(tree.stats["degenerate_no_tree"])
        self.assertEqual(tree.n_layers, 1)
        self.assertEqual(tree.summary_nodes(), [])

    def test_flag_is_clear_above_the_boundary(self):
        """Inert on a healthy build, or the flag stops meaning anything."""
        tree = self._tiny_tree(40)
        self.assertFalse(tree.stats["degenerate_no_tree"])
        self.assertTrue(tree.summary_nodes())

    def test_analyse_shouts_when_every_corpus_was_flat(self):
        """The variant-A shape. A fully flat M4 retrieves ZERO summary
        units, so it never reaches the App. I gate block — the warning
        has to survive independently of it."""
        rows = [
            {
                "system_id": "M4", "n_retrieved": 10,
                "retrieved_unit_types": {"chunk": 10},
                "retrieval": {"skipped": False, "f1": 0.4},
                "answer": {"value": 0.3}, "predicted_answer": "x",
                "question_type": "t",
                "metadata": {"m4_tree_degenerate": True},
            }
            for _ in range(5)
        ]
        nl = _aggregate(rows)["systems"]["M4"]["non_leaf"]
        self.assertEqual(nl["degenerate_rows"], 5)
        self.assertIsNone(nl["in_band"], "no gate applies to a flat index")

    def test_bic_failure_rows_are_counted(self):
        """Guard (v) trips get hammered by thousands of tiny builds in
        variant A; the rate has to be visible."""
        rows = [
            {
                "system_id": "M4", "n_retrieved": 10,
                "retrieved_unit_types": {"chunk": 8, "summary_low": 2},
                "retrieval": {"skipped": False, "f1": 0.4},
                "answer": {"value": 0.3}, "predicted_answer": "x",
                "question_type": "t",
                "metadata": {"m4_bic_fit_failures": 3 if i % 2 else 0},
            }
            for i in range(6)
        ]
        nl = _aggregate(rows)["systems"]["M4"]["non_leaf"]
        self.assertEqual(nl["bic_failure_rows"], 3)


class TestExpansionIsNotInTheCacheKey(unittest.TestCase):
    def test_toggling_expansion_does_not_move_the_substrate_key(self):
        """Query-time only: the twin must reuse the same tree, or running
        it would cost a full rebuild."""
        from src.cache import compute_cache_key
        from src.components import resolve_components
        from src.raptor_paper import paper_substrate_extra

        def key(cfg):
            r = resolve_components(cfg.m4, cfg, default_reranker=None)
            return compute_cache_key(
                chunking_config=r.chunker_config, embedder_model=r.embedder_id,
                corpus_hash="C", parsing_identity={},
                extra=paper_substrate_extra(
                    params=cfg.m4.paper, summary_model=cfg.m4.summary_model,
                    summary_prompt_version=cfg.m4.summary_prompt_version,
                    summary_max_tokens=cfg.m4.summary_max_tokens,
                    summary_batch_size=cfg.m4.summary_batch_size,
                    summary_max_padded_tokens=cfg.m4.summary_max_padded_tokens,
                    rrf_k=cfg.m4.rrf_k,
                    include_root=cfg.m4.include_root_in_flat_index,
                ),
            )

        base = DEFAULT_CONFIG
        twin = replace(base, m4=replace(base.m4, expand_summary_nodes=True))
        self.assertEqual(key(base), key(twin))


if __name__ == "__main__":
    unittest.main()
