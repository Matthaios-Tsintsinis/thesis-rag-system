"""Tests for M4's token-budget retrieval, driven over a hand-built tree."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

import src.retrievers.m4_raptor as M
from src.config import DEFAULT_CONFIG
from src.chunking import Chunk
from src.raptor_paper import (
    PaperNode,
    PaperTree,
    build_collapsed_index,
    count_tokens_plain,
)


DIM = 8


def _unit(i: int) -> np.ndarray:
    """Return a unit vector whose cosine to `_query()` falls as i grows."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = 1.0
    v[1 + (i % (DIM - 1))] = 0.15 * (i + 1)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _query() -> np.ndarray:
    """Return the fixed query vector every node is scored against."""
    v = np.zeros((1, DIM), dtype=np.float32)
    v[0, 0] = 1.0
    return v


def _build_system(budget, n_leaves=10, words_per_node=40):
    """Build an M4 system over a hand-made two-layer tree with the given budget."""
    text = " ".join(["token"] * words_per_node) + "."
    nodes: dict[str, PaperNode] = {}
    layer0: list[str] = []
    # Leaves of equal length, ranked by their index; one summary over all.
    for i in range(n_leaves):
        nid = f"L0_{i:06d}"
        nodes[nid] = PaperNode(node_id=nid, layer=0, text=text,
                               leaf_indices=[i], embedding=_unit(i))
        layer0.append(nid)
    summary = PaperNode(
        node_id="L1_000000", layer=1, text=text,
        children=list(layer0), leaf_indices=list(range(n_leaves)),
        embedding=_unit(n_leaves),
    )
    nodes[summary.node_id] = summary

    tree = PaperTree(
        nodes=nodes,
        layer_to_nodes={0: layer0, 1: ["L1_000000"]},
        n_layers=2,
        params=DEFAULT_CONFIG.m4.paper,
    )

    # Populate the system's internals directly, skipping the tree build.
    cfg = replace(
        DEFAULT_CONFIG,
        m4=replace(DEFAULT_CONFIG.m4, retrieval_budget_tokens=budget),
    )
    sysm = M.RaptorSystem(config=cfg)
    sysm._tree = tree
    sysm._flat = build_collapsed_index(tree)
    sysm.chunks = [
        Chunk(chunk_id=f"c{i}", doc_id="d", text=text,
              n_words=words_per_node, position=i)
        for i in range(n_leaves)
    ]
    sysm._resolved = type("R", (), {"embedder_id": "fake"})()
    sysm._indexed = True
    return sysm, count_tokens_plain(text)


class TestBudgetFill(unittest.TestCase):
    """Pins how the token budget governs the number of nodes returned."""

    def setUp(self):
        # Stub the embedder so the query is the fixed vector.
        self._orig = M.embed_texts
        M.embed_texts = lambda texts, model_name=None: _query()

    def tearDown(self):
        M.embed_texts = self._orig

    def test_fills_up_to_but_not_over_the_budget(self):
        """Packing stops at the first node that overflows the budget."""
        sysm, per_node = _build_system(budget=5 * 50)
        out = sysm.retrieve("q")
        used = sysm.last_trace["budget_tokens_used"]
        self.assertLessEqual(used, 5 * 50)
        self.assertEqual(used, len(out) * per_node)
        # One more node would overflow.
        self.assertGreater(used + per_node, 5 * 50)

    def test_budget_governs_instead_of_top_k_final(self):
        """A budget that admits fewer nodes than top_k_final wins."""
        sysm, per_node = _build_system(budget=2 * 60)
        out = sysm.retrieve("q")
        self.assertLess(len(out), DEFAULT_CONFIG.m4.top_k_final)

    def test_budget_can_exceed_top_k_final(self):
        """A large budget returns more nodes than top_k_final."""
        sysm, per_node = _build_system(budget=10_000, n_leaves=30)
        out = sysm.retrieve("q")
        self.assertGreater(len(out), DEFAULT_CONFIG.m4.top_k_final)

    def test_explicit_k_overrides_the_budget(self):
        """An explicit k returns exactly k nodes and leaves the budget unused."""
        sysm, _ = _build_system(budget=100)
        out = sysm.retrieve("q", k=7)
        self.assertEqual(len(out), 7)
        self.assertIsNone(sysm.last_trace["budget_tokens_used"])

    def test_budget_none_restores_count_mode(self):
        """With no budget, retrieval returns top_k_final nodes."""
        # More leaves than top_k_final, so the count rule is what caps.
        sysm, _ = _build_system(budget=None, n_leaves=25)
        out = sysm.retrieve("q")
        self.assertEqual(len(out), DEFAULT_CONFIG.m4.top_k_final)
        self.assertIsNone(sysm.last_trace["budget_tokens_limit"])

    def test_first_node_is_kept_even_if_it_alone_overflows(self):
        """The first node is admitted even when it alone overflows the budget."""
        # harness choice: unreachable at ~110-token nodes
        sysm, per_node = _build_system(budget=1)
        out = sysm.retrieve("q")
        self.assertEqual(len(out), 1)
        self.assertGreater(sysm.last_trace["budget_tokens_used"], 1)

    def test_default_config_carries_the_paper_budget(self):
        """The default M4 budget is the paper's 2,000 tokens."""
        # RAPTOR paper §3: "2000 maximum tokens ... top-20 nodes" (paper over repo): see METHODS §A.4.3
        self.assertEqual(DEFAULT_CONFIG.m4.retrieval_budget_tokens, 2000)

    def test_trace_reports_both_used_and_limit(self):
        """The trace records both the budget limit and the tokens used."""
        sysm, _ = _build_system(budget=300)
        sysm.retrieve("q")
        self.assertEqual(sysm.last_trace["budget_tokens_limit"], 300)
        self.assertLessEqual(sysm.last_trace["budget_tokens_used"], 300)


if __name__ == "__main__":
    unittest.main()
