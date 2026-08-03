"""Tests for the paper-faithful bottom-up RAPTOR tree (src/raptor_paper.py).

Two tiers. The bulk use an injected `cluster_fn` so the deterministic
bookkeeping — node ids, multi-parent links, leaf-index closure, stop
condition, batch-shape invariance of the topology, serialisation — is
tested without paying for UMAP fits. A small number exercise the REAL UMAP+GMM path, since a
clustering port whose clustering was never run is not verified.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.raptor import raptor_substrate_extra
from src.config import M4Config, RaptorBuildParams
from src.raptor_paper import (
    PAPER_TREE_SCHEMA_VERSION,
    PaperNode,
    PaperTreeParams,
    build_collapsed_index,
    build_paper_tree,
    get_text,
    load_collapsed_index,
    load_paper_tree,
    paper_substrate_extra,
    perform_clustering,
    save_collapsed_index,
    save_paper_tree,
    tree_stats,
)


EMB_DIM = 8


def _fake_embed_dim(texts: list[str], dim: int) -> np.ndarray:
    """Deterministic unit vectors, one per text. No model, no randomness.

    Uses a stable digest rather than hash() — PYTHONHASHSEED randomises
    str hashing per process, which would make the determinism tests lie.
    """
    import hashlib

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        for j in range(dim):
            out[i, j] = digest[j % len(digest)] / 255.0 + 0.05 * (j + 1)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-9)).astype(np.float32)


def _fake_embed(texts: list[str]) -> np.ndarray:
    return _fake_embed_dim(texts, EMB_DIM)


def _pairwise_cluster_fn(nodes, params, stats):
    """Consecutive pairs, with node 0 deliberately in TWO clusters.

    The double membership stands in for the paper's soft clustering, so
    the multi-parent bookkeeping is exercised without a GMM.
    """
    clusters = [list(nodes[i : i + 2]) for i in range(0, len(nodes), 2)]
    if len(clusters) > 1:
        clusters[1] = [nodes[0]] + clusters[1]
    return clusters


def _leaf_inputs(n: int) -> tuple[list[str], np.ndarray]:
    texts = [f"leaf chunk number {i} with some body text." for i in range(n)]
    return texts, _fake_embed(texts)


class TestBottomUpConstruction(unittest.TestCase):
    def setUp(self):
        self.params = PaperTreeParams(reduction_dimension=2, num_layers=5)
        self.texts, self.embs = _leaf_inputs(16)

    def _build(self, **kw):
        return build_paper_tree(
            self.texts,
            self.embs,
            params=self.params,
            summarize_batch_fn=lambda cs: [f"summary of {len(c)} chars" for c in cs],
            embed_fn=_fake_embed,
            cluster_fn=_pairwise_cluster_fn,
            **kw,
        )

    def test_leaves_are_layer_zero_and_all_present(self):
        tree = self._build()
        self.assertEqual(len(tree.layer_to_nodes[0]), 16)
        for nid in tree.layer_to_nodes[0]:
            node = tree.nodes[nid]
            self.assertTrue(node.is_leaf)
            self.assertEqual(node.layer, 0)
            self.assertEqual(node.children, [])

    def test_tree_grows_upward_and_shrinks(self):
        tree = self._build()
        sizes = [len(tree.layer_to_nodes[k]) for k in sorted(tree.layer_to_nodes)]
        self.assertGreater(len(sizes), 1, "no summary layer was built")
        for lo, hi in zip(sizes, sizes[1:]):
            self.assertLess(hi, lo, f"layer did not shrink: {sizes}")

    def test_stop_condition_matches_reference(self):
        # Reference breaks when a layer holds <= reduction_dimension + 1.
        tree = self._build()
        top = max(tree.layer_to_nodes)
        self.assertLessEqual(
            len(tree.layer_to_nodes[top]), self.params.reduction_dimension + 1
        )

    def test_num_layers_is_an_upper_bound(self):
        params = PaperTreeParams(reduction_dimension=1, num_layers=2)
        tree = build_paper_tree(
            self.texts, self.embs, params=params,
            summarize_batch_fn=lambda cs: ["s"] * len(cs), embed_fn=_fake_embed,
            cluster_fn=_pairwise_cluster_fn,
        )
        self.assertLessEqual(max(tree.layer_to_nodes), 2)

    def test_summary_nodes_have_children_and_text(self):
        tree = self._build()
        for node in tree.summary_nodes():
            self.assertTrue(node.children)
            self.assertTrue(node.text)
            self.assertIsNotNone(node.embedding)

    def test_multi_parent_is_recorded(self):
        tree = self._build()
        first_leaf = tree.layer_to_nodes[0][0]
        self.assertGreater(
            len(tree.nodes[first_leaf].parent_ids), 1,
            "soft-cluster double membership did not produce two parents",
        )
        stats = tree_stats(tree)
        self.assertGreater(stats["n_multi_parent_nodes"], 0)

    def test_leaf_index_closure_is_transitive_and_sorted(self):
        tree = self._build()
        for node in tree.summary_nodes():
            expected: set[int] = set()
            for cid in node.children:
                expected.update(tree.nodes[cid].leaf_indices)
            self.assertEqual(node.leaf_indices, sorted(expected))
            self.assertEqual(node.leaf_indices, sorted(set(node.leaf_indices)))

    def test_parent_child_links_are_consistent(self):
        tree = self._build()
        for node in tree.nodes.values():
            for cid in node.children:
                self.assertIn(node.node_id, tree.nodes[cid].parent_ids)
            for pid in node.parent_ids:
                self.assertIn(node.node_id, tree.nodes[pid].children)

    def test_summary_call_count_is_tracked(self):
        calls = []
        build_paper_tree(
            self.texts, self.embs, params=self.params,
            summarize_batch_fn=lambda cs: (calls.extend(cs), ["s"] * len(cs))[1],
            embed_fn=_fake_embed, cluster_fn=_pairwise_cluster_fn,
        )
        self.assertGreater(len(calls), 0)

    def test_empty_summary_node_survives_but_stops_clustering(self):
        tree = build_paper_tree(
            self.texts, self.embs, params=self.params,
            summarize_batch_fn=lambda cs: [""] * len(cs),
            embed_fn=_fake_embed, cluster_fn=_pairwise_cluster_fn,
        )
        # Layer 1 exists, carries no embeddings, and nothing built above it.
        self.assertIn(1, tree.layer_to_nodes)
        self.assertTrue(tree.layer_to_nodes[1])
        for nid in tree.layer_to_nodes[1]:
            self.assertIsNone(tree.nodes[nid].embedding)
        self.assertEqual(max(tree.layer_to_nodes), 1)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            build_paper_tree([], np.zeros((0, EMB_DIM)), params=self.params,
                             summarize_batch_fn=lambda cs: ["s"] * len(cs), embed_fn=_fake_embed)
        with self.assertRaises(ValueError):
            build_paper_tree(["a", "b"], _fake_embed(["a"]), params=self.params,
                             summarize_batch_fn=lambda cs: ["s"] * len(cs), embed_fn=_fake_embed)
        with self.assertRaises(ValueError):
            build_paper_tree(["a"], np.zeros(EMB_DIM), params=self.params,
                             summarize_batch_fn=lambda cs: ["s"] * len(cs), embed_fn=_fake_embed)


class TestDeterminism(unittest.TestCase):
    """The cache-identity contract: same inputs -> byte-identical artifact."""

    def setUp(self):
        self.params = PaperTreeParams(reduction_dimension=2)
        self.texts, self.embs = _leaf_inputs(16)

    def _ids_and_texts(self, **kw):
        tree = build_paper_tree(
            self.texts, self.embs, params=self.params,
            summarize_batch_fn=lambda cs: [f"S:{len(c)}" for c in cs],
            embed_fn=_fake_embed, cluster_fn=_pairwise_cluster_fn, **kw
        )
        return [(nid, tree.nodes[nid].text) for nid in tree.all_node_ids()]

    def test_repeat_builds_are_identical(self):
        self.assertEqual(self._ids_and_texts(), self._ids_and_texts())

    def test_batch_shape_cannot_move_ids_or_topology(self):
        """What batching IS allowed to change, and what it is not.

        Replaces the old `max_workers` throughput-knob test, whose
        subject was deleted when the ThreadPoolExecutor went away (it was
        unsafe against a local model — one lru_cached tokenizer, mutated
        per call).

        THE PROPERTY, and it is narrower than the old one on purpose:
        node IDS and TOPOLOGY are a function of member position in the
        input layer, computed before any summary call is dispatched, so
        NO batching decision can perturb them. The reference, by
        contrast, assigns indices inside a Lock-guarded dict while
        summarising on a thread pool, making its ids a function of
        completion order.

        WHAT IS DELIBERATELY NOT ASSERTED: that summary TEXT is invariant
        under batch shape. It may not be — padding and batched-matmul
        reduction order can flip argmax on near-ties even at temperature
        0. That is precisely the open question, and the project's answer
        is to NAME batch_size and max_padded_tokens in M4's substrate key
        (raptor_paper.paper_substrate_extra) rather than to assume
        invariance here. Asserting text invariance with a fake
        summariser would prove nothing about the real one and would read
        as a guarantee the code does not make.
        """
        def ids_and_shape(chunk_size):
            # A summariser that batches its input differently every time,
            # while returning position-stable text, isolates SHAPE from
            # TEXT: any id or topology difference would be caused by the
            # grouping alone.
            def summarize(cs):
                out = []
                for s in range(0, len(cs), chunk_size):
                    out.extend(f"S:{len(c)}" for c in cs[s : s + chunk_size])
                return out

            tree = build_paper_tree(
                self.texts, self.embs, params=self.params,
                summarize_batch_fn=summarize, embed_fn=_fake_embed,
                cluster_fn=_pairwise_cluster_fn,
            )
            return (
                tree.all_node_ids(),
                {nid: sorted(tree.nodes[nid].children) for nid in tree.nodes},
                tree.layer_to_nodes,
            )

        self.assertEqual(ids_and_shape(1), ids_and_shape(4))
        self.assertEqual(ids_and_shape(1), ids_and_shape(1000))

    def test_misaligned_summary_count_is_refused(self):
        """A short return would silently attach summaries to the wrong
        clusters — positional alignment fails quietly, so it must raise."""
        with self.assertRaises(RuntimeError):
            build_paper_tree(
                self.texts, self.embs, params=self.params,
                summarize_batch_fn=lambda cs: ["s"] * (len(cs) - 1),
                embed_fn=_fake_embed, cluster_fn=_pairwise_cluster_fn,
            )


class TestGetText(unittest.TestCase):
    """Port fidelity of the summariser's input format (utils.get_text)."""

    def test_collapses_newlines_and_joins_on_blank_lines(self):
        nodes = [
            PaperNode(node_id="a", layer=0, text="one\ntwo"),
            PaperNode(node_id="b", layer=0, text="three"),
        ]
        self.assertEqual(get_text(nodes), "one two\n\nthree\n\n")

    def test_trailing_separator_is_reference_behaviour(self):
        nodes = [PaperNode(node_id="a", layer=0, text="x")]
        self.assertTrue(get_text(nodes).endswith("\n\n"))


class TestCollapsedIndex(unittest.TestCase):
    def setUp(self):
        self.params = PaperTreeParams(reduction_dimension=2)
        texts, embs = _leaf_inputs(16)
        self.tree = build_paper_tree(
            texts, embs, params=self.params,
            summarize_batch_fn=lambda cs: [f"S:{len(c)}" for c in cs],
            embed_fn=_fake_embed, cluster_fn=_pairwise_cluster_fn,
        )

    def test_index_contains_every_embedded_node_all_layers(self):
        idx = build_collapsed_index(self.tree)
        embedded = [
            n for n in self.tree.nodes.values() if n.embedding is not None
        ]
        self.assertEqual(len(idx.refs), len(embedded))
        self.assertEqual(idx.faiss_index.ntotal, len(embedded))
        # Paper collapses the ENTIRE tree: leaves AND summaries present.
        layers = {r["layer"] for r in idx.refs}
        self.assertIn(0, layers)
        self.assertGreater(max(layers), 0)
        self.assertTrue(any(r["is_leaf"] for r in idx.refs))
        self.assertTrue(any(not r["is_leaf"] for r in idx.refs))

    def test_no_root_is_excluded(self):
        """Contrast with src/raptor.py, which drops the synthetic root."""
        idx = build_collapsed_index(self.tree)
        top = max(self.tree.layer_to_nodes)
        top_ids = {
            nid for nid in self.tree.layer_to_nodes[top]
            if self.tree.nodes[nid].embedding is not None
        }
        ref_ids = {r["node_id"] for r in idx.refs}
        self.assertTrue(top_ids <= ref_ids)

    def test_refs_align_positionally_with_the_index(self):
        idx = build_collapsed_index(self.tree)
        node = self.tree.nodes[idx.refs[0]["node_id"]]
        q = np.asarray(node.embedding, dtype=np.float32).reshape(1, -1)
        _, found = idx.faiss_index.search(q, 1)
        self.assertEqual(int(found[0][0]), 0)


class TestSerialisation(unittest.TestCase):
    def setUp(self):
        self.params = PaperTreeParams(reduction_dimension=2)
        texts, embs = _leaf_inputs(16)
        self.tree = build_paper_tree(
            texts, embs, params=self.params,
            summarize_batch_fn=lambda cs: [f"S:{len(c)}" for c in cs],
            embed_fn=_fake_embed, cluster_fn=_pairwise_cluster_fn,
        )

    def test_tree_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            tj, ep = Path(td) / "t.json", Path(td) / "e.npy"
            save_paper_tree(self.tree, tj, ep)
            back = load_paper_tree(tj, ep)

        self.assertEqual(back.n_layers, self.tree.n_layers)
        self.assertEqual(back.params, self.tree.params)
        self.assertEqual(set(back.nodes), set(self.tree.nodes))
        self.assertEqual(back.layer_to_nodes, self.tree.layer_to_nodes)
        for nid, node in self.tree.nodes.items():
            other = back.nodes[nid]
            self.assertEqual(other.text, node.text)
            self.assertEqual(other.children, node.children)
            self.assertEqual(other.parent_ids, node.parent_ids)
            self.assertEqual(other.leaf_indices, node.leaf_indices)
            if node.embedding is None:
                self.assertIsNone(other.embedding)
            else:
                np.testing.assert_allclose(other.embedding, node.embedding)

    def test_schema_mismatch_refuses_to_load(self):
        import json as _json

        with tempfile.TemporaryDirectory() as td:
            tj, ep = Path(td) / "t.json", Path(td) / "e.npy"
            save_paper_tree(self.tree, tj, ep)
            obj = _json.loads(tj.read_text())
            obj["schema"] = "some_other_schema_v9"
            tj.write_text(_json.dumps(obj))
            with self.assertRaises(ValueError):
                load_paper_tree(tj, ep)

    def test_collapsed_index_round_trip(self):
        idx = build_collapsed_index(self.tree)
        with tempfile.TemporaryDirectory() as td:
            fp, mp = Path(td) / "i.faiss", Path(td) / "m.json"
            save_collapsed_index(idx, fp, mp)
            back = load_collapsed_index(fp, mp)
        self.assertEqual(back.dim, idx.dim)
        self.assertEqual(back.refs, idx.refs)
        self.assertEqual(back.faiss_index.ntotal, idx.faiss_index.ntotal)


class TestCacheIdentity(unittest.TestCase):
    """Lever B, strict reading: own extras function, raptor.py never opened."""

    def _extra(self, **kw):
        base = dict(
            params=PaperTreeParams(),
            summary_model="gpt-4o-mini",
            summary_prompt_version="raptor_paper_v1",
        )
        base.update(kw)
        return paper_substrate_extra(**base)

    def test_emits_the_same_seven_base_fields_as_the_shared_extras(self):
        shared = raptor_substrate_extra(
            build=RaptorBuildParams(),
            summary_model="gpt-4o-mini",
            summary_prompt_version="v1",
            include_root=False,
            rrf_k=60,
        )
        mine = self._extra()
        self.assertTrue(
            set(shared) <= set(mine),
            f"missing base fields: {set(shared) - set(mine)}",
        )

    def test_carries_the_m4_only_keys_that_close_the_landmine(self):
        mine = self._extra()
        # The original landmine: the shared extras fold tree PARAMETERS
        # but never the clustering ALGORITHM, so swapping KMeans for
        # UMAP+GMM would have changed artifacts without moving the key.
        self.assertEqual(mine["clustering"]["algo"], "umap_gmm_bic")
        self.assertEqual(mine["tree_schema"], PAPER_TREE_SCHEMA_VERSION)
        self.assertIn("chunker_impl", mine)
        self.assertIn("summary_max_tokens", mine)

    def test_batch_shape_is_named_in_the_key(self):
        """Summaries are CACHED, so batch composition is not free.

        Batch composition can change generated text at temperature 0.
        Answers are not cached, so their variance is documented noise;
        summaries ARE the artifact this key names, so the shape that
        produced them has to be in it. Both knobs, because
        token_budget_batches takes batch_size as a count cap and so both
        participate in composition.
        """
        mine = self._extra()
        self.assertIn("summary_batch_size", mine)
        self.assertIn("summary_max_padded_tokens", mine)
        self.assertNotEqual(mine, self._extra(summary_batch_size=8))
        self.assertNotEqual(mine, self._extra(summary_max_padded_tokens=8000))

    def test_extras_defaults_match_m4config(self):
        """The extras defaults and M4Config must not drift apart.

        m4_raptor passes every value explicitly, but the test helpers and
        any future caller lean on these defaults; a silent divergence
        would make a test compute a key production never uses.
        """
        m4 = M4Config()
        mine = self._extra()
        self.assertEqual(mine["summary_batch_size"], m4.summary_batch_size)
        self.assertEqual(
            mine["summary_max_padded_tokens"], m4.summary_max_padded_tokens
        )
        self.assertEqual(mine["summary_max_tokens"], m4.summary_max_tokens)

    def test_tree_field_is_the_paper_params_not_raptor_build_params(self):
        mine = self._extra()
        self.assertEqual(mine["tree"], asdict(PaperTreeParams()))
        self.assertNotEqual(mine["tree"], asdict(RaptorBuildParams()))

    def test_params_change_moves_the_extras(self):
        a = self._extra()
        b = self._extra(params=PaperTreeParams(gmm_threshold=0.2))
        self.assertNotEqual(a, b)

    def test_collapses_the_entire_tree_by_default(self):
        self.assertTrue(self._extra()["include_root_in_flat_index"])

    def test_reference_defaults_are_the_documented_ones(self):
        p = PaperTreeParams()
        self.assertEqual(p.reduction_dimension, 10)
        self.assertEqual(p.gmm_threshold, 0.1)
        self.assertEqual(p.max_length_in_cluster, 3500)
        self.assertEqual(p.num_layers, 5)
        self.assertEqual(p.local_n_neighbors, 10)
        self.assertEqual(p.metric, "cosine")
        self.assertEqual(p.bic_max_clusters, 50)
        # Ruling 3: the reference's seed inconsistency, reproduced.
        self.assertEqual(p.bic_random_state, 224)
        self.assertEqual(p.gmm_random_state, 0)


class TestRealClustering(unittest.TestCase):
    """The real UMAP+GMM path. Slow; kept deliberately small."""

    @staticmethod
    def _structured(n_per: int = 12, dim: int = 16, groups: int = 3):
        rng = np.random.RandomState(7)
        centres = rng.randn(groups, dim) * 3.0
        rows = [
            centres[g] + rng.randn(dim) * 0.25
            for g in range(groups)
            for _ in range(n_per)
        ]
        X = np.vstack(rows).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return X

    def test_perform_clustering_returns_soft_clusters(self):
        X = self._structured()
        nodes = [
            PaperNode(node_id=f"L0_{i}", layer=0, text=f"node {i} text",
                      leaf_indices=[i], embedding=X[i])
            for i in range(len(X))
        ]
        params = PaperTreeParams(reduction_dimension=3, bic_max_clusters=6)
        stats: dict = {}
        clusters = perform_clustering(nodes, params, stats)

        self.assertTrue(clusters)
        for c in clusters:
            self.assertTrue(c)
        covered = {n.node_id for c in clusters for n in c}
        self.assertEqual(covered, {n.node_id for n in nodes},
                         "clustering dropped nodes")

    def test_recluster_guard_trips_and_is_counted(self):
        X = self._structured(n_per=6, dim=16, groups=3)
        nodes = [
            PaperNode(node_id=f"L0_{i}", layer=0, text="word " * 50,
                      leaf_indices=[i], embedding=X[i])
            for i in range(len(X))
        ]
        # Every cluster busts a 1-token budget, and depth 0 is already at
        # the bound -> the guard must fire rather than recursing forever.
        params = PaperTreeParams(
            reduction_dimension=3, bic_max_clusters=5,
            max_length_in_cluster=1, max_recluster_depth=0,
        )
        stats: dict = {}
        clusters = perform_clustering(nodes, params, stats)
        self.assertTrue(clusters)
        self.assertGreater(stats.get("recluster_guard_trips", 0), 0)

    def test_end_to_end_real_tree(self):
        X = self._structured(n_per=12, dim=16, groups=3)
        texts = [f"document sentence number {i}." for i in range(len(X))]
        params = PaperTreeParams(reduction_dimension=3, bic_max_clusters=6)

        tree = build_paper_tree(
            texts, X, params=params,
            summarize_batch_fn=lambda cs: [f"summary covering {len(c.split())} words" for c in cs],
            embed_fn=lambda ts: _fake_embed_dim(ts, X.shape[1]),
        )

        self.assertGreaterEqual(tree.n_layers, 2, "no summary layer built")
        self.assertEqual(len(tree.layer_to_nodes[0]), len(X))
        s = tree_stats(tree)
        self.assertGreater(s["n_summary_nodes"], 0)
        self.assertGreater(s["mean_children_per_parent"], 1.0)
        # Every leaf must be reachable from some summary node.
        reached: set[int] = set()
        for node in tree.summary_nodes():
            reached.update(node.leaf_indices)
        self.assertEqual(reached, set(range(len(X))))

        idx = build_collapsed_index(tree)
        self.assertGreater(len(idx.refs), len(X), "summaries missing from index")


if __name__ == "__main__":
    unittest.main()
