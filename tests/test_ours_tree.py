"""Pins ours' own tree: k-means from the top down, summaries from the bottom up,
no tree over a small unit, and that the query steps read it as they read any tree."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from src.config import DEFAULT_CONFIG
from src.eval.types import CorpusItem
from src.raptor_paper import load_paper_tree, save_paper_tree
from tests.test_ours_smoke import (
    DECOMPOSITIONS,
    DIM,
    FakeReader,
    fake_embed,
    llama_config,
    ours_fakes,
)

PARAMS = DEFAULT_CONFIG.m4.paper


def _blobs(sizes, dim=DIM, seed=0):
    """Unit vectors in tight groups, one group per axis, shuffled together."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for axis, size in enumerate(sizes):
        for _ in range(size):
            v = rng.normal(0, 0.02, dim)
            v[axis] += 1.0
            rows.append(v / np.linalg.norm(v))
            truth.append(axis)
    order = rng.permutation(len(rows))
    return np.asarray(rows, dtype=np.float32)[order], [truth[i] for i in order]


def _embed(texts):
    return fake_embed(texts)


class _Summaries:
    """Records every context and every batch; each summary names its call."""

    def __init__(self):
        self.batches = []
        self.n = 0

    def __call__(self, contexts):
        self.batches.append(list(contexts))
        out = []
        for _ in contexts:
            out.append(f"summary number {self.n}")
            self.n += 1
        return out


def _tree(sizes, summarize=None, **overrides):
    vectors, truth = _blobs(sizes)
    texts = [f"leaf {i} of group {t}" for i, t in enumerate(truth)]
    summarize = summarize or _Summaries()
    with mock.patch.multiple(settings, **overrides) if overrides else _noop():
        tree = om.kmeans_tree(texts, vectors, summarize, _embed, PARAMS)
    return tree, truth, summarize


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _above(tree):
    return [n for n in tree.nodes.values() if n.layer > 0]


def _root(tree):
    return tree.nodes[tree.layer_to_nodes[max(tree.layer_to_nodes)][0]]


class TestSmallUnits(unittest.TestCase):

    def test_a_unit_under_the_minimum_has_leaves_only(self):
        calls = _Summaries()
        tree, _, _ = _tree([10, 13], calls)
        self.assertEqual(sorted(tree.layer_to_nodes), [0])
        self.assertEqual(len(tree.nodes), 23)
        self.assertEqual(calls.batches, [])
        self.assertTrue(tree.stats["degenerate_no_tree"])
        self.assertEqual(tree.stats["n_summary_calls"], 0)
        self.assertTrue(all(not n.parent_ids for n in tree.nodes.values()))

    def test_every_leaf_is_its_own_cluster_there(self):
        """Step 5's cap of 4 a cluster must not bite on a unit with no clusters."""
        vectors, _ = _blobs([10, 13])
        texts = [f"leaf {i}" for i in range(23)]
        tree = om.kmeans_tree(texts, vectors, _Summaries(), _embed, PARAMS)
        chunks = [om.Chunk(chunk_id=f"d::{i}", doc_id="d", text=t, n_words=2, position=i)
                  for i, t in enumerate(texts)]
        index = om.build_tree_index(chunks, vectors, tree, {"d": ("d", "d", "")})
        self.assertEqual(len(set(index.cluster)), 23)
        self.assertIsNone(index.root)
        self.assertEqual(index.top_nodes, [])


class TestTopology(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tree, cls.truth, cls.summaries = _tree([40, 40, 40, 40])

    def test_the_groups_are_found(self):
        """Four tight groups of 40: the root's four children are those groups."""
        children = [self.tree.nodes[c] for c in _root(self.tree).children]
        self.assertEqual(len(children), 4)
        for child in children:
            self.assertEqual(len({self.truth[i] for i in child.leaf_indices}), 1)

    def test_every_leaf_has_one_parent_and_the_root_none(self):
        for node in self.tree.nodes.values():
            self.assertEqual(len(node.parent_ids), 0 if node is _root(self.tree) else 1)

    def test_children_share_out_their_parents_leaves(self):
        for node in _above(self.tree):
            under = [i for c in node.children for i in self.tree.nodes[c].leaf_indices]
            self.assertEqual(sorted(under), sorted(node.leaf_indices))

    def test_a_layer_is_the_height_above_the_leaves(self):
        for node in _above(self.tree):
            self.assertEqual(node.layer,
                             1 + max(self.tree.nodes[c].layer for c in node.children))

    def test_every_node_above_the_leaves_has_a_summary_and_a_vector(self):
        for node in _above(self.tree):
            self.assertTrue(node.text.startswith("summary number"))
            self.assertEqual(node.embedding.shape, (DIM,))

    def test_the_same_input_builds_the_same_tree(self):
        again, _, _ = _tree([40, 40, 40, 40])
        self.assertEqual({k: (n.children, n.text) for k, n in self.tree.nodes.items()},
                         {k: (n.children, n.text) for k, n in again.nodes.items()})

    def test_stats(self):
        self.assertFalse(self.tree.stats["degenerate_no_tree"])
        self.assertEqual(self.tree.stats["n_summary_calls"], len(_above(self.tree)))


class TestStops(unittest.TestCase):

    def test_two_groups_when_the_size_supports_only_two(self):
        """30 leaves make two groups, the most that hold about 24 is one, the least is two."""
        tree, _, _ = _tree([15, 15])
        self.assertEqual(len(_root(tree).children), 2)

    def test_three_groups_at_three_minimum_sizes(self):
        tree, _, _ = _tree([24, 24, 24])
        self.assertEqual(len(_root(tree).children), 3)

    def test_never_more_than_the_branching_factor(self):
        tree, _, _ = _tree([40] * 8)
        for node in _above(tree):
            if node.layer > 1:
                self.assertLessEqual(len(node.children), settings.TREE_BRANCHING_FACTOR)

    def test_a_group_under_the_minimum_stops_at_its_leaves(self):
        tree, _, _ = _tree([15, 15])
        for node in _above(tree):
            if len(node.leaf_indices) < settings.TREE_MIN_CLUSTER_SIZE:
                self.assertEqual(node.layer, 1)
                self.assertEqual(len(node.children), len(node.leaf_indices))

    def test_the_depth_limit_stops_the_split(self):
        tree, _, _ = _tree([40] * 4, TREE_MAX_DEPTH=1)
        self.assertEqual(sorted(tree.layer_to_nodes), [0, 1, 2])
        for child in _root(tree).children:
            self.assertEqual(tree.nodes[child].layer, 1)
            self.assertEqual(len(tree.nodes[child].children), 40)

    def test_identical_vectors_do_not_loop(self):
        """k-means finds one group: the node stops at its leaves."""
        vectors = np.ones((30, 4), dtype=np.float32) / 2.0
        texts = [f"leaf {i}" for i in range(30)]
        with mock.patch("ours.model.kmeans_groups", return_value=[list(range(30))]):
            tree = om.kmeans_tree(texts, vectors, _Summaries(), _embed, PARAMS)
        self.assertEqual(sorted(tree.layer_to_nodes), [0, 1])
        self.assertEqual(len(_root(tree).children), 30)


class TestSummaries(unittest.TestCase):

    def test_built_from_the_bottom_up_one_batch_a_layer(self):
        tree, _, calls = _tree([40, 40, 40, 40])
        n_layers = len([k for k in tree.layer_to_nodes if k > 0])
        self.assertEqual(len(calls.batches), n_layers)
        self.assertEqual([len(b) for b in calls.batches],
                         [len(tree.layer_to_nodes[k]) for k in sorted(tree.layer_to_nodes) if k > 0])

    def test_low_nodes_read_leaves_and_high_nodes_read_summaries(self):
        tree, _, calls = _tree([40, 40, 40, 40])
        self.assertTrue(all(ctx.startswith("leaf") for ctx in calls.batches[0]))
        self.assertTrue(all(ctx.startswith("summary number") for ctx in calls.batches[-1]))

    def test_the_root_is_summarised_too(self):
        tree, _, _ = _tree([40, 40, 40, 40])
        self.assertTrue(_root(tree).text)

    def _node(self, vectors, texts):
        nodes = {f"c{i}": om.PaperNode(node_id=f"c{i}", layer=0, text=t,
                                       embedding=np.asarray(v, dtype=np.float32))
                 for i, (v, t) in enumerate(zip(vectors, texts))}
        return om.PaperNode(node_id="p", layer=1, text="", children=list(nodes)), nodes

    def test_the_context_is_the_childrens_text_in_order(self):
        node, nodes = self._node([[1, 0], [1, 0]], ["first one", "second one"])
        self.assertEqual(om.summary_context(node, nodes), "first one\n\nsecond one")

    def test_the_limit_keeps_those_nearest_the_centre_in_order(self):
        far, near = [0.0, 1.0], [1.0, 0.0]
        node, nodes = self._node([near, far, near, near], ["aa bb", "far away", "cc dd", "ee ff"])
        with mock.patch.object(settings, "SUMMARY_CONTEXT_TOKENS", 4):
            self.assertEqual(om.summary_context(node, nodes), "aa bb\n\ncc dd")

    def test_one_child_is_kept_however_long(self):
        node, nodes = self._node([[1, 0]], ["word " * 50])
        with mock.patch.object(settings, "SUMMARY_CONTEXT_TOKENS", 3):
            self.assertEqual(om.summary_context(node, nodes), "word " * 50)


class TestAsTheQueryStepsReadIt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tree, cls.truth, _ = _tree([40, 40, 40, 40])
        cls.vectors = np.vstack([cls.tree.nodes[f"L0_{i:06d}"].embedding for i in range(160)])
        cls.chunks = [om.Chunk(chunk_id=f"d::{i}", doc_id="d",
                               text=cls.tree.nodes[f"L0_{i:06d}"].text, n_words=5, position=i)
                      for i in range(160)]

    def _index(self, tree):
        return om.build_tree_index(self.chunks, self.vectors, tree, {"d": ("d", "d", "")})

    def test_the_root_is_found_and_never_searched(self):
        index = self._index(self.tree)
        root = _root(self.tree)
        self.assertEqual(index.root, root.node_id)
        self.assertNotIn(root.node_id, index.node_ids)
        self.assertEqual(index.top_nodes, root.children)

    def test_a_leafs_cluster_is_its_parent(self):
        index = self._index(self.tree)
        for i in range(160):
            self.assertEqual(index.cluster[i],
                             self.tree.nodes[f"L0_{i:06d}"].parent_ids[0])

    def test_it_survives_the_file_format(self):
        with tempfile.TemporaryDirectory() as td:
            js, emb = Path(td) / "t.json", Path(td) / "t.npy"
            save_paper_tree(self.tree, js, emb)
            back = load_paper_tree(js, emb)
        self.assertEqual(sorted(back.nodes), sorted(self.tree.nodes))
        for key, node in self.tree.nodes.items():
            self.assertEqual((back.nodes[key].layer, back.nodes[key].children,
                              back.nodes[key].parent_ids, back.nodes[key].text),
                             (node.layer, node.children, node.parent_ids, node.text))
            np.testing.assert_allclose(back.nodes[key].embedding, node.embedding)


class TestThroughTheSystem(unittest.TestCase):
    """A unit big enough for a tree goes through ours end to end."""

    @classmethod
    def setUpClass(cls):
        items = []
        for k in range(40):
            title = f"Place{k}"
            for s in range(6):
                items.append(CorpusItem(
                    item_id=f"{title}::{s}", parent_id=title, span_id=f"sent{s}",
                    text=f"The place{k} record {s} says topic{k} lived near river{k} and "
                         f"worked with friend{k} on the farm{k} for {1800 + s} years "
                         f"before moving to harbour{k} with family{k} and animal{k}.",
                    metadata={"title": title, "sent_id": s}))
        cls.question = "Where did topic3 live?"
        cls.td = tempfile.TemporaryDirectory()
        reader = FakeReader(DECOMPOSITIONS, {cls.question: "near river3"})
        with ours_fakes(Path(cls.td.name), reader):
            cls.ours = om.OursSystem(config=llama_config())
            cls.ours.index_items(items)
            cls.prepared = cls.ours.prepare(cls.question)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_the_unit_has_a_tree(self):
        self.assertFalse(self.ours._builder.index_stats["degenerate_no_tree"])
        self.assertGreater(self.ours._builder.index_stats["n_summary_calls_at_index"], 1)

    def test_summaries_are_searched_and_shown_but_never_the_root(self):
        index = self.ours._tree_index
        self.assertTrue(any(index.layer[n] > 0 for n in index.node_ids))
        self.assertNotIn(index.root, index.node_ids)
        self.assertNotIn(index.root, [r.chunk.chunk_id for r in self.prepared.packed])

    def test_the_context_has_leaves_within_the_budget(self):
        self.assertGreater(len(self.prepared.retrieved), 0)
        self.assertLessEqual(len(self.prepared.retrieved), settings.FINAL_CONTEXT_CHUNKS)
        self.assertTrue(all(r.chunk.gold_provenance for r in self.prepared.retrieved))

    def test_summaries_use_the_designs_prompt(self):
        from tests.test_ours_smoke import fake_generate_batch

        seen = []

        def spy(system_prompts, user_prompts, cfg=None, **kw):
            seen.append((set(system_prompts), cfg, kw))
            return fake_generate_batch(system_prompts, user_prompts, cfg=cfg, **kw)

        with tempfile.TemporaryDirectory() as td, \
                ours_fakes(Path(td), FakeReader({}, {})), \
                mock.patch("src.models.generate_batch", spy):
            items = [CorpusItem(item_id=f"t{k}", parent_id=f"t{k}", span_id="<whole>",
                                text=f"topic{k} " * 100 + ".", metadata={"title": f"t{k}"})
                     for k in range(30)]
            om.OursSystem(config=llama_config()).index_items(items)
        self.assertTrue(seen)
        for prompts, cfg, kw in seen:
            self.assertEqual(prompts, {settings.SUMMARY_PROMPT})
            self.assertEqual(cfg.max_new_tokens, settings.SUMMARY_MAX_TOKENS)
            self.assertEqual(cfg.temperature, 0.0)
            self.assertEqual(cfg.model, settings.READER)
            self.assertEqual(kw["batch_size"], llama_config().m4.summary_batch_size)


if __name__ == "__main__":
    unittest.main()
