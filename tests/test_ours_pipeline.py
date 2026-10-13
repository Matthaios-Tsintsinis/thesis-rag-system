"""Pins ours' query steps on a small hand-built tree whose similarities are
known: budgets, expansion by layer, multi-branch, structure, cluster cap,
merge and backfill, orientation, and the document ranking. Also pins that
the hand-written tokenizer, RRF and token counter agree with the harness."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from src.chunking import Chunk
from src.raptor_paper import PaperNode, PaperTree, PaperTreeParams

# Cosine of each leaf to the view (1, 0, 0). Leaves 0-5 are document d0,
# leaves 6-11 document d1. Layer 1: A B C D (three leaves each); layer 2:
# P = A+B, Q = C+D; layer 3: R = P+Q, the single top node (a root).
LEAF_COS = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.95, 0.15, 0.6, 0.05, 0.5, 0.4]
NODE_COS = {"A": 0.85, "B": 0.45, "C": 0.75, "D": 0.35, "P": 0.6, "Q": 0.55,
            "R": 0.5}
CHILDREN = {"A": [0, 1, 2], "B": [3, 4, 5], "C": [6, 7, 8], "D": [9, 10, 11],
            "P": ["A", "B"], "Q": ["C", "D"], "R": ["P", "Q"]}
LAYER = {"A": 1, "B": 1, "C": 1, "D": 1, "P": 2, "Q": 2, "R": 3}
VIEW = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _vec(c: float) -> np.ndarray:
    """A unit vector whose cosine to VIEW is c."""
    return np.array([c, math.sqrt(1 - c * c), 0.0], dtype=np.float32)


def _leaf_id(i: int) -> str:
    return f"L0_{i:06d}"


def _tree(with_summaries: bool = True) -> PaperTree:
    """The hand-built tree, or its leaves alone when with_summaries is False."""
    nodes = {}
    for i, c in enumerate(LEAF_COS):
        nodes[_leaf_id(i)] = PaperNode(node_id=_leaf_id(i), layer=0,
                                       text=f"leaf {i}", leaf_indices=[i],
                                       embedding=_vec(c))
    layers = {0: [_leaf_id(i) for i in range(len(LEAF_COS))]}
    if with_summaries:
        for name in ["A", "B", "C", "D", "P", "Q", "R"]:
            kids = [_leaf_id(k) if isinstance(k, int) else k
                    for k in CHILDREN[name]]
            leaves = sorted({i for k in kids for i in nodes[k].leaf_indices})
            nodes[name] = PaperNode(node_id=name, layer=LAYER[name],
                                    text=f"summary {name} " + "word " * 120,
                                    children=kids, leaf_indices=leaves,
                                    embedding=_vec(NODE_COS[name]))
            for k in kids:
                nodes[k].parent_ids.append(name)
            layers.setdefault(LAYER[name], []).append(name)
    return PaperTree(nodes=nodes, layer_to_nodes=layers,
                     n_layers=len(layers), params=PaperTreeParams())


def _chunks() -> list[Chunk]:
    """M4-style leaf chunks with HotpotQA-style provenance."""
    out = []
    for i in range(len(LEAF_COS)):
        doc = "d0.txt" if i < 6 else "d1.txt"
        title = "Alpha Lake" if i < 6 else "Beta Mountain"
        out.append(Chunk(chunk_id=f"{doc}::{i % 6:04d}", doc_id=doc,
                         text=f"leaf {i} text", n_words=3, position=i % 6,
                         gold_provenance=((title, f"sent{i % 6}"),)))
    return out


DOCS = {"d0.txt": ("Alpha Lake", "Alpha Lake", "leaf text"),
        "d1.txt": ("Beta Mountain", "Beta Mountain", "leaf text")}


def _index(with_summaries: bool = True) -> om.TreeIndex:
    leaf_vectors = np.vstack([_vec(c) for c in LEAF_COS])
    return om.build_tree_index(_chunks(), leaf_vectors,
                               _tree(with_summaries), DOCS)


def _no_bm25(index) -> np.ndarray:
    """A view that shares no word with any leaf."""
    return om.bm25_scores(index, "zzz")


class TestTreeIndex(unittest.TestCase):
    """What the index keeps from M4's tree."""

    def test_single_top_node_is_the_root_and_never_searched(self):
        """R is the root: not searchable; multi-branch starts at its children."""
        index = _index()
        self.assertEqual(index.root, "R")
        self.assertNotIn("R", index.node_ids)
        self.assertEqual(index.top_nodes, ["P", "Q"])

    def test_leaves_come_first_in_search_order(self):
        """Search position i is leaf i, so BM25 leaf indices line up."""
        index = _index()
        self.assertEqual(index.node_ids[:12], [_leaf_id(i) for i in range(12)])

    def test_tree_without_summaries_has_no_branches(self):
        """A degenerate unit (leaves only) gives multi-branch nothing."""
        index = _index(with_summaries=False)
        self.assertEqual(index.top_nodes, [])
        self.assertEqual(om.multi_branch_candidates(index, VIEW, _no_bm25(index)), [])


class TestHarnessAgreement(unittest.TestCase):
    """The hand-written helpers agree with the harness's own."""

    def test_tokenizer_matches_m3(self):
        """Lowercase \\w+ tokens, as M3's BM25 uses."""
        from src.retrievers.m3_hybrid import _tokenize
        text = "Scott Derrickson's film, 'Sinister' (2012) — Été!"
        self.assertEqual(om.tokenize(text), _tokenize(text))

    def test_rrf_matches_m3(self):
        """Same scores and order as M3's rrf_fuse on rankings without ties."""
        from src.retrievers.m3_hybrid import rrf_fuse
        rankings = [[3, 1, 2, 7], [1, 5, 3]]
        self.assertEqual(om.rrf(rankings), rrf_fuse(rankings, k=60))

    def test_token_count_matches_the_harness(self):
        """Ours counts evidence tokens exactly as the packer does."""
        from src.prompt_packing import count_tokens
        text = "[Aspect: main] (retrieval_confidence: 0.73)\n  Chunk 1: Paris."
        self.assertEqual(om.count_tokens(text), count_tokens(text))


class TestBudgets(unittest.TestCase):
    """Step 3: 13 aspect slots; one aspect takes all of them."""

    def _aspects(self, scores):
        out = []
        for i, s in enumerate(scores):
            a = om.Aspect(name=f"a{i}", text=f"a{i}", importance=1.0)
            a.score = s
            out.append(a)
        return out

    def test_one_aspect_takes_all_thirteen(self):
        aspects = self._aspects([0.3])
        om.allocate_budgets(aspects)
        self.assertEqual([a.budget for a in aspects], [13])

    def test_two_aspects_are_clamped_to_eight(self):
        aspects = self._aspects([0.9, 0.1])
        om.allocate_budgets(aspects)
        self.assertEqual([a.budget for a in aspects], [8, 5])

    def test_three_equal_aspects_split_thirteen(self):
        aspects = self._aspects([0.5, 0.5, 0.5])
        om.allocate_budgets(aspects)
        self.assertEqual([a.budget for a in aspects], [5, 4, 4])

    def test_budgets_always_sum_to_thirteen(self):
        for scores in ([0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.7, 0.1]):
            aspects = self._aspects(scores)
            om.allocate_budgets(aspects)
            self.assertEqual(sum(a.budget for a in aspects), 13, scores)
            self.assertTrue(all(2 <= a.budget <= 8 for a in aspects), scores)


class TestExpansion(unittest.TestCase):
    """Step 4a: a summary hit becomes leaves according to its layer."""

    def test_low_summary_gives_its_best_leaves(self):
        with mock.patch.object(settings, "SUMMARY_EXPANSION_TOP_K_CHUNKS", 2):
            self.assertEqual(om.expand_summary(_index(), "A", VIEW), [0, 2])

    def test_mid_summary_over_few_leaves_goes_direct(self):
        self.assertEqual(om.expand_summary(_index(), "P", VIEW), [0, 2, 4])

    def test_mid_summary_over_many_leaves_routes_to_children(self):
        with mock.patch.object(
                settings, "MAX_DESCENDANT_CHUNKS_FOR_DIRECT_EXPANSION", 5), \
                mock.patch.object(
                    settings, "MAX_CHILDREN_TO_FOLLOW_FROM_BROAD_SUMMARY", 1):
            self.assertEqual(om.expand_summary(_index(), "P", VIEW), [0, 2, 1])

    def test_high_summary_routes_through_its_two_best_children(self):
        # R's children are P and Q; each goes direct over its six leaves.
        self.assertEqual(om.expand_summary(_index(), "R", VIEW),
                         [0, 2, 4, 6, 8, 10])

    def test_recursion_limit_goes_direct(self):
        with mock.patch.object(settings, "MAX_EXPANSION_RECURSION_DEPTH", 0):
            self.assertEqual(om.expand_summary(_index(), "R", VIEW), [6, 0, 2])

    def test_collapsed_search_ranks_a_leaf_by_its_parents_hit(self):
        """Leaf 1 (cosine 0.1) enters through A's hit and outranks leaf 4."""
        index = _index()
        ranked = [leaf for leaf, _ in
                  om.collapsed_candidates(index, VIEW, _no_bm25(index))]
        self.assertEqual(sorted(ranked), list(range(12)))
        self.assertLess(ranked.index(1), ranked.index(4))


class TestMultiBranch(unittest.TestCase):
    """Step 4b: top-down walk keeping several branches alive."""

    def _leaves(self, **patches):
        index = _index()
        with mock.patch.multiple(settings, **patches) if patches \
                else mock.patch.object(settings, "MULTI_BRANCH_MAX_DEPTH", 4):
            return {leaf for leaf, _ in
                    om.multi_branch_candidates(index, VIEW, _no_bm25(index))}

    def test_default_walk_reaches_every_cluster(self):
        self.assertEqual(self._leaves(), set(range(12)))

    def test_one_first_level_branch_stays_in_its_subtree(self):
        self.assertEqual(self._leaves(MULTI_BRANCH_TOP_K_DEPTH_1=1),
                         set(range(6)))

    def test_one_branch_per_level(self):
        self.assertEqual(self._leaves(MULTI_BRANCH_TOP_K_PER_LEVEL=1),
                         {0, 1, 2, 6, 7, 8})

    def test_leaf_cap_per_terminal(self):
        self.assertEqual(self._leaves(MULTI_BRANCH_LEAVES_PER_PATH=1),
                         {0, 4, 6, 10})

    def test_bm25_enters_the_ordering(self):
        """A key strong only in BM25 still ranks in the fused order."""
        order = om.hybrid_order(["x", "y", "z"],
                                dense={"x": 0.9, "y": 0.8, "z": 0.1},
                                sparse={"x": 0.0, "y": 0.0, "z": 5.0})
        self.assertEqual(order[0], "z")


class TestStructure(unittest.TestCase):
    """Step 4c: section cap, title bias and neighbours."""

    def test_cap_three_per_section_then_neighbours(self):
        index = _index()
        scored = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (6, 0.5)]
        out = dict(om.apply_structure(index, scored, "zzz"))
        # Leaf 3 is cut by the cap, then returns as leaf 2's neighbour.
        self.assertEqual(set(out), {0, 1, 2, 3, 6, 7})
        self.assertAlmostEqual(out[3], 0.7 * 0.9)
        self.assertAlmostEqual(out[7], 0.5 * 0.9)

    def test_title_word_biases_its_section(self):
        index = _index()
        out = dict(om.apply_structure(index, [(0, 0.5), (6, 0.5)],
                                      "the mountain"))
        self.assertAlmostEqual(out[6], 0.5 * 1.15)
        self.assertAlmostEqual(out[0], 0.5)

    def test_stopwords_alone_do_not_bias(self):
        index = _index()
        out = dict(om.apply_structure(index, [(6, 0.5)], "the of and"))
        self.assertAlmostEqual(out[6], 0.5)


class TestPoolAndMerge(unittest.TestCase):
    """Step 5 cluster cap and step 7 merge with backfill."""

    def test_cluster_cap(self):
        with mock.patch.object(settings, "MAX_CHUNKS_PER_RAPTOR_CLUSTER", 2):
            pool = om.aspect_pool(_index(), [[(0, 0.9), (1, 0.8), (2, 0.7),
                                              (6, 0.6)]])
        self.assertEqual([leaf for leaf, _ in pool], [0, 1, 6])

    def test_views_merge_by_best_score(self):
        pool = om.aspect_pool(_index(), [[(0, 0.2)], [(0, 0.9), (6, 0.5)]])
        self.assertEqual(pool, [(0, 0.9), (6, 0.5)])

    def test_backfill_skips_leaves_taken_earlier(self):
        rankings = [[(0, 0.9), (1, 0.8), (2, 0.7)],
                    [(0, 0.95), (6, 0.6), (7, 0.5)],
                    [(6, 0.9), (8, 0.4)]]
        got = om.merge_and_backfill(rankings, [2, 2, 2])
        self.assertEqual([[leaf for leaf, _ in g] for g in got],
                         [[0, 1], [6, 7], [8]])


class TestOrientation(unittest.TestCase):
    """Step 8: titles and parent summaries, never the root, within budget."""

    def _summaries(self, groups, chunk_tokens=10_000):
        return [[key for kind, key, _ in lines if kind == "summary"]
                for lines in om.orientation(_index(), groups, chunk_tokens)]

    def test_nearest_parent_and_one_above_never_the_root(self):
        self.assertEqual(self._summaries([[0, 1], [6]]), [["A", "P"], ["C", "Q"]])

    def test_shared_parents_appear_once(self):
        self.assertEqual(self._summaries([[0], [2]]), [["A", "P"], []])

    def test_per_aspect_cap(self):
        with mock.patch.object(settings, "MAX_PARENT_SUMMARIES_PER_ASPECT", 1):
            self.assertEqual(self._summaries([[0, 3]]), [["A"]])

    def test_titles_come_first(self):
        lines = om.orientation(_index(), [[6]], 10_000)[0]
        self.assertEqual(lines[0][:2], ("title", ("d1.txt", 0)))
        self.assertEqual(lines[0][2], "Beta Mountain")

    def test_summaries_are_cut_to_eighty_tokens(self):
        lines = om.orientation(_index(), [[0]], 10_000)[0]
        for kind, _, text in lines:
            if kind == "summary":
                self.assertLessEqual(om.count_tokens(text),
                                     settings.MAX_PARENT_SUMMARY_TOKENS)

    def test_orientation_stays_within_its_share(self):
        chunk_tokens = 300
        lines = om.orientation(_index(), [[0, 1], [6]], chunk_tokens)
        used = sum(om.count_tokens(t) for group in lines for _, _, t in group)
        self.assertLessEqual(used, chunk_tokens * 0.15 / 0.85)


class TestDocRanking(unittest.TestCase):
    """The row's document ranking follows the scorer's rule."""

    def test_first_occurrence_and_summaries_rank_nothing(self):
        chunks = _chunks()
        summary = Chunk(chunk_id="A", doc_id="", text="s", n_words=1,
                        position=1)
        ranking = om.doc_ranking([chunks[6], summary, chunks[7], chunks[0]])
        self.assertEqual(ranking, ["Beta Mountain", "Alpha Lake"])

    def test_agrees_with_the_hotpotqa_scorer(self):
        """hit@K and MRR from the ranking equal the scorer's on titles."""
        from scripts.replay_retrieval import rank_stats_from_ranking
        from src.eval.alignment import score_retrieval_rank_aware
        from src.eval.hotpotqa import _TITLE_SPAN, _project_to_titles
        from src.retrievers.base import RetrievedChunk

        chunks = _chunks()
        ordered = [chunks[i] for i in (7, 8, 1, 2)]
        gold = frozenset({("Alpha Lake", _TITLE_SPAN)})
        scorer = score_retrieval_rank_aware(
            _project_to_titles([RetrievedChunk(chunk=c, score=1.0, rank=n)
                                for n, c in enumerate(ordered)]),
            gold, k_values=(1, 2, 5))
        mine = rank_stats_from_ranking(
            [(p, _TITLE_SPAN) for p in om.doc_ranking(ordered)], gold,
            (1, 2, 5))
        self.assertEqual(mine["hit_at_k"], scorer["hit_at_k"])
        self.assertEqual(mine["mrr"], scorer["mrr"])


if __name__ == "__main__":
    unittest.main()
