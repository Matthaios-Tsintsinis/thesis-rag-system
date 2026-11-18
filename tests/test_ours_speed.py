"""v20.1: three speed-ups that leave every output as v20's (OURS_VERSIONS "v20.1").

BM25 scored from a postings table gives rank_bm25's own scores, bit for bit; the paraphrase
view is searched once for steps 2 and 4 and the evidence is what searching it twice gave;
a large unit's second-embedder vectors are kept in ours' own cache and read back exactly."""

from __future__ import annotations

import contextlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from rank_bm25 import BM25Okapi

from ours import config as settings
from ours import model as om
from tests.test_ours_second_embedder import second_embed
from tests.test_ours_smoke import DECOMPOSITIONS, FakeReader, Q0, hotpot_unit, llama_config, ours_fakes
from tests.test_ours_sources import NEWS, prepared as news_prepared

WORDS = [f"w{i}" for i in range(40)]


class TestLeafBM25(unittest.TestCase):
    """The postings table scores every leaf as rank_bm25's BM25Okapi does, to the last bit."""

    def assert_same_scores(self, docs, queries):
        okapi, leaves = BM25Okapi(docs), om.LeafBM25(docs)
        for query in queries:
            want, got = okapi.get_scores(query), leaves.get_scores(query)
            self.assertEqual(got.dtype, want.dtype, query)
            self.assertEqual(got.shape, want.shape, query)
            self.assertEqual(got.tobytes(), want.tobytes(), query)

    def test_random_corpora_score_bit_for_bit(self):
        rng = random.Random(20261118)
        for n_docs in (1, 2, 3, 17, 250):
            docs = [[rng.choice(WORDS) for _ in range(rng.randint(1, 30))] for _ in range(n_docs)]
            queries = [[rng.choice(WORDS + ["unseen"]) for _ in range(rng.randint(0, 25))]
                       for _ in range(40)]
            self.assert_same_scores(docs, queries)

    def test_repeated_unknown_and_common_words(self):
        # "a" sits in most documents, so its idf is negative and floored, as rank_bm25 floors it.
        docs = [["a", "a", "b"], ["a", "c"], ["a", "d", "d", "d"], ["e"], ["a"]]
        self.assert_same_scores(docs, [[], ["a"], ["a", "a"], ["zzz"], ["d", "zzz", "d"],
                                       ["e", "a", "b", "c"], ["b"] * 9])

    def test_a_unit_index_scores_through_it(self):
        with tempfile.TemporaryDirectory() as td, ours_fakes(td, FakeReader(DECOMPOSITIONS, {})):
            system = om.OursSystem(config=llama_config())
            system.index_items(hotpot_unit(0).corpus)
            index = system._tree_index
        self.assertIsInstance(index.bm25, om.LeafBM25)
        okapi = BM25Okapi([om.tokenize(c.text) for c in index.chunks])
        for text in (Q0, "topic0x1 castle 1807 friend", "nothing here at all", ""):
            want = np.asarray(okapi.get_scores(om.tokenize(text)), dtype=float)
            self.assertEqual(om.bm25_scores(index, text).tobytes(), want.tobytes(), text)


def search_twice():
    """Patches that make steps 2 and 4 search the paraphrase view each, as v20 did."""
    real_view, real_confidence = om.view_candidates, om.preliminary_confidence

    def view_again(index, view_text, view_vector, aspect_text, focus=None, second_vector=None,
                   searched=None):
        return real_view(index, view_text, view_vector, aspect_text, focus, second_vector)

    def confidence_again(index, aspect, paraphrase_vector, second_vector=None, searched=None):
        return real_confidence(index, aspect, paraphrase_vector, second_vector)

    return [mock.patch("ours.model.view_candidates", view_again),
            mock.patch("ours.model.preliminary_confidence", confidence_again)]


def hotpot_prepared(patches=(), mode=None):
    reader = FakeReader(DECOMPOSITIONS, {})
    with tempfile.TemporaryDirectory() as td, ours_fakes(td, reader), contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        if mode is not None:
            stack.enter_context(mock.patch.object(settings, "SECOND_EMBEDDER_MODE", mode))
        system = om.OursSystem(config=llama_config())
        system.index_items(hotpot_unit(0).corpus)
        return system.prepare(Q0)


def evidence(query):
    """Everything a prepared query hands on: chunks, scores, ranking, prompt, diagnostics."""
    return ([(r.chunk.chunk_id, r.score) for r in query.retrieved],
            [(r.chunk.chunk_id, r.score) for r in query.packed],
            [(r.chunk.chunk_id, r.score) for r in query.scoring_ranking],
            query.user_prompt, query.evidence_tokens, query.n_retrieval_calls,
            json.dumps(query.extra, sort_keys=True, default=str))


class TestParaphraseSearchedOnce(unittest.TestCase):
    def test_each_aspect_runs_one_collapsed_search_fewer(self):
        calls = []
        real = om.collapsed_candidates

        def recording(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        query = hotpot_prepared([mock.patch("ours.model.collapsed_candidates", recording)])
        n_aspects = len(query.extra["ours_aspects"])
        self.assertEqual(n_aspects, 2)
        # Two views an aspect and the global view; step 2 adds none.
        self.assertEqual(len(calls), 2 * n_aspects + 1)

    def test_the_evidence_is_what_searching_twice_gave(self):
        for mode in ("only", "both", "off"):
            once = hotpot_prepared(mode=mode)
            twice = hotpot_prepared(search_twice(), mode=mode)
            self.assertEqual(evidence(once), evidence(twice), mode)

    def test_a_named_source_search_is_unchanged_too(self):
        question = "Did The Verge and TechCrunch agree on the merger of the firms?"
        once, _ = news_prepared(NEWS, question)
        with contextlib.ExitStack() as stack:
            for patch in search_twice():
                stack.enter_context(patch)
            twice, _ = news_prepared(NEWS, question)
        self.assertEqual(evidence(once), evidence(twice))

    def test_the_shared_search_is_the_view_s_own(self):
        with tempfile.TemporaryDirectory() as td, ours_fakes(td, FakeReader(DECOMPOSITIONS, {})):
            system = om.OursSystem(config=llama_config())
            system.index_items(hotpot_unit(0).corpus)
            index = system._tree_index
            aspect = om.Aspect("a", "Where did topic0x0 travel?", 1.0)
            aspect.views = ["Where did topic0x0 go?", "topic0x0 went to the river."]
            vector = om.embed_query([aspect.views[0]])[0][0]
            found = om.paraphrase_search(index, aspect, vector)
            self.assertEqual(om.view_candidates(index, aspect.views[0], vector, aspect.text, searched=found),
                             om.view_candidates(index, aspect.views[0], vector, aspect.text))
            self.assertEqual(om.preliminary_confidence(index, aspect, vector, searched=found),
                             om.preliminary_confidence(index, aspect, vector))


def index_twice(threshold, mode="only", corrupt=False):
    """Index one unit in two systems sharing a cache: (first index, second, embedder calls each, kept files)."""
    calls = []

    def recording(texts, model_name=None):
        calls.append(model_name)
        return second_embed(texts, model_name)

    with tempfile.TemporaryDirectory() as td, ours_fakes(td, FakeReader(DECOMPOSITIONS, {})), \
            mock.patch("src.models.embed_texts", recording), \
            mock.patch.object(settings, "SECOND_EMBEDDER_MODE", mode), \
            mock.patch.object(settings, "SECOND_VECTORS_CACHE_MIN_TEXTS", threshold):
        first = om.OursSystem(config=llama_config())
        first.index_items(hotpot_unit(0).corpus)
        n_first = calls.count(settings.SECOND_EMBEDDER)
        kept = sorted((Path(td) / settings.SECOND_VECTORS_NAMESPACE).glob("*")) \
            if (Path(td) / settings.SECOND_VECTORS_NAMESPACE).exists() else []
        if corrupt and kept:
            np.save(kept[0], np.zeros((3, 2), dtype=np.float32))
        second = om.OursSystem(config=llama_config())
        second.index_items(hotpot_unit(0).corpus)
        n_second = calls.count(settings.SECOND_EMBEDDER) - n_first
        return first._tree_index, second._tree_index, n_first, n_second, [p.name for p in kept]


class TestSecondVectorsKept(unittest.TestCase):
    def test_a_large_unit_is_embedded_once_and_read_back_exactly(self):
        first, second, n_first, n_second, kept = index_twice(threshold=1)
        self.assertEqual((n_first, n_second), (1, 0))
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].endswith(".npy"))
        self.assertEqual(second.node_vectors.dtype, np.float32)
        self.assertEqual(second.node_vectors.tobytes(), first.node_vectors.tobytes())
        self.assertEqual(second.leaf_vectors.tobytes(), first.leaf_vectors.tobytes())

    def test_a_small_unit_is_embedded_again_and_nothing_is_kept(self):
        _, _, n_first, n_second, kept = index_twice(threshold=10**9)
        self.assertEqual((n_first, n_second, kept), (1, 1, []))

    def test_the_vectors_beside_m4s_also_come_back_exactly(self):
        first, second, _, n_second, _ = index_twice(threshold=1, mode="both")
        self.assertEqual(n_second, 0)
        self.assertEqual(second.second_vectors.tobytes(), first.second_vectors.tobytes())

    def test_a_kept_file_that_does_not_fit_is_made_again(self):
        first, second, _, n_second, _ = index_twice(threshold=1, corrupt=True)
        self.assertEqual(n_second, 1)
        self.assertEqual(second.node_vectors.tobytes(), first.node_vectors.tobytes())

    def test_other_texts_or_another_embedder_revision_find_another_file(self):
        with tempfile.TemporaryDirectory() as td, mock.patch("src.paths.cache_dir", return_value=Path(td)):
            a = om.second_vectors_path(["one", "two"])
            self.assertNotEqual(a, om.second_vectors_path(["one", "two", "three"]))
            self.assertNotEqual(a, om.second_vectors_path(["onetwo"]))
            with mock.patch.object(settings, "SECOND_EMBEDDER_REVISION", "another"):
                self.assertNotEqual(a, om.second_vectors_path(["one", "two"]))
            self.assertEqual(a.parent, Path(td) / settings.SECOND_VECTORS_NAMESPACE)


if __name__ == "__main__":
    unittest.main()
