"""Pins the design's embedder beside M4's (development v18). "both" adds its ranking to the
collapsed search's fusion; "only" puts its vectors in every search, over M4's tree; "off" leaves
all as it was. The tree, its clusters and M4's vectors and cache are never touched."""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from tests.test_ours_pipeline import LEAF_COS, NODE_COS, VIEW, _index, _no_bm25, _vec
from tests.test_ours_smoke import DECOMPOSITIONS, FakeReader, Q0, fake_embed, hotpot_unit, llama_config, ours_fakes

SECOND_DIM = 32


def second_embed(texts, model_name=None):
    """A vector of another size for the second embedder, so nothing can mix them unnoticed."""
    if model_name != settings.SECOND_EMBEDDER:
        return fake_embed(texts, model_name)
    out = np.zeros((len(texts), SECOND_DIM), dtype=np.float32)
    for r, text in enumerate(texts):
        for w in re.findall(r"\w+", text.lower()):
            out[r, int(hashlib.sha1(w.encode()).hexdigest(), 16) % SECOND_DIM] += 1.0
        norm = float(np.linalg.norm(out[r]))
        out[r, 0 if norm == 0 else slice(None)] = 1.0 if norm == 0 else out[r] / norm
    return out


def run(mode, question=Q0):
    """Index and prepare under a mode: (prepared, index, system, the embedder calls, the cache's files)."""
    calls = []

    def recording(texts, model_name=None):
        calls.append((model_name, list(texts)))
        return second_embed(texts, model_name)

    reader = FakeReader(DECOMPOSITIONS, {})
    with tempfile.TemporaryDirectory() as td, ours_fakes(td, reader), \
            mock.patch("src.models.embed_texts", recording), \
            mock.patch.object(settings, "SECOND_EMBEDDER_MODE", mode):
        system = om.OursSystem(config=llama_config())
        system.index_items(hotpot_unit(0).corpus)
        prepared = system.prepare(question)
        files = sorted(str(p.relative_to(td)) for p in Path(td).rglob("*") if p.is_file())
        return prepared, system._tree_index, system, calls, files


def second_view_index():
    """The hand-built tree with second vectors that put leaf 11 nearest and leaf 6, mpnet's best, last."""
    index = _index()
    second = np.vstack([_vec(c) for c in LEAF_COS] + [_vec(NODE_COS[n]) for n in index.node_ids[len(LEAF_COS):]])
    second[11] = _vec(1.0)
    second[6] = _vec(0.0)
    index.second_vectors = second
    return index


class TestBoth(unittest.TestCase):
    def test_every_node_gets_a_second_vector_and_m4s_stay(self):
        _, index, system, _, _ = run("both")
        self.assertEqual(index.second_vectors.shape, (len(index.node_ids), SECOND_DIM))
        self.assertEqual(index.node_vectors.shape[1], fake_embed(["x"]).shape[1])
        self.assertTrue(np.array_equal(index.leaf_vectors, np.asarray(system._tree_system.chunk_embeddings, dtype=np.float32)))

    def test_the_leaves_are_embedded_as_the_chunks_read_with_no_name(self):
        _, index, _, calls, _ = run("both")
        texts = next(t for m, t in calls if m == settings.SECOND_EMBEDDER)
        self.assertEqual(texts[: len(index.chunks)], [c.text for c in index.chunks])

    def test_its_ranking_joins_the_fusion(self):
        index = second_view_index()
        without = [leaf for leaf, _ in om.collapsed_candidates(index, VIEW, _no_bm25(index))]
        with_second = [leaf for leaf, _ in om.collapsed_candidates(index, VIEW, _no_bm25(index), None, VIEW)]
        self.assertLess(with_second.index(11), without.index(11))
        self.assertGreater(with_second.index(6), without.index(6))

    def test_it_keeps_to_a_focus(self):
        index = second_view_index()
        focus = om.focus_on(index, {6, 7, 8})
        got = {leaf for leaf, _ in om.collapsed_candidates(index, VIEW, _no_bm25(index), focus, VIEW)}
        self.assertTrue(got)
        self.assertLessEqual(got, {6, 7, 8})

    def test_every_collapsed_search_gets_its_second_vector(self):
        seconds = []
        real = om.collapsed_candidates

        def recording(index, view_vector, view_bm25, focus=None, second_vector=None):
            seconds.append(second_vector)
            return real(index, view_vector, view_bm25, focus, second_vector)

        with mock.patch("ours.model.collapsed_candidates", recording):
            run("both")
        self.assertTrue(seconds)
        self.assertTrue(all(s is not None and s.shape == (SECOND_DIM,) for s in seconds))

    def test_a_search_has_the_second_vector_of_its_own_text(self):
        seen = []
        real = om.view_candidates

        def recording(index, view_text, view_vector, aspect_text, focus=None, second_vector=None):
            seen.append((view_text, second_vector))
            return real(index, view_text, view_vector, aspect_text, focus, second_vector)

        with mock.patch("ours.model.view_candidates", recording):
            run("both")
        for text, second in seen:
            self.assertTrue(np.allclose(second, second_embed([text], settings.SECOND_EMBEDDER)[0]), text)

    def test_each_query_text_is_embedded_by_both(self):
        prepared, _, _, calls, _ = run("both")
        firsts = [t for m, t in calls if m == settings.EMBEDDER and Q0 in t]
        seconds = [t for m, t in calls if m == settings.SECOND_EMBEDDER and Q0 in t]
        self.assertEqual(firsts, seconds)
        self.assertTrue(prepared.retrieved)


class TestOnly(unittest.TestCase):
    def test_every_search_reads_the_second_vectors(self):
        _, index, _, _, _ = run("only")
        self.assertEqual(index.node_vectors.shape[1], SECOND_DIM)
        self.assertEqual(index.leaf_vectors.shape[1], SECOND_DIM)
        self.assertTrue(all(v.shape[0] == SECOND_DIM for v in index.vector_of.values()))
        self.assertIsNone(index.second_vectors)

    def test_no_query_text_goes_to_m4s_embedder(self):
        _, _, _, calls, _ = run("only")
        self.assertFalse([t for m, t in calls if m == settings.EMBEDDER and Q0 in t])

    def test_it_runs_end_to_end_over_m4s_tree(self):
        prepared, index, system, _, _ = run("only")
        self.assertTrue(prepared.retrieved)
        self.assertEqual(system._tree_system.chunk_embeddings.shape[1], fake_embed(["x"]).shape[1])


class TestOff(unittest.TestCase):
    def test_the_second_embedder_is_never_called(self):
        _, index, _, calls, _ = run("off")
        self.assertFalse([1 for m, _ in calls if m == settings.SECOND_EMBEDDER])
        self.assertIsNone(index.second_vectors)


class TestTheModes(unittest.TestCase):
    def test_a_mode_it_does_not_know_is_refused(self):
        with self.assertRaisesRegex(ValueError, "SECOND_EMBEDDER_MODE"):
            run("sometimes")

    def test_the_cache_holds_the_same_files_in_every_mode(self):
        _, _, _, _, off = run("off")
        _, _, _, _, both = run("both")
        _, _, _, _, only = run("only")
        self.assertEqual(len(off), len(both))
        self.assertEqual(len(off), len(only))

    def test_the_tree_stays_the_same(self):
        _, a, _, _, _ = run("off")
        _, b, _, _, _ = run("both")
        _, c, _, _, _ = run("only")
        self.assertEqual(a.children, b.children)
        self.assertEqual(a.children, c.children)
        self.assertEqual(a.cluster, c.cluster)


class TestPinnedRevisions(unittest.TestCase):
    def _served(self, shas):
        api = mock.Mock()
        api.model_info.side_effect = lambda name: SimpleNamespace(sha=shas[name])
        return mock.patch("huggingface_hub.HfApi", return_value=api), api

    def _good(self):
        return {settings.READER: settings.READER_REVISION, settings.EMBEDDER: settings.EMBEDDER_REVISION,
                settings.SECOND_EMBEDDER: settings.SECOND_EMBEDDER_REVISION}

    def test_the_second_embedder_is_pinned_when_it_is_used(self):
        for mode in ("both", "only"):
            shas = dict(self._good(), **{settings.SECOND_EMBEDDER: "another"})
            patch, _ = self._served(shas)
            with patch, mock.patch.object(settings, "SECOND_EMBEDDER_MODE", mode), self.assertRaises(SystemExit):
                om.check_model_revisions()

    def test_off_it_is_not_asked_for(self):
        patch, api = self._served(self._good())
        with patch, mock.patch.object(settings, "SECOND_EMBEDDER_MODE", "off"):
            om.check_model_revisions()
        self.assertNotIn(mock.call(settings.SECOND_EMBEDDER), api.model_info.call_args_list)


if __name__ == "__main__":
    unittest.main()
