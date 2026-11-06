"""Pins that each aspect also searches under its own words (development v17): the untouched
question, when there is one aspect. No more reader calls; switched off, it searches as v16 did."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from tests.test_ours_smoke import DECOMPOSITIONS, FakeReader, Q0, fake_embed, hotpot_unit, llama_config, ours_fakes


def run(question=Q0, decompositions=DECOMPOSITIONS, **flags):
    """prepare() with every model faked and the searches recorded."""
    reader = FakeReader(decompositions, {})
    calls = []
    real = om.view_candidates

    def recording(index, view_text, view_vector, aspect_text, focus=None, **kwargs):
        calls.append((view_text, np.array(view_vector), aspect_text))
        return real(index, view_text, view_vector, aspect_text, focus, **kwargs)

    with tempfile.TemporaryDirectory() as td, ours_fakes(td, reader), \
            mock.patch("ours.model.view_candidates", recording), \
            mock.patch.multiple(settings, **(flags or {"QUESTION_VIEW": True})):
        system = om.OursSystem(config=llama_config())
        system.index_items(hotpot_unit(0).corpus)
        return system.prepare(question), calls, reader


class TestItsOwnWords(unittest.TestCase):
    def test_each_aspect_searches_under_its_own_words(self):
        _, calls, _ = run()
        for aspect in ("Where did topic0x0 travel?", "Who is topic0x1?"):
            own = [c for c in calls if c[0] == aspect and c[2] == aspect]
            self.assertEqual(len(own), 1, aspect)

    def test_each_aspect_still_has_its_two_rewrites(self):
        _, calls, _ = run()
        for aspect in ("Where did topic0x0 travel?", "Who is topic0x1?"):
            self.assertEqual(sum(c[2] == aspect for c in calls), 3, aspect)

    def test_one_aspect_searches_the_untouched_question(self):
        _, calls, _ = run(decompositions={})
        # Once for the aspect, once for the global view.
        self.assertEqual(sum(c[0] == Q0 for c in calls), 2)

    def test_the_vector_is_its_own(self):
        _, calls, _ = run()
        for text, vector, aspect in calls:
            if text == aspect:
                self.assertTrue(np.allclose(vector, fake_embed([text])[0]), text)

    def test_all_three_reach_the_pool(self):
        pools = []
        real = om.aspect_pool

        def recording(index, rankings):
            pools.append(len(rankings))
            return real(index, rankings)

        with mock.patch("ours.model.aspect_pool", recording):
            run()
        # Two aspects with three searches each, then the global view with one.
        self.assertEqual(pools, [3, 3, 1])

    def test_no_more_reader_calls(self):
        _, _, on = run(QUESTION_VIEW=True)
        _, _, off = run(QUESTION_VIEW=False)
        self.assertEqual(len(on.prompts), len(off.prompts))

    def test_the_row_counts_the_searches(self):
        on, _, _ = run(QUESTION_VIEW=True)
        off, _, _ = run(QUESTION_VIEW=False)
        self.assertEqual(on.n_retrieval_calls, 3 * 2 + 1)
        self.assertEqual(off.n_retrieval_calls, 2 * 2 + 1)


class TestSwitchedOff(unittest.TestCase):
    def test_it_searches_as_before(self):
        _, calls, _ = run(QUESTION_VIEW=False)
        self.assertEqual(len(calls), 2 * 2 + 1)
        self.assertFalse(any(c[0] == c[2] and c[0] != Q0 for c in calls))


if __name__ == "__main__":
    unittest.main()
