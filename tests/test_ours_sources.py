"""Pins that a question naming a source or a date steers retrieval (development v16): each
aspect searches the sources it names, or the question's, the global view stays open, and the
evidence holds some of every named source and dated article. A corpus without sources, as
HotpotQA and NarrativeQA are, stays exactly as it was."""

from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from src.eval.types import CorpusItem, EvalUnit
from tests.test_ours_pipeline import VIEW, _index, _no_bm25
from tests.test_ours_smoke import DECOMPOSITIONS, FakeReader, Q0, hotpot_unit, llama_config, ours_fakes

STORY = " ".join(f"Sentence {i} of this report covers the merger of the two firms in week {i}."
                 for i in range(30))


def article(url, source, date, topic):
    return CorpusItem(item_id=url, parent_id=url, span_id="<whole>", text=f"{topic} {STORY}",
                      metadata={"title": f"On {topic}", "source": source, "published_at": date})


ARTICLES = (
    article("u0", "The Verge", "2023-10-07T08:00:00+00:00", "Apples"),
    article("u1", "The Verge", "2023-10-30T08:00:00+00:00", "Bananas"),
    article("u2", "TechCrunch", "2023-11-27T08:00:00+00:00", "Cherries"),
    article("u3", "TechCrunch", "2023-11-28T08:00:00+00:00", "Dates"),
    article("u4", "Fortune", "2023-12-24T08:00:00+00:00", "Elderberries"),
    article("u5", "Fortune", "2023-12-25T08:00:00+00:00", "Figs"),
)
NEWS = EvalUnit(corpus_id="news", corpus=ARTICLES, queries=())
SOURCE_OF = {a.parent_id: a.metadata["source"] for a in ARTICLES}


def prepared(unit, question, decompositions=None, **flags):
    """prepare() with every model faked: (the prepared query, the unit's index)."""
    reader = FakeReader(decompositions or {}, {})
    with tempfile.TemporaryDirectory() as td, ours_fakes(td, reader), \
            mock.patch.multiple(settings, **(flags or {"SOURCE_FOCUS": True})):
        system = om.OursSystem(config=llama_config())
        system.index_items(unit.corpus)
        return system.prepare(question), system._tree_index


def sources_read(query):
    return [SOURCE_OF[str(r.chunk.gold_provenance[0][0])] for r in query.retrieved]


def plain_index(names=(), sources=None):
    """An index holding only what the naming functions read."""
    return mock.Mock(source_names=sorted(names, key=len, reverse=True))


class TestNamedSources(unittest.TestCase):
    def test_a_source_is_found_as_the_corpus_writes_it(self):
        index = plain_index(["The Verge", "Fortune"])
        self.assertEqual(om.named_sources(index, "Did The Verge report it?"), ["The Verge"])

    def test_a_plain_word_is_no_source(self):
        index = plain_index(["The Age", "Rivals", "Fortune"])
        self.assertEqual(om.named_sources(index, "Do the rivals of the age make a fortune?"), [])

    def test_a_name_inside_a_longer_word_or_name_does_not_count(self):
        index = plain_index(["Business World", "Music Business Worldwide", "BBC News", "BBC News - Technology"])
        self.assertEqual(om.named_sources(index, "Music Business Worldwide said so."), ["Music Business Worldwide"])
        self.assertEqual(om.named_sources(index, "BBC News - Technology said so."), ["BBC News - Technology"])

    def test_several_are_all_found(self):
        index = plain_index(["The Verge", "TechCrunch", "Fortune"])
        self.assertCountEqual(om.named_sources(index, "Did TechCrunch and The Verge agree?"),
                              ["The Verge", "TechCrunch"])

    def test_no_sources_no_names(self):
        self.assertEqual(om.named_sources(plain_index(), "Did The Verge report it?"), [])


class TestNamedDates(unittest.TestCase):
    def test_in_the_form_a_document_gives(self):
        self.assertEqual(om.named_dates("After the report on October 7, 2023, what changed?"), ["October 7, 2023"])
        self.assertEqual(om.published_date("2023-10-07T08:00:00+00:00"), "October 7, 2023")

    def test_forms_of_writing_it(self):
        self.assertEqual(om.named_dates("on October 07, 2023"), ["October 7, 2023"])
        self.assertEqual(om.named_dates("on October 7 2023"), ["October 7, 2023"])

    def test_part_of_a_date_is_no_date(self):
        self.assertEqual(om.named_dates("In October 2023 it changed, in May of 2020 too."), [])

    def test_once_each(self):
        self.assertEqual(om.named_dates("October 7, 2023 and again October 7, 2023 and May 1, 2022"),
                         ["October 7, 2023", "May 1, 2022"])


class TestFocus(unittest.TestCase):
    """On the hand-built tree: leaves 0-5 under A and B, 6-11 under C and D."""

    FOCUS_LEAVES = {6, 7, 8}

    def test_nodes_holding_a_leaf_are_allowed(self):
        index = _index()
        focus = om.focus_on(index, self.FOCUS_LEAVES)
        allowed = {n for n, ok in zip(index.node_ids, focus.nodes) if ok}
        self.assertIn("C", allowed)
        self.assertIn("Q", allowed)
        self.assertNotIn("A", allowed)
        self.assertNotIn("P", allowed)
        self.assertNotIn(index.node_ids[0], allowed)
        self.assertIn(index.node_ids[6], allowed)

    def test_collapsed_search_stays_inside(self):
        index = _index()
        focus = om.focus_on(index, self.FOCUS_LEAVES)
        got = {leaf for leaf, _ in om.collapsed_candidates(index, VIEW, _no_bm25(index), focus)}
        self.assertTrue(got)
        self.assertLessEqual(got, self.FOCUS_LEAVES)

    def test_a_broad_summary_expands_inside(self):
        index = _index()
        focus = om.focus_on(index, {6})
        self.assertEqual(om.expand_summary(index, "Q", VIEW, focus=focus), [6])

    def test_the_tree_walk_stays_inside(self):
        index = _index()
        focus = om.focus_on(index, self.FOCUS_LEAVES)
        got = {leaf for leaf, _ in om.multi_branch_candidates(index, VIEW, _no_bm25(index), focus)}
        self.assertTrue(got)
        self.assertLessEqual(got, self.FOCUS_LEAVES)

    def test_bm25_stays_inside(self):
        index = _index()
        focus = om.focus_on(index, self.FOCUS_LEAVES)
        scores = om.bm25_scores(index, "leaf text", focus)
        self.assertTrue((scores[[0, 1, 2, 3, 4, 5, 9, 10, 11]] == 0).all())
        self.assertEqual(len(scores), 12)

    def test_no_focus_searches_as_before(self):
        index = _index()
        self.assertEqual(om.collapsed_candidates(index, VIEW, _no_bm25(index), None),
                         om.collapsed_candidates(index, VIEW, _no_bm25(index)))


class TestCoverTargets(unittest.TestCase):
    def _search(self, by_target):
        return lambda target: by_target[frozenset(target)]

    def test_a_target_held_needs_nothing(self):
        groups = [[(1, .9), (2, .8), (3, .7)], [(4, .6)]]
        placed = om.cover_targets(groups, [{1, 2}], self._search({}))
        self.assertEqual(placed, 0)
        self.assertEqual(groups, [[(1, .9), (2, .8), (3, .7)], [(4, .6)]])

    def test_free_slots_go_first(self):
        groups = [[(1, .9)], [(2, .8)]]
        placed = om.cover_targets(groups, [{7, 8}], self._search({frozenset({7, 8}): [(7, .5), (8, .4)]}))
        self.assertEqual(placed, 2)
        self.assertEqual({leaf for g in groups for leaf, _ in g}, {1, 2, 7, 8})

    def test_with_no_slot_the_weakest_of_the_largest_aspect_gives_way(self):
        aspect = [(i, 1 - i / 100) for i in range(9)]
        short = [(20, .9), (21, .8), (22, .7), (23, .6)]
        groups = [aspect, short, [(30, .9), (31, .8)]]
        om.cover_targets(groups, [{40, 41}], self._search({frozenset({40, 41}): [(40, .5), (41, .4)]}))
        self.assertEqual([leaf for leaf, _ in groups[0]], [0, 1, 2, 3, 4, 5, 6, 40, 41])
        self.assertEqual(groups[1], short)
        self.assertEqual(groups[2], [(30, .9), (31, .8)])

    def test_what_a_target_needs_is_never_taken(self):
        groups = [[(i, 1 - i / 100) for i in range(13)], [(30, .9), (31, .8)]]
        om.cover_targets(groups, [{11, 12}, {40, 41}], self._search({frozenset({40, 41}): [(40, .5), (41, .4)]}))
        leaves = {leaf for leaf, _ in groups[0]}
        self.assertLessEqual({11, 12, 40, 41}, leaves)
        self.assertEqual(len(groups[0]), 13)

    def test_a_chunk_already_read_is_not_placed_twice(self):
        groups = [[(7, .9)], [(2, .8)]]
        om.cover_targets(groups, [{7, 8}], self._search({frozenset({7, 8}): [(7, .9), (8, .4)]}))
        self.assertEqual(sum(leaf == 7 for g in groups for leaf, _ in g), 1)
        self.assertIn(8, {leaf for g in groups for leaf, _ in g})

    def test_a_target_with_nothing_found_is_left(self):
        groups = [[(1, .9)], [(2, .8)]]
        self.assertEqual(om.cover_targets(groups, [{7}], self._search({frozenset({7}): []})), 0)
        self.assertEqual(groups, [[(1, .9)], [(2, .8)]])


class TestSearchTargets(unittest.TestCase):
    def _index(self, names=("The Verge", "Fortune")):
        return SimpleNamespace(
            source_names=list(names),
            source=["The Verge", "The Verge", "Fortune", "Fortune", "TechCrunch"],
            date=["October 7, 2023", "May 1, 2022", "October 7, 2023", "May 1, 2022", "October 7, 2023"])

    def test_a_dated_article_then_each_source(self):
        targets = om.search_targets(self._index(), "Did The Verge on October 7, 2023 agree with Fortune?")
        self.assertEqual(targets, [{0, 2}, {0, 1}, {2, 3}])

    def test_a_date_keeps_to_the_named_sources(self):
        targets = om.search_targets(self._index(), "Did The Verge report it on October 7, 2023?")
        self.assertEqual(targets, [{0}, {0, 1}])

    def test_a_date_with_no_source_takes_any_article_of_it(self):
        targets = om.search_targets(self._index(), "What happened on October 7, 2023?")
        self.assertEqual(targets, [{0, 2, 4}])

    def test_a_date_no_article_has_is_no_target(self):
        targets = om.search_targets(self._index(), "Did Fortune report it on October 8, 2023?")
        self.assertEqual(targets, [{2, 3}])

    def test_no_sources_in_the_corpus_no_targets(self):
        self.assertEqual(om.search_targets(self._index(names=()), "What happened on October 7, 2023?"), [])

    def test_switched_off_no_targets(self):
        with mock.patch.object(settings, "SOURCE_FOCUS", False):
            self.assertEqual(om.search_targets(self._index(), "Did Fortune report it?"), [])


class TestSourceFocus(unittest.TestCase):
    def _index(self):
        index = _index()
        index.source = ["The Verge"] * 6 + ["Fortune"] * 6
        index.source_names = ["The Verge", "Fortune"]
        return index

    def test_the_leaves_of_the_sources_named(self):
        self.assertEqual(om.source_focus(self._index(), "Did Fortune report it?").leaves, frozenset(range(6, 12)))

    def test_naming_none_searches_everywhere(self):
        self.assertIsNone(om.source_focus(self._index(), "Did anyone report it?"))

    def test_switched_off_searches_everywhere(self):
        with mock.patch.object(settings, "SOURCE_FOCUS", False):
            self.assertIsNone(om.source_focus(self._index(), "Did Fortune report it?"))


def searches(question, decompositions=None):
    """What each search was limited to: (view text, aspect text, its leaves or None); and the index."""
    calls = []
    real = om.view_candidates

    def recording(index, view_text, view_vector, aspect_text, focus=None, **kwargs):
        calls.append((view_text, aspect_text, None if focus is None else focus.leaves))
        return real(index, view_text, view_vector, aspect_text, focus, **kwargs)

    with mock.patch("ours.model.view_candidates", recording):
        _, index = prepared(NEWS, question, decompositions)
    return calls, index


def leaves_of(index, *sources):
    return frozenset(i for i, s in enumerate(index.source) if s in sources)


class TestWhatEachSearchIsLimitedTo(unittest.TestCase):
    QUESTION = "Did The Verge and TechCrunch agree on the merger of the firms?"

    def test_an_aspect_searches_the_source_it_names(self):
        decomposed = {self.QUESTION: json.dumps({"type": "multi_aspect", "aspects": [
            {"name": "What did The Verge say?", "importance": 0.5},
            {"name": "What did TechCrunch say?", "importance": 0.5}]})}
        calls, index = searches(self.QUESTION, decomposed)
        verge = [c for c in calls if c[1] == "What did The Verge say?"]
        crunch = [c for c in calls if c[1] == "What did TechCrunch say?"]
        self.assertTrue(verge and crunch)
        self.assertEqual({c[2] for c in verge}, {leaves_of(index, "The Verge")})
        self.assertEqual({c[2] for c in crunch}, {leaves_of(index, "TechCrunch")})

    def test_an_aspect_naming_none_searches_the_questions_sources(self):
        decomposed = {self.QUESTION: json.dumps({"type": "multi_aspect", "aspects": [
            {"name": "What was said on the merger?", "importance": 0.5},
            {"name": "Who agreed on the merger?", "importance": 0.5}]})}
        calls, index = searches(self.QUESTION, decomposed)
        aspects = [c for c in calls if c[1] != self.QUESTION]
        self.assertTrue(aspects)
        self.assertEqual({c[2] for c in aspects}, {leaves_of(index, "The Verge", "TechCrunch")})

    def test_the_global_view_is_open(self):
        calls, _ = searches(self.QUESTION)
        self.assertIn((self.QUESTION, self.QUESTION, None), calls)


class TestCoverage(unittest.TestCase):
    QUESTION = "Did Fortune and TechCrunch agree on the merger of the firms?"

    def test_the_steps_give_it_the_questions_targets_and_a_search_inside_each(self):
        seen = {}
        real = om.cover_targets

        def recording(groups, targets, search):
            seen["targets"] = targets
            seen["found"] = [{leaf for leaf, _ in search(t)} for t in targets]
            return real(groups, targets, search)

        with mock.patch("ours.model.cover_targets", recording):
            _, index = prepared(NEWS, self.QUESTION)
        self.assertEqual({frozenset(t) for t in seen["targets"]},
                         {leaves_of(index, "Fortune"), leaves_of(index, "TechCrunch")})
        for target, found in zip(seen["targets"], seen["found"]):
            self.assertTrue(found)
            self.assertLessEqual(found, set(target))

    def test_a_source_crowded_out_of_the_evidence_comes_back(self):
        real = om.merge_and_backfill
        leaf_source = {}

        def crowded(rankings, quotas):
            groups = real(rankings, quotas)
            return [[pair for pair in group if leaf_source.get(pair[0]) != "TechCrunch"] for group in groups]

        real_index = om.build_tree_index

        def remembering(*args, **kwargs):
            index = real_index(*args, **kwargs)
            leaf_source.update(enumerate(index.source))
            return index

        with mock.patch("ours.model.merge_and_backfill", crowded), \
                mock.patch("ours.model.build_tree_index", remembering):
            query, _ = prepared(NEWS, self.QUESTION)
        self.assertGreaterEqual(sources_read(query).count("TechCrunch"), settings.MIN_CHUNKS_PER_TARGET)
        self.assertGreaterEqual(query.extra["ours_n_chunks_placed"], settings.MIN_CHUNKS_PER_TARGET)

    def test_a_dated_article_crowded_out_comes_back(self):
        question = "Did the report from The Verge on October 30, 2023 mention the merger of the firms?"
        real = om.merge_and_backfill
        leaf_doc = {}

        def crowded(rankings, quotas):
            groups = real(rankings, quotas)
            return [[pair for pair in group if leaf_doc.get(pair[0]) != "u1"] for group in groups]

        real_index = om.build_tree_index

        def remembering(chunks, *args, **kwargs):
            leaf_doc.update({i: str(c.gold_provenance[0][0]) for i, c in enumerate(chunks)})
            return real_index(chunks, *args, **kwargs)

        with mock.patch("ours.model.merge_and_backfill", crowded), \
                mock.patch("ours.model.build_tree_index", remembering):
            query, _ = prepared(NEWS, question)
        dated = [r for r in query.retrieved if str(r.chunk.gold_provenance[0][0]) == "u1"]
        self.assertGreaterEqual(len(dated), settings.MIN_CHUNKS_PER_TARGET)


class TestTheSteps(unittest.TestCase):
    QUESTION = "What did The Verge report about the merger of the firms in its Apples story?"

    def test_a_named_source_fills_the_aspects(self):
        query, _ = prepared(NEWS, self.QUESTION)
        n_global = query.extra["ours_global_selected"]
        aspects = sources_read(query)[: len(query.retrieved) - n_global]
        self.assertTrue(aspects)
        self.assertEqual(set(aspects), {"The Verge"})
        self.assertEqual(query.extra["ours_named_sources"], ["The Verge"])

    def test_the_global_view_stays_open(self):
        with mock.patch("ours.model.view_candidates", wraps=om.view_candidates) as search:
            prepared(NEWS, self.QUESTION)
        open_calls = [c for c in search.call_args_list
                      if len(c.args) < 5 and c.kwargs.get("focus") is None or (len(c.args) >= 5 and c.args[4] is None)]
        self.assertTrue(open_calls, "the global view searched inside the focus")

    def test_an_aspect_naming_a_source_searches_that_source(self):
        question = "Did The Verge and TechCrunch agree on the merger of the firms?"
        decomposed = {question: json.dumps({"type": "multi_aspect", "aspects": [
            {"name": "What did The Verge say on the merger of the firms?", "importance": 0.5},
            {"name": "What did TechCrunch say on the merger of the firms?", "importance": 0.5}]})}
        query, _ = prepared(NEWS, question, decomposed)
        read = sources_read(query)
        first = query.extra["ours_aspects"][0]["n_selected"]
        second = query.extra["ours_aspects"][1]["n_selected"]
        self.assertEqual(set(read[:first]), {"The Verge"})
        self.assertEqual(set(read[first: first + second]), {"TechCrunch"})

    def test_every_named_source_gets_chunks_even_when_no_aspect_names_it(self):
        question = "Did Fortune and TechCrunch agree on the merger of the firms?"
        decomposed = {question: json.dumps({"type": "multi_aspect", "aspects": [
            {"name": "What did Fortune say on the merger of the firms?", "importance": 0.9},
            {"name": "What did Fortune add on the merger of the firms?", "importance": 0.8}]})}
        query, _ = prepared(NEWS, question, decomposed)
        read = sources_read(query)
        self.assertGreaterEqual(read.count("TechCrunch"), settings.MIN_CHUNKS_PER_TARGET)
        self.assertGreaterEqual(read.count("Fortune"), settings.MIN_CHUNKS_PER_TARGET)

    def test_a_dated_article_gets_chunks(self):
        question = "Did the report from The Verge on October 30, 2023 mention the merger of the firms?"
        query, _ = prepared(NEWS, question)
        dated = [r for r in query.retrieved if str(r.chunk.gold_provenance[0][0]) == "u1"]
        self.assertGreaterEqual(len(dated), settings.MIN_CHUNKS_PER_TARGET)
        self.assertEqual(query.extra["ours_named_dates"], ["October 30, 2023"])

    def test_never_more_than_the_design_reads(self):
        query, _ = prepared(NEWS, "Did Fortune, TechCrunch and The Verge agree on October 30, 2023?")
        self.assertLessEqual(len(query.retrieved), settings.FINAL_CONTEXT_CHUNKS)

    def test_a_question_naming_nothing_searches_as_before(self):
        question = "What was said about the merger of the firms?"
        on, _ = prepared(NEWS, question, SOURCE_FOCUS=True)
        off, _ = prepared(NEWS, question, SOURCE_FOCUS=False)
        self.assertEqual(on.user_prompt, off.user_prompt)

    def test_switched_off_it_only_records_the_names(self):
        off, _ = prepared(NEWS, self.QUESTION, SOURCE_FOCUS=False)
        self.assertEqual(off.extra["ours_named_sources"], ["The Verge"])
        self.assertEqual(off.extra["ours_n_chunks_placed"], 0)

    def test_a_corpus_without_sources_stays_exactly_as_it_was(self):
        on, on_index = prepared(hotpot_unit(0), Q0, DECOMPOSITIONS, SOURCE_FOCUS=True)
        off, _ = prepared(hotpot_unit(0), Q0, DECOMPOSITIONS, SOURCE_FOCUS=False)
        self.assertEqual(on_index.source_names, [])
        self.assertEqual(on.user_prompt, off.user_prompt)
        self.assertEqual(on.extra["ours_named_sources"], [])
        self.assertEqual(on.extra["ours_named_dates"], [])
        self.assertEqual(on.extra["ours_n_chunks_placed"], 0)
        self.assertEqual([r.chunk.chunk_id for r in on.packed], [r.chunk.chunk_id for r in off.packed])


if __name__ == "__main__":
    unittest.main()
