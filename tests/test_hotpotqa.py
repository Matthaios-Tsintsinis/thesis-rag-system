"""HotpotQA, both variants. No network, no model — a fake dataset stands in.

The loader's job is corpus SHAPE and gold PROVENANCE, and both are
testable against a hand-built dataset. What a real download would add is
confidence that the HF field names are right, which is one line of the
loader and the first thing a smoke run reveals.

The properties pinned here are the ones the design decisions turned on:

  * variant A gives ONE unit per question over that question's own 10
    paragraphs — the published setting;
  * variant B pools across a shard and DEDUPLICATES by title, because
    the same Wikipedia paragraph is a distractor for many questions;
  * gold is SENTENCE-level for both (ruling (ii)), so a title's
    sentences are separate CorpusItems under one parent;
  * rank-aware is projected to TITLE level, and the projection COPIES
    rather than mutates, because those Chunks are the system's live
    index state.
"""

from __future__ import annotations

import unittest

from src.chunking import Chunk
from src.eval.hotpotqa import (
    HotpotQABenchmark,
    HotpotQAPooledBenchmark,
    _project_to_titles,
    _sentence_items,
)
from src.eval.types import EvalQuery, GoldAnswer
from src.retrievers.base import RetrievedChunk


def _ctx(*paragraphs):
    return {"title": [t for t, _ in paragraphs],
            "sentences": [s for _, s in paragraphs]}


def _row(qid, titles_and_sents, sf_title, sf_ids, answer="Paris",
         qtype="bridge", level="hard"):
    return {
        "id": qid,
        "question": f"question {qid}?",
        "answer": answer,
        "type": qtype,
        "level": level,
        "context": _ctx(*titles_and_sents),
        "supporting_facts": {"title": sf_title, "sent_id": sf_ids},
    }


class _FakeDS(list):
    """Enough of a datasets.Dataset for the loader: len, iter, select."""

    def select(self, idxs):
        return _FakeDS([self[i] for i in idxs])


def _fake_rows(n=4):
    rows = []
    for i in range(n):
        rows.append(_row(
            f"q{i}",
            [("Alpha", ["A one.", "A two."]),
             (f"Beta{i}", ["B one.", "B two."]),
             ("Gamma", ["G one."])],
            ["Alpha", f"Beta{i}"], [0, 1],
        ))
    return _FakeDS(rows)


def _install(bench, rows):
    bench._rows = rows           # skip _load / _measure entirely
    bench.stats["n_questions"] = len(rows)
    return bench


class TestSentenceItems(unittest.TestCase):
    def test_one_item_per_sentence_parented_by_title(self):
        items = _sentence_items(_ctx(("T", ["one.", "two.", "three."])))
        self.assertEqual(len(items), 3)
        self.assertEqual({i.parent_id for i in items}, {"T"})
        self.assertEqual([i.span_id for i in items],
                         ["sent0", "sent1", "sent2"])

    def test_item_id_is_unique_per_sentence(self):
        items = _sentence_items(_ctx(("T", ["a.", "b."]), ("U", ["c."])))
        self.assertEqual(len({i.item_id for i in items}), 3)

    def test_blank_sentences_are_dropped_without_shifting_ids(self):
        """A dropped sentence must not renumber the ones after it, or
        gold sent_ids would point at the wrong text."""
        items = _sentence_items(_ctx(("T", ["a.", "   ", "c."])))
        self.assertEqual([i.span_id for i in items], ["sent0", "sent2"])
        self.assertEqual([i.text for i in items], ["a.", "c."])


class TestVariantA(unittest.TestCase):
    def setUp(self):
        self.b = _install(HotpotQABenchmark(), _fake_rows(3))
        self.units = list(self.b.iter_eval_units(split="validation"))

    def test_one_unit_per_question(self):
        self.assertEqual(len(self.units), 3)
        for u in self.units:
            self.assertEqual(len(u.queries), 1)

    def test_corpus_is_only_that_questions_paragraphs(self):
        titles = {i.parent_id for i in self.units[0].corpus}
        self.assertEqual(titles, {"Alpha", "Beta0", "Gamma"})

    def test_gold_is_sentence_level(self):
        gold = self.units[0].queries[0].gold_passage_sets[0]
        self.assertEqual(gold, frozenset({("Alpha", "sent0"), ("Beta0", "sent1")}))

    def test_parent_scope_is_none(self):
        """The corpus IS the candidate set; a further restriction would
        be a second, silent filter."""
        self.assertIsNone(self.units[0].queries[0].parent_scope)

    def test_question_type_carries_the_bridge_comparison_split(self):
        self.assertEqual(self.units[0].queries[0].question_type, "bridge")

    def test_max_units_caps_questions(self):
        b = _install(HotpotQABenchmark(), _fake_rows(5))
        self.assertEqual(len(list(b.iter_eval_units(split="validation",
                                                    max_units=2))), 2)

    def test_a_bad_split_is_refused(self):
        with self.assertRaises(ValueError):
            list(self.b.iter_eval_units(split="train"))


class TestVariantBPooling(unittest.TestCase):
    def setUp(self):
        self.b = _install(
            HotpotQAPooledBenchmark(shard_questions=2), _fake_rows(4))
        self.units = list(self.b.iter_eval_units(split="validation"))

    def test_one_unit_per_shard_holding_all_its_questions(self):
        self.assertEqual(len(self.units), 2)
        self.assertEqual(len(self.units[0].queries), 2)

    def test_shared_titles_are_deduplicated(self):
        """Alpha and Gamma appear in EVERY question. Indexing them once
        per question would waste the build and put exact-duplicate
        vectors into the clustering."""
        titles = [i.parent_id for i in self.units[0].corpus]
        self.assertEqual(titles.count("Alpha"), 2)   # 2 sentences, once
        self.assertEqual(titles.count("Gamma"), 1)
        self.assertEqual(set(titles), {"Alpha", "Beta0", "Beta1", "Gamma"})

    def test_every_question_keeps_its_own_gold(self):
        golds = [q.gold_passage_sets[0] for q in self.units[0].queries]
        self.assertIn(("Beta0", "sent1"), golds[0])
        self.assertIn(("Beta1", "sent1"), golds[1])

    def test_pooled_corpus_is_larger_than_any_single_question(self):
        pooled = len({i.parent_id for i in self.units[0].corpus})
        single = len({i.parent_id for i in
                      _install(HotpotQABenchmark(), _fake_rows(4))
                      .iter_eval_units(split="validation").__next__().corpus})
        self.assertGreater(pooled, single)

    def test_variant_is_recorded_on_every_query(self):
        """A row must say which variant produced it; the two are not
        comparable to each other or to published numbers."""
        self.assertEqual(self.units[0].queries[0].metadata["variant"], "pooled")


class TestTitleProjection(unittest.TestCase):
    @staticmethod
    def _rc(chunk_id, prov):
        return RetrievedChunk(
            chunk=Chunk(chunk_id=chunk_id, doc_id="d", text="t", n_words=1,
                        position=0, gold_provenance=prov),
            score=1.0, rank=0, source_unit_type="chunk")

    def test_sentence_atoms_collapse_to_one_title_atom(self):
        out = _project_to_titles([
            self._rc("c0", (("Alpha", "sent0"), ("Alpha", "sent1")))])
        self.assertEqual(out[0].chunk.gold_provenance, (("Alpha", "<title>"),))

    def test_a_chunk_spanning_two_titles_keeps_both(self):
        out = _project_to_titles([
            self._rc("c0", (("Alpha", "sent2"), ("Beta", "sent0")))])
        self.assertEqual(out[0].chunk.gold_provenance,
                         (("Alpha", "<title>"), ("Beta", "<title>")))

    def test_the_original_chunks_are_not_mutated(self):
        """These Chunks are the system's LIVE index state, and CK-2 runs
        on the same objects — rewriting provenance in place would corrupt
        the set-F1 computed alongside."""
        original = self._rc("c0", (("Alpha", "sent0"),))
        _project_to_titles([original])
        self.assertEqual(original.chunk.gold_provenance, (("Alpha", "sent0"),))


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.b = HotpotQABenchmark()
        self.q = EvalQuery(
            query_id="q", question_text="?", parent_scope=None,
            gold_answers=(GoldAnswer(answer_type="free_form",
                                     free_form="the Eiffel Tower"),),
            gold_passage_sets=(frozenset({("Alpha", "sent0"),
                                          ("Beta", "sent1")}),),
            question_type="bridge")

    def test_exact_match_uses_the_official_normalisation(self):
        """Lowercase, strip articles, drop punctuation — so 'Eiffel
        Tower' matches 'the Eiffel Tower'."""
        s = self.b.score_answer("Eiffel Tower.", self.q)
        self.assertEqual(s.metadata["exact_match"], 1.0)
        self.assertEqual(s.value, 1.0)

    def test_token_f1_is_the_primary_value(self):
        s = self.b.score_answer("the Eiffel", self.q)
        self.assertEqual(s.method, "token_f1")
        self.assertGreater(s.value, 0.0)
        self.assertLess(s.value, 1.0)
        self.assertEqual(s.metadata["exact_match"], 0.0)

    def test_retrieval_reports_rank_aware_at_title_level(self):
        chunks = [
            RetrievedChunk(
                chunk=Chunk(chunk_id="c0", doc_id="d", text="t", n_words=1,
                            position=0,
                            gold_provenance=(("Alpha", "sent0"),)),
                score=1.0, rank=0, source_unit_type="chunk"),
            RetrievedChunk(
                chunk=Chunk(chunk_id="c1", doc_id="d", text="t", n_words=1,
                            position=1,
                            gold_provenance=(("Beta", "sent1"),)),
                score=0.9, rank=1, source_unit_type="chunk"),
        ]
        score = self.b.score_retrieval(chunks, self.q)
        self.assertFalse(score.skipped)
        # Both gold TITLES retrieved in the top 2.
        self.assertEqual(score.hit_at_k.get(2), 1.0)
        self.assertGreater(score.mrr, 0.0)

    def test_empty_gold_skips_rank_aware_without_crashing(self):
        q = EvalQuery(query_id="q", question_text="?", parent_scope=None,
                      gold_answers=(), gold_passage_sets=(frozenset(),),
                      question_type="bridge")
        self.assertIsNotNone(self.b.score_retrieval([], q))


if __name__ == "__main__":
    unittest.main()
