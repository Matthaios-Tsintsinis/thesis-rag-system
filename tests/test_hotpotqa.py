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
    PREREGISTERED_Q,
    SUBSAMPLE_SEED,
    HotpotQABenchmark,
    HotpotQAPooledBenchmark,
    _project_to_titles,
    _sentence_items,
    hotpot_token_f1,
    subsample_indices,
)
from src.eval.scorers.extractive import normalize_qasper_answer, token_f1
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


class TestSeededSubsample(unittest.TestCase):
    """Seeded random, NOT the head of the file.

    HotpotQA dev is not guaranteed randomly ordered — it can be grouped
    by type and level — so a head slice risks skewing the sample on
    exactly the bridge/comparison axis that justifies including the
    benchmark.
    """

    def test_it_is_not_the_head(self):
        idx = subsample_indices(7405, 1000)
        self.assertNotEqual(idx, list(range(1000)))

    def test_same_seed_is_reproducible(self):
        self.assertEqual(subsample_indices(7405, 1000),
                         subsample_indices(7405, 1000))

    def test_a_different_seed_gives_a_different_sample(self):
        self.assertNotEqual(subsample_indices(7405, 1000),
                            subsample_indices(7405, 1000, seed=1))

    def test_indices_are_sorted_unique_and_in_range(self):
        """Sorted so the subsample keeps DATASET order — variant B's
        shard boundaries must be a function of the sample, not of the
        order sample() happened to emit."""
        idx = subsample_indices(7405, 1000)
        self.assertEqual(idx, sorted(idx))
        self.assertEqual(len(idx), len(set(idx)))
        self.assertEqual(len(idx), 1000)
        self.assertTrue(all(0 <= i < 7405 for i in idx))

    def test_both_variants_draw_the_SAME_sample(self):
        """If A and B answered different questions, an A-vs-B comparison
        would confound the pooling change with a change of question set —
        and comparing them is the whole reason both exist."""
        a = HotpotQABenchmark(max_questions=500)
        b = HotpotQAPooledBenchmark(max_questions=500)
        self.assertEqual(
            subsample_indices(7405, a.max_questions),
            subsample_indices(7405, b.max_questions),
        )

    def test_asking_for_everything_is_the_identity(self):
        self.assertEqual(subsample_indices(50, 50), list(range(50)))
        self.assertEqual(subsample_indices(50, 999), list(range(50)))

    def test_the_seed_matches_the_project_convention(self):
        self.assertEqual(SUBSAMPLE_SEED, 20260805)

    def test_the_registered_sample_is_the_DEFAULT_not_a_flag(self):
        """The runner constructs benchmarks with NO arguments. If the
        subsample only fired when max_questions was passed, no real run
        would ever subsample, and --max-units 1000 would silently take
        the first 1,000 of 7,405 — the head slice the seeding exists to
        avoid, reintroduced through a different door."""
        for cls in (HotpotQABenchmark, HotpotQAPooledBenchmark):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls().max_questions, PREREGISTERED_Q)
        self.assertEqual(PREREGISTERED_Q, 1000)

    def test_the_full_split_is_still_reachable_explicitly(self):
        self.assertIsNone(HotpotQABenchmark(max_questions=None).max_questions)


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


def _yn_query(gold: str, qtype: str = "comparison") -> EvalQuery:
    return EvalQuery(
        query_id="q", question_text="?", parent_scope=None,
        gold_answers=(GoldAnswer(answer_type="free_form", free_form=gold),),
        gold_passage_sets=(frozenset({("Alpha", "sent0")}),),
        question_type=qtype)


class TestOfficialYesNoGuard(unittest.TestCase):
    """HotpotQA's yes/no/noanswer guard, against a transcribed reference.

    The reference below is transcribed from the published
    `hotpot_evaluate_v1.f1_score`, so each assertion carries its own
    source instead of pointing at one. It uses THIS harness's normaliser
    rather than the official `normalize_answer`, which is legitimate here
    and only here: the two compose identically and our documented
    NFKC/Unicode extension is provably inert on the ASCII inputs below.
    """

    @staticmethod
    def _official_f1(prediction: str, ground_truth: str) -> float:
        """Shape of the official f1_score, returning the F1 term only."""
        from collections import Counter

        normalized_prediction = normalize_qasper_answer(prediction)
        normalized_ground_truth = normalize_qasper_answer(ground_truth)

        ZERO_METRIC = 0.0

        if (normalized_prediction in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth):
            return ZERO_METRIC
        if (normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth):
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    # The case measured on the preregistered sample, 2026-08-18.
    MEASURED_PRED = "Yes, both films were directed by the same person."
    MEASURED_GOLD = "yes"

    def test_the_measured_case_scores_zero(self):
        self.assertEqual(
            hotpot_token_f1(self.MEASURED_PRED, self.MEASURED_GOLD), 0.0)

    def test_the_guard_is_load_bearing_on_that_case(self):
        """Without the guard this pair earns 2/9 — so the test above
        cannot be passing for some incidental reason."""
        self.assertAlmostEqual(
            token_f1(self.MEASURED_PRED, self.MEASURED_GOLD), 0.2222, places=4)

    def test_it_scores_zero_THROUGH_the_benchmark_scorer(self):
        """The property is about the scorer a cell actually runs, not
        about a helper reachable from a test."""
        s = HotpotQABenchmark().score_answer(
            self.MEASURED_PRED, _yn_query(self.MEASURED_GOLD))
        self.assertEqual(s.value, 0.0)
        self.assertEqual(s.method, "token_f1")

    def test_the_pooled_variant_inherits_it(self):
        """All ten HotpotQA cells score alike or the column is two
        populations wearing one name."""
        s = HotpotQAPooledBenchmark().score_answer(
            self.MEASURED_PRED, _yn_query(self.MEASURED_GOLD))
        self.assertEqual(s.value, 0.0)

    def test_an_exactly_correct_yes_still_scores_one(self):
        for pred in ("yes", "Yes", "Yes."):
            self.assertEqual(hotpot_token_f1(pred, "yes"), 1.0, pred)

    def test_the_opposite_sentinel_scores_zero(self):
        self.assertEqual(hotpot_token_f1("no", "yes"), 0.0)
        self.assertEqual(hotpot_token_f1("yes", "no"), 0.0)

    def test_a_sentinel_prediction_against_a_real_gold_scores_zero(self):
        """The FIRST official branch: prediction is the sentinel."""
        self.assertEqual(hotpot_token_f1("yes", "the Eiffel Tower"), 0.0)

    def test_non_sentinel_pairs_are_untouched(self):
        """The guard must not disturb the 93.9% of rows it does not
        concern."""
        for pred, gold in (
            ("Eiffel Tower", "the Eiffel Tower"),
            ("Paris France", "Paris"),
            ("nothing in common", "the Eiffel Tower"),
        ):
            self.assertEqual(hotpot_token_f1(pred, gold),
                             token_f1(pred, gold), (pred, gold))

    def test_agreement_with_the_transcribed_official_scorer(self):
        battery = [
            (self.MEASURED_PRED, "yes"),
            ("yes", "yes"), ("no", "no"), ("yes", "no"), ("no", "yes"),
            ("yes", "the Eiffel Tower"), ("noanswer", "yes"),
            ("No, they were not.", "no"),
            ("Eiffel Tower", "the Eiffel Tower"),
            ("Paris", "Paris"), ("Paris", "London"),
            ("the same person directed both", "yes"),
        ]
        for pred, gold in battery:
            self.assertAlmostEqual(
                hotpot_token_f1(pred, gold), self._official_f1(pred, gold),
                places=12, msg=f"{pred!r} vs {gold!r}")

    def test_exact_match_is_NOT_routed_through_the_guard(self):
        """Official `exact_match_score` has no guard, and this harness
        already matched it. Wrapping EM would create a divergence where
        none existed."""
        s = HotpotQABenchmark().score_answer("Yes", _yn_query("yes"))
        self.assertEqual(s.metadata["exact_match"], 1.0)
        s2 = HotpotQABenchmark().score_answer(
            self.MEASURED_PRED, _yn_query("yes"))
        self.assertEqual(s2.metadata["exact_match"], 0.0)


class TestTheGuardIsHotpotLocal(unittest.TestCase):
    """It must not reach the other two live benchmarks.

    BEHAVIOURAL, not a source grep: each scorer is driven on the exact
    pair the guard would zero, and asserted to return the UNGUARDED
    value. A grep for the import would pass just as well if the shared
    `token_f1` had been edited instead, which is the change this test
    exists to forbid.
    """

    PRED = "Yes, both films were directed by the same person."
    GOLD = "yes"

    def test_the_shared_token_f1_is_unchanged(self):
        self.assertAlmostEqual(token_f1(self.PRED, self.GOLD), 0.2222,
                               places=4)

    def test_multihop_still_scores_it_unguarded(self):
        from src.eval.multihop import MultiHopBenchmark
        from src.eval.types import ANSWER_TYPE_FREE_FORM

        q = EvalQuery(
            query_id="m", question_text="?", parent_scope=None,
            gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                     free_form=self.GOLD),),
            gold_passage_sets=(frozenset(),), question_type="comparison_query")
        s = MultiHopBenchmark().score_answer(self.PRED, q)
        self.assertAlmostEqual(s.value, 0.2222, places=4)

    def test_narrativeqa_still_scores_it_unguarded(self):
        from src.eval.narrativeqa import NarrativeQABenchmark
        from src.eval.types import ANSWER_TYPE_FREE_FORM

        q = EvalQuery(
            query_id="n", question_text="?", parent_scope=None,
            gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                     free_form=self.GOLD),),
            gold_passage_sets=(), question_type="movie")
        s = NarrativeQABenchmark().score_answer(self.PRED, q)
        self.assertAlmostEqual(s.value, 0.2222, places=4)


if __name__ == "__main__":
    unittest.main()
