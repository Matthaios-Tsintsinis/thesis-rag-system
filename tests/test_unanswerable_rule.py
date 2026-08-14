"""The null-query pure-refusal rule (P2).

Detection alone used to decide the MultiHop null_query score, and it
credited fabrication: "I don't know the year, but the answer is Tesla."
scored 1.0 — full marks for naming an entity the corpus does not support
(docs/EVAL_AUDIT.md ISSUE-2).

The rule now has two parts: the anchored detector must fire, AND the
utterance with the hedge clause removed must assert nothing.

WHY NOT AN ENTITY DETECTOR. The alternative proposal credited 1.0 unless
the remainder held a digit or a non-sentence-initial capitalised token.
It fails in both directions, and the fourth and last cases below are the
counter-examples: a refusal that echoes a year would have scored 0.0, and
a refusal naming a sentence-initial entity would have scored 1.0.
"""

from __future__ import annotations

import unittest

from src.eval.multihop import MultiHopBenchmark
from src.eval.scorers import score_unanswerable
from src.eval.types import ANSWER_TYPE_UNANSWERABLE, EvalQuery, GoldAnswer


# The eight cases fixed before generation begins, with the eighth
# (the digit-bearing pure refusal) added because it is what ruled the
# entity/digit heuristic out.
CASES = (
    ("No answer available.", 1.0),
    ("I don't know.", 1.0),
    ("The evidence does not contain this information.", 1.0),
    ("The evidence does not cover 2023.", 1.0),
    ("I don't know the year, but the answer is Tesla.", 0.0),
    ("Insufficient information; the CEO is Tim Cook.", 0.0),
    ("Tim Cook.", 0.0),
    ("There were 4 acquisitions, but I am not certain.", 0.0),
)


def _null_query() -> EvalQuery:
    return EvalQuery(
        query_id="null1",
        question_text="who?",
        parent_scope=None,
        gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_UNANSWERABLE,
                                 free_form="", unanswerable=True),),
        gold_passage_sets=(frozenset(),),
        question_type="null_query",
        metadata={},
    )


class TestTheEightCases(unittest.TestCase):
    def test_scorer_matches_every_expected_outcome(self):
        for pred, expected in CASES:
            self.assertEqual(score_unanswerable(pred), expected, pred)

    def test_the_rule_reaches_the_benchmark_through_score_answer(self):
        """The scorer being right is not enough if the benchmark does not
        call it."""
        bench = MultiHopBenchmark()
        q = _null_query()
        for pred, expected in CASES:
            score = bench.score_answer(pred, q)
            self.assertEqual(score.value, expected, pred)
            self.assertEqual(score.method, "unanswerable_rule", pred)


class TestFabricationIsNotCredited(unittest.TestCase):
    def test_a_hedge_wrapping_an_entity_scores_zero(self):
        self.assertEqual(
            score_unanswerable("I don't know the year, but the answer is Tesla."),
            0.0)

    def test_a_hedge_wrapping_a_person_scores_zero(self):
        self.assertEqual(
            score_unanswerable("Insufficient information; the CEO is Tim Cook."),
            0.0)

    def test_a_bare_fabrication_scores_zero(self):
        self.assertEqual(score_unanswerable("Tim Cook."), 0.0)


class TestPureRefusalsAreCredited(unittest.TestCase):
    def test_a_refusal_echoing_a_year_is_still_pure(self):
        """The entity/digit heuristic would have scored this 0.0. The
        year names what the evidence does NOT cover; nothing is
        asserted."""
        self.assertEqual(score_unanswerable("The evidence does not cover 2023."),
                         1.0)

    def test_a_refusal_naming_a_sentence_initial_entity_is_still_pure(self):
        """The mirror failure: a capitalisation rule would have credited
        this for the wrong reason. Here it is credited for the right one
        — the entity is the object of an explicit non-mention."""
        self.assertEqual(
            score_unanswerable("Tesla is not mentioned in the evidence."), 0.0)
        self.assertEqual(
            score_unanswerable("The evidence does not mention Tesla."), 1.0)

    def test_the_canonical_prompted_response(self):
        self.assertEqual(score_unanswerable("No answer available."), 1.0)


class TestRuleComposition(unittest.TestCase):
    def test_a_non_abstaining_prediction_scores_zero_without_a_remainder_test(self):
        """If the detector does not fire, the rule short-circuits: there
        is no hedge to strip, so nothing can be a pure refusal."""
        self.assertEqual(score_unanswerable("The CEO is Tim Cook."), 0.0)

    def test_empty_prediction_scores_zero(self):
        self.assertEqual(score_unanswerable(""), 0.0)
        self.assertEqual(score_unanswerable("   "), 0.0)


if __name__ == "__main__":
    unittest.main()
