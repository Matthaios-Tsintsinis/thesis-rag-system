"""Pins the null-query pure-refusal rule: the anchored detector must fire
and the remainder after the hedge clause must assert nothing.
"""

from __future__ import annotations

import unittest

from src.eval.multihop import MultiHopBenchmark
from src.eval.scorers import score_unanswerable
from src.eval.types import ANSWER_TYPE_UNANSWERABLE, EvalQuery, GoldAnswer


# harness addition: see METHODS §C.9
# The eight pinned cases; the fourth and last show why a digit or
# capitalised-token check cannot stand in for the remainder test.
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
    """Build one MultiHop null query with an unanswerable gold."""
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
    """Pins the eight cases through the scorer and the benchmark."""

    def test_scorer_matches_every_expected_outcome(self):
        """score_unanswerable returns the pinned value for every case."""
        for pred, expected in CASES:
            self.assertEqual(score_unanswerable(pred), expected, pred)

    def test_the_rule_reaches_the_benchmark_through_score_answer(self):
        """MultiHopBenchmark.score_answer applies the rule to null queries."""
        bench = MultiHopBenchmark()
        q = _null_query()
        for pred, expected in CASES:
            score = bench.score_answer(pred, q)
            self.assertEqual(score.value, expected, pred)
            self.assertEqual(score.method, "unanswerable_rule", pred)


class TestFabricationIsNotCredited(unittest.TestCase):
    """Pins that a hedge around an assertion earns nothing."""

    def test_a_hedge_wrapping_an_entity_scores_zero(self):
        """A hedge followed by a named entity scores 0.0."""
        self.assertEqual(
            score_unanswerable("I don't know the year, but the answer is Tesla."),
            0.0)

    def test_a_hedge_wrapping_a_person_scores_zero(self):
        """A hedge followed by a named person scores 0.0."""
        self.assertEqual(
            score_unanswerable("Insufficient information; the CEO is Tim Cook."),
            0.0)

    def test_a_bare_fabrication_scores_zero(self):
        """An answer with no hedge at all scores 0.0."""
        self.assertEqual(score_unanswerable("Tim Cook."), 0.0)


class TestPureRefusalsAreCredited(unittest.TestCase):
    """Pins that a refusal asserting nothing scores 1.0."""

    def test_a_refusal_echoing_a_year_is_still_pure(self):
        """A year inside the refusal clause itself does not break purity."""
        self.assertEqual(score_unanswerable("The evidence does not cover 2023."),
                         1.0)

    def test_a_refusal_naming_a_sentence_initial_entity_is_still_pure(self):
        """An entity that is the object of a non-mention keeps purity."""
        self.assertEqual(
            score_unanswerable("Tesla is not mentioned in the evidence."), 0.0)
        self.assertEqual(
            score_unanswerable("The evidence does not mention Tesla."), 1.0)

    def test_the_canonical_prompted_response(self):
        """The prompted refusal string scores 1.0."""
        self.assertEqual(score_unanswerable("No answer available."), 1.0)


class TestRuleComposition(unittest.TestCase):
    """Pins how the detector and the remainder test compose."""

    def test_a_non_abstaining_prediction_scores_zero_without_a_remainder_test(self):
        """Without a detector hit there is no hedge to strip, so 0.0."""
        self.assertEqual(score_unanswerable("The CEO is Tim Cook."), 0.0)

    def test_empty_prediction_scores_zero(self):
        """An empty or whitespace-only prediction scores 0.0."""
        self.assertEqual(score_unanswerable(""), 0.0)
        self.assertEqual(score_unanswerable("   "), 0.0)


if __name__ == "__main__":
    unittest.main()
