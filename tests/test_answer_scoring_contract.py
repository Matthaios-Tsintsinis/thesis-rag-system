"""Pins the shared answer-scoring contract across all three benchmarks:
token-F1 is always computed, abstention is metadata only, and the method
names the scoring rule (METHODS §C.1, §C.9).
"""

from __future__ import annotations

import unittest

from src.eval.hotpotqa import HotpotQABenchmark
from src.eval.multihop import MultiHopBenchmark
from src.eval.narrativeqa import NarrativeQABenchmark
from src.eval.scorers import is_abstention
from src.eval.scorers.extractive import token_f1
from src.eval.types import (
    ANSWER_TYPE_FREE_FORM,
    ANSWER_TYPE_UNANSWERABLE,
    EvalQuery,
    GoldAnswer,
)


# A hedged prediction that still contains the exact gold string. After
# normalisation it has 10 tokens, the gold 2, all 2 shared: P 0.2, R 1.
# official: hotpot_evaluate_v1.py::f1_score @ 36358534
HEDGED = "The evidence does not give a date, but the person is Sam Bankman-Fried."
GOLD = "Sam Bankman-Fried"
EXPECTED = 1 / 3  # 2PR / (P + R) with P = 0.2, R = 1

ALLOWED_METHODS = {"token_f1", "exact_match", "unanswerable_rule", "no_references"}


def _query(gold: str, *, answer_type: str = ANSWER_TYPE_FREE_FORM,
           n_refs: int = 1) -> EvalQuery:
    """Build a query with n_refs identical gold references."""
    return EvalQuery(
        query_id="q1",
        question_text="who?",
        parent_scope=None,
        gold_answers=tuple(
            GoldAnswer(answer_type=answer_type, free_form=gold,
                       unanswerable=(answer_type == ANSWER_TYPE_UNANSWERABLE))
            for _ in range(n_refs)
        ),
        gold_passage_sets=(frozenset({("d", "<whole>")}),),
        question_type="inference_query",
        metadata={},
    )


def _scorers():
    """Return the score_answer callable of each benchmark by name."""
    return {
        "multihop": MultiHopBenchmark().score_answer,
        "narrativeqa": NarrativeQABenchmark().score_answer,
        "hotpotqa": HotpotQABenchmark.__new__(HotpotQABenchmark).score_answer,
    }


class TestThreeBenchmarkEquality(unittest.TestCase):
    """The three benchmarks score one hedged prediction the same way."""

    def test_hedged_correct_answer_scores_identically_everywhere(self):
        """Every benchmark scores the hedged prediction at its token-F1."""
        q = _query(GOLD, n_refs=1)
        scores = {name: fn(HEDGED, q).value for name, fn in _scorers().items()}
        for name, value in scores.items():
            self.assertAlmostEqual(
                value, EXPECTED, places=10,
                msg=f"{name} scored {value}, expected {EXPECTED} — a gate is back",
            )
        self.assertEqual(len(set(round(v, 12) for v in scores.values())), 1,
                         f"benchmarks disagree: {scores}")

    def test_the_prediction_really_does_read_as_an_abstention(self):
        """The detector fires on HEDGED, so the equality test is real."""
        self.assertTrue(is_abstention(HEDGED))

    def test_the_expected_value_is_the_independently_computed_f1(self):
        """EXPECTED equals token_f1 of the fixture."""
        self.assertAlmostEqual(token_f1(HEDGED, GOLD), EXPECTED, places=10)


class TestAbstentionIsMetadataOnly(unittest.TestCase):
    """Abstention detection writes metadata and never moves the score."""

    def test_abstained_is_recorded_and_the_score_is_unaffected(self):
        """metadata.abstained is True while the value stays at token-F1."""
        q = _query(GOLD, n_refs=1)
        for name, fn in _scorers().items():
            score = fn(HEDGED, q)
            self.assertTrue(score.metadata.get("abstained"),
                            f"{name} did not record abstained=True")
            self.assertAlmostEqual(score.value, EXPECTED, places=10,
                                   msg=f"{name} let abstention move the score")

    def test_score_always_equals_the_computed_token_f1(self):
        """The value tracks token-F1 whether or not the prediction hedges."""
        cases = [
            ("Sam Bankman-Fried", GOLD),
            ("The evidence does not say, but it is Sam Bankman-Fried.", GOLD),
            ("I don't know the year, but the answer is Tesla.", "Tesla"),
            ("There is insufficient information about the CEO.", "Tim Cook"),
            ("No answer available.", GOLD),
        ]
        mh = MultiHopBenchmark().score_answer
        for pred, gold in cases:
            score = mh(pred, _query(gold, n_refs=1))
            self.assertAlmostEqual(
                score.value, token_f1(pred, gold), places=10,
                msg=f"value diverged from token_f1 for {pred!r}",
            )

    def test_metadata_token_f1_is_the_real_value_not_a_post_gate_zero(self):
        """metadata.token_f1 holds the computed F1."""
        score = MultiHopBenchmark().score_answer(HEDGED, _query(GOLD, n_refs=1))
        self.assertAlmostEqual(score.metadata["token_f1"], EXPECTED, places=10)


class TestPureRefusal(unittest.TestCase):
    """A refusal on an answerable query scores through token-F1."""

    def test_a_refusal_with_no_overlap_scores_zero_via_token_f1(self):
        """A refusal with no gold overlap scores 0.0 under method token_f1."""
        q = _query("Tim Cook", n_refs=1)
        for name, fn in _scorers().items():
            score = fn("No answer available.", q)
            self.assertEqual(score.value, 0.0, f"{name}")
            # The zero comes from the metric; the method says so.
            self.assertEqual(score.method, "token_f1", f"{name}")


class TestMethodNamesTheRuleNotTheOutcome(unittest.TestCase):
    """answer.method names the scoring rule, never the outcome."""

    def test_no_method_encodes_an_outcome(self):
        """No benchmark emits a method containing 'abstain'."""
        predictions = [
            "Sam Bankman-Fried",
            HEDGED,
            "No answer available.",
            "I don't know.",
            "",
        ]
        for name, fn in _scorers().items():
            for pred in predictions:
                method = fn(pred, _query(GOLD, n_refs=1)).method
                self.assertNotIn("abstain", method,
                                 f"{name} emitted outcome-encoding method {method!r}")
                self.assertIn(method, ALLOWED_METHODS, f"{name}: {method!r}")

    def test_null_queries_use_the_unanswerable_rule(self):
        """A null query scores under unanswerable_rule (METHODS §C.9)."""
        q = _query("", answer_type=ANSWER_TYPE_UNANSWERABLE, n_refs=1)
        score = MultiHopBenchmark().score_answer("No answer available.", q)
        self.assertEqual(score.method, "unanswerable_rule")


if __name__ == "__main__":
    unittest.main()
