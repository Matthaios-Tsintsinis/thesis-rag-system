"""The unified answer-scoring contract (P1), pinned across all three benchmarks.

WHAT THIS EXISTS TO PREVENT. Both MultiHop and NarrativeQA used to return
0.0 the moment a prediction contained a hedging phrase, BEFORE computing
token-F1. A prediction carrying the exact gold string scored 0.0000 where
its real F1 was 0.3333, and the identical prediction scored 0.3333 on
HotpotQA, which never had the gate — so the same answer was worth
different amounts depending on which benchmark it was scored under.
Measured in docs/EVAL_AUDIT.md ISSUE-1.

THE CONTRACT:
  1. The score for an answerable query is token-F1 against gold, max over
     references, ALWAYS COMPUTED.
  2. Abstention detection is metadata.abstained and reaches nothing else.
  3. metadata.token_f1 holds the real F1, never a post-gate value.
  4. answer.method names the RULE, not the OUTCOME.

The equality test below is the load-bearing one: it fails if any single
benchmark reintroduces a gate, because the three would stop agreeing.
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


# The hedged-but-correct prediction from the audit. It carries the exact
# gold string AND a hedge, which is precisely the case the deleted gate
# scored as a refusal.
HEDGED = "The evidence does not give a date, but the person is Sam Bankman-Fried."
GOLD = "Sam Bankman-Fried"
EXPECTED = 1 / 3  # 3 shared tokens; 15 predicted, 3 gold -> F1 = 1/3

ALLOWED_METHODS = {"token_f1", "exact_match", "unanswerable_rule", "no_references"}


def _query(gold: str, *, answer_type: str = ANSWER_TYPE_FREE_FORM,
           n_refs: int = 1) -> EvalQuery:
    """A query carrying `n_refs` references.

    n_refs=1 by design for the equality test: NarrativeQA scores
    max-over-references, so a two-reference fixture would compare a max
    against two single-reference scores and the equality would be an
    artifact of the fixture rather than of the contract.
    """
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
    return {
        "multihop": MultiHopBenchmark().score_answer,
        "narrativeqa": NarrativeQABenchmark().score_answer,
        "hotpotqa": HotpotQABenchmark.__new__(HotpotQABenchmark).score_answer,
    }


class TestThreeBenchmarkEquality(unittest.TestCase):
    def test_hedged_correct_answer_scores_identically_everywhere(self):
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
        """Guards the guard. If the detector stopped firing on this
        prediction, the equality test above would pass for the wrong
        reason — it would no longer exercise a gate at all."""
        self.assertTrue(is_abstention(HEDGED))

    def test_the_expected_value_is_the_independently_computed_f1(self):
        self.assertAlmostEqual(token_f1(HEDGED, GOLD), EXPECTED, places=10)


class TestAbstentionIsMetadataOnly(unittest.TestCase):
    def test_abstained_is_recorded_and_the_score_is_unaffected(self):
        q = _query(GOLD, n_refs=1)
        for name, fn in _scorers().items():
            score = fn(HEDGED, q)
            self.assertTrue(score.metadata.get("abstained"),
                            f"{name} did not record abstained=True")
            self.assertAlmostEqual(score.value, EXPECTED, places=10,
                                   msg=f"{name} let abstention move the score")

    def test_score_always_equals_the_computed_token_f1(self):
        """Behavioural proof the gate is gone: across predictions whose
        hedging and whose overlap vary independently, the value tracks the
        F1 and nothing else."""
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
        score = MultiHopBenchmark().score_answer(HEDGED, _query(GOLD, n_refs=1))
        self.assertAlmostEqual(score.metadata["token_f1"], EXPECTED, places=10)


class TestPureRefusal(unittest.TestCase):
    def test_a_refusal_with_no_overlap_scores_zero_via_token_f1(self):
        q = _query("Tim Cook", n_refs=1)
        for name, fn in _scorers().items():
            score = fn("No answer available.", q)
            self.assertEqual(score.value, 0.0, f"{name}")
            # The 0.0 must come from the metric, not from a gate: the
            # method names the rule that produced it.
            self.assertEqual(score.method, "token_f1", f"{name}")


class TestMethodNamesTheRuleNotTheOutcome(unittest.TestCase):
    def test_no_method_encodes_an_outcome(self):
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
        q = _query("", answer_type=ANSWER_TYPE_UNANSWERABLE, n_refs=1)
        score = MultiHopBenchmark().score_answer("No answer available.", q)
        self.assertEqual(score.method, "unanswerable_rule")


if __name__ == "__main__":
    unittest.main()
