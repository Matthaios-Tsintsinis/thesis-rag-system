"""What the `abstained` flag is catching, classified not eyeballed.

The classifier reuses the null-query rule's own machinery — hedge span
from `detect_abstention`, `is_filler_only` on the remainder — so it
cannot disagree with the scorer about what a pure refusal is. These
tests pin that agreement against the SAME eight predictions P2 pinned,
which is what makes them evidence rather than a restatement.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.inspect_abstentions import (
    DETECTOR_DISAGREES,
    HEDGED_CONTENT,
    PURE_REFUSAL,
    classify,
    inspect,
)


class TestTheClassifierAgreesWithTheNullRule(unittest.TestCase):
    """P2's eight pinned cases. A pure refusal scores 1.0 under
    `score_unanswerable`; those are exactly the ones that must classify
    as PURE_REFUSAL here."""

    def test_pure_refusals(self):
        for pred in ("No answer available.",
                     "I don't know.",
                     "The evidence does not contain this information.",
                     "The evidence does not cover 2023."):
            self.assertEqual(classify(pred)[0], PURE_REFUSAL, pred)

    def test_hedge_then_assertion_is_NOT_a_refusal(self):
        """The case that makes `abstained` a vocabulary marker: flagged,
        but it answers."""
        for pred in ("I don't know the year, but the answer is Tesla.",
                     "Insufficient information; the CEO is Tim Cook."):
            verdict, hedge, remainder = classify(pred)
            self.assertEqual(verdict, HEDGED_CONTENT, pred)
            self.assertTrue(hedge, pred)
            self.assertTrue(remainder, pred)

    def test_the_remainder_keeps_the_asserted_entity(self):
        """This is why such a row can score well on token-F1 while
        carrying abstained=True."""
        _, _, remainder = classify(
            "I don't know the year, but the answer is Tesla.")
        self.assertIn("tesla", remainder)

    def test_a_prediction_the_detector_does_not_flag_is_drift(self):
        """Recorded abstained but no hedge span under the current
        detector — counted separately, never folded into a real
        category."""
        self.assertEqual(
            classify("I'm not certain, but it's Sam Bankman-Fried.")[0],
            DETECTOR_DISAGREES)


def _write(rows):
    f = Path(tempfile.mkdtemp()) / "cell.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return f


def _row(qid, predicted, value, *, method="token_f1", abstained=True):
    return {
        "system_id": "M1", "benchmark": "multihop_rag", "query_id": qid,
        "predicted_answer": predicted,
        "answer": {"value": value, "method": method,
                   "metadata": {"abstained": abstained}},
    }


class TestInspectSeparatesTheTwoMeanings(unittest.TestCase):
    def setUp(self):
        self.out = inspect(_write([
            _row("a", "No answer available.", 0.0),
            _row("b", "I don't know.", 0.0),
            _row("c", "I don't know the year, but the answer is Tesla.", 0.8),
            _row("d", "Insufficient information; the CEO is Tim Cook.", 0.6),
            # not flagged -> outside the bucket entirely
            _row("e", "Tim Cook.", 0.9, abstained=False),
            # null row -> excluded: a refusal there is CORRECT
            _row("n1", "No answer available.", 1.0,
                 method="unanswerable_rule"),
        ]))

    def test_null_rows_are_excluded_from_the_population(self):
        self.assertEqual(self.out["n_answerable_rows"], 5)
        self.assertEqual(self.out["n_flagged_abstained"], 4)

    def test_the_two_meanings_get_different_means(self):
        """The whole finding in one assertion: refusals score 0, hedged
        answers score well, and averaging them together is what produced
        the misleading 0.198."""
        self.assertEqual(self.out["pure_refusal"]["n"], 2)
        self.assertAlmostEqual(self.out["pure_refusal"]["mean_score"], 0.0)
        self.assertEqual(self.out["hedged_content"]["n"], 2)
        self.assertAlmostEqual(self.out["hedged_content"]["mean_score"], 0.7)

    def test_the_blended_mean_is_what_analyse_would_have_shown(self):
        pr, hc = self.out["pure_refusal"], self.out["hedged_content"]
        blended = (pr["n"] * pr["mean_score"] + hc["n"] * hc["mean_score"]) / 4
        self.assertAlmostEqual(blended, 0.35)
        self.assertGreater(blended, pr["mean_score"])

    def test_hedges_are_reported_with_their_counts(self):
        hedges = {h["hedge"] for h in self.out["top_hedges"]}
        self.assertIn("no answer available", hedges)
        self.assertIn("insufficient information", hedges)


if __name__ == "__main__":
    unittest.main()
