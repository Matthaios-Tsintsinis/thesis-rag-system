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


class TestCreditedRefusalsAreSurfaced(unittest.TestCase):
    """A PURE REFUSAL asserts nothing, so on an ANSWERABLE question it
    must score 0.0. Anything above zero means the answer scorer credited
    a refusal, and that has to be visible rather than averaged in.

    MEASURED on MultiHop: the canonical "No answer available." normalises
    to ["no", "answer", "available"], and among all 2,255 answerable
    golds the ONLY one it overlaps is the bare token "no" (561 of them,
    24.9%), which yields exactly 0.5. So the score alone diagnoses it and
    no join to the gold is needed.
    """

    def test_the_canonical_refusal_scores_exactly_half_against_gold_no(self):
        from src.eval.scorers.extractive import token_f1

        self.assertAlmostEqual(token_f1("No answer available.", "no"), 0.5)

    def test_no_other_short_gold_credits_the_refusal(self):
        from src.eval.scorers.extractive import token_f1

        for gold in ("yes", "before", "after", "Apple", "Google"):
            self.assertEqual(token_f1("No answer available.", gold), 0.0, gold)

    def test_credited_refusals_are_counted_and_massed(self):
        out = inspect(_write([
            _row("a", "No answer available.", 0.0),
            _row("b", "No answer available.", 0.5),
            _row("c", "No answer available.", 0.5),
            _row("d", "No answer available.", 0.0),
        ]))
        pr = out["pure_refusal"]
        self.assertEqual(pr["n"], 4)
        self.assertEqual(pr["n_credited"], 2)
        self.assertAlmostEqual(pr["credited_share"], 0.5)
        self.assertAlmostEqual(pr["credited_score_mass"], 1.0)

    def test_the_histogram_shows_the_two_populations(self):
        """Identical strings scoring differently is the signature; a mean
        alone hides it completely."""
        out = inspect(_write([
            _row("a", "No answer available.", 0.0),
            _row("b", "No answer available.", 0.5),
        ]))
        self.assertEqual(out["pure_refusal"]["score_histogram"],
                         {0.0: 1, 0.5: 1})
        self.assertAlmostEqual(out["pure_refusal"]["mean_score"], 0.25)

    def test_a_clean_cell_reports_no_credited_refusals(self):
        out = inspect(_write([
            _row("a", "No answer available.", 0.0),
            _row("b", "I don't know.", 0.0),
        ]))
        self.assertEqual(out["pure_refusal"]["n_credited"], 0)


class TestTheSupplementaryFigure(unittest.TestCase):
    """Required beside the primary on every MultiHop cell.

    The primary stays token-F1 exactly as frozen. This is the same rows
    with the credited-refusal mass removed, computed post-hoc — no scorer
    change, no freeze violation, no matrix split.
    """

    def setUp(self):
        rows = [
            _row("a", "No answer available.", 0.0),
            _row("b", "No answer available.", 0.5),
            _row("c", "No answer available.", 0.5),
            _row("d", "No answer available.", 0.0),
        ]
        rows += [_row(f"e{i}", "Google", 0.1, abstained=False)
                 for i in range(6)]
        self.out = inspect(_write(rows))

    def test_the_primary_is_the_plain_answerable_mean(self):
        """(4 refusals contributing 1.0 + 6 answers contributing 0.6)/10."""
        self.assertAlmostEqual(self.out["answerable_mean"], 0.16)

    def test_the_supplementary_removes_only_the_credited_mass(self):
        self.assertAlmostEqual(
            self.out["answerable_mean_excluding_credited"], 0.06)

    def test_the_credited_share_is_reported(self):
        self.assertAlmostEqual(
            self.out["credited_share_of_answerable_mass"], 1.0 / 1.6)

    def test_the_share_depends_on_refusal_rate_so_it_cannot_cancel(self):
        """The confound in one assertion: two systems with the SAME
        answers but different refusal rates get different credited
        shares, so the artifact does not subtract out of a comparison."""
        few = inspect(_write(
            [_row("a", "No answer available.", 0.5)]
            + [_row(f"e{i}", "Google", 0.1, abstained=False)
               for i in range(6)]))
        many = inspect(_write(
            [_row(f"a{i}", "No answer available.", 0.5) for i in range(4)]
            + [_row(f"e{i}", "Google", 0.1, abstained=False)
               for i in range(6)]))
        self.assertNotAlmostEqual(
            few["credited_share_of_answerable_mass"],
            many["credited_share_of_answerable_mass"])
        self.assertGreater(many["credited_share_of_answerable_mass"],
                           few["credited_share_of_answerable_mass"])

    def test_a_clean_cell_leaves_the_two_figures_equal(self):
        out = inspect(_write([
            _row("a", "No answer available.", 0.0),
            _row("b", "Google", 0.4, abstained=False),
        ]))
        self.assertAlmostEqual(out["answerable_mean"],
                               out["answerable_mean_excluding_credited"])


if __name__ == "__main__":
    unittest.main()
