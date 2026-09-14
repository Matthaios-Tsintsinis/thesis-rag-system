"""Tests for the anchored abstention detector: a hedge counts only as the
whole utterance or the leading clause, and its span is returned so the
null rule can strip it (METHODS §C.9)."""

from __future__ import annotations

import unittest

from src.eval.scorers.unanswerable import (
    detect_abstention,
    is_abstention,
    is_filler_only,
    score_abstention,
)


class TestAnchoring(unittest.TestCase):
    def test_hedge_inside_an_informative_clause_does_not_abstain(self):
        """A hedge buried inside a factual clause is not an abstention."""
        text = ("The report gives insufficient information about Q3, "
                "but revenue rose to 4.1 billion.")
        m = detect_abstention(text)
        self.assertFalse(m.matched)
        self.assertIsNone(m.span)

    def test_whole_utterance_hedge_spans_everything(self):
        """A whole-utterance hedge spans all of the text, remainder empty."""
        m = detect_abstention("There is insufficient information.")
        self.assertTrue(m.matched)
        self.assertEqual(m.text[m.span[0]:m.span[1]],
                         "there is insufficient information")
        self.assertEqual(m.remainder, "")

    def test_leading_clause_hedge_spans_only_that_clause(self):
        """A leading-clause hedge spans only that clause; the rest survives."""
        m = detect_abstention("I don't know, but the answer is Tesla.")
        self.assertTrue(m.matched)
        self.assertEqual(m.text[m.span[0]:m.span[1]], "i don't know")
        self.assertIn("tesla", m.remainder)


class TestNegativeExistenceFramesConsumeTheirObject(unittest.TestCase):
    """A "does not mention X" frame consumes X: X names what is absent."""

    def test_a_digit_bearing_object_is_still_a_pure_refusal(self):
        """An object with digits leaves an empty remainder."""
        m = detect_abstention("The evidence does not cover 2023.")
        self.assertTrue(m.matched)
        self.assertEqual(m.remainder, "")

    def test_an_entity_bearing_object_is_still_a_pure_refusal(self):
        """An object naming an entity leaves an empty remainder."""
        m = detect_abstention("The context does not mention Tesla.")
        self.assertTrue(m.matched)
        self.assertEqual(m.remainder, "")

    def test_but_content_after_the_clause_survives_in_the_remainder(self):
        """Content after the frame's clause stays in the remainder."""
        m = detect_abstention(
            "The context does not mention Tesla, but the answer is Tesla.")
        self.assertTrue(m.matched)
        self.assertIn("tesla", m.remainder)


class TestTrailingHedgeIsNotLeading(unittest.TestCase):
    def test_a_hedge_after_content_does_not_abstain(self):
        """A hedge that follows content is not an abstention."""
        self.assertFalse(
            is_abstention("There were 4 acquisitions, but I am not certain."))

    def test_a_bare_factual_answer_does_not_abstain(self):
        """A bare factual answer is not an abstention."""
        self.assertFalse(is_abstention("Tim Cook."))


class TestCanonicalResponse(unittest.TestCase):
    def test_the_prompted_canonical_string_always_matches(self):
        """The prompted refusal string matches and scores 1.0."""
        m = detect_abstention("No answer available.")
        self.assertTrue(m.matched)
        self.assertEqual(m.remainder, "")
        self.assertEqual(score_abstention("No answer available."), 1.0)


class TestFillerPrimitive(unittest.TestCase):
    def test_filler_only_accepts_function_words(self):
        """Function words, punctuation and the empty string are filler."""
        self.assertTrue(is_filler_only("there is"))
        self.assertTrue(is_filler_only(""))
        self.assertTrue(is_filler_only(", but"))

    def test_filler_only_rejects_content(self):
        """Content words, names and digits are not filler."""
        self.assertFalse(is_filler_only("the report gives about q3"))
        self.assertFalse(is_filler_only("tim cook"))
        self.assertFalse(is_filler_only("2023"))


class TestDegenerate(unittest.TestCase):
    def test_empty_prediction_does_not_abstain(self):
        """An empty or blank prediction is not an abstention."""
        self.assertFalse(is_abstention(""))
        self.assertFalse(is_abstention("   "))

    def test_span_indexes_the_normalised_text(self):
        """The span indexes the normalised text, not the raw input."""
        m = detect_abstention("  No Answer Available.  ")
        self.assertTrue(m.matched)
        self.assertEqual(m.text[m.span[0]:m.span[1]], m.text)


if __name__ == "__main__":
    unittest.main()
