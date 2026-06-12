"""Offline unit tests for the multiple-choice scorer (no GPU/API/network).

Run from the repo root:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from src.eval.scorers.multiple_choice import (
    RULE3_MIN_MARGIN,
    extract_choice,
    score_multiple_choice,
)


# Four options with no shared content — letter/abstention cases must not
# accidentally resolve via text rules.
OPTS = (
    "The dog barked at strangers all night",
    "The cat slept on the warm windowsill",
    "The bird flew south before the winter storm",
    "The fish swam in slow circles",
)


class TestLetterExtraction(unittest.TestCase):
    def test_bare_letter_fullmatch(self) -> None:
        for text, want in [("B", "B"), ("(B)", "B"), ("b", "B"),
                           ("**C**", "C"), ("A", "A"), ("  d  ", "D")]:
            letter, method = extract_choice(text, OPTS)
            self.assertEqual(letter, want, text)
            self.assertEqual(method, "letter_leading", text)

    def test_leading_letter_with_punctuation(self) -> None:
        for text, want in [("B) because of the storm", "B"),
                           ("b. as stated", "B"),
                           ("D: because...", "D"),
                           ("A: the dog", "A")]:
            letter, method = extract_choice(text, OPTS)
            self.assertEqual(letter, want, text)
            self.assertEqual(method, "letter_leading", text)

    def test_leading_bare_letter_plus_words_b_through_d(self) -> None:
        letter, method = extract_choice("B is correct because the text says so", OPTS)
        self.assertEqual(letter, "B")
        self.assertEqual(method, "letter_leading")

    def test_markers(self) -> None:
        for text, want in [("Option C", "C"),
                           ("Answer: D", "D"),
                           ("The answer is (b)", "B"),
                           ("my choice is A", "A"),
                           ("I think the correct option is B here", "B")]:
            letter, method = extract_choice(text, OPTS)
            self.assertEqual(letter, want, text)
            self.assertEqual(method, "letter_marker", text)

    def test_marker_does_not_misfire_on_ordinary_words(self) -> None:
        # "answer is correct" — the c of "correct" must not extract as C.
        letter, method = extract_choice(
            "The answer is correctly described in the passage", OPTS
        )
        self.assertNotEqual((letter, method), ("C", "letter_marker"))


class TestLeadingAGuard(unittest.TestCase):
    def test_article_trap_not_extracted(self) -> None:
        # "A good story..." — leading A is the English article here.
        letter, method = extract_choice("A good story about war.", OPTS)
        self.assertNotEqual(letter, "A")
        self.assertEqual(method, "unparseable")

    def test_bare_a_plus_words_falls_through_by_design(self) -> None:
        """Leading-A fall-through pin.

        This fall-through is the INTENDED cost of the leading-A
        article-collision guard. A bare leading "A" followed by words
        is indistinguishable from the English article without a marker
        word, so rule 1 skips it BY DESIGN; such answers resolve via
        option-text/token-F1 when content helps, else land in
        unparseable and surface in the analyser's per-system
        unparseable rate. Do NOT "fix" this test by adding a bare ^A
        pattern — that reintroduces the article collision
        ("A good story about..." -> false A).
        """
        # Options constructed so no later rule can match (OPTS shares no
        # content with the prediction).
        letter, method = extract_choice("A is correct because...", OPTS)
        self.assertIsNone(letter)
        self.assertEqual(method, "unparseable")

    def test_companion_marker_recovers_a(self) -> None:
        # Same sentence with a marker word: the recoverable path.
        letter, method = extract_choice("Answer: A is correct because...", OPTS)
        self.assertEqual(letter, "A")
        self.assertEqual(method, "letter_marker")


class TestAbstention(unittest.TestCase):
    def test_abstention_phrases(self) -> None:
        for text in ["I don't know",
                     "This cannot be answered from the evidence",
                     "No answer available."]:
            letter, method = extract_choice(text, OPTS)
            self.assertIsNone(letter, text)
            self.assertEqual(method, "abstention", text)

    def test_explicit_letter_beats_abstention_phrasing(self) -> None:
        letter, method = extract_choice(
            "I don't know for sure, but the answer is B", OPTS
        )
        self.assertEqual(letter, "B")
        self.assertEqual(method, "letter_marker")


class TestTextMatch(unittest.TestCase):
    def test_exact_normalised_restatement(self) -> None:
        letter, method = extract_choice("the cat slept on the warm windowsill.", OPTS)
        self.assertEqual(letter, "B")
        self.assertEqual(method, "text_exact")

    def test_unique_containment(self) -> None:
        letter, method = extract_choice(
            "Based on the passage, the bird flew south before the winter "
            "storm, which is what the narrator records.",
            OPTS,
        )
        self.assertEqual(letter, "C")
        self.assertEqual(method, "text_containment")

    def test_multi_containment_falls_through(self) -> None:
        # Prediction contained in BOTH options -> ambiguous -> must NOT
        # resolve via text_containment (falls to token-F1 / unparseable).
        opts = (
            "the red apple fell down",
            "the red apple fell up",
            "a completely different option",
            "another unrelated option",
        )
        letter, method = extract_choice("red apple fell", opts)
        self.assertNotEqual(method, "text_containment")


class TestTokenF1Fallback(unittest.TestCase):
    def test_strong_paraphrase_picked(self) -> None:
        opts = (
            "the dog barked at strangers all night",
            "quantum entanglement of photon pairs",
            "medieval trade routes across the alps",
            "deep sea bioluminescent organisms",
        )
        letter, method = extract_choice(
            "it was that the dog barked at strangers during the night", opts
        )
        self.assertEqual(letter, "A")
        self.assertEqual(method, "token_f1")

    def test_near_tie_falls_through_to_unparseable(self) -> None:
        # REQUIRED: best-vs-runner-up F1 gap below RULE3_MIN_MARGIN must
        # not guess. "red apple fell" scores identically against both
        # near-twin options (margin 0 < RULE3_MIN_MARGIN).
        opts = (
            "the red apple fell down",
            "the red apple fell up",
            "a completely different option",
            "another unrelated option",
        )
        self.assertGreater(RULE3_MIN_MARGIN, 0.0)
        letter, method = extract_choice("red apple fell", opts)
        self.assertIsNone(letter)
        self.assertEqual(method, "unparseable")

    def test_below_threshold_unparseable(self) -> None:
        letter, method = extract_choice(
            "completely unrelated rambling text with nothing useful", OPTS
        )
        self.assertIsNone(letter)
        self.assertEqual(method, "unparseable")


class TestScoring(unittest.TestCase):
    def test_correct_letter_scores_one(self) -> None:
        value, method, md = score_multiple_choice("B", OPTS, gold_label=2)
        self.assertEqual(value, 1.0)
        self.assertEqual(md["predicted_letter"], "B")
        self.assertEqual(md["gold_letter"], "B")
        self.assertFalse(md["abstained"])
        self.assertFalse(md["unparseable"])

    def test_wrong_letter_scores_zero(self) -> None:
        value, _, md = score_multiple_choice("B", OPTS, gold_label=3)
        self.assertEqual(value, 0.0)
        self.assertEqual(md["gold_letter"], "C")

    def test_gold_label_is_one_indexed(self) -> None:
        value, _, md = score_multiple_choice("A", OPTS, gold_label=1)
        self.assertEqual(value, 1.0)
        self.assertEqual(md["gold_letter"], "A")
        with self.assertRaises(ValueError):
            score_multiple_choice("A", OPTS, gold_label=0)
        with self.assertRaises(ValueError):
            score_multiple_choice("A", OPTS, gold_label=5)

    def test_abstention_scores_zero_but_flagged(self) -> None:
        value, method, md = score_multiple_choice("I don't know", OPTS, gold_label=1)
        self.assertEqual(value, 0.0)
        self.assertEqual(method, "abstention")
        self.assertTrue(md["abstained"])
        self.assertFalse(md["unparseable"])

    def test_unparseable_scores_zero_but_flagged(self) -> None:
        value, method, md = score_multiple_choice(
            "A is correct because...", OPTS, gold_label=1
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(method, "unparseable")
        self.assertTrue(md["unparseable"])
        self.assertFalse(md["abstained"])


if __name__ == "__main__":
    unittest.main()
