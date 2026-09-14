"""Pins the answer normaliser to the official HotpotQA/SQuAD composition
and its Unicode extension.
"""

from __future__ import annotations

import re
import string
import unittest

from src.eval.scorers.extractive import (
    assert_gold_not_empty,
    normalize_qasper_answer,
    token_f1,
)


# official: hotpot_evaluate_v1.py::normalize_answer @ 36358534
def official_normalize(s: str) -> str:
    """Transcribes the official normaliser; SQuAD 2.0 uses the same one."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


class TestMatchesTheOfficialComposition(unittest.TestCase):
    def test_the_case_that_exposed_the_inverted_order(self):
        """Punctuation goes before articles, so "the-cat" stays "thecat"."""
        self.assertEqual(normalize_qasper_answer("the-cat"), "thecat")
        self.assertEqual(normalize_qasper_answer("the-cat"),
                         official_normalize("the-cat"))

    def test_agreement_on_a_battery_of_ascii_cases(self):
        """Ours equals the official normaliser on hand-picked ASCII cases."""
        cases = [
            "The Answer, is: 42.",
            "Sam Bankman-Fried",
            "the cat sat on a mat",
            "U.S.-based",
            "an apple, the pear",
            "The-Boy-Who-Lived",
            "THE END.",
            "a/an the",
            "",
            "   ",
        ]
        for c in cases:
            self.assertEqual(normalize_qasper_answer(c), official_normalize(c), c)

    def test_agreement_under_ascii_fuzz(self):
        """Ours equals the official normaliser on 3000 random ASCII strings."""
        import random

        rng = random.Random(20260814)
        alphabet = string.ascii_letters + string.digits + string.punctuation + "   "
        for _ in range(3000):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
            self.assertEqual(normalize_qasper_answer(s), official_normalize(s),
                             repr(s))


# harness extension (inert on ASCII): see METHODS §C.11
class TestUnicodeFolding(unittest.TestCase):
    """The Unicode fold runs on every benchmark and changes no ASCII text."""

    def test_curly_and_straight_apostrophes_tokenise_identically(self):
        """A curly apostrophe folds to the straight one before tokenising."""
        self.assertEqual(normalize_qasper_answer("don’t"),
                         normalize_qasper_answer("don't"))
        self.assertEqual(token_f1("don’t know", "don't know"), 1.0)

    def test_curly_quotes_and_dashes_fold(self):
        """Curly quotes and en dashes fold to their ASCII forms."""
        self.assertEqual(normalize_qasper_answer("“Paris”"),
                         normalize_qasper_answer('"Paris"'))
        self.assertEqual(normalize_qasper_answer("2020–2021"),
                         normalize_qasper_answer("2020-2021"))

    def test_nfkc_alone_would_not_have_sufficed(self):
        """U+2019 survives NFKC and sits outside string.punctuation."""
        import unicodedata

        self.assertEqual(unicodedata.normalize("NFKC", "’"), "’")
        self.assertNotIn("’", string.punctuation)

    def test_non_ascii_symbols_survive_as_the_official_pipeline_leaves_them(self):
        """Non-ASCII symbols outside string.punctuation are kept."""
        self.assertIn("€", normalize_qasper_answer("€5"))


# SQuAD 2.0 evaluate-v2.0.py rule; unreachable, loaders refuse empty gold
class TestBothEmptyDivergence(unittest.TestCase):
    """The two official evaluators disagree on both-empty; we take SQuAD."""

    def test_squad_rule_is_adopted(self):
        """Two empty normalised strings score 1.0."""
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("...", "!!!"), 1.0)

    def test_hotpotqa_reference_would_return_zero_here(self):
        """The HotpotQA scorer returns 0.0 on two empty token lists."""
        # official: hotpot_evaluate_v1.py::f1_score @ 36358534
        def hotpot_f1(prediction, ground_truth):
            np_, ng = official_normalize(prediction), official_normalize(ground_truth)
            if np_ in ["yes", "no", "noanswer"] and np_ != ng:
                return 0.0
            if ng in ["yes", "no", "noanswer"] and np_ != ng:
                return 0.0
            pt, gt = np_.split(), ng.split()
            from collections import Counter
            num_same = sum((Counter(pt) & Counter(gt)).values())
            if num_same == 0:
                return 0.0
            precision = num_same / len(pt)
            recall = num_same / len(gt)
            return 2 * precision * recall / (precision + recall)

        self.assertEqual(hotpot_f1("", ""), 0.0)
        self.assertEqual(token_f1("", ""), 1.0)

    def test_the_branch_is_unreachable_in_this_pipeline(self):
        """The loader refuses empty gold; an empty pred alone scores 0."""
        with self.assertRaises(ValueError):
            assert_gold_not_empty("q1", "...", benchmark="test")
        self.assertEqual(token_f1("", "Tim Cook"), 0.0)


class TestGoldAssertion(unittest.TestCase):
    def test_it_fires_on_a_gold_that_normalises_to_empty(self):
        """A gold that normalises to empty raises ValueError."""
        for bad in ("", "   ", "...", "the", "a an the"):
            with self.assertRaises(ValueError, msg=bad):
                assert_gold_not_empty("q1", bad, benchmark="multihop_rag")

    def test_the_message_names_the_offending_query(self):
        """The error message carries the query id."""
        with self.assertRaises(ValueError) as ctx:
            assert_gold_not_empty("multihop_000042", "", benchmark="multihop_rag")
        self.assertIn("multihop_000042", str(ctx.exception))

    def test_it_passes_a_real_gold(self):
        """A non-empty gold passes silently."""
        assert_gold_not_empty("q1", "Sam Bankman-Fried", benchmark="multihop_rag")


if __name__ == "__main__":
    unittest.main()
