"""P4: the answer normaliser matches the published evaluators.

The composition was INVERTED before 2026-08-14 — lower, strip articles,
drop punctuation — while its docstring claimed a verbatim port and
asserted that the ordering mattered. It does. Both official evaluators
compose `white_space_fix(remove_articles(remove_punc(lower(s))))`, and
the divergence is observable on any article adjacent to punctuation.

The reference below is transcribed from the published sources. It moves
to match them, never the reverse.
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


def official_normalize(s: str) -> str:
    """Verbatim from `hotpot_evaluate_v1.py` and SQuAD 2.0
    `evaluate-v2.0.py`, which agree character for character."""
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
        """Official removes the hyphen first, so no word boundary is left
        for the article regex. The old order matched "the" against the
        hyphen and returned "cat"."""
        self.assertEqual(normalize_qasper_answer("the-cat"), "thecat")
        self.assertEqual(normalize_qasper_answer("the-cat"),
                         official_normalize("the-cat"))

    def test_agreement_on_a_battery_of_ascii_cases(self):
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
        """Ten cases can agree by luck. This cannot."""
        import random

        rng = random.Random(20260814)
        alphabet = string.ascii_letters + string.digits + string.punctuation + "   "
        for _ in range(3000):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
            self.assertEqual(normalize_qasper_answer(s), official_normalize(s),
                             repr(s))


class TestUnicodeFolding(unittest.TestCase):
    """Ours, beyond the official pipeline, applied uniformly to all three
    live benchmarks and inert on ASCII."""

    def test_curly_and_straight_apostrophes_tokenise_identically(self):
        self.assertEqual(normalize_qasper_answer("don’t"),
                         normalize_qasper_answer("don't"))
        self.assertEqual(token_f1("don’t know", "don't know"), 1.0)

    def test_curly_quotes_and_dashes_fold(self):
        self.assertEqual(normalize_qasper_answer("“Paris”"),
                         normalize_qasper_answer('"Paris"'))
        self.assertEqual(normalize_qasper_answer("2020–2021"),
                         normalize_qasper_answer("2020-2021"))

    def test_nfkc_alone_would_not_have_sufficed(self):
        """Recorded because it is counter-intuitive: U+2019 is not a
        compatibility character, so NFKC leaves it, and it is not in
        string.punctuation, so the official table leaves it too."""
        import unicodedata

        self.assertEqual(unicodedata.normalize("NFKC", "’"), "’")
        self.assertNotIn("’", string.punctuation)

    def test_non_ascii_symbols_survive_as_the_official_pipeline_leaves_them(self):
        self.assertIn("€", normalize_qasper_answer("€5"))


class TestBothEmptyDivergence(unittest.TestCase):
    """THE TWO OFFICIAL REFERENCES DISAGREE, and this records that the
    conflict was known and neutralised upstream rather than resolved by
    preference."""

    def test_squad_rule_is_adopted(self):
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("...", "!!!"), 1.0)

    def test_hotpotqa_reference_would_return_zero_here(self):
        """hotpot_evaluate_v1.f1_score has no no-answer branch: two empty
        token lists fall through to `num_same == 0` and it returns
        ZERO_METRIC. Transcribed and asserted so the divergence is on the
        record, not in a comment."""
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
        """Gold can never normalise to empty (the loader asserts it), and
        a pred-only-empty case is where every implementation agrees."""
        with self.assertRaises(ValueError):
            assert_gold_not_empty("q1", "...", benchmark="test")
        self.assertEqual(token_f1("", "Tim Cook"), 0.0)


class TestGoldAssertion(unittest.TestCase):
    def test_it_fires_on_a_gold_that_normalises_to_empty(self):
        for bad in ("", "   ", "...", "the", "a an the"):
            with self.assertRaises(ValueError, msg=bad):
                assert_gold_not_empty("q1", bad, benchmark="multihop_rag")

    def test_the_message_names_the_offending_query(self):
        with self.assertRaises(ValueError) as ctx:
            assert_gold_not_empty("multihop_000042", "", benchmark="multihop_rag")
        self.assertIn("multihop_000042", str(ctx.exception))

    def test_it_passes_a_real_gold(self):
        assert_gold_not_empty("q1", "Sam Bankman-Fried", benchmark="multihop_rag")


if __name__ == "__main__":
    unittest.main()
