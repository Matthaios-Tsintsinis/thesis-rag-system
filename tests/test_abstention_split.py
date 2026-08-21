"""The answer score split by abstention.

WHY THIS EXISTS. On a free-form benchmark a refusal scores 0.0 against
every reference BY CONSTRUCTION, so a low mean answer score has two
completely different readings — the system answered badly, or it declined
often — and the micro-mean cannot tell them apart. Cell 2
(M4/narrativeqa, mean 0.0692 over 1,230 answerable rows) is the worked
example that motivated it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.analyse import _aggregate, _iter_records


def _row(qid, value, *, predicted, abstained=None,
         system="M4", benchmark="narrativeqa"):
    md = {} if abstained is None else {"abstained": abstained}
    return {
        "system_id": system, "benchmark": benchmark, "split": "validation",
        "query_id": qid, "predicted_answer": predicted,
        "retrieval": {"skipped": True},
        "answer": {"value": value, "method": "token_f1", "metadata": md},
    }


def _write(rows):
    d = Path(tempfile.mkdtemp())
    f = d / "cell.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return f


class TestTheSplit(unittest.TestCase):
    def setUp(self):
        self.f = _write([
            _row("a", 0.0, predicted="No answer available.", abstained=True),
            _row("b", 0.0, predicted="I don't know.", abstained=True),
            _row("c", 0.4, predicted="Micky felt outboxed.", abstained=False),
            _row("d", 0.2, predicted="He lost on points.", abstained=False),
        ])
        self.s = _aggregate(_iter_records([self.f]))["systems"]

    def _sys(self):
        (only,) = self.s.values()
        return only

    def test_both_sides_carry_n_AND_mean(self):
        """A mean without its n is the same defect as a guard without its
        comparison."""
        st = self._sys()
        self.assertEqual(st["n_abstained"], 2)
        self.assertEqual(st["n_answered"], 2)
        self.assertAlmostEqual(st["ans_score_abstained_mean"], 0.0)
        self.assertAlmostEqual(st["ans_score_answered_mean"], 0.3)

    def test_the_split_partitions_the_rows(self):
        st = self._sys()
        self.assertEqual(st["n_abstained"] + st["n_answered"], st["n_queries"])

    def test_the_overall_mean_hides_what_the_split_shows(self):
        """0.15 overall against 0.30 among answered rows — the whole
        point of reporting both."""
        st = self._sys()
        self.assertAlmostEqual(st["ans_score_mean"], 0.15)
        self.assertAlmostEqual(st["ans_score_answered_mean"], 0.30)
        self.assertNotAlmostEqual(st["ans_score_mean"],
                                  st["ans_score_answered_mean"])


class TestTheRecordedFlagWins(unittest.TestCase):
    """`answer.metadata.abstained` is what the scorer wrote AT RUN TIME.

    Recomputing it at analysis time would silently re-classify banked
    rows if the detector ever moved — drift that is invisible in a mean.
    This drives the distinction with a row whose text the detector would
    NOT flag but whose recorded flag says it abstained.
    """

    def test_the_recorded_flag_is_preferred_over_recomputation(self):
        f = _write([
            _row("x", 0.0, predicted="Paris is the capital.", abstained=True),
            _row("y", 0.5, predicted="Berlin.", abstained=False),
        ])
        (st,) = _aggregate(_iter_records([f]))["systems"].values()
        self.assertEqual(st["n_abstained"], 1)
        self.assertEqual(st["abstention_rate"], 0.5)

    def test_rows_predating_the_field_fall_back_to_the_detector(self):
        f = _write([
            _row("x", 0.0, predicted="No answer available."),
            _row("y", 0.5, predicted="Berlin."),
        ])
        (st,) = _aggregate(_iter_records([f]))["systems"].values()
        self.assertEqual(st["n_abstained"], 1)


def _null_row(qid, value, *, abstained, system="M1",
              benchmark="multihop_rag"):
    """A null query: the loader marked it unanswerable, so P1's contract
    records method == 'unanswerable_rule' and a CORRECT refusal scores
    1.0 on a refusal-correctness scale, not a token-F1 one."""
    return {
        "system_id": system, "benchmark": benchmark, "split": "validation",
        "query_id": qid, "predicted_answer": "x",
        "retrieval": {"skipped": True},
        "answer": {"value": value, "method": "unanswerable_rule",
                   "metadata": {"abstained": abstained}},
    }


class TestNullRowsAreExcludedFromTheSplit(unittest.TestCase):
    """THE DEFECT THIS FORBIDS, measured on M1 x MultiHop.

    With nulls included the split INVERTED — abstained rows scored 0.303
    against answered rows at 0.018, 17x the wrong way — purely because
    301 correct null refusals landed on the abstained side. The printed
    guidance ("read mean_ans as answer QUALITY") was then false on the
    one benchmark that has nulls.
    """

    def setUp(self):
        self.f = _write([
            _row("a", 0.0, predicted="No answer available.", abstained=True,
                 system="M1", benchmark="multihop_rag"),
            _row("b", 0.0, predicted="I don't know.", abstained=True,
                 system="M1", benchmark="multihop_rag"),
            _row("c", 0.4, predicted="answered", abstained=False,
                 system="M1", benchmark="multihop_rag"),
            _row("d", 0.2, predicted="answered", abstained=False,
                 system="M1", benchmark="multihop_rag"),
            # Correct refusals on nulls — 1.0 each, and poison the split
            # if they are allowed into it.
            _null_row("n1", 1.0, abstained=True),
            _null_row("n2", 1.0, abstained=True),
            _null_row("n3", 0.0, abstained=False),
        ])
        (self.st,) = _aggregate(_iter_records([self.f]))["systems"].values()

    def test_the_split_counts_only_answerable_rows(self):
        self.assertEqual(self.st["n_answerable_rows"], 4)
        self.assertEqual(self.st["n_null_rows"], 3)
        self.assertEqual(self.st["n_abstained"] + self.st["n_answered"],
                         self.st["n_answerable_rows"])

    def test_correct_null_refusals_do_NOT_inflate_the_abstained_mean(self):
        """The regression in one assertion: 0.0, not (0+0+1+1)/4."""
        self.assertAlmostEqual(self.st["ans_score_abstained_mean"], 0.0)

    def test_the_split_is_no_longer_inverted(self):
        """With nulls in, mean_abs exceeded mean_ans. It must not."""
        self.assertLess(self.st["ans_score_abstained_mean"],
                        self.st["ans_score_answered_mean"])

    def test_null_abstentions_are_counted_but_kept_separate(self):
        self.assertEqual(self.st["n_abstained_null"], 2)

    def test_both_abstention_rates_are_reported_with_their_populations(self):
        """`abstention_rate` is over ALL rows; the answerable one is over
        answerable rows. Neither may borrow the other's denominator."""
        self.assertAlmostEqual(self.st["abstention_rate"], 4 / 7)
        self.assertAlmostEqual(self.st["abstention_rate_answerable"], 2 / 4)


class TestBenchmarksWithoutNullsAreUnaffected(unittest.TestCase):
    """NarrativeQA has no nulls, so the split is over every row and the
    numbers must not move."""

    def test_no_null_rows_means_the_split_covers_everything(self):
        f = _write([
            _row("a", 0.0, predicted="No answer available.", abstained=True),
            _row("c", 0.4, predicted="answered", abstained=False),
        ])
        (st,) = _aggregate(_iter_records([f]))["systems"].values()
        self.assertEqual(st["n_null_rows"], 0)
        self.assertEqual(st["n_answerable_rows"], 2)
        self.assertAlmostEqual(st["abstention_rate"],
                               st["abstention_rate_answerable"])


if __name__ == "__main__":
    unittest.main()
