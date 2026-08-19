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


def _row(qid, value, *, predicted, abstained=None):
    md = {} if abstained is None else {"abstained": abstained}
    return {
        "system_id": "M4", "benchmark": "narrativeqa", "split": "validation",
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


if __name__ == "__main__":
    unittest.main()
