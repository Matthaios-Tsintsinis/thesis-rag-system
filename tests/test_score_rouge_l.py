"""The post-hoc ROUGE-L join.

No `rouge` package and no dataset download: the scorer is injected and
the gold map is passed in, so what is tested is the JOIN and the
reporting shape rather than the metric arithmetic (which
`test_rouge_l_config.py` pins against the AllenNLP source).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.score_rouge_l import score_file


def _row(qid, predicted, abstained):
    return {
        "system_id": "M4", "benchmark": "narrativeqa", "query_id": qid,
        "predicted_answer": predicted,
        "answer": {"value": 0.0, "metadata": {"abstained": abstained}},
    }


def _write(rows):
    f = Path(tempfile.mkdtemp()) / "cell.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return f


def _fake_scorer(scores_by_pred):
    def scorer(prediction, references):
        return scores_by_pred[prediction]
    return scorer


class TestTheJoinAndTheSplit(unittest.TestCase):
    def setUp(self):
        self.path = _write([
            _row("q0", "No answer available.", True),
            _row("q1", "He was outboxed.", False),
            _row("q2", "He lost on points.", False),
        ])
        self.gold = {"q0": ("a",), "q1": ("b",), "q2": ("c",)}
        self.scorer = _fake_scorer({
            "No answer available.": {"f": 0.0, "p": 0.0, "r": 0.0},
            "He was outboxed.": {"f": 0.6, "p": 0.5, "r": 0.7},
            "He lost on points.": {"f": 0.4, "p": 0.3, "r": 0.5},
        })

    def test_overall_means_cover_every_row(self):
        out = score_file(self.path, self.gold, scorer=self.scorer)
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["rouge_l_f_mean"], (0.0 + 0.6 + 0.4) / 3)

    def test_the_abstention_split_mirrors_analyse(self):
        """A refusal scores ~0 on ROUGE-L for the same structural reason
        it scores 0.0 on token-F1, so both columns need the same split."""
        out = score_file(self.path, self.gold, scorer=self.scorer)
        self.assertEqual(out["n_abstained"], 1)
        self.assertEqual(out["n_answered"], 2)
        self.assertAlmostEqual(out["rouge_l_f_abstained_mean"], 0.0)
        self.assertAlmostEqual(out["rouge_l_f_answered_mean"], 0.5)

    def test_references_reach_the_scorer(self):
        seen = {}

        def scorer(prediction, references):
            seen[prediction] = references
            return {"f": 0.1, "p": 0.1, "r": 0.1}

        score_file(self.path, self.gold, scorer=scorer)
        self.assertEqual(seen["He was outboxed."], ("b",))


class TestAPartialJoinIsRefused(unittest.TestCase):
    def test_a_missing_query_id_aborts_rather_than_scoring_a_subset(self):
        """Scoring the remainder would report a mean over an unnamed
        subset — a mean without its population."""
        path = _write([_row("q0", "x", False), _row("ghost", "y", False)])
        with self.assertRaises(SystemExit) as ctx:
            score_file(path, {"q0": ("a",)},
                       scorer=lambda p, r: {"f": 1.0, "p": 1.0, "r": 1.0})
        self.assertIn("ghost", str(ctx.exception))

    def test_a_complete_join_does_not_abort(self):
        path = _write([_row("q0", "x", False)])
        out = score_file(path, {"q0": ("a",)},
                         scorer=lambda p, r: {"f": 1.0, "p": 1.0, "r": 1.0})
        self.assertEqual(out["n"], 1)


if __name__ == "__main__":
    unittest.main()
