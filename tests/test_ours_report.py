"""Pins ours' report: the Phase-2 target formula (and that it reproduces the
published Llama targets before a new bar is trusted), Holm, the paired
tests and the directional verdict, and the gates on ours' cells."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import report_ours as ro


def _row(qid, answer, retr=None, map10=None, hit=None, mrr=0.0, meta=None):
    """A banked row reduced to what the report reads."""
    retrieval = ({"skipped": True} if retr is None else
                 {"skipped": False, "f1": retr, "map_at_k": {"10": map10 or 0.0},
                  "hit_at_k": hit or {}, "mrr": mrr})
    return {"query_id": qid, "answer": {"value": answer, "method": "token_f1"},
            "retrieval": retrieval, "metadata": meta or {}}


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class TestTarget(unittest.TestCase):
    """2.8 x median SD of the paired differences / sqrt(median n)."""

    def test_formula(self):
        d1 = np.array([0.0, 1.0, 0.0, 1.0])
        d2 = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
        sds = [float(np.std(d1, ddof=1)), float(np.std(d2, ddof=1))]
        expected = 2.8 * float(np.median(sds)) / np.sqrt(np.median([4, 6]))
        self.assertAlmostEqual(ro.target([d1, d2]), expected)

    def test_published_targets_must_reproduce(self):
        """A bank whose families give other targets refuses the new bar."""
        with tempfile.TemporaryDirectory() as td:
            bank = Path(td)
            rng = np.random.default_rng(0)
            for s in ("M1", "M2", "M3", "M4"):
                _write(bank / f"tiny_{s}_validation.jsonl",
                       [_row(f"q{i}", float(v)) for i, v in
                        enumerate(rng.random(50))])
            got = ro.family_target(bank, "tiny", "answer")
            ro.check_published(bank, {("tiny", "answer"): round(got, 4)})
            with self.assertRaises(SystemExit):
                ro.check_published(bank, {("tiny", "answer"): round(got, 4) + 0.001})


class TestTests(unittest.TestCase):
    """The grid's paired permutation, bootstrap and Holm rules."""

    def test_holm(self):
        self.assertEqual([round(p, 10) for p in ro.holm([0.01, 0.04, 0.03])],
                         [0.03, 0.06, 0.06])

    def test_permutation_detects_a_clear_shift_and_not_zero(self):
        rng = np.random.default_rng(1)
        self.assertLess(ro.permutation_p(np.full(200, 0.1), rng), 0.001)
        self.assertEqual(ro.permutation_p(np.zeros(50), rng), 1.0)

    def test_declared_direction_is_part_of_the_verdict(self):
        """Ours better: SUPPORTED. Ours worse by as much: NOT SUPPORTED."""
        base = {f"q{i}": 0.5 for i in range(300)}
        better = {k: 0.7 for k in base}
        worse = {k: 0.3 for k in base}
        rows = ro.run_family([("up", base, better, None), ("down", base, worse, None)],
                             np.random.default_rng(2), directional=True)
        verdicts = {r["name"]: r["verdict"] for r in rows}
        self.assertEqual(verdicts, {"up": "SUPPORTED", "down": "NOT SUPPORTED"})

    def test_a_delta_below_its_bar_is_undetectable_not_a_loss(self):
        base = {f"q{i}": 0.5 + 0.1 * (i % 2) for i in range(40)}
        tiny = {k: v + 0.001 for k, v in base.items()}
        rows = ro.run_family([("small", base, tiny, 0.02)],
                             np.random.default_rng(3), directional=True)
        self.assertTrue(rows[0]["below_bar"])


class TestOursCellGates(unittest.TestCase):
    """The report refuses ours' cells that are not what the plan says."""

    def _rows(self, ranking=("A", "B"), hit5=1.0, code="c1"):
        meta = {"ours_doc_ranking": list(ranking), "ours_code": code}
        return [_row("q0", 1.0, retr=0.5, hit={"1": 1.0, "5": hit5, "10": hit5},
                     mrr=1.0, meta=meta)]

    def test_ranking_must_reproduce_the_scorers_hit_and_mrr(self):
        ro.check_doc_ranking(self._rows(), {"q0": frozenset({"A"})})
        with self.assertRaises(SystemExit):
            ro.check_doc_ranking(self._rows(ranking=("B", "C")),
                                 {"q0": frozenset({"A"})})

    def test_one_code_fingerprint(self):
        ro.check_code({"x": self._rows()}, frozen=None)
        mixed = {"x": self._rows(), "y": self._rows(code="c2")}
        with self.assertRaises(SystemExit):
            ro.check_code(mixed, frozen=None)
        with self.assertRaises(SystemExit):
            ro.check_code({"x": self._rows()}, frozen="c9")

    def test_frozen_code_is_the_model_in_the_tree(self):
        """The latest round's freeze names the model as it stands; a later edit
        to ours/ fails here. A round still in development has no freeze yet."""
        from ours.model import CODE_FINGERPRINT
        latest = list(ro.FROZEN_CODE.values())[-1]
        if latest is None:
            self.skipTest("the latest round's version is in development")
        self.assertEqual(latest, CODE_FINGERPRINT)

    def test_every_round_has_its_batteries_and_round_one_stays_frozen(self):
        self.assertEqual(list(ro.RECORDED_OURS), list(ro.FROZEN_CODE))
        self.assertEqual(ro.FROZEN_CODE["ours"], "d975038e7513cf11")

    def test_only_a_frozen_rounds_bank_is_reported(self):
        self.assertEqual(ro.frozen_code_of(Path("outputs/ours")), "d975038e7513cf11")
        for bank in ("outputs/ours_dev", "outputs/p11"):
            with self.assertRaisesRegex(SystemExit, "not a scored round's bank"):
                ro.frozen_code_of(Path(bank))
        with mock.patch.dict(ro.FROZEN_CODE, {"ours_v9": None}):
            with self.assertRaisesRegex(SystemExit, "not frozen"):
                ro.frozen_code_of(Path("outputs/ours_v9"))

    def test_unrecorded_refusal_battery_refuses_and_says_what_to_record(self):
        rows = [{"query_id": "q0", "predicted_answer": "No answer available.",
                 "answer": {"value": 0.5, "method": "token_f1",
                            "metadata": {"abstained": True}}}]
        self.assertEqual(ro.credited(rows), (1, 0.5))
        with self.assertRaisesRegex(SystemExit, r"\['ours_v20'\]\['multihop_rag'\] = \(1, 0\.5\)"):
            ro.recorded_battery("ours_v20", "multihop_rag", rows, recorded={})
        self.assertEqual(ro.recorded_battery("ours", "multihop_rag", rows), (418, 209.0))

    def test_round_two_is_v20_frozen(self):
        self.assertEqual(list(ro.FROZEN_CODE), ["ours", "ours_v20"])
        self.assertEqual(ro.frozen_code_of(Path("outputs/ours_v20")), "e77ef1d48101c109")
        self.assertEqual(ro.RECORDED_OURS["ours_v20"],
                         {"hotpotqa": (0, 0.0), "hotpotqa_pooled": (0, 0.0)})


class _Scorer:
    """An exact-match scorer: a null query scores the refusal 1, as the null rule does."""

    def score_answer(self, predicted, query):
        from src.eval.types import AnswerScore

        abstained = predicted == "No answer available."
        if query == "null":
            return AnswerScore(float(abstained), ro.NULL_METHOD, metadata={"abstained": abstained})
        return AnswerScore(float(predicted == query), "token_f1", metadata={"abstained": abstained})


def _answered(qid, final, plain, value, method="token_f1"):
    row = _row(qid, value, meta={"ours_plain_answer": plain})
    row["predicted_answer"], row["answer"]["method"] = final, method
    return row


class TestPlainAnswer(unittest.TestCase):
    """Ours' final answer beside the baselines' prompt on ours' own evidence."""

    QUERIES = {"q0": "Paris", "q1": "Rome", "q2": "null"}
    GOLDS = {"q0": ["Paris"], "q1": ["Rome"], "q2": []}

    def rows(self):
        return [_answered("q0", "Paris", "Paris is the capital", 1.0),
                _answered("q1", "No answer available.", "Rome", 0.0),
                _answered("q2", "Lyon", "No answer available.", 0.0, ro.NULL_METHOD)]

    def test_both_answers_scored_by_the_cells_scorer(self):
        got = ro.plain_numbers(self.rows(), _Scorer(), self.QUERIES, self.GOLDS)
        self.assertEqual(got, {"n_answerable": 2, "n_null": 1,
                               "final_f1": 0.5, "plain_f1": 0.5,
                               "final_em": 0.5, "plain_em": 0.5,
                               "final_refused": 0.5, "plain_refused": 0.0,
                               "final_null": 0.0, "plain_null": 1.0})

    def test_the_scorer_must_give_the_rows_own_score_back(self):
        rows = self.rows()
        rows[0]["answer"]["value"] = 0.9
        with self.assertRaisesRegex(SystemExit, "rescoring the final answer"):
            ro.plain_numbers(rows, _Scorer(), self.QUERIES, self.GOLDS)

    def test_a_row_without_its_plain_answer_refuses(self):
        rows = self.rows()
        del rows[1]["metadata"]["ours_plain_answer"]
        with self.assertRaisesRegex(SystemExit, "no ours_plain_answer"):
            ro.plain_numbers(rows, _Scorer(), self.QUERIES, self.GOLDS)

    def test_no_null_question_no_null_score(self):
        got = ro.plain_numbers(self.rows()[:2], _Scorer(), self.QUERIES, self.GOLDS)
        self.assertIsNone(got["final_null"])
        self.assertIsNone(got["plain_null"])

    def test_the_markdown_shows_both(self):
        mechanics = {"multihop_rag": {
            "parse_failure_rate": 0.0, "mean_llm_calls": 7.0, "mean_aspects": 2.0,
            "low_confidence_share": 0.1, "mean_evidence_chunks": 15.0,
            "mean_evidence_tokens": 1500.0, "tree_cache_hit_share": 1.0,
            "section_detector_share": {"fixed": 1.0}}}
        plains = {"multihop_rag": ro.plain_numbers(self.rows(), _Scorer(), self.QUERIES, self.GOLDS)}
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(ro, "checksum_line", return_value="checksum"):
                ro.write_outputs([], mechanics, Path(td), plains)
                text = (Path(td) / "COMPARISON_OURS.md").read_text(encoding="utf-8")
                ro.write_outputs([], mechanics, Path(td))
                without = (Path(td) / "COMPARISON_OURS.md").read_text(encoding="utf-8")
        self.assertIn("| MHR | 2 | 0.5000 / 0.5000 | 0.5000 / 0.5000 | 50.0% / 0.0% | "
                      "0.0000 / 1.0000 (1) |", text)
        self.assertNotIn("plain answer", without)


class TestOursRowEndToEnd(unittest.TestCase):
    """A cell the runner writes for ours reads through the exporter's functions."""

    def test_ours_row_from_a_runner_cell(self):
        from scripts.dev_ours import golds_of
        from tests.test_ours_registry import TestRegistry
        from tests.test_ours_smoke import TwoHotpotUnits

        case = TestRegistry("test_registry_resolves_ours")
        case.setUp()
        try:
            ours_dir = case.dir / "ours"
            case._main(ours_dir / "hotpotqa_OURS_validation.jsonl")
            rows = ro.read_rows(ours_dir / "hotpotqa_OURS_validation.jsonl")
            loader = TwoHotpotUnits()
            golds, docs = golds_of(loader)
            row = ro.ours_row(ours_dir, "hotpotqa", golds, docs, rows)
            # The real HotpotQA scorer gives each runner row its own score back.
            plain = ro.plain_numbers(rows, loader, ro.queries_of(loader), golds)
        finally:
            case.tearDown()
        self.assertEqual((plain["n_answerable"], plain["n_null"]), (2, 0))
        self.assertAlmostEqual(plain["final_f1"],
                               sum(float(r["answer"]["value"]) for r in rows) / 2)
        self.assertEqual((row["system"], row["label"], row["n_rank_population"]),
                         ("OURS", "ours", 2))
        self.assertEqual(row["em"], repr(0.5))
        expected = sum(ro.recall_at(r["metadata"]["ours_doc_ranking"],
                                    docs[r["query_id"]], 5) for r in rows) / 2
        self.assertEqual(row["recall_at_5"], repr(round(expected, 10)))


if __name__ == "__main__":
    unittest.main()
