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
        with self.assertRaisesRegex(SystemExit, r"\['ours_v4'\]\['multihop_rag'\] = \(1, 0\.5\)"):
            ro.recorded_battery("ours_v4", "multihop_rag", rows, recorded={})
        self.assertEqual(ro.recorded_battery("ours", "multihop_rag", rows), (418, 209.0))


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
            golds, docs = golds_of(TwoHotpotUnits())
            row = ro.ours_row(ours_dir, "hotpotqa", golds, docs, rows)
        finally:
            case.tearDown()
        self.assertEqual((row["system"], row["label"], row["n_rank_population"]),
                         ("OURS", "ours", 2))
        self.assertEqual(row["em"], repr(0.5))
        expected = sum(ro.recall_at(r["metadata"]["ours_doc_ranking"],
                                    docs[r["query_id"]], 5) for r in rows) / 2
        self.assertEqual(row["recall_at_5"], repr(round(expected, 10)))


if __name__ == "__main__":
    unittest.main()
