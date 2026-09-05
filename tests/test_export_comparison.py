"""Tests for scripts/export_comparison: read_cell and the F1 | EM | R@5 export.
Runs on a synthetic bank; RECORDED_CREDITED is pinned to its banked values.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.eval.scorers.extractive import normalize_qasper_answer
from src.eval.types import EvalQuery, GoldAnswer
from scripts.export_comparison import (
    BENCHMARKS,
    LLAMA,
    QWEN,
    RECORDED_CREDITED,
    ROW_ORDER,
    SYSTEMS,
    build_comparison,
    checksum_line,
    gold_texts,
    read_cell,
    write_outputs,
)

CANON = "No answer available."


def _row(qid, value, pred, method="token_f1", abstained=False,
         rank_hit=None, exact_match=None):
    """Build one banked JSONL row with optional rank and EM metadata."""
    retrieval = ({"skipped": True} if rank_hit is None
                 else {"skipped": False, "hit_at_k": {"5": rank_hit}})
    metadata = {"abstained": abstained}
    if exact_match is not None:
        metadata["exact_match"] = exact_match
    return {"query_id": qid, "predicted_answer": pred,
            "answer": {"value": value, "method": method,
                       "metadata": metadata},
            "retrieval": retrieval}


# Banked EM per HotpotQA row (answer.metadata.exact_match), consistent
# with GOLD_QUERIES: q0 refusal vs "no" 0; q1 refusal vs "paris" 0;
# q2 "Paris." vs ("paris", "france") 1; q3 "an answer" vs "answer" 1.
# MultiHop and NarrativeQA rows carry no banked EM.
BANKED_EM = {"q0": 0.0, "q1": 0.0, "q2": 1.0, "q3": 1.0}


def _cell_rows(benchmark, system):
    """Build a cell's four answerable rows, plus a null row on MultiHop."""
    ranked = system != "M1" and benchmark != "narrativeqa"
    hp = benchmark.startswith("hotpotqa")
    def rh(v):
        return v if ranked else None
    def em(qid):
        return BANKED_EM[qid] if hp else None
    rows = [
        _row("q0", 0.5, CANON, abstained=True, rank_hit=rh(1.0),
             exact_match=em("q0")),
        _row("q1", 0.0, CANON, abstained=True, rank_hit=rh(0.0),
             exact_match=em("q1")),
        _row("q2", 0.8, "Paris.", rank_hit=rh(1.0), exact_match=em("q2")),
        _row("q3", 0.4, "an answer", rank_hit=rh(1.0),
             exact_match=em("q3")),
    ]
    if benchmark == "multihop_rag":
        rows.append(_row("qnull", 1.0, CANON, method="unanswerable_rule",
                         abstained=True, rank_hit=None))
    return rows


def _query(qid, texts):
    """Build a production EvalQuery holding GoldAnswer objects."""
    return EvalQuery(
        query_id=qid, question_text="q?", parent_scope=None,
        gold_answers=tuple(GoldAnswer(answer_type="free_form",
                                      free_form=t) for t in texts),
        gold_passage_sets=(), question_type="free_form", metadata={})


GOLD_QUERIES = (
    _query("q0", ("no",)),              # credited under token-F1; EM must be 0
    _query("q1", ("paris",)),
    _query("q2", ("paris", "france")),  # max over references
    _query("q3", ("answer",)),          # article stripped by the normaliser
)
GOLDS = {str(q.query_id): gold_texts(q) for q in GOLD_QUERIES}


def _write_cell(bank: Path, generator, benchmark, system, *,
                partial=False, wrong_generator=None, retrieval_f1=0.5):
    """Write one cell's summary and JSONL into the bank directory."""
    rows = _cell_rows(benchmark, system)
    n_null = 1 if benchmark == "multihop_rag" else 0
    ranked = system != "M1" and benchmark != "narrativeqa"
    primary = (0.5 + 0.0 + 0.8 + 0.4) / 4
    summary = {
        "system": system, "benchmark": benchmark,
        "generator": wrong_generator or generator,
        "partial_run": partial,
        "n_queries_scored": 4 + n_null, "expected_n_queries": 4 + n_null,
        "n_answerable": 4,
        "n_retrieval_scored": 4 if ranked else 0,
        "mean_retrieval_f1": retrieval_f1, "mean_answer_score": primary,
        "mean_answer_score_answerable": primary,
        "mean_answer_score_null": (1.0 if n_null else None),
        "elapsed_s": 1.0, "git_commit": "abc", "timestamp": "t",
        "environment": {"python": "3.12.13", "lockfile_hash": "x"},
        "model_revisions": {"revisions": {"generator": "sha"}},
    }
    stem = f"{benchmark}_{system}_validation"
    (bank / f"{stem}.summary.json").write_text(json.dumps(summary),
                                               encoding="utf-8")
    with (bank / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_sidecar(bank: Path, benchmark, system):
    """Write a rankings sidecar: four rank-scored rows, recall@5 0.5 each."""
    stem = f"{benchmark}_{system}_validation"
    with (bank / f"rankings.{stem}.jsonl").open("w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "query_id": f"q{i}", "n_gold": 2,
                "gold": [["a", "<w>"], ["b", "<w>"]],
                "doc_ranking": [["a", "<w>"], ["x", "<w>"]],
                "recall_at_k": {"1": 0.5, "5": 0.5, "10": 0.5},
            }) + chr(10))


def _fixture(tmp: Path):
    """Build both banks, sidecars, gold maps and the recorded table."""
    p10, p11 = tmp / "p10", tmp / "p11"
    p10.mkdir()
    p11.mkdir()
    for generator, bank in ((QWEN, p10), (LLAMA, p11)):
        for benchmark in BENCHMARKS:
            for system in SYSTEMS:
                _write_cell(bank, generator, benchmark, system)
                if system != "M1" and benchmark != "narrativeqa":
                    _write_sidecar(bank, benchmark, system)
    recorded = {(g, b, s): (1, 0.5)
                for g in (QWEN, LLAMA) for b in BENCHMARKS for s in SYSTEMS}
    gold_maps = {b: dict(GOLDS) for b in BENCHMARKS}
    return p10, p11, gold_maps, recorded


class TestReadCell(unittest.TestCase):
    """read_cell, the per-cell reader every table row starts from."""

    def test_arithmetic_from_one_cell(self):
        """Supplementary mean, abstain share and plain mean come out right."""
        with TemporaryDirectory() as td:
            p10, p11, _, rec = _fixture(Path(td))
            r = read_cell(p10, QWEN, "multihop_rag", "M1", rec)
            self.assertEqual(r["n_credited"], 1)
            # harness choice: primary minus credited mass over the same denominator (METHODS §C.1)
            # supplementary = 0.425 - 0.125
            self.assertAlmostEqual(float(r["supplementary_mean"]), 0.3, places=12)
            # 2 of 4 answerable abstain; plain mean = (0.8 + 0.4) / 2
            self.assertEqual(float(r["abstain_pct"]), 50.0)
            self.assertAlmostEqual(float(r["mean_plain"]), 0.6, places=12)
            # The null mean is read from the summary, not recomputed.
            self.assertEqual(float(r["mean_answer_score_null"]), 1.0)

    def test_absence_encoding(self):
        """Retrieval absence is labelled no_retrieval, no_gold or empty."""
        with TemporaryDirectory() as td:
            p10, p11, _, rec = _fixture(Path(td))
            _write_cell(p10, QWEN, "hotpotqa", "M2",
                        retrieval_f1=0.1234567890123456)
            m1 = read_cell(p10, QWEN, "hotpotqa", "M1", rec)
            self.assertEqual((m1["mean_retrieval_f1"], m1["retrieval_absence"]),
                             ("", "no_retrieval"))
            # For M1 on NarrativeQA the system-level absence wins.
            self.assertEqual(read_cell(p10, QWEN, "narrativeqa", "M1", rec)
                             ["retrieval_absence"], "no_retrieval")
            self.assertEqual(read_cell(p10, QWEN, "narrativeqa", "M2", rec)
                             ["retrieval_absence"], "no_gold")
            m2 = read_cell(p10, QWEN, "hotpotqa", "M2", rec)
            # Full precision survives the string round-trip.
            self.assertEqual(float(m2["mean_retrieval_f1"]), 0.1234567890123456)
            self.assertEqual(m2["retrieval_absence"], "")

    def test_refuses_missing_summary(self):
        """A cell without a summary file is refused."""
        with TemporaryDirectory() as td:
            p10, p11, _, rec = _fixture(Path(td))
            (p11 / "hotpotqa_M4_validation.summary.json").unlink()
            with self.assertRaises(SystemExit) as cm:
                read_cell(p11, LLAMA, "hotpotqa", "M4", rec)
            self.assertIn("missing summary", str(cm.exception))

    def test_refuses_partial_run(self):
        """A summary flagged partial_run is refused."""
        with TemporaryDirectory() as td:
            p10, p11, _, rec = _fixture(Path(td))
            _write_cell(p10, QWEN, "hotpotqa", "M2", partial=True)
            with self.assertRaises(SystemExit) as cm:
                read_cell(p10, QWEN, "hotpotqa", "M2", rec)
            self.assertIn("partial_run", str(cm.exception))

    def test_refuses_generator_mismatch(self):
        """A summary naming the other bank's generator is refused."""
        with TemporaryDirectory() as td:
            p10, p11, _, rec = _fixture(Path(td))
            _write_cell(p10, QWEN, "hotpotqa", "M3", wrong_generator=LLAMA)
            with self.assertRaises(SystemExit) as cm:
                read_cell(p10, QWEN, "hotpotqa", "M3", rec)
            self.assertIn("generator", str(cm.exception))

    def test_refuses_credited_disagreement(self):
        """A credited count off the recorded table is refused."""
        with TemporaryDirectory() as td:
            p10, p11, _, rec = _fixture(Path(td))
            rec[(QWEN, "multihop_rag", "M1")] = (2, 1.0)  # wrong on purpose
            with self.assertRaises(SystemExit) as cm:
                read_cell(p10, QWEN, "multihop_rag", "M1", rec)
            self.assertIn("recorded battery", str(cm.exception))

    def test_the_table_builder_reads_every_cell_through_it(self):
        """A refusal inside read_cell stops build_comparison too."""
        with TemporaryDirectory() as td:
            p10, p11, gold_maps, rec = _fixture(Path(td))
            _write_cell(p11, LLAMA, "narrativeqa", "M4", partial=True)
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gold_maps, recorded=rec)
            self.assertIn("partial_run", str(cm.exception))


class TestRecordedBattery(unittest.TestCase):
    """RECORDED_CREDITED is a commitment: 32 cells with the banked values."""

    def test_shape_and_guard_zeros(self):
        """The table has 32 cells and the HotpotQA family is all zero."""
        self.assertEqual(len(RECORDED_CREDITED), 32)
        for (g, b, s), (n, mass) in RECORDED_CREDITED.items():
            if b in ("hotpotqa", "hotpotqa_pooled"):
                self.assertEqual((n, mass), (0, 0.0), (g, b, s))

    def test_spot_values_from_the_living_record(self):
        """Four spot cells hold their banked (n, mass) values."""
        self.assertEqual(RECORDED_CREDITED[(QWEN, "multihop_rag", "M4")],
                         (504, 252.0))
        self.assertEqual(RECORDED_CREDITED[(LLAMA, "multihop_rag", "M1")],
                         (558, 279.0))
        self.assertEqual(RECORDED_CREDITED[(QWEN, "narrativeqa", "M4")],
                         (6, 1.36))
        self.assertEqual(RECORDED_CREDITED[(LLAMA, "narrativeqa", "M3")],
                         (6, 1.44))


class TestComparison(unittest.TestCase):
    """build_comparison, its gates, and the written outputs."""

    def test_em_immunity_of_the_credited_refusal(self):
        """The refusal string never equals a "no" gold under EM."""
        # The production normaliser sees all three gold surface forms
        # that pay token-F1 0.5 on MultiHop.
        for gold in ("no", "No", "no."):
            self.assertNotEqual(normalize_qasper_answer(CANON),
                                normalize_qasper_answer(gold), gold)
        # The scorer's normaliser is the same function object, not a copy.
        from src.eval.scorers.extractive import (
            normalize_qasper_answer as frozen)
        self.assertIs(normalize_qasper_answer, frozen)

    def test_gold_texts_extracts_strings_from_real_gold_objects(self):
        """gold_texts returns plain strings and drops empty free_form."""
        q = _query("qx", ("alpha", "beta"))
        out = gold_texts(q)
        self.assertEqual(out, ("alpha", "beta"))
        for t in out:
            self.assertIsInstance(t, str)
        # An empty free_form is filtered out, never normalised.
        empty = EvalQuery(
            query_id="qe", question_text="q?", parent_scope=None,
            gold_answers=(GoldAnswer(answer_type="free_form", free_form=""),),
            gold_passage_sets=(), question_type="free_form", metadata={})
        self.assertEqual(gold_texts(empty), ())

    def test_happy_path(self):
        """A clean bank yields 32 ordered rows with the expected EM and R@5."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            rows = build_comparison(p10, p11, gm, recorded=rec)
            self.assertEqual(len(rows), 32)
            # Row order within each (generator, benchmark) group.
            self.assertEqual([r["system"] for r in rows[:4]],
                             [s for _, s in ROW_ORDER])
            r0 = rows[0]  # Qwen / multihop / M1
            # official: hotpot_evaluate_v1.py::exact_match_score @ 36358534
            # EM per row 0, 0, 1, 1 -> 0.5; the null row is excluded
            # (n_em 4, not 5).
            self.assertEqual(float(r0["em"]), 0.5)
            self.assertEqual(r0["n_em_population"], 4)
            self.assertEqual(r0["hit_at_5"], "")       # M1: no retrieval
            by = {(r["generator"], r["benchmark"], r["system"]): r
                  for r in rows}
            m2 = by[(QWEN, "hotpotqa", "M2")]
            self.assertEqual(float(m2["hit_at_5"]), 0.75)
            self.assertEqual(m2["n_rank_population"], 4)
            self.assertEqual(float(m2["recall_at_5"]), 0.5)
            self.assertEqual(float(m2["recall_at_1"]), 0.5)
            self.assertEqual(by[(QWEN, "narrativeqa", "M2")]["hit_at_5"], "")
            self.assertEqual(by[(QWEN, "narrativeqa", "M2")]["recall_at_5"], "")

    def test_checksum_counts(self):
        """The checksum line counts rows and populated metric columns."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            rows = build_comparison(p10, p11, gm, recorded=rec)
            line = checksum_line(rows)
            self.assertIn("rows=32", line)
            # 3 ranked systems x 3 ranked benchmarks x 2 generators.
            self.assertIn("hit_at_5:18", line)
            self.assertIn("recall_at_5:18", line)
            self.assertIn("recall_at_1:18", line)
            self.assertIn("em:32", line)

    def test_md_output(self):
        """The Markdown output has eight tables, the header and checksum."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            rows = build_comparison(p10, p11, gm, recorded=rec)
            _, md_path = write_outputs(rows, Path(td) / "out")
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("POST-HOC", md)
            self.assertEqual(md.count("## "), 8)      # eight tables
            self.assertIn(checksum_line(rows), md)
            self.assertIn("bm25-hybrid", md)
            self.assertIn("| system | F1 | EM | R@5 |", md)
            self.assertNotIn("| system | F1 | EM | hit@5 |", md)

    def test_refuses_missing_sidecar(self):
        """A ranked cell without its rankings sidecar is refused."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            (p11 / "rankings.hotpotqa_M4_validation.jsonl").unlink()
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("sidecar", str(cm.exception))

    def test_refuses_recall_bound_violation(self):
        """A sidecar whose recall@5 exceeds hit@5 is refused."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            stem = p10 / "rankings.hotpotqa_M2_validation.jsonl"
            rows_ = [json.loads(l) for l in
                     stem.read_text(encoding="utf-8").strip().splitlines()]
            for r in rows_:
                r["recall_at_k"]["5"] = 1.0    # mean 1.0 > hit@5 0.75
            stem.write_text(
                chr(10).join(json.dumps(r) for r in rows_) + chr(10),
                encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("impossible", str(cm.exception))

    def test_refuses_missing_gold(self):
        """A scored row with no gold in the map is refused."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            del gm["hotpotqa"]["q2"]
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("no gold", str(cm.exception))

    def test_refuses_self_inconsistent_summary(self):
        """n_retrieval_scored disagreeing with the rows is refused."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            stem = p10 / "hotpotqa_M3_validation.summary.json"
            s = json.loads(stem.read_text(encoding="utf-8"))
            s["n_retrieval_scored"] = 9
            stem.write_text(json.dumps(s), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("self-inconsistent", str(cm.exception))

    def test_old_schema_without_n_retrieval_scored_passes(self):
        """A summary lacking n_retrieval_scored derives the count from rows."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            stem = p10 / "multihop_rag_M4_validation.summary.json"
            s = json.loads(stem.read_text(encoding="utf-8"))
            del s["n_retrieval_scored"]
            stem.write_text(json.dumps(s), encoding="utf-8")
            rows = build_comparison(p10, p11, gm, recorded=rec)
            by = {(r["generator"], r["benchmark"], r["system"]): r
                  for r in rows}
            self.assertEqual(
                by[(QWEN, "multihop_rag", "M4")]["n_rank_population"], 4)

    def test_hotpotqa_em_gate_refuses_a_disagreeing_banked_row(self):
        """A banked HotpotQA EM differing from the recomputed EM is refused."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            jpath = p10 / "hotpotqa_M2_validation.jsonl"
            rows = [json.loads(l) for l in
                    jpath.read_text(encoding="utf-8").strip().splitlines()]
            rows[2]["answer"]["metadata"]["exact_match"] = 0.0  # q2 is 1
            jpath.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("recomputed EM", str(cm.exception))

    def test_hotpotqa_em_gate_refuses_a_row_without_the_field(self):
        """A HotpotQA row with no banked exact_match field is refused."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            jpath = p11 / "hotpotqa_pooled_M4_validation.jsonl"
            rows = [json.loads(l) for l in
                    jpath.read_text(encoding="utf-8").strip().splitlines()]
            del rows[1]["answer"]["metadata"]["exact_match"]
            jpath.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("no answer.metadata.exact_match", str(cm.exception))

    def test_non_hotpotqa_rows_carry_no_banked_em_and_pass(self):
        """MultiHop and NarrativeQA rows need no exact_match field."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            r = json.loads((p10 / "multihop_rag_M2_validation.jsonl")
                           .read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("exact_match", r["answer"]["metadata"])
            self.assertEqual(len(build_comparison(p10, p11, gm,
                                                  recorded=rec)), 32)

    def test_refuses_row_side_rank_shortfall(self):
        """Fewer ranked rows than the summary claims is refused."""
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            jpath = p10 / "hotpotqa_M2_validation.jsonl"
            lines = jpath.read_text(encoding="utf-8").strip().splitlines()
            rows = [json.loads(l) for l in lines]
            rows[0]["retrieval"] = {"skipped": True}   # lose one ranked row
            jpath.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("rank population", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
