"""export_comparison: the post-hoc EM + hit@5 comparison exporter.

Synthetic fixture, measured expectations, injected recorded table —
same discipline as test_export_matrix. The EM-immunity case runs the
REAL frozen normaliser against the real canonical refusal string.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.eval.scorers.extractive import normalize_qasper_answer
from scripts.export_comparison import (
    ROW_ORDER,
    build_comparison,
    checksum_line,
    write_outputs,
)
from scripts.export_matrix import BENCHMARKS, LLAMA, QWEN, SYSTEMS

CANON = "No answer available."


def _row(qid, value, pred, method="token_f1", abstained=False,
         rank_hit=None):
    retrieval = ({"skipped": True} if rank_hit is None
                 else {"skipped": False, "hit_at_k": {"5": rank_hit}})
    return {"query_id": qid, "predicted_answer": pred,
            "answer": {"value": value, "method": method,
                       "metadata": {"abstained": abstained}},
            "retrieval": retrieval}


def _cell_rows(benchmark, system):
    ranked = system != "M1" and benchmark != "narrativeqa"
    def rh(v):
        return v if ranked else None
    rows = [
        _row("q0", 0.5, CANON, abstained=True, rank_hit=rh(1.0)),
        _row("q1", 0.0, CANON, abstained=True, rank_hit=rh(0.0)),
        _row("q2", 0.8, "Paris.", rank_hit=rh(1.0)),
        _row("q3", 0.4, "an answer", rank_hit=rh(1.0)),
    ]
    if benchmark == "multihop_rag":
        rows.append(_row("qnull", 1.0, CANON, method="unanswerable_rule",
                         abstained=True, rank_hit=None))
    return rows


GOLDS = {"q0": ("no",),                 # credited under token-F1; EM must be 0
         "q1": ("paris",),
         "q2": ("paris", "france"),     # max over references
         "q3": ("answer",)}             # article stripped by the normaliser


def _write_cell(bank: Path, generator, benchmark, system):
    rows = _cell_rows(benchmark, system)
    n_null = 1 if benchmark == "multihop_rag" else 0
    ranked = system != "M1" and benchmark != "narrativeqa"
    primary = (0.5 + 0.0 + 0.8 + 0.4) / 4
    summary = {
        "system": system, "benchmark": benchmark, "generator": generator,
        "partial_run": False,
        "n_queries_scored": 4 + n_null, "expected_n_queries": 4 + n_null,
        "n_answerable": 4,
        "n_retrieval_scored": 4 if ranked else 0,
        "mean_retrieval_f1": 0.5, "mean_answer_score": primary,
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


def _fixture(tmp: Path):
    p10, p11 = tmp / "p10", tmp / "p11"
    p10.mkdir()
    p11.mkdir()
    for generator, bank in ((QWEN, p10), (LLAMA, p11)):
        for benchmark in BENCHMARKS:
            for system in SYSTEMS:
                _write_cell(bank, generator, benchmark, system)
    recorded = {(g, b, s): (1, 0.5)
                for g in (QWEN, LLAMA) for b in BENCHMARKS for s in SYSTEMS}
    gold_maps = {b: dict(GOLDS) for b in BENCHMARKS}
    return p10, p11, gold_maps, recorded


class TestComparison(unittest.TestCase):
    def test_em_immunity_of_the_credited_refusal(self):
        self.assertNotEqual(normalize_qasper_answer(CANON),
                            normalize_qasper_answer("no"))

    def test_happy_path(self):
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            rows = build_comparison(p10, p11, gm, recorded=rec)
            self.assertEqual(len(rows), 32)
            # row order within each (generator, benchmark) group
            self.assertEqual([r["system"] for r in rows[:4]],
                             [s for _, s in ROW_ORDER])
            r0 = rows[0]  # Qwen / multihop / M1
            # EM: q0 refusal-vs-"no" 0; q1 refusal 0; q2 "Paris." vs
            # ("paris","france") 1; q3 "an answer" vs "answer" 1 -> 0.5;
            # the null row is excluded (n_em 4, not 5)
            self.assertEqual(float(r0["em"]), 0.5)
            self.assertEqual(r0["n_em_population"], 4)
            self.assertEqual(r0["hit_at_5"], "")       # M1: no retrieval
            by = {(r["generator"], r["benchmark"], r["system"]): r
                  for r in rows}
            m2 = by[(QWEN, "hotpotqa", "M2")]
            self.assertEqual(float(m2["hit_at_5"]), 0.75)
            self.assertEqual(m2["n_rank_population"], 4)
            self.assertEqual(by[(QWEN, "narrativeqa", "M2")]["hit_at_5"], "")
            for r in rows:
                self.assertEqual(r["recall_at_5"], "")

    def test_checksum_counts(self):
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            rows = build_comparison(p10, p11, gm, recorded=rec)
            line = checksum_line(rows)
            self.assertIn("rows=32", line)
            # hit@5 filled for 3 systems x 3 ranked benchmarks x 2 gens
            self.assertIn("hit_at_5:18", line)
            self.assertIn("recall_at_5:0", line)
            self.assertIn("em:32", line)

    def test_md_output(self):
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            rows = build_comparison(p10, p11, gm, recorded=rec)
            _, md_path = write_outputs(rows, Path(td) / "out")
            md = md_path.read_text(encoding="utf-8")
            self.assertIn("POST-HOC", md)
            self.assertEqual(md.count("## "), 8)      # eight tables
            self.assertIn(checksum_line(rows), md)
            self.assertIn("bm25-hybrid", md)

    def test_refuses_missing_gold(self):
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            del gm["hotpotqa"]["q2"]
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("no gold", str(cm.exception))

    def test_refuses_rank_population_mismatch(self):
        with TemporaryDirectory() as td:
            p10, p11, gm, rec = _fixture(Path(td))
            stem = p10 / "hotpotqa_M3_validation.summary.json"
            s = json.loads(stem.read_text(encoding="utf-8"))
            s["n_retrieval_scored"] = 9
            stem.write_text(json.dumps(s), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                build_comparison(p10, p11, gm, recorded=rec)
            self.assertIn("rank population", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
