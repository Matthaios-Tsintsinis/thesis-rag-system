"""export_matrix: the 32-cell supplementary-data exporter.

The fixture is SYNTHETIC and its expected credited table is MEASURED from
the fixture's own rows (the fixture-parameters-measured-from-data
discipline) and injected through `build_rows(recorded=...)` — the
production default `RECORDED_CREDITED` is tested separately as the
recorded-battery commitment it is, plus behaviourally: the mismatch test
proves the pipeline READS the table and refuses on disagreement, so the
constants cannot go inert.
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.export_matrix import (
    BENCHMARKS,
    COLUMNS,
    LLAMA,
    QWEN,
    RECORDED_CREDITED,
    SYSTEMS,
    build_rows,
    checksum_line,
    write_outputs,
)

CANON = "No answer available."


def _row(value, method="token_f1", abstained=False, pred="an answer"):
    return {
        "predicted_answer": pred,
        "answer": {"value": value, "method": method,
                   "metadata": {"abstained": abstained}},
    }


def _cell_rows(benchmark):
    """Four answerable rows (one credited refusal, one plain refusal at 0,
    two plain answers) + one null row on multihop."""
    rows = [
        _row(0.5, abstained=True, pred=CANON),      # credited refusal
        _row(0.0, abstained=True, pred=CANON),      # refusal, uncredited
        _row(0.8),                                   # plain
        _row(0.4),                                   # plain
    ]
    if benchmark == "multihop_rag":
        rows.append(_row(1.0, method="unanswerable_rule", abstained=True,
                         pred=CANON))               # null row: excluded
    return rows


def _write_cell(bank: Path, generator, benchmark, system, *,
                partial=False, wrong_generator=None):
    n_ans = 4
    n_null = 1 if benchmark == "multihop_rag" else 0
    rows = _cell_rows(benchmark)
    primary = (0.5 + 0.0 + 0.8 + 0.4) / n_ans     # 0.425
    summary = {
        "system": system, "benchmark": benchmark,
        "generator": wrong_generator or generator,
        "partial_run": partial,
        "n_queries_scored": n_ans + n_null,
        "expected_n_queries": n_ans + n_null,
        "n_answerable": n_ans,
        "mean_retrieval_f1": 0.1234567890123456,
        "mean_answer_score": primary,
        "mean_answer_score_answerable": primary,
        "mean_answer_score_null": (1.0 if n_null else None),
        "elapsed_s": 123.4,
        "git_commit": "abc1234",
        "timestamp": "20260830-000000",
        "environment": {"python": "3.12.13", "lockfile_hash": "feedbeef"},
        "model_revisions": {"revisions": {"generator": "sha000"}},
    }
    stem = f"{benchmark}_{system}_validation"
    (bank / f"{stem}.summary.json").write_text(json.dumps(summary),
                                               encoding="utf-8")
    with (bank / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _fixture(tmp: Path, **cell_kwargs):
    p10, p11 = tmp / "p10", tmp / "p11"
    p10.mkdir()
    p11.mkdir()
    for generator, bank in ((QWEN, p10), (LLAMA, p11)):
        for benchmark in BENCHMARKS:
            for system in SYSTEMS:
                _write_cell(bank, generator, benchmark, system)
    # measured from the fixture rows: one credited refusal at 0.5 per cell
    recorded = {(g, b, s): (1, 0.5)
                for g in (QWEN, LLAMA) for b in BENCHMARKS for s in SYSTEMS}
    return p10, p11, recorded


class TestExport(unittest.TestCase):
    def test_happy_path_32_rows_and_arithmetic(self):
        with TemporaryDirectory() as td:
            p10, p11, rec = _fixture(Path(td))
            rows = build_rows(p10, p11, recorded=rec)
            self.assertEqual(len(rows), 32)
            r = rows[0]  # Qwen / multihop / M1
            self.assertEqual(r["n_credited"], 1)
            # supplementary = primary - mass/n_answerable = 0.425 - 0.125
            self.assertAlmostEqual(float(r["supplementary_mean"]), 0.3, places=12)
            # abstain: 2 of 4 answerable; plain mean = (0.8+0.4)/2
            self.assertEqual(float(r["abstain_pct"]), 50.0)
            self.assertAlmostEqual(float(r["mean_plain"]), 0.6, places=12)
            # null mean comes from the summary, never recomputed
            self.assertEqual(float(r["mean_answer_score_null"]), 1.0)

    def test_absence_encoding(self):
        with TemporaryDirectory() as td:
            p10, p11, rec = _fixture(Path(td))
            rows = build_rows(p10, p11, recorded=rec)
            by = {(r["generator"], r["benchmark"], r["system"]): r for r in rows}
            m1 = by[(QWEN, "hotpotqa", "M1")]
            self.assertEqual((m1["mean_retrieval_f1"], m1["retrieval_absence"]),
                             ("", "no_retrieval"))
            # M1 on NarrativeQA: the system-level absence wins
            self.assertEqual(by[(QWEN, "narrativeqa", "M1")]["retrieval_absence"],
                             "no_retrieval")
            self.assertEqual(by[(QWEN, "narrativeqa", "M2")]["retrieval_absence"],
                             "no_gold")
            m2 = by[(QWEN, "hotpotqa", "M2")]
            # full precision survives the string round-trip
            self.assertEqual(float(m2["mean_retrieval_f1"]), 0.1234567890123456)
            self.assertEqual(m2["retrieval_absence"], "")

    def test_written_files_and_checksum(self):
        with TemporaryDirectory() as td:
            p10, p11, rec = _fixture(Path(td))
            rows = build_rows(p10, p11, recorded=rec)
            csv_path, md_path = write_outputs(rows, Path(td) / "out")
            with csv_path.open(encoding="utf-8", newline="") as f:
                back = list(csv.DictReader(f))
            self.assertEqual(len(back), 32)
            self.assertEqual(list(back[0].keys()), COLUMNS)
            line = checksum_line(rows)
            self.assertIn("rows=32", line)
            # retrieval f1 is empty on M1 (8) + non-M1 narrativeqa (6) = 18 filled
            self.assertIn("mean_retrieval_f1:18", line)
            self.assertIn(line, md_path.read_text(encoding="utf-8"))

    def test_refuses_missing_summary(self):
        with TemporaryDirectory() as td:
            p10, p11, rec = _fixture(Path(td))
            (p11 / "hotpotqa_M4_validation.summary.json").unlink()
            with self.assertRaises(SystemExit) as cm:
                build_rows(p10, p11, recorded=rec)
            self.assertIn("missing summary", str(cm.exception))

    def test_refuses_partial_run(self):
        with TemporaryDirectory() as td:
            p10, p11, rec = _fixture(Path(td))
            _write_cell(p10, QWEN, "hotpotqa", "M2", partial=True)
            with self.assertRaises(SystemExit) as cm:
                build_rows(p10, p11, recorded=rec)
            self.assertIn("partial_run", str(cm.exception))

    def test_refuses_generator_mismatch(self):
        with TemporaryDirectory() as td:
            p10, p11, rec = _fixture(Path(td))
            _write_cell(p10, QWEN, "hotpotqa", "M3", wrong_generator=LLAMA)
            with self.assertRaises(SystemExit) as cm:
                build_rows(p10, p11, recorded=rec)
            self.assertIn("generator", str(cm.exception))

    def test_refuses_credited_disagreement(self):
        """The pipeline READS the recorded table -- the gate cannot go inert."""
        with TemporaryDirectory() as td:
            p10, p11, rec = _fixture(Path(td))
            rec[(QWEN, "multihop_rag", "M1")] = (2, 1.0)  # wrong on purpose
            with self.assertRaises(SystemExit) as cm:
                build_rows(p10, p11, recorded=rec)
            self.assertIn("recorded battery", str(cm.exception))


class TestRecordedBattery(unittest.TestCase):
    """The default table is a recorded-battery commitment: 32 cells, the
    HotpotQA family all zero (the sentinel guard), the spot values the
    living record banked. Breaking on change is these tests working."""

    def test_shape_and_guard_zeros(self):
        self.assertEqual(len(RECORDED_CREDITED), 32)
        for (g, b, s), (n, mass) in RECORDED_CREDITED.items():
            if b in ("hotpotqa", "hotpotqa_pooled"):
                self.assertEqual((n, mass), (0, 0.0), (g, b, s))

    def test_spot_values_from_the_living_record(self):
        self.assertEqual(RECORDED_CREDITED[(QWEN, "multihop_rag", "M4")],
                         (504, 252.0))
        self.assertEqual(RECORDED_CREDITED[(LLAMA, "multihop_rag", "M1")],
                         (558, 279.0))
        self.assertEqual(RECORDED_CREDITED[(QWEN, "narrativeqa", "M4")],
                         (6, 1.36))
        self.assertEqual(RECORDED_CREDITED[(LLAMA, "narrativeqa", "M3")],
                         (6, 1.44))


if __name__ == "__main__":
    unittest.main()
