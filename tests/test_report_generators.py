"""The generator sweep's report and its template check (v20.2): every finished cell read through
the exporter's functions, each generator against the reader's round-2 cells on the same
questions, and the code of every row checked against the sweep's model."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ours import config as settings
from scripts import dev_ours
from scripts import report_generators as rg
from scripts import report_ours as ro

V20 = "e77ef1d48101c109"
FAST = "1234abcd1234abcd"
TINY, SMALL = "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
GOLDS = {"n0": ("Paris",), "n1": ("Rome",), "m0": ("Reuters",), "m1": ("BBC",), "m2": ()}
DOCS = {"n0": frozenset(), "n1": frozenset(), "m0": frozenset({"A"}),
        "m1": frozenset({"B"}), "m2": frozenset()}


def _row(qid, value, predicted, code, ranking=None, method="token_f1", abstained=False):
    meta = {"ours_code": code, "ours_aspects": [{"low_confidence": False}],
            "ours_section_detectors": {"fixed": 1}, "ours_parse_failed": False,
            "ours_n_llm_calls": 6, "ours_tree_cache_hit": True,
            "ours_doc_ranking": list(ranking or [])}
    if ranking is None:
        retrieval = {"skipped": True}
    else:
        gold = DOCS[qid]
        stats = {k: float(any(d in gold for d in ranking[:k])) for k in (1, 5, 10)}
        first = next((i for i, d in enumerate(ranking) if d in gold), None)
        retrieval = {"skipped": False, "f1": stats[5] / 2, "map_at_k": {"10": stats[10]},
                     "hit_at_k": {str(k): v for k, v in stats.items()},
                     "mrr": 0.0 if first is None else 1.0 / (first + 1)}
    return {"query_id": qid, "predicted_answer": predicted,
            "answer": {"value": value, "method": method, "metadata": {"abstained": abstained}},
            "retrieval": retrieval, "metadata": meta, "n_retrieved": 15,
            "evidence_tokens": 1500, "latency_s": 2.0}


def _cell(bank: Path, generator: str, benchmark: str, rows, gpu="NVIDIA L4"):
    bank.mkdir(parents=True, exist_ok=True)
    stem = f"{benchmark}_OURS_validation"
    (bank / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    answerable = [r for r in rows if r["answer"]["method"] != ro.NULL_METHOD]
    null = [r for r in rows if r["answer"]["method"] == ro.NULL_METHOD]
    summary = {"system": "OURS", "benchmark": benchmark, "generator": generator,
               "n_queries_scored": len(rows), "expected_n_queries": len(rows),
               "n_answerable": len(answerable), "partial_run": False, "elapsed_s": 1.0,
               "mean_answer_score": sum(r["answer"]["value"] for r in rows) / len(rows),
               "mean_answer_score_answerable": sum(r["answer"]["value"] for r in answerable) / len(answerable),
               "mean_answer_score_null": (sum(r["answer"]["value"] for r in null) / len(null)) if null else None,
               "environment": {"gpu": gpu}}
    if benchmark == "multihop_rag":
        summary["n_retrieval_scored"] = len(answerable)
        summary["mean_retrieval_f1"] = sum(r["retrieval"]["f1"] for r in answerable) / len(answerable)
    (bank / f"{stem}.summary.json").write_text(json.dumps(summary), encoding="utf-8")


def narrativeqa(code, right):
    """Two stories' questions; `right` of them answered."""
    return [_row("n0", 1.0 if right > 0 else 0.0, "Paris" if right > 0 else "Lyon", code),
            _row("n1", 1.0 if right > 1 else 0.0, "Rome" if right > 1 else "Milan", code)]


def multihop(code, first):
    """Two answerable questions ranked with gold first or second, one null question refused."""
    order = (["A", "B"], ["B", "A"]) if first else (["B", "A"], ["A", "B"])
    return [_row("m0", 1.0 if first else 0.0, "Reuters" if first else "AP", code, ranking=order[0]),
            _row("m1", 1.0, "BBC", code, ranking=order[1]),
            _row("m2", 1.0, "No answer available.", code, method=ro.NULL_METHOD, abstained=True)]


class _Banks:
    """The reader's round-2 bank and a sweep with one generator finished and one half way."""

    def __init__(self, root: Path, sweep_code=FAST, reader_code=V20):
        self.reader = root / "ours_v20"
        self.sweep = root / "ours_v20_generators"
        _cell(self.reader, settings.READER, "narrativeqa", narrativeqa(reader_code, 1))
        _cell(self.reader, settings.READER, "multihop_rag", multihop(reader_code, False))
        small = dev_ours.sweep_bank(self.sweep, SMALL)
        _cell(small, SMALL, "narrativeqa", narrativeqa(sweep_code, 2))
        _cell(small, SMALL, "multihop_rag", multihop(sweep_code, True), gpu="NVIDIA L4")
        _cell(dev_ours.sweep_bank(self.sweep, TINY), TINY, "narrativeqa", narrativeqa(sweep_code, 0))


@contextlib.contextmanager
def _reading():
    with mock.patch.object(rg, "golds_of", lambda loader: (GOLDS, DOCS)), \
            mock.patch.dict(ro.SAME_OUTPUTS, {FAST: V20}), \
            contextlib.redirect_stdout(io.StringIO()):
        yield


LOADERS = {"narrativeqa": None, "multihop_rag": None}


class TestRegistry(unittest.TestCase):
    def test_the_sweep_runs_v20_and_the_reader_is_round_twos(self):
        self.assertEqual(rg.SWEEP_CODE, {"ours_v20_generators": V20})
        self.assertEqual(ro.FROZEN_CODE[rg.READER_BANK], V20)

    def test_sizes_in_run_order_the_reader_last(self):
        self.assertEqual(list(rg.PARAMS_B), [*settings.SWEEP_GENERATORS, settings.READER])
        sizes = [rg.PARAMS_B[g] for g in settings.SWEEP_GENERATORS]
        self.assertEqual(sizes, sorted(sizes))


class TestCollect(unittest.TestCase):
    def test_finished_cells_against_the_readers_on_the_same_questions(self):
        with tempfile.TemporaryDirectory() as td, _reading():
            banks = _Banks(Path(td))
            rows = rg.collect(banks.sweep, banks.reader, loaders=LOADERS)
        by = {(r["generator"], r["benchmark"]): r for r in rows}
        # Seven generators and the reader, per benchmark.
        self.assertEqual(len(rows), 16)
        self.assertTrue(by[(TINY, "multihop_rag")]["not_run"])
        self.assertTrue(by[(PHI, "narrativeqa")]["not_run"])
        small_nqa = by[(SMALL, "narrativeqa")]
        self.assertEqual(small_nqa["f1"], 1.0)
        self.assertAlmostEqual(small_nqa["answer_delta_vs_reader"], 0.5)
        self.assertEqual(small_nqa["n_paired"], 2)
        self.assertAlmostEqual(by[(TINY, "narrativeqa")]["answer_delta_vs_reader"], -0.5)
        small_mhr = by[(SMALL, "multihop_rag")]
        # Answerable F1 over m0 and m1; the null question's refusal scores 1.
        self.assertEqual((small_mhr["f1"], small_mhr["null_score"]), (1.0, 1.0))
        self.assertEqual(small_mhr["recall_at_5"], 1.0)
        self.assertAlmostEqual(small_mhr["answer_delta_vs_reader"], 1 / 3)
        self.assertAlmostEqual(small_mhr["retrieval_delta_vs_reader"], 0.0)
        reader = by[(settings.READER, "multihop_rag")]
        self.assertEqual(reader["f1"], 0.5)
        self.assertNotIn("answer_delta_vs_reader", reader)

    def test_the_table_shows_what_has_run(self):
        with tempfile.TemporaryDirectory() as td, _reading():
            banks = _Banks(Path(td))
            rows = rg.collect(banks.sweep, banks.reader, loaders=LOADERS)
            rg.write_outputs(rows, banks.sweep)
            text = (banks.sweep / "GENERATORS_OURS.md").read_text(encoding="utf-8")
            n_csv = len((banks.sweep / "GENERATORS_OURS.csv").read_text(encoding="utf-8").splitlines())
        self.assertIn("| Llama-3.2-1B-Instruct | 1.24 | not run yet |", text)
        self.assertIn("| Qwen2.5-1.5B-Instruct | 1.54 | 1.0000 | 1.0000 | 0.0% | +0.5000 [", text)
        self.assertIn("(reference)", text)
        # A header and the five finished cells.
        self.assertEqual(n_csv, 6)

    def test_rows_of_another_model_are_refused(self):
        for sweep_code, reader_code in (("0" * 16, V20), (FAST, FAST)):
            with tempfile.TemporaryDirectory() as td, _reading():
                banks = _Banks(Path(td), sweep_code=sweep_code, reader_code=reader_code)
                with self.assertRaisesRegex(SystemExit, "not the sweep's model"):
                    rg.collect(banks.sweep, banks.reader, loaders=LOADERS)

    def test_only_a_registered_sweep_and_round_twos_bank(self):
        with tempfile.TemporaryDirectory() as td, _reading():
            banks = _Banks(Path(td))
            with self.assertRaisesRegex(SystemExit, "not a registered sweep"):
                rg.collect(banks.reader, banks.reader, loaders=LOADERS)
            with self.assertRaisesRegex(SystemExit, "round 2's bank"):
                rg.collect(banks.sweep, banks.sweep, loaders=LOADERS)

    def test_a_cell_of_another_generator_is_refused(self):
        with tempfile.TemporaryDirectory() as td, _reading():
            banks = _Banks(Path(td))
            _cell(dev_ours.sweep_bank(banks.sweep, TINY), SMALL, "narrativeqa", narrativeqa(FAST, 0))
            with self.assertRaisesRegex(SystemExit, "summary generator"):
                rg.collect(banks.sweep, banks.reader, loaders=LOADERS)


class TestQwenBaselines(unittest.TestCase):
    def test_ours_under_qwen_against_its_column(self):
        qwen = "Qwen/Qwen2.5-7B-Instruct"
        with tempfile.TemporaryDirectory() as td, _reading():
            root = Path(td)
            p10 = root / "p10"
            for system in ro.SYSTEMS:
                rows = narrativeqa("x", 1)
                p10.mkdir(exist_ok=True)
                (p10 / f"narrativeqa_{system}_validation.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            sweep = root / "ours_v20_generators"
            _cell(dev_ours.sweep_bank(sweep, qwen), qwen, "narrativeqa", narrativeqa(FAST, 2))
            got = rg.baselines(sweep, p10, [])
        self.assertEqual([(g["system"], g["metric"]) for g in got],
                         [(s, "answer") for s in ro.SYSTEMS])
        self.assertTrue(all(abs(g["delta"] - 0.5) < 1e-12 and g["n"] == 2 for g in got))


class _Template:
    """A chat template: dated with a default (Llama), or without a date at all."""

    def __init__(self, default=None, dated=True):
        self.default, self.dated = default, dated

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        head = f"Today Date: {kwargs.get('date_string', self.default)}\n" if self.dated else ""
        return head + "|".join(m["content"] for m in messages)


class TestTemplates(unittest.TestCase):
    """The reader's prompt is the one it always read; Llama 3.2's date is the fixed one."""

    def _templates(self, reader_default="26 Jul 2024"):
        def load(model, revision):
            if model == settings.READER:
                return _Template(reader_default)
            if model.startswith("meta-llama/Llama-3.2"):
                return _Template("19 Nov 2026")
            return _Template(dated=False)
        return load

    def test_each_model_read_at_its_pin(self):
        seen = []

        def load(model, revision):
            seen.append((model, revision))
            return _Template(dated=False)

        with contextlib.redirect_stdout(io.StringIO()):
            dev_ours.check_templates(load, lambda m, r: {})
        self.assertEqual(seen, [(settings.READER, settings.READER_REVISION),
                                *settings.SWEEP_GENERATORS.items()])

    def test_only_llama_3_2s_date_is_replaced(self):
        with contextlib.redirect_stdout(io.StringIO()):
            lines = dev_ours.check_templates(self._templates(), lambda m, r: {"repetition_penalty": 1.05})
        replaced = [line["model"] for line in lines if line["date_replaced"]]
        self.assertEqual(replaced, ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"])
        self.assertTrue(all("26 Jul 2024" in line["prompt"] for line in lines
                            if line["model"].startswith("meta-llama")))
        self.assertEqual(lines[0]["generation"], {"repetition_penalty": 1.05})

    def test_a_reader_whose_prompt_would_change_is_refused(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(SystemExit, "prompt changes with the fixed date"):
            dev_ours.check_templates(self._templates(reader_default="01 Jan 2025"), lambda m, r: {})


PHI = "microsoft/Phi-4-mini-instruct"


if __name__ == "__main__":
    unittest.main()
