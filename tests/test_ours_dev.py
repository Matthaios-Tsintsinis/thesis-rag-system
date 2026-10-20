"""Pins ours' development tooling: the held-out draws never touch a
registered question or story, the census reads the trees without building,
the numbers come from the rows as the report reads them, and development
runs refuse the banked cache."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import dev_ours
from src.eval.narrativeqa import select_units
from src.eval.sampling import subsample_indices
from tests.test_ours_smoke import (
    ANSWERS,
    DECOMPOSITIONS,
    FakeReader,
    TwoHotpotUnits,
    ours_fakes,
    run_cell,
)


def _hotpot_rows(n: int) -> list[dict]:
    """Rows in the HotpotQA distractor schema, one question each."""
    return [{"id": f"id{i}", "question": f"Question {i}?", "answer": "yes",
             "type": "bridge", "level": "hard",
             "supporting_facts": {"title": [f"T{i}a"], "sent_id": [0]},
             "context": {"title": [f"T{i}a", f"T{i}b"],
                         "sentences": [["Sentence one of a.", "Two."],
                                       ["Sentence of b."]]}}
            for i in range(n)]


def _story_rows(n_stories: int) -> list[dict]:
    """NarrativeQA rows: two questions per story."""
    rows = []
    for s in range(n_stories):
        doc = {"id": f"s{s}", "text": f"Story text {s}. " * 20, "kind": "gutenberg",
               "url": "", "summary": {"title": f"Story {s}"}}
        for q in range(2):
            rows.append({"document": doc, "question": {"text": f"Who {s} {q}?"},
                         "answers": [{"text": "someone"}]})
    return rows


class TestHeldOutDraws(unittest.TestCase):
    """Development data comes only from outside the registered samples."""

    def test_question_draw_avoids_the_registered_thousand(self):
        draw = dev_ours.held_out_indices(7405)
        self.assertEqual(len(draw), dev_ours.N_DEV_QUESTIONS)
        self.assertEqual(len(set(draw)), len(draw))
        self.assertFalse(set(draw) & set(subsample_indices(7405, 1000)))
        self.assertEqual(draw, dev_ours.held_out_indices(7405))

    def test_hotpotqa_loader_yields_the_held_out_questions(self):
        from datasets import Dataset

        rows = _hotpot_rows(1500)
        printed = io.StringIO()
        with mock.patch("datasets.load_dataset",
                        return_value=Dataset.from_list(rows)), \
                contextlib.redirect_stdout(printed):
            units = list(dev_ours.HeldOutHotpotQA().iter_eval_units(split="validation"))
            pooled = list(dev_ours.HeldOutHotpotQAPooled()
                          .iter_eval_units(split="validation"))
        # The loader's own line names the registered seed; ours says what was drawn.
        self.assertIn(f"seed {dev_ours.DEV_SEED}", printed.getvalue())
        expected = [f"hotpot::id{i}" for i in dev_ours.held_out_indices(1500)]
        self.assertEqual([u.corpus_id for u in units], expected)
        registered = {f"hotpot::id{i}" for i in subsample_indices(1500, 1000)}
        self.assertFalse(registered & set(expected))
        self.assertEqual(len(pooled), 3)

    def test_narrativeqa_loader_avoids_the_registered_forty(self):
        loader = dev_ours.HeldOutNarrativeQA()
        with mock.patch.object(loader, "_load_split", return_value=_story_rows(115)):
            units = list(loader.iter_eval_units(split="validation"))
        ids = [u.corpus_id for u in units]
        registered = set(select_units([f"s{i}" for i in range(115)], None))
        self.assertEqual(len(ids), dev_ours.N_DEV_STORIES)
        self.assertFalse(registered & set(ids))
        self.assertEqual(loader.stats["n_queries"], 2 * dev_ours.N_DEV_STORIES)


class TestNumbers(unittest.TestCase):
    """The numbers a cell's rows give, as the report and the dev log read them."""

    def test_recall_is_not_padded(self):
        self.assertEqual(dev_ours.recall_at(["a"], {"a", "b"}, 5), 0.5)
        self.assertEqual(dev_ours.recall_at([], {"a"}, 5), 0.0)
        self.assertEqual(dev_ours.recall_at(["c", "a", "x", "y", "z", "b"],
                                            {"a", "b"}, 5), 0.5)

    def test_numbers_from_smoke_rows(self):
        with tempfile.TemporaryDirectory() as td:
            rows = run_cell(Path(td) / "cache", Path(td) / "o.jsonl")
        golds, docs = dev_ours.golds_of(TwoHotpotUnits())
        n = dev_ours.ours_numbers(rows, golds, docs)
        self.assertEqual(n["n_rows"], 2)
        self.assertEqual(n["parse_failure_rate"], 0.5)
        self.assertEqual(n["mean_llm_calls"], 5.0)
        self.assertEqual(n["section_detector_share"], {"fixed": 1.0})
        self.assertEqual(len(n["code"]), 1)
        self.assertEqual(n["em"], 0.5)
        expected = sum(dev_ours.recall_at(r["metadata"]["ours_doc_ranking"],
                                          docs[r["query_id"]], 5) for r in rows) / 2
        self.assertEqual(n["recall_at_5"], expected)
        self.assertEqual(n["map_at_10"],
                         sum(r["retrieval"]["map_at_k"]["10"] for r in rows) / 2)


def _answer_row(qid, value, predicted, abstained=False):
    """A row reduced to what the version comparison reads."""
    return {"query_id": qid, "predicted_answer": predicted,
            "answer": {"value": value, "method": "token_f1",
                       "metadata": {"abstained": abstained}}}


class TestCompareVersions(unittest.TestCase):
    """Two versions compared question by question."""

    def test_paired_deltas_and_changed_answers(self):
        golds = {f"q{i}": ("Paris",) for i in range(40)}
        a = [_answer_row(f"q{i}", 0.0, "No answer available.", True)
             for i in range(40)]
        b = [_answer_row(f"q{i}", 1.0, "Paris") if i < 30 else a[i]
             for i in range(40)]
        out = dev_ours.compare_versions(a, b, golds)
        self.assertEqual(out["n"], 40)
        self.assertEqual(out["n_answers_changed"], 30)
        self.assertEqual(out["f1"]["delta"], 0.75)
        self.assertEqual(out["em"]["delta"], 0.75)
        self.assertEqual((out["refusal_a"], out["refusal_b"]), (1.0, 0.25))
        self.assertLess(out["f1"]["p"], 0.001)
        self.assertGreater(out["f1"]["ci"][0], 0.0)

    def test_identical_versions_show_no_difference(self):
        golds = {"q0": ("Paris",), "q1": ("Rome",)}
        rows = [_answer_row("q0", 1.0, "Paris"), _answer_row("q1", 0.0, "Oslo")]
        out = dev_ours.compare_versions(rows, rows, golds)
        self.assertEqual((out["f1"]["delta"], out["f1"]["p"], out["n_answers_changed"]),
                         (0.0, 1.0, 0))


class TestInspect(unittest.TestCase):
    """The answer-side breakdown read from a cell's rows."""

    def test_breakdown_from_smoke_rows(self):
        with tempfile.TemporaryDirectory() as td:
            rows = run_cell(Path(td) / "cache", Path(td) / "o.jsonl")
        golds, docs = dev_ours.golds_of(TwoHotpotUnits())
        report, samples = dev_ours.inspect_rows(rows, golds, docs, n_samples=2)
        # Unit 0 answers "the river" (F1 1.0); unit 1 refuses.
        self.assertEqual(report["abstained_share"], 0.5)
        self.assertEqual(report["exact_refusal_share"], 0.5)
        self.assertEqual(report["f1_answered"], 1.0)
        self.assertEqual(report["answer_words_mean"], 2.5)
        self.assertEqual(report["f1_by_type"], {"bridge": [2, 0.5]})
        self.assertEqual(report["f1_by_n_aspects"], {"1": [1, 0.0], "2": [1, 1.0]})
        self.assertEqual(report["map_at_10"],
                         sum(r["retrieval"]["map_at_k"]["10"] for r in rows) / 2)
        self.assertEqual([s["query_id"] for s in samples], ["q0", "q1"])
        self.assertEqual(samples[1]["predicted"], "No answer available.")


class TestCensus(unittest.TestCase):
    """The census counts warm trees under the recorded topology string."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.p11 = self.root / "p11"
        self.p11.mkdir()
        self.cache = self.root / "cache"
        # Build the first unit's tree only, as an earlier M4 cell would have.
        unit = TwoHotpotUnits().units[0]
        with ours_fakes(self.cache, FakeReader(DECOMPOSITIONS, ANSWERS)):
            from ours.model import OursSystem
            OursSystem(config=dev_ours.llama_config()).index_items(unit.corpus)

    def tearDown(self):
        self.td.cleanup()

    def _summary(self, env):
        (self.p11 / "hotpotqa_M4_validation.summary.json").write_text(
            json.dumps({"tree_build_env": env, "model_revisions": {"revisions": {}}}),
            encoding="utf-8")

    def _census(self):
        out = io.StringIO()
        with mock.patch.dict("src.eval.runner.BENCHMARK_REGISTRY",
                             {"hotpotqa": TwoHotpotUnits}), \
                mock.patch("src.paths.cache_dir", return_value=self.cache), \
                contextlib.redirect_stdout(out):
            env = dev_ours.census(self.p11, benchmarks=("hotpotqa",))
        return env, out.getvalue()

    def test_counts_warm_and_cold_without_building(self):
        self._summary("test-env")
        files_before = sorted(self.cache.rglob("*"))
        env, text = self._census()
        self.assertEqual(env, "test-env")
        self.assertIn("hotpotqa: 1 units warm, 1 to build", text)
        self.assertEqual(sorted(self.cache.rglob("*")), files_before)

    def test_another_topology_finds_nothing(self):
        self._summary("other-env")
        _, text = self._census()
        self.assertIn("hotpotqa: 0 units warm, 2 to build", text)


class TestDevRun(unittest.TestCase):
    """A development run logs its numbers and refuses the banked cache."""

    def test_refuses_without_a_development_cache(self):
        with mock.patch.dict(os.environ, {"THESIS_CACHE_DIR": ""}):
            with self.assertRaises(SystemExit):
                dev_ours.dev_run("hotpotqa", "v0", Path("requirements.lock"), False)

    def test_run_appends_one_log_line(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock = root / "requirements.lock"
            lock.write_text("# lock\nnumpy==0.0.0\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"THESIS_CACHE_DIR": str(root / "cache"),
                                              "THESIS_OUTPUT_DIR": str(root / "out")}), \
                    mock.patch.dict(dev_ours.HELD_OUT, {"hotpotqa": TwoHotpotUnits}), \
                    mock.patch("scripts.pin_environment.check_lockfile", return_value=0), \
                    ours_fakes(root / "cache", FakeReader(DECOMPOSITIONS, ANSWERS)), \
                    contextlib.redirect_stdout(io.StringIO()):
                dev_ours.dev_run("hotpotqa", "v0", lock, False)
            log = (root / "out" / "ours_dev" / "DEV_LOG.jsonl").read_text().splitlines()
        self.assertEqual(len(log), 1)
        line = json.loads(log[0])
        self.assertEqual((line["version"], line["benchmark"], line["n_rows"]),
                         ("v0", "hotpotqa", 2))
        from ours.model import CODE_FINGERPRINT
        self.assertEqual(line["ours_code"], CODE_FINGERPRINT)


if __name__ == "__main__":
    unittest.main()
