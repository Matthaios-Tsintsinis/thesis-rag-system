"""P8: analyse and significance_diagnostic must agree about what a file says.

`analyse` did not deduplicate by query_id; `significance_diagnostic` did,
by keying a dict. On a three-row fixture holding one duplicate they
reported 0.6667 and 0.5000 for the same input, and the truth was 0.5000
(docs/EVAL_AUDIT.md ISSUE-3). Two tools in one repo disagreeing about a
file is not a rounding difference.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.significance_diagnostic import load_scores
from src.eval import analyse as an


def _row(qid: str, *, f1: float, ans: float, system: str = "M2",
         benchmark: str = "multihop_rag") -> dict:
    return {
        "system_id": system, "benchmark": benchmark, "split": "validation",
        "query_id": qid, "parent_scope": None, "question_text": "?",
        "predicted_answer": "x", "question_type": "inference_query",
        "latency_s": 1.0, "n_retrieved": 15, "n_packed": 15,
        "evidence_tokens": 100, "n_input_tokens": 150,
        "retrieved_unit_types": {"chunk": 15}, "packed_unit_types": {"chunk": 15},
        "retrieval": {"skipped": False, "f1": f1, "recall": f1,
                      "precision": f1, "n_gold": 1, "n_covered": 1,
                      "n_retrieved_atoms": 1, "per_annotator": [],
                      "hit_at_k": {}, "map_at_k": {}, "mrr": 0.0},
        "answer": {"value": ans, "method": "token_f1", "per_annotator": [ans],
                   "metadata": {}},
        "metadata": {},
    }


def _write(tmp: Path, rows: list[dict], *, summary: dict | None = None) -> Path:
    p = tmp / "multihop_rag_M2.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if summary is not None:
        (tmp / "multihop_rag_M2.summary.json").write_text(
            json.dumps(summary), encoding="utf-8")
    return p


class TestTheTwoToolsAgree(unittest.TestCase):
    def test_identical_n_and_means_on_a_file_with_a_duplicate(self):
        rows = [_row("dup1", f1=1.0, ans=1.0),
                _row("dup1", f1=1.0, ans=1.0),
                _row("uniq2", f1=0.0, ans=0.0)]
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), rows)
            recs = an._iter_records([p])
            agg = an._aggregate(recs)["systems"]["M2"]
            sc = load_scores(p)
            ans = [v["answer"] for v in sc.values()]
            retr = [v["retrieval"] for v in sc.values()
                    if v["retrieval"] is not None]

            self.assertEqual(agg["n_queries"], len(sc))
            self.assertAlmostEqual(agg["ans_score_mean"], sum(ans) / len(ans))
            self.assertAlmostEqual(agg["retr_f1_mean"], sum(retr) / len(retr))
            # And both equal the truth, not merely each other.
            self.assertEqual(agg["n_queries"], 2)
            self.assertAlmostEqual(agg["ans_score_mean"], 0.5)


class TestDeduplication(unittest.TestCase):
    def test_last_row_wins(self):
        rows = [_row("q1", f1=1.0, ans=1.0), _row("q1", f1=0.0, ans=0.0)]
        with tempfile.TemporaryDirectory() as td:
            recs = an._iter_records([_write(Path(td), rows)])
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["answer"]["value"], 0.0)


class TestRefusesMixedInputs(unittest.TestCase):
    def test_two_systems_in_one_invocation_raise(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), [_row("q1", f1=1.0, ans=1.0, system="M2"),
                                  _row("q2", f1=1.0, ans=1.0, system="M3")])
            with self.assertRaises(ValueError) as ctx:
                an._iter_records([p])
            self.assertIn("REFUSING", str(ctx.exception))

    def test_two_benchmarks_in_one_invocation_raise(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), [
                _row("q1", f1=1.0, ans=1.0, benchmark="multihop_rag"),
                _row("q2", f1=1.0, ans=1.0, benchmark="narrativeqa")])
            with self.assertRaises(ValueError):
                an._iter_records([p])


class TestExpectedCountGate(unittest.TestCase):
    """Derived from the loader at run time, never a literal — which is
    what stops it aborting NarrativeQA after P7 re-drew the sample."""

    def test_it_aborts_on_a_short_cell(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), [_row("q1", f1=1.0, ans=1.0)],
                       summary={"expected_n_queries": 2556,
                                "max_units": None, "max_queries": None})
            with self.assertRaises(ValueError) as ctx:
                an._check_expected_n([p], 1)
            self.assertIn("partial mean", str(ctx.exception))

    def test_it_passes_a_complete_cell(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), [_row("q1", f1=1.0, ans=1.0)],
                       summary={"expected_n_queries": 1,
                                "max_units": None, "max_queries": None})
            an._check_expected_n([p], 1)

    def test_a_deliberately_capped_run_is_exempt(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), [_row("q1", f1=1.0, ans=1.0)],
                       summary={"expected_n_queries": 2556,
                                "max_units": None, "max_queries": 50})
            an._check_expected_n([p], 1)

    def test_a_missing_summary_warns_rather_than_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            p = _write(Path(td), [_row("q1", f1=1.0, ans=1.0)])
            an._check_expected_n([p], 1)

    def test_the_runner_records_a_loader_derived_count(self):
        """BEHAVIOURAL, not a source grep.

        This asserted the literal text of an inline `.get("n_queries")`
        expression, so it broke the moment the read moved into a named
        function — while proving nothing about where the number comes
        from. It now drives the resolver with two different loaders and
        checks the value TRACKS the loader, which a constant could not.
        """
        from src.eval.runner import resolve_expected_n_queries

        class _Loader:
            def __init__(self, n):
                self.stats = {"n_queries": n}

        self.assertEqual(resolve_expected_n_queries(_Loader(2556)), 2556)
        self.assertEqual(resolve_expected_n_queries(_Loader(1208)), 1208)

    def test_no_literal_query_count_is_baked_into_the_runner(self):
        """The original intent of the grep above, kept and made precise:
        a hardcoded 1208 would abort every NarrativeQA cell the moment P7
        re-drew the sample."""
        import inspect

        from src.eval import runner

        src = inspect.getsource(runner)
        for literal in ("1208", "2556"):
            self.assertNotIn(
                f"== {literal}", src, f"literal {literal} compared in runner"
            )


if __name__ == "__main__":
    unittest.main()
