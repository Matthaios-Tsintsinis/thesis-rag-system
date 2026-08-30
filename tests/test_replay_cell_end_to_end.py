"""replay_cell driven end to end through fakes built from REAL types.

The registries are patched but every object crossing replay_cell's
boundary is the production dataclass (EvalUnit, EvalQuery,
PreparedQuery, RetrievalScore, RetrievedChunk, Chunk) — the GoldAnswer
standard. Covers the evidence the pre-flight audit demanded: a
single-row mismatch refuses the WHOLE cell and writes no sidecar; a
missing warm substrate refuses before anything indexes; the pass path
writes glob-safe sidecars whose names match none of the bank's
discovery patterns.
"""

from __future__ import annotations

import fnmatch
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.chunking import Chunk
from src.eval.types import EvalQuery, EvalUnit, RetrievalScore
from src.retrievers.base import PreparedQuery, RetrievedChunk
from scripts.export_matrix import QWEN
from scripts.replay_retrieval import replay_cell

ATOM = ("docA", "<whole>")


def _chunk_hit():
    c = Chunk(chunk_id="c0", doc_id="d", text="t", n_words=1, position=0,
              gold_provenance=(ATOM,))
    return RetrievedChunk(chunk=c, score=1.0, rank=0)


def _prepared(query):
    hit = [_chunk_hit()]
    return PreparedQuery(query=query, retrieved=hit, packed=hit,
                         scoring_ranking=hit, system_prompt="s",
                         user_prompt="u", evidence_tokens=1,
                         n_input_tokens=2)


REPLAYED = RetrievalScore(
    skipped=False, recall=1.0, precision=1.0, f1=1.0, n_gold=1,
    hit_at_k={1: 1.0, 5: 1.0, 10: 1.0},
    map_at_k={1: 1.0, 5: 1.0, 10: 1.0}, mrr=1.0)


class FakeSystem:
    warm = "warm-path"

    def __init__(self, config=None):
        self.config = config
        self.tree_cache_hit = True

    def substrate_warm_path(self, items):
        return self.warm

    def index_items(self, items):
        pass

    def prepare(self, query, k=None):
        return _prepared(query)


class FakeBenchmark:
    def iter_eval_units(self, split):
        q = EvalQuery(query_id="q0", question_text="who?",
                      parent_scope=None, gold_answers=(),
                      gold_passage_sets=(frozenset({ATOM}),),
                      question_type="free_form", metadata={})
        yield EvalUnit(corpus_id="u0", corpus=(), queries=(q,))

    def score_retrieval(self, retrieved, q, scoring_ranking=None):
        return REPLAYED


def _write_bank(bank: Path, banked_retr: dict):
    stem = "multihop_rag_M2_validation"
    summary = {"system": "M2", "benchmark": "multihop_rag",
               "generator": QWEN, "partial_run": False,
               "n_answerable": 1, "n_queries_scored": 1,
               "tree_build_env": None}
    (bank / f"{stem}.summary.json").write_text(json.dumps(summary),
                                               encoding="utf-8")
    row = {"query_id": "q0",
           "answer": {"value": 1.0, "method": "token_f1", "metadata": {}},
           "retrieval": banked_retr}
    (bank / f"{stem}.jsonl").write_text(json.dumps(row) + "\n",
                                        encoding="utf-8")


BANKED_MATCH = {"skipped": False, "f1": 1.0, "recall": 1.0,
                "precision": 1.0, "mrr": 1.0,
                "hit_at_k": {"1": 1.0, "5": 1.0, "10": 1.0},
                "map_at_k": {"1": 1.0, "5": 1.0, "10": 1.0}}

# every discovery pattern found in the glob audit (pre-flight item C8)
BANK_DISCOVERY_PATTERNS = (
    "*.summary.json",                       # bank gates + aggregate rglob
    "multihop_rag_M2_validation_*.jsonl",   # significance stamped glob
    "multihop_rag_M2_*.jsonl",              # significance loose glob
)


def _patched(fn):
    return mock.patch.dict(
        __import__("src.eval.runner", fromlist=["SYSTEM_REGISTRY"]
                   ).SYSTEM_REGISTRY, {"M2": FakeSystem})(
        mock.patch.dict(
            __import__("src.eval.runner", fromlist=["BENCHMARK_REGISTRY"]
                       ).BENCHMARK_REGISTRY,
            {"multihop_rag": FakeBenchmark})(fn))


class TestReplayCellEndToEnd(unittest.TestCase):
    @_patched
    def test_pass_path_writes_glob_safe_sidecars(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH))
            out = replay_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertEqual(out["n_rows"], 1)
            self.assertEqual(out["recall_at_5"], 1.0)
            rows_f = bank / "rankings.multihop_rag_M2_validation.jsonl"
            sum_f = bank / "rankings.multihop_rag_M2_validation.json"
            self.assertTrue(rows_f.is_file() and sum_f.is_file())
            for f in (rows_f, sum_f):
                for pat in BANK_DISCOVERY_PATTERNS:
                    self.assertFalse(fnmatch.fnmatch(f.name, pat),
                                     (f.name, pat))
            side = json.loads(rows_f.read_text(encoding="utf-8"))
            self.assertEqual(side["recall_at_k"]["5"], 1.0)
            self.assertEqual(side["doc_ranking"], [list(ATOM)])

    @_patched
    def test_single_row_mismatch_refuses_cell_and_writes_nothing(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            banked = dict(BANKED_MATCH)
            banked["f1"] = 0.9        # one field, one row
            _write_bank(bank, banked)
            with self.assertRaises(SystemExit) as cm:
                replay_cell(bank, QWEN, "multihop_rag", "M2")
            msg = str(cm.exception)
            self.assertIn("GATE FAILED", msg)
            self.assertIn("q0", msg)              # the refusal names the row
            self.assertEqual(
                list(bank.glob("rankings.*")), [])  # nothing written

    @_patched
    def test_missing_warm_substrate_refuses(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH))
            with mock.patch.object(FakeSystem, "warm", None):
                with self.assertRaises(SystemExit) as cm:
                    replay_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertIn("warm substrate", str(cm.exception))
            self.assertEqual(list(bank.glob("rankings.*")), [])

    @_patched
    def test_existing_sidecar_refuses_without_force(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH))
            replay_cell(bank, QWEN, "multihop_rag", "M2")
            with self.assertRaises(SystemExit) as cm:
                replay_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertIn("--force", str(cm.exception))
            replay_cell(bank, QWEN, "multihop_rag", "M2", force=True)

    @_patched
    def test_env_mismatch_refuses(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH))
            stem = bank / "multihop_rag_M2_validation.summary.json"
            s = json.loads(stem.read_text(encoding="utf-8"))
            s["tree_build_env"] = "python=9.9;umap-learn=0.0.0"
            stem.write_text(json.dumps(s), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                replay_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertIn("topology env", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
