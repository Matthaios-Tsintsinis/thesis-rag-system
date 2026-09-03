"""replay_cell driven end to end through fakes built from REAL types.

The registries are patched but every object crossing replay_cell's
boundary is the production dataclass (EvalUnit, EvalQuery,
PreparedQuery, RetrievalScore, RetrievedChunk, Chunk) — the GoldAnswer
standard. Covers the evidence the pre-flight audit demanded — a
single-row mismatch refuses the WHOLE cell and writes no sidecar; a
missing warm substrate refuses before anything indexes; the pass path
writes glob-safe sidecars — plus the corrected per-system key logic:
M2/M3 carry no topology component and ignore tree_build_env entirely;
M4 asserts host COMPATIBILITY (component versions + python
major.minor) and injects the RECORDED env string verbatim through the
replay-only override, so a pre-e907d68 token-less record resolves the
old key.
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
from src.raptor_paper import PAPER_TREE_BUILD_ENV
from src.retrievers.base import PreparedQuery, RetrievedChunk
from scripts.export_comparison import QWEN
from scripts.replay_retrieval import _parse_env, replay_cell

ATOM = ("docA", "<whole>")

# host env, MEASURED from the real constant (the
# fixture-parameters-from-real-data discipline): the compatible cases
# are compatible on ANY host running these tests.
HOST = _parse_env(PAPER_TREE_BUILD_ENV)
TOKENLESS_ENV = ";".join(
    f"{k}={v}" for k, v in HOST.items() if k != "python")
HOST_PY_FULL = HOST.get("python", "3.12") + ".13"


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
    def __init__(self, config=None):
        self.config = config
        self.tree_cache_hit = True
        self.topology_env_override = None

    def index_items(self, items):
        pass

    def prepare(self, query, k=None):
        return _prepared(query)


def _fake_resolution(bank_holder, warm=True):
    """Patch resolve_substrate: a real temp warm dir with a manifest
    whose corpus_hash matches the derived one -- so replay_cell's
    key-identity assertion runs against real files."""
    def fake(system, system_id, items):
        d = bank_holder["dir"] / "cachekey0000"
        if warm:
            d.mkdir(exist_ok=True)
            (d / "manifest.json").write_text(
                json.dumps({"corpus_hash": "HASH0"}), encoding="utf-8")
            return d, "HASH0", d
        return None, "HASH0", d
    return fake


class FakeBenchmark:
    def iter_eval_units(self, split):
        q = EvalQuery(query_id="q0", question_text="who?",
                      parent_scope=None, gold_answers=(),
                      gold_passage_sets=(frozenset({ATOM}),),
                      question_type="free_form", metadata={})
        yield EvalUnit(corpus_id="u0", corpus=(), queries=(q,))

    def score_retrieval(self, retrieved, q, scoring_ranking=None):
        return REPLAYED


def _write_bank(bank: Path, banked_retr: dict, system="M2",
                tree_build_env=None, env_python=None):
    stem = f"multihop_rag_{system}_validation"
    summary = {"system": system, "benchmark": "multihop_rag",
               "generator": QWEN, "partial_run": False,
               "n_answerable": 1, "n_queries_scored": 1,
               "environment": {"python": env_python or HOST_PY_FULL},
               "tree_build_env": tree_build_env}
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
    "multihop_rag_M4_validation_*.jsonl",
    "multihop_rag_M4_*.jsonl",
)


def _patched(fn):
    import src.eval.runner as runner
    return mock.patch.dict(
        runner.SYSTEM_REGISTRY, {"M2": FakeSystem, "M4": FakeSystem})(
        mock.patch.dict(
            runner.BENCHMARK_REGISTRY,
            {"multihop_rag": FakeBenchmark})(fn))


class TestReplayCellEndToEnd(unittest.TestCase):
    @_patched
    def test_pass_path_writes_glob_safe_sidecars(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH))
            holder = {"dir": bank}
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            _fake_resolution(holder)):
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
            holder = {"dir": bank}
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            _fake_resolution(holder)):
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
            holder = {"dir": bank}
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            _fake_resolution(holder, warm=False)):
                with self.assertRaises(SystemExit) as cm:
                    replay_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertIn("NO COMPLETE substrate", str(cm.exception))
            self.assertIn("cachekey0000", str(cm.exception))
            self.assertEqual(list(bank.glob("rankings.*")), [])

    @_patched
    def test_manifest_hash_disagreement_refuses(self):
        # key identity SEEN, not inferred: a resolved dir whose manifest
        # records a different corpus_hash refuses by name
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH))
            d = bank / "cachekey0000"
            d.mkdir()
            (d / "manifest.json").write_text(
                json.dumps({"corpus_hash": "OTHER"}), encoding="utf-8")

            def fake(system, system_id, items):
                return d, "HASH0", d
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            fake):
                with self.assertRaises(SystemExit) as cm:
                    replay_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertIn("corpus_hash", str(cm.exception))
            self.assertIn("OTHER", str(cm.exception))

    @_patched
    def test_existing_sidecar_refuses_until_deleted_by_hand(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH))
            holder = {"dir": bank}
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            _fake_resolution(holder)):
                replay_cell(bank, QWEN, "multihop_rag", "M2")
                with self.assertRaises(SystemExit) as cm:
                    replay_cell(bank, QWEN, "multihop_rag", "M2")
                self.assertIn("already exists", str(cm.exception))
                (bank / "rankings.multihop_rag_M2_validation.jsonl").unlink()
                replay_cell(bank, QWEN, "multihop_rag", "M2")


class TestPerSystemKeyLogic(unittest.TestCase):
    """The corrected item-4 logic, after the first-run refusal."""

    @_patched
    def test_m2_ignores_tree_build_env_entirely(self):
        # the defect that refused cell 1: an M2 summary whose recorded
        # env disagrees with the host in every component must PASS --
        # M2's key carries no topology component
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH), system="M2",
                        tree_build_env="umap-learn=0.0.0;"
                                       "scikit-learn=0.0.0;numpy=0.0.0",
                        env_python="3.13.15")
            holder = {"dir": bank}
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            _fake_resolution(holder)):
                out = replay_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertEqual(out["n_rows"], 1)

    @_patched
    def test_m4_pre_token_record_resolves_old_key(self):
        # a pre-e907d68 cell: recorded env is TOKEN-LESS, host has the
        # token -> compatible -> the override receives the recorded
        # string VERBATIM (token-less), reconstructing the old key
        captured = {}
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH), system="M4",
                        tree_build_env=TOKENLESS_ENV)
            holder = {"dir": bank}
            base = _fake_resolution(holder)

            def spy(system, system_id, items):
                captured["override"] = system.topology_env_override
                return base(system, system_id, items)
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            spy):
                out = replay_cell(bank, QWEN, "multihop_rag", "M4")
            self.assertEqual(out["n_rows"], 1)
            self.assertEqual(captured["override"], TOKENLESS_ENV)
            self.assertNotIn("python", captured["override"])

    @_patched
    def test_m4_tokened_record_passes_and_injects_verbatim(self):
        captured = {}
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH), system="M4",
                        tree_build_env=PAPER_TREE_BUILD_ENV)
            holder = {"dir": bank}
            base = _fake_resolution(holder)

            def spy(system, system_id, items):
                captured["override"] = system.topology_env_override
                return base(system, system_id, items)
            with mock.patch("scripts.replay_retrieval.resolve_substrate",
                            spy):
                replay_cell(bank, QWEN, "multihop_rag", "M4")
            self.assertEqual(captured["override"], PAPER_TREE_BUILD_ENV)

    @_patched
    def test_m4_component_mismatch_refuses(self):
        bad_env = TOKENLESS_ENV.replace(
            "scikit-learn=" + HOST["scikit-learn"], "scikit-learn=0.0.0")
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH), system="M4",
                        tree_build_env=bad_env)
            with self.assertRaises(SystemExit) as cm:
                replay_cell(bank, QWEN, "multihop_rag", "M4")
            self.assertIn("INCOMPATIBLE", str(cm.exception))
            self.assertIn("scikit-learn", str(cm.exception))

    @_patched
    def test_m4_python_major_minor_mismatch_refuses(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH), system="M4",
                        tree_build_env=TOKENLESS_ENV,
                        env_python="9.99.0")
            with self.assertRaises(SystemExit) as cm:
                replay_cell(bank, QWEN, "multihop_rag", "M4")
            self.assertIn("INCOMPATIBLE", str(cm.exception))

    @_patched
    def test_m4_missing_record_refuses(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, dict(BANKED_MATCH), system="M4",
                        tree_build_env=None)
            with self.assertRaises(SystemExit) as cm:
                replay_cell(bank, QWEN, "multihop_rag", "M4")
            self.assertIn("no tree_build_env", str(cm.exception))


class TestOverrideLever(unittest.TestCase):
    """The replay-only injection cannot affect the runner."""

    def test_runner_construction_leaves_override_none(self):
        # the runner builds systems as system_cls(config=cfg) -- the
        # same construction leaves the lever at None, and None is
        # byte-identical to the pre-lever key (proven on the extra dict)
        from src.config import DEFAULT_CONFIG
        from src.raptor_paper import paper_substrate_extra
        from src.retrievers.m4_raptor import RaptorSystem
        sy = RaptorSystem(config=DEFAULT_CONFIG)
        self.assertIsNone(sy.topology_env_override)
        m4 = DEFAULT_CONFIG.m4
        base = dict(params=m4.paper, summary_model=m4.summary_model,
                    summary_prompt_version=m4.summary_prompt_version,
                    summary_max_tokens=m4.summary_max_tokens,
                    summary_batch_size=m4.summary_batch_size,
                    summary_max_padded_tokens=m4.summary_max_padded_tokens,
                    rrf_k=m4.rrf_k,
                    include_root=m4.include_root_in_flat_index)
        self.assertEqual(paper_substrate_extra(**base),
                         paper_substrate_extra(**base, build_env=None))
        self.assertEqual(
            paper_substrate_extra(**base)["build_env"],
            PAPER_TREE_BUILD_ENV)
        injected = paper_substrate_extra(**base, build_env="x=1")
        self.assertEqual(injected["build_env"], "x=1")


if __name__ == "__main__":
    unittest.main()
