"""The cold-tree gate enumerates ALL warm units before indexing anything.

WHAT THIS FORBIDS. The per-unit gate aborted on the FIRST warm unit, so
discovering N warm substrates cost N session starts. On M4/hotpotqa —
1,000 units, ~50 warm from the query slice — that is up to fifty aborts.

THE TEST THAT MATTERS is `test_BOTH_warm_units_are_named_in_ONE_abort`:
naming only the first is exactly the old behaviour passing under a new
name, so a test that merely asserts "it aborted" would not distinguish
them.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.eval.base import BenchmarkRunner
from src.eval.types import CorpusItem, EvalQuery, EvalUnit, GoldAnswer


def _unit(cid, qids):
    return EvalUnit(
        corpus_id=cid,
        corpus=(CorpusItem(item_id=f"{cid}::i0", parent_id=cid,
                           span_id="<whole>", text=f"text for {cid}"),),
        queries=tuple(
            EvalQuery(query_id=q, question_text="?", parent_scope=cid,
                      gold_answers=(GoldAnswer(answer_type="free_form",
                                               free_form="x"),),
                      gold_passage_sets=(), question_type="k")
            for q in qids
        ),
    )


class _StubSystem:
    """Reports a warm substrate for the corpus_ids it was told about."""

    system_id = "M4"
    has_cacheable_substrate = True

    def __init__(self, warm_ids):
        self.warm_ids = set(warm_ids)
        self.checked: list[str] = []
        self.indexed: list[str] = []

    def substrate_warm_path(self, items):
        cid = items[0].parent_id
        self.checked.append(cid)
        return f"/cache/M4_RAPTOR/{cid}" if cid in self.warm_ids else None

    def index_items(self, items):  # pragma: no cover - must never run
        self.indexed.append(items[0].parent_id)


def _runner(**kw):
    out = Path(tempfile.mkdtemp()) / "cell.jsonl"
    return BenchmarkRunner(output_path=out, require_cold_tree=True,
                           verbose=False, **kw)


class TestPreflightEnumeratesEverything(unittest.TestCase):
    def setUp(self):
        self.units = [_unit("u1", ["a"]), _unit("u2", ["b"]),
                      _unit("u3", ["c"]), _unit("u4", ["d"])]

    def test_BOTH_warm_units_are_named_in_ONE_abort(self):
        """The whole point. Naming only the first is the old behaviour."""
        sys_ = _StubSystem({"u2", "u4"})
        with self.assertRaises(SystemExit) as ctx:
            _runner()._cold_tree_preflight(sys_, self.units, set())
        msg = str(ctx.exception)
        self.assertIn("u2", msg)
        self.assertIn("u4", msg)
        self.assertIn("/cache/M4_RAPTOR/u2", msg)
        self.assertIn("/cache/M4_RAPTOR/u4", msg)
        self.assertIn("2 of 4", msg)

    def test_every_unit_is_checked_before_the_abort(self):
        """It does not stop scanning at the first warm one."""
        sys_ = _StubSystem({"u2", "u4"})
        with self.assertRaises(SystemExit):
            _runner()._cold_tree_preflight(sys_, self.units, set())
        self.assertEqual(sys_.checked, ["u1", "u2", "u3", "u4"])

    def test_nothing_is_indexed_when_it_aborts(self):
        sys_ = _StubSystem({"u2"})
        with self.assertRaises(SystemExit):
            _runner()._cold_tree_preflight(sys_, self.units, set())
        self.assertEqual(sys_.indexed, [])

    def test_all_cold_passes_silently(self):
        sys_ = _StubSystem(set())
        _runner()._cold_tree_preflight(sys_, self.units, set())
        self.assertEqual(sys_.checked, ["u1", "u2", "u3", "u4"])


class TestPreflightScope(unittest.TestCase):
    def test_units_already_banked_are_not_checked(self):
        """A resumed pass does not index them, so a warm substrate there
        is not a finding."""
        units = [_unit("u1", ["a"]), _unit("u2", ["b"])]
        sys_ = _StubSystem({"u1"})
        _runner()._cold_tree_preflight(sys_, units, {"a"})
        self.assertEqual(sys_.checked, ["u2"])

    def test_systems_without_a_substrate_are_skipped_entirely(self):
        class _NoSubstrate(_StubSystem):
            has_cacheable_substrate = False

        sys_ = _NoSubstrate({"u1"})
        _runner()._cold_tree_preflight(sys_, [_unit("u1", ["a"])], set())
        self.assertEqual(sys_.checked, [])

    def test_the_gate_is_inert_when_no_cold_tree_is_required(self):
        """M1/M2/M3 build no tree; the runner constructs them with
        require_cold_tree=False. There is no such setting for M4."""
        sys_ = _StubSystem({"u1"})
        r = BenchmarkRunner(
            output_path=Path(tempfile.mkdtemp()) / "c.jsonl",
            require_cold_tree=False, verbose=False)
        r._cold_tree_preflight(sys_, [_unit("u1", ["a"])], set())
        self.assertEqual(sys_.checked, [])


if __name__ == "__main__":
    unittest.main()
