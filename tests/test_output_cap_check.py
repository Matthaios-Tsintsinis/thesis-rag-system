"""The fixed-cap output check runs on every answer.

Before the repo reduction `_check_output_length` ran only when a
`--max-new-tokens` override was passed — which no banked cell did — so
the check existed, was correct, and was inert on the run path (the
project's recurring defect shape). It now reads the configured cap
(`GenerationConfig.max_new_tokens`) from the system at the call site and
runs on every answer. These tests drive it two ways: the method alone,
and end to end through `BenchmarkRunner.run` with a system whose cap is
1 and whose answer is long, so the abort is proven to come from the
pipeline READING the cap rather than from the method existing.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.chunking import Chunk
from src.config import DEFAULT_CONFIG
from src.eval.base import BenchmarkRunner
from src.eval.types import (
    ANSWER_TYPE_FREE_FORM,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)
from src.retrievers.base import AnswerResult, BaseSystem, RetrievedChunk

LONG = "this is a full length answer that ignored the cap entirely " * 4


class _ChattySystem(BaseSystem):
    system_id = "CHATTY"

    def index(self, corpus_path):  # pragma: no cover - unused
        self._indexed = True

    def index_items(self, items):
        self.chunks = [
            Chunk(chunk_id=i.item_id, doc_id=i.item_id, text=i.text,
                  n_words=3, position=0,
                  gold_provenance=((i.parent_id, i.span_id),))
            for i in items
        ]
        self._indexed = True

    def retrieve(self, query, k=None):
        return [RetrievedChunk(chunk=c, score=1.0, rank=n)
                for n, c in enumerate(self.chunks)]

    def answer(self, query, k=None):
        r = self.retrieve(query)
        return AnswerResult(query=query, answer=LONG, retrieved=r, packed=r,
                            latency_s=0.01, n_retrieval_calls=1,
                            n_input_tokens=10, evidence_tokens=5, extra={})


class _OneUnitBenchmark:
    name = "one"

    def __init__(self):
        self.stats = {"n_units": 1, "n_queries": 1}

    def iter_eval_units(self, split, max_units=None):
        yield EvalUnit(
            corpus_id="unit0",
            corpus=(CorpusItem(item_id="p0::<whole>", parent_id="p0",
                               span_id="<whole>", text="text here"),),
            queries=(EvalQuery(
                query_id="q0", question_text="q?", parent_scope=None,
                gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                         free_form="x"),),
                gold_passage_sets=(frozenset({("p0", "s0")}),),
                question_type="factoid"),),
        )

    def score_retrieval(self, retrieved, query, scoring_ranking=None):
        return RetrievalScore(skipped=False, f1=1.0)

    def score_answer(self, answer, query):
        return AnswerScore(value=1.0, method="stub")


def _cfg(cap: int):
    return replace(DEFAULT_CONFIG,
                   generation=replace(DEFAULT_CONFIG.generation,
                                      max_new_tokens=cap))


class TestTheMethod(unittest.TestCase):
    def _runner(self):
        self._td = tempfile.TemporaryDirectory()
        return BenchmarkRunner(output_path=Path(self._td.name) / "o.jsonl",
                               verbose=False)

    def tearDown(self):
        td = getattr(self, "_td", None)
        if td is not None:
            td.cleanup()

    def test_an_overlong_answer_raises(self):
        with self.assertRaises(RuntimeError) as cm:
            self._runner()._check_output_length(LONG, "q1", 1)
        self.assertIn("NOT APPLIED", str(cm.exception))

    def test_the_tolerance_is_cap_times_1_25_plus_2(self):
        """tiktoken and the generator's BPE disagree by ~10-20% on the
        same text, so a tight bound would abort real runs."""
        self._runner()._check_output_length("hi", "q1", 1)  # within 1*1.25+2


class TestSpecialTokenLiterals(unittest.TestCase):
    """tiktoken refuses special-token literals by default. A prompt keeps
    that refusal (a corpus must not smuggle control tokens); an ANSWER
    is data the reader emitted and must be counted, never crash a cell."""

    def test_an_answer_carrying_a_special_literal_is_counted(self):
        r = BenchmarkRunner(output_path=Path(tempfile.mkdtemp()) / "o.jsonl",
                            verbose=False)
        r._check_output_length("<|endoftext|> the answer", "q1", 512)

    def test_the_prompt_counter_still_refuses_special_literals(self):
        from src.prompt_packing import count_tokens

        with self.assertRaises(ValueError):
            count_tokens("<|endoftext|>")
        self.assertGreater(count_tokens("<|endoftext|>", allow_special=True), 0)


class TestThePipelineReadsTheCap(unittest.TestCase):
    def _run(self, cap: int):
        with tempfile.TemporaryDirectory() as td:
            system = _ChattySystem(config=_cfg(cap))
            runner = BenchmarkRunner(output_path=Path(td) / "o.jsonl",
                                     verbose=False)
            return list(runner.run(system, _OneUnitBenchmark(),
                                   split="validation"))

    def test_a_tiny_cap_aborts_the_run_on_the_first_answer(self):
        with self.assertRaises(RuntimeError) as cm:
            self._run(1)
        self.assertIn("max_new_tokens=1", str(cm.exception))

    def test_the_matrix_cap_passes_a_long_but_legal_answer(self):
        rows = self._run(DEFAULT_CONFIG.generation.max_new_tokens)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
