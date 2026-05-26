"""Benchmark protocol + BenchmarkRunner orchestration.

A `Benchmark` exposes:
  * `name` — string id used in output filenames.
  * `iter_eval_units(split, max_units)` — generator of `EvalUnit`s
    (one per per-paper corpus for QASPER, one shared-corpus
    EvalUnit for MultiHop-RAG).
  * `score_answer(predicted, query)` — benchmark-specific scoring of
    the predicted answer text against the query's gold annotations.

`BenchmarkRunner` drives one (system, benchmark, split) pass:

  for unit in benchmark.iter_eval_units(split, max_units):
      system.index_items(unit.corpus)
      for q in unit.queries:
          ar = system.answer(q.question_text)
          retr = score_retrieval_ck2(ar.retrieved, q.gold_passage_sets)
          ans  = benchmark.score_answer(ar.answer, q)
          yield ScoredQuery(...)

Each ScoredQuery is written as one JSONL line. The driver is
deliberately simple: one process, one system, one benchmark, one
split. Parallelism across systems/benchmarks happens at the CLI runner
level by sharding (run M2 + qasper, M3 + qasper, ... separately).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from ..retrievers.base import BaseSystem, RetrievedChunk
from .alignment import score_retrieval_ck2
from .types import AnswerScore, EvalQuery, EvalUnit, RetrievalScore, ScoredQuery


class Benchmark(Protocol):
    """Protocol every benchmark loader must satisfy."""

    name: str

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]: ...

    def score_answer(
        self,
        predicted: str,
        query: EvalQuery,
    ) -> AnswerScore: ...

    def score_retrieval(
        self,
        retrieved: list[RetrievedChunk],
        query: EvalQuery,
    ) -> RetrievalScore: ...


class BenchmarkRunner:
    """Runs one (system, benchmark, split) pass and emits ScoredQuery JSONL.

    Output schema (one JSON object per line):
      {system, benchmark, split, query_id, parent_scope, question_text,
       predicted_answer, retrieval: {...}, answer: {...},
       question_type, latency_s, n_retrieved, metadata: {...}}

    The runner does not store anything in memory beyond the current
    EvalUnit + a small running counter — long benchmark runs stream
    directly to disk so a crash mid-run loses only the in-flight
    EvalUnit.
    """

    def __init__(
        self,
        *,
        output_path: Path,
        verbose: bool = True,
    ) -> None:
        self.output_path = output_path
        self.verbose = verbose

    def run(
        self,
        system: BaseSystem,
        benchmark: Benchmark,
        *,
        split: str,
        max_units: int | None = None,
        max_queries: int | None = None,
    ) -> Iterator[ScoredQuery]:
        """Drive one (system, benchmark, split) pass to JSONL.

        `max_units` caps the EvalUnits processed (one per paper for
        QASPER; MultiHop has one shared-corpus EvalUnit so max_units<=1
        is meaningful there). `max_queries` caps TOTAL queries across
        units — useful for MultiHop where the natural EvalUnit holds
        2556 queries; pass `--max-queries 50` for a small-sample
        shared-corpus validation before the full run.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        n_units = 0
        n_queries = 0
        t_start = time.perf_counter()

        with self.output_path.open("w", encoding="utf-8") as fout:
            stopped = False
            for unit_idx, unit in enumerate(
                benchmark.iter_eval_units(split=split, max_units=max_units)
            ):
                if stopped:
                    break
                if self.verbose:
                    print(
                        f"[eval] unit {unit_idx + 1}: corpus_id={unit.corpus_id!r}  "
                        f"n_items={len(unit.corpus)}  n_queries={len(unit.queries)}"
                    )

                t_index = time.perf_counter()
                system.index_items(unit.corpus)
                index_s = time.perf_counter() - t_index

                if self.verbose:
                    print(f"  index_s={index_s:.2f}")

                for q in unit.queries:
                    if max_queries is not None and n_queries >= max_queries:
                        stopped = True
                        break
                    t_q = time.perf_counter()
                    ar = system.answer(q.question_text)
                    latency_s = time.perf_counter() - t_q

                    # Retrieval scoring is benchmark-specific: each
                    # benchmark implements score_retrieval to combine
                    # set-F1 (CK-2) with any rank-aware metrics it
                    # cares about (MultiHop adds Hit@K / MAP@K / MRR).
                    # Independent of CK-4 packing — reads ar.retrieved
                    # (full ranking).
                    retr = benchmark.score_retrieval(ar.retrieved, q)
                    ans = benchmark.score_answer(ar.answer, q)

                    # CK-4: collect unit-type distributions for analysis.
                    retrieved_unit_types: dict[str, int] = {}
                    for r in ar.retrieved:
                        ut = getattr(r, "source_unit_type", "chunk")
                        retrieved_unit_types[ut] = retrieved_unit_types.get(ut, 0) + 1
                    packed_unit_types: dict[str, int] = {}
                    for r in ar.packed:
                        ut = getattr(r, "source_unit_type", "chunk")
                        packed_unit_types[ut] = packed_unit_types.get(ut, 0) + 1

                    scored = ScoredQuery(
                        system_id=system.system_id,
                        benchmark=benchmark.name,
                        split=split,
                        query_id=q.query_id,
                        parent_scope=q.parent_scope,
                        question_text=q.question_text,
                        predicted_answer=ar.answer,
                        retrieval=retr,
                        answer=ans,
                        question_type=q.question_type,
                        latency_s=latency_s,
                        n_retrieved=len(ar.retrieved),
                        n_packed=len(ar.packed),
                        evidence_tokens=int(ar.evidence_tokens),
                        n_input_tokens=int(ar.n_input_tokens),
                        retrieved_unit_types=retrieved_unit_types,
                        packed_unit_types=packed_unit_types,
                        metadata=q.metadata,
                    )
                    fout.write(
                        json.dumps(asdict(scored), ensure_ascii=False) + "\n"
                    )
                    fout.flush()
                    n_queries += 1
                    yield scored

                n_units += 1

        if self.verbose:
            elapsed = time.perf_counter() - t_start
            print(
                f"[eval] done: {n_units} units, {n_queries} queries, "
                f"elapsed={elapsed:.1f}s, output={self.output_path}"
            )


__all__ = ["Benchmark", "BenchmarkRunner"]
