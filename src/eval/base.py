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

from ..retrievers.base import AnswerResult, BaseSystem, RetrievedChunk
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
        batch_size: int | None = None,
        resume: bool = False,
        max_padded_tokens: int | None = None,
        verify_max_new_tokens: int | None = None,
    ) -> None:
        self.output_path = output_path
        self.verbose = verbose
        # Resume an interrupted pass: append to the existing JSONL and
        # skip query_ids already present. Index caches survive a dead
        # session on their own, but without this the answers do not —
        # and a re-run would TRUNCATE the partial output. On this project
        # sessions have died to a reclaimed runtime, a Drive disconnect,
        # an RPD ceiling and a broken torch install, so treating an
        # interrupted pass as normal rather than exceptional is the
        # correct default posture even though the flag is opt-in.
        self.resume = resume
        # Upper bound on n * longest-prompt within a batch. When set,
        # it governs batch shape and `batch_size` becomes an upper
        # bound on COUNT rather than the batch size itself. This is
        # the knob that survives ragged real prompts; a fixed count
        # measured on uniform synthetic prompts does not.
        self.max_padded_tokens = max_padded_tokens
        # None -> sequential answering (the historic path, and the right
        # one against an API). An int enables TWO-PHASE answering for
        # systems that support it: retrieve every query first, then
        # generate the whole unit in batches. Necessary because the
        # harness now generates with a LOCAL model, where sequential
        # answering wastes ~90% of throughput and generation is ~90% of
        # the cost.
        self.batch_size = batch_size
        # When a generation cap is being asserted, CHECK IT. A probe that
        # silently did not apply its cap produced a wrong number rather
        # than an error once already; the caller should not have to be the
        # one who notices. None disables the check entirely.
        self.verify_max_new_tokens = verify_max_new_tokens
        self._verify_tok = None

    def _check_output_length(self, answer: str, query_id: str, model: str) -> None:
        """Abort if a generated answer overran the cap that was requested.

        The cap is re-measured from the DECODED text, so allow one token
        of slack: decode -> strip -> re-encode is not an exact inverse of
        generation (a leading space can vanish, a multi-byte glyph can
        re-split). That slack is irrelevant to the failure this catches,
        which is a cap of 1 producing a 100-token answer.
        """
        if self.verify_max_new_tokens is None:
            return
        if self._verify_tok is None:
            from ..models import load_generator

            self._verify_tok = load_generator(model)[0]
        n = len(self._verify_tok(answer, add_special_tokens=False)["input_ids"])
        if n > self.verify_max_new_tokens + 1:
            raise RuntimeError(
                f"generation cap NOT APPLIED: query {query_id!r} returned "
                f"~{n} tokens against max_new_tokens="
                f"{self.verify_max_new_tokens}. The run is measuring "
                f"something other than what it claims. Answer began: "
                f"{answer[:80]!r}"
            )

    def _existing_query_ids(self) -> set[str]:
        """query_ids already banked in the output file, for --resume.

        Tolerates a truncated final line: a session killed mid-write
        leaves a partial JSON object, and that is exactly the case
        resume exists to handle, so it must not raise. The partial row's
        query is simply re-answered.
        """
        if not self.resume or not self.output_path.exists():
            return set()
        done: set[str] = set()
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # torn final line
        return done

    def _answer_unit(
        self,
        system: BaseSystem,
        queries: list[EvalQuery],
    ) -> Iterator[tuple[EvalQuery, "AnswerResult", float]]:
        """Yield (query, AnswerResult, latency_s) for one unit's queries.

        Sequential unless batching is enabled AND the system supports it
        (M1 and M7 still override answer() wholesale — see
        BaseSystem.supports_batched_answer). Falling back rather than
        guessing keeps an unconverted system correct instead of subtly
        wrong.

        Output order is ALWAYS the input query order, regardless of the
        length-sorted order generation actually ran in.
        """
        if not queries:
            return
        if self.batch_size is None or not getattr(
            system, "supports_batched_answer", False
        ):
            for q in queries:
                t_q = time.perf_counter()
                ar = system.answer(q.question_text)
                yield q, ar, time.perf_counter() - t_q
            return

        from ..models import (
            deterministic_batch_order,
            generate_batch,
            token_budget_batches,
        )

        # PHASE A — retrieval + any query-time LLM work, sequential.
        t_a = time.perf_counter()
        prepared = [system.prepare(q.question_text) for q in queries]
        phase_a_s = time.perf_counter() - t_a
        if self.verbose:
            print(f"  phase_a(retrieve)={phase_a_s:.1f}s  n={len(queries)}")

        # PHASE B — batched generation, YIELDED PER BATCH so the caller
        # can write and flush incrementally.
        #
        # DURABILITY over ordering, deliberately. Generating the whole
        # unit before yielding anything would mean MultiHop (ONE unit,
        # 2,556 queries) writes nothing for well over an hour and loses
        # everything to a reclaimed runtime or a Drive disconnect — both
        # of which have happened on this project. Yielding per batch caps
        # the loss at one batch.
        #
        # Length sorting happens HERE, across the whole unit, rather than
        # inside generate_batch: sorting per-batch would only homogenise
        # within an already-arbitrary group and lose most of the padding
        # saving. The consequence is that rows are emitted in
        # LENGTH-SORTED order, not query order. That is safe — every row
        # carries query_id and every downstream consumer (analyse,
        # aggregate, the significance diagnostic) parses per line into
        # dicts and never depends on file order. Sorting is by
        # n_input_tokens, already computed by prepare(), so it costs no
        # extra tokenisation.
        lengths = [p.n_input_tokens for p in prepared]
        order, _ = deterministic_batch_order(lengths)

        # BATCH SHAPE. A fixed count must be sized for the worst-case
        # batch, because padding makes the cost n * longest rather than
        # sum(len) — and length sorting, while it cuts total padding
        # waste, CONCENTRATES the longest prompts into one batch, which
        # is precisely the batch that OOMs. Bounding padded tokens
        # instead adapts to raggedness and bounds peak memory by
        # construction, with one knob covering both M4's ~2k prompts and
        # M2/M3/M9's ~4k.
        if self.max_padded_tokens is not None:
            groups = token_budget_batches(
                order, lengths,
                max_padded_tokens=self.max_padded_tokens,
                max_batch_size=self.batch_size,
            )
        else:
            groups = [
                order[s : s + self.batch_size]
                for s in range(0, len(order), self.batch_size)
            ]

        t_b = time.perf_counter()
        n_done = 0
        for idxs in groups:
            t_batch = time.perf_counter()
            answers = generate_batch(
                [prepared[i].system_prompt for i in idxs],
                [prepared[i].user_prompt for i in idxs],
                cfg=system.config.generation,
                batch_size=self.batch_size,
                # Already globally sorted; re-sorting inside the call
                # would be a no-op on a homogeneous group.
                sort_by_length=False,
            )
            batch_s = time.perf_counter() - t_batch
            # Per-query generation time is not observable once batched;
            # amortise it rather than invent a measured-looking number.
            per_query_gen_s = batch_s / max(1, len(idxs))
            for i, ans in zip(idxs, answers):
                ar = system.finish(prepared[i], ans, generate_s=per_query_gen_s)
                yield queries[i], ar, ar.latency_s
            n_done += len(idxs)
            if self.verbose:
                elapsed = time.perf_counter() - t_b
                print(
                    f"  phase_b {n_done}/{len(order)}  "
                    f"{n_done / max(elapsed, 1e-9):.2f} gen-req/s"
                )

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

        already_done = self._existing_query_ids()
        if already_done and self.verbose:
            print(
                f"[eval] resuming: {len(already_done)} queries already in "
                f"{self.output_path.name}, skipping them"
            )
        mode = "a" if (self.resume and already_done) else "w"

        with self.output_path.open(mode, encoding="utf-8") as fout:
            stopped = False
            for unit_idx, unit in enumerate(
                benchmark.iter_eval_units(split=split, max_units=max_units)
            ):
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

                # Select this unit's queries up front so phase A and
                # phase B iterate the same list.
                unit_queries: list[EvalQuery] = []
                for q in unit.queries:
                    if max_queries is not None and (
                        n_queries + len(unit_queries)
                    ) >= max_queries:
                        stopped = True
                        break
                    if q.query_id in already_done:
                        continue
                    unit_queries.append(q)

                for q, ar, latency_s in self._answer_unit(system, unit_queries):

                    # Retrieval scoring is benchmark-specific: each
                    # benchmark implements score_retrieval to combine
                    # set-F1 (CK-2) with any rank-aware metrics it
                    # cares about (MultiHop adds Hit@K / MAP@K / MRR).
                    # Independent of CK-4 packing — reads ar.retrieved
                    # (full ranking).
                    self._check_output_length(
                        ar.answer, q.query_id, system.config.generation.model
                    )
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
                        # Loader-provided query metadata merged with any
                        # per-query system diagnostics the AnswerResult
                        # carried (e.g. M9's corrective-action logging).
                        # System keys are namespaced by convention
                        # ("m9_*") so loader metadata can't collide.
                        metadata={**q.metadata, **(ar.extra or {})},
                    )
                    fout.write(
                        json.dumps(asdict(scored), ensure_ascii=False) + "\n"
                    )
                    fout.flush()
                    n_queries += 1
                    yield scored

                n_units += 1
                # Stop check at the BOTTOM of the body: breaking here
                # (not at the top) means the max-queries stop never
                # pulls another unit from the loader's generator. A
                # top-of-loop check advances the generator first, so
                # the loader builds one extra unit that is immediately
                # discarded — inflating loader-side benchmark_stats
                # (n_stories / n_queries) past what was actually
                # processed. The runner summary's n_queries_scored is
                # authoritative either way.
                if stopped:
                    break

        if self.verbose:
            elapsed = time.perf_counter() - t_start
            print(
                f"[eval] done: {n_units} units, {n_queries} queries, "
                f"elapsed={elapsed:.1f}s, output={self.output_path}"
            )


__all__ = ["Benchmark", "BenchmarkRunner"]
