"""Benchmark protocol + BenchmarkRunner orchestration.

A `Benchmark` exposes:
  * `name` — string id used in output filenames.
  * `iter_eval_units(split)` — generator of `EvalUnit`s (one shared
    corpus for MultiHop-RAG, one per story for NarrativeQA, one per
    question for HotpotQA-distractor, one per shard for the pooled
    variant).
  * `score_answer(predicted, query)` — benchmark-specific scoring of
    the predicted answer text against the query's gold annotations.

`BenchmarkRunner` drives one (system, benchmark, split) pass:

  for unit in benchmark.iter_eval_units(split):
      todo = [q for q in unit.queries if q.query_id not in already_done]
      if not todo:
          continue            # NOT indexed - see run()
      system.index_items(unit.corpus)
      for q in todo:
          ar = system.answer(q.question_text)
          retr = score_retrieval_ck2(ar.retrieved, q.gold_passage_sets)
          ans  = benchmark.score_answer(ar.answer, q)
          yield ScoredQuery(...)

Each ScoredQuery is written as one JSONL line. The driver is
deliberately simple: one process, one system, one benchmark, one
split. Parallelism across systems/benchmarks happens at the CLI runner
level by sharding (run M2 + multihop_rag, M3 + multihop_rag, ...
separately).
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
        scoring_ranking: list[RetrievedChunk] | None = None,
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
        resume: bool = False,
        require_cold_tree: bool = False,
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
        # P10's cold-tree rule, enforced rather than remembered. A warm
        # M4 substrate may have been built under a different topology
        # stack, and nothing in the output says which — so a matrix can
        # end up holding two tree populations with no error anywhere.
        # This session measured the same story three times and twice
        # served a cache read while reporting a build; `probe_cell_costs`
        # aborts on exactly this and the runner did not.
        self.require_cold_tree = require_cold_tree

    def _check_output_length(self, answer: str, query_id: str, cap: int) -> None:
        """Abort if a generated answer overran the configured cap.

        The cap is `GenerationConfig.max_new_tokens` (512 on every matrix
        cell), read from the system's config at the call site so it is
        the value generation actually consumed. Before the repo reduction
        this ran only when a `--max-new-tokens` override was passed —
        which no banked cell did — so it now runs on every answer as the
        fixed-cap check.

        Counted with the harness-wide tiktoken counter, NOT the
        generator's own tokenizer. An earlier version called
        `load_generator` for its tokenizer, which pulls ~15 GB of weights
        into memory purely to count tokens — free when the generator is
        already resident, and a surprise mid-run load when it is not
        (retrieval-only passes, stubbed systems, any CPU host).

        The cost of the substitution is that tiktoken and Qwen's BPE
        disagree by roughly 10-20% on the same text, so the tolerance is
        `cap * 1.25 + 2` rather than an exact bound. That is ample for
        the failure this exists to catch, which is a cap of 1 emitting a
        hundred tokens; it is deliberately NOT a precise assertion about
        generation length.
        """
        from ..prompt_packing import count_tokens

        n = count_tokens(answer, allow_special=True)
        if n > cap * 1.25 + 2:
            raise RuntimeError(
                f"generation cap NOT APPLIED: query {query_id!r} returned "
                f"~{n} tokens against max_new_tokens={cap}. The run is "
                "measuring something other than what it claims. Answer "
                f"began: {answer[:80]!r}"
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

        SEQUENTIAL, by construction. Answer generation was measured
        faster sequentially than batched on the 512-token answer path
        (M2 x MultiHop, 64 queries, L4: 4.2558 s/query against 5.1654 at
        the best batched cap), because a batch runs until its LONGEST
        member stops; every banked cell answered sequentially. M4's tree
        summaries batch separately through `models.generate_batch`, where
        the 100-token cap makes batching the right call.
        """
        for q in queries:
            t_q = time.perf_counter()
            ar = system.answer(q.question_text)
            yield q, ar, time.perf_counter() - t_q

    def _cold_tree_preflight(
        self, system, units: list, already_done: set
    ) -> None:
        """Scan EVERY unit for a warm substrate before indexing anything.

        WHY THIS IS NOT JUST THE PER-UNIT GATE MOVED EARLIER. The
        per-unit gate aborts on the FIRST warm unit, so discovering N
        warm substrates costs N session starts. On M4/hotpotqa — 1,000
        units with ~50 warm from the query slice — that is up to fifty
        aborts, the worst case in the matrix by a wide margin.

        THE DEEPER REASON, and it is the one that decided the design: the
        runbook carried a hand-maintained list of warm substrates to
        delete. On M4/narrativeqa it named TWO and there were THREE. A
        documented list is a thing that goes stale; an enumeration
        cannot. So the operator is never asked to consult a list again —
        the gate computes the set and prints it.

        Read-only and index-free: `substrate_warm_path` writes the corpus
        layout to a temp dir and hashes it. No embedder, no clustering,
        no summariser, no GPU.

        Units whose queries are all already banked are SKIPPED, matching
        the resume rule downstream — a resumed pass does not index them,
        so a warm substrate there is not a finding.
        """
        if not self.require_cold_tree:
            return
        if not getattr(system, "has_cacheable_substrate", False):
            return

        warm: list[tuple[str, str]] = []
        n_checked = 0
        for unit in units:
            if all(q.query_id in already_done for q in unit.queries):
                continue
            n_checked += 1
            path = system.substrate_warm_path(unit.corpus)
            if path:
                warm.append((str(unit.corpus_id), path))

        if self.verbose:
            print(
                f"[eval] cold-tree preflight: {n_checked} unit(s) checked, "
                f"{len(warm)} warm"
            )
        if not warm:
            return

        listing = "\n".join(f"  {cid}\n    {p}" for cid, p in warm)
        raise SystemExit(
            f"COLD-TREE PREFLIGHT FAILED: {len(warm)} of {n_checked} units "
            "already have a complete substrate on disk.\n"
            "ALL of them are listed here so this costs ONE session start, "
            "not one per warm unit:\n"
            f"{listing}\n"
            "A warm substrate may have been built under a different "
            "topology stack, and nothing in the output records which — so "
            "the matrix could hold two tree populations with no error "
            "anywhere. Delete the directories above and re-run; a "
            "deliberate re-derivation uses a throwaway THESIS_CACHE_DIR, "
            "never the banked cache.\n"
            "Nothing was indexed; no GPU work was done."
        )

    def run(
        self,
        system: BaseSystem,
        benchmark: Benchmark,
        *,
        split: str,
    ) -> Iterator[ScoredQuery]:
        """Drive one (system, benchmark, split) pass to JSONL.

        Always the full cell: the loader draws its declared population
        (NarrativeQA's seeded 40 stories, HotpotQA's registered 1,000
        questions, MultiHop's single unit) and every unit with a query
        not already banked is indexed and answered.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Exposed on the instance so main() can check the resolved
        # population after the pass. A local would leave the count
        # trapped inside a generator.
        self.n_units_processed = 0
        n_units = 0
        # Counted separately rather than folded into n_units: a unit that
        # was skipped without being indexed is not a unit that was
        # processed, and collapsing the two would hide the very saving
        # the ordering exists to produce.
        n_units_skipped = 0
        n_queries = 0
        t_start = time.perf_counter()

        already_done = self._existing_query_ids()
        # (preflight runs below, once `already_done` is known)
        if already_done and self.verbose:
            print(
                f"[eval] resuming: {len(already_done)} queries already in "
                f"{self.output_path.name}, skipping them"
            )
        mode = "a" if (self.resume and already_done) else "w"

        units = list(benchmark.iter_eval_units(split=split))
        self._cold_tree_preflight(system, units, already_done)

        with self.output_path.open(mode, encoding="utf-8") as fout:
            for unit_idx, unit in enumerate(units):
                if self.verbose:
                    print(
                        f"[eval] unit {unit_idx + 1}: corpus_id={unit.corpus_id!r}  "
                        f"n_items={len(unit.corpus)}  n_queries={len(unit.queries)}"
                    )

                # Select this unit's queries up front so phase A and
                # phase B iterate the same list — and, critically, BEFORE
                # indexing. Indexing is work done on behalf of queries;
                # a unit with nothing left to answer must not be indexed
                # at all.
                unit_queries: list[EvalQuery] = [
                    q for q in unit.queries if q.query_id not in already_done
                ]

                # INDEX ORDERING IS LOAD-BEARING, not tidiness. With the
                # index call above this filter, a resumed pass rebuilt
                # the tree for every unit it was about to skip. On a
                # Drive-resident cache that is a cache read; on
                # HotpotQA-A, whose cache lives on session-local disk
                # that dies with the runtime, it is a COLD M4 TREE BUILD
                # per skipped unit — and tree builds are the dominant
                # cost in this harness (one NarrativeQA story has
                # measured 20,691 s). P10 is expected to span several
                # Colab sessions, so resume is the normal path, and the
                # waste would have been paid on every one of them.
                if not unit_queries:
                    if self.verbose:
                        print(
                            f"  no queries outstanding for "
                            f"{unit.corpus_id!r} — NOT indexed"
                        )
                    n_units_skipped += 1
                    continue

                t_index = time.perf_counter()
                system.index_items(unit.corpus)
                index_s = time.perf_counter() - t_index

                # BACKSTOP ONLY since the preflight landed. The preflight
                # enumerates every unit before anything is indexed, so a
                # warm substrate should be impossible here — this catches
                # one that appeared mid-run, or a system whose
                # `substrate_warm_path` disagrees with what `index()`
                # actually did.
                if self.require_cold_tree and getattr(
                    system, "tree_cache_hit", None
                ):
                    raise SystemExit(
                        "COLD-TREE GATE FAILED: tree_cache_hit=True on "
                        f"unit {unit.corpus_id!r} after {index_s:.2f}s. "
                        "A warm substrate may have been built under a "
                        "different topology stack, and nothing in the "
                        "output records which — so the matrix could hold "
                        "two tree populations with no error anywhere. "
                        "Delete that substrate directory and re-run; a "
                        "deliberate re-derivation uses a throwaway "
                        "THESIS_CACHE_DIR, never the banked cache."
                    )

                if self.verbose:
                    print(f"  index_s={index_s:.2f}")

                for q, ar, latency_s in self._answer_unit(system, unit_queries):

                    # Retrieval scoring is benchmark-specific: each
                    # benchmark implements score_retrieval to combine
                    # set-F1 (CK-2) with any rank-aware metrics it
                    # cares about (MultiHop adds Hit@K / MAP@K / MRR).
                    # Independent of CK-4 packing — reads ar.retrieved
                    # (full ranking).
                    self._check_output_length(
                        ar.answer, q.query_id,
                        system.config.generation.max_new_tokens,
                    )
                    retr = benchmark.score_retrieval(
                        ar.retrieved, q,
                        scoring_ranking=getattr(
                            ar, "scoring_ranking", None) or None,
                    )
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
                        # carried (M4's non-leaf share, budget use,
                        # degenerate-tree flag). System keys are
                        # namespaced by convention ("m4_*") so loader
                        # metadata can't collide.
                        #
                        # packed_ids: the IDENTITY of the packed set, in
                        # prompt order. Added 2026-08-24 after a per-row
                        # M2-vs-M3 set comparison turned out to be
                        # UNRECOVERABLE from banked rows — they carried
                        # counts and unit-type distributions, but the ids
                        # existed only at run time and nothing recorded
                        # them (the recurring lesson, again). Eval-time
                        # metadata only; NOT in any cache key.
                        metadata={
                            **q.metadata,
                            **(ar.extra or {}),
                            "packed_ids": [
                                r.chunk.chunk_id for r in ar.packed
                            ],
                        },
                    )
                    fout.write(
                        json.dumps(asdict(scored), ensure_ascii=False) + "\n"
                    )
                    fout.flush()
                    n_queries += 1
                    yield scored

                n_units += 1
                self.n_units_processed = n_units

        if self.verbose:
            elapsed = time.perf_counter() - t_start
            skipped = (
                f", {n_units_skipped} units skipped un-indexed"
                if n_units_skipped
                else ""
            )
            print(
                f"[eval] done: {n_units} units, {n_queries} queries"
                f"{skipped}, elapsed={elapsed:.1f}s, "
                f"output={self.output_path}"
            )


__all__ = ["Benchmark", "BenchmarkRunner"]
