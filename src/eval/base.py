"""Benchmark protocol and the runner that drives one (system, benchmark,
split) pass, writing one ScoredQuery per JSONL line."""

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
    """What every benchmark loader provides: units and two scorers."""

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
    """Streams one (system, benchmark, split) pass to JSONL, unit by unit."""

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
        # resume appends to the existing JSONL and skips banked query_ids;
        # without it a re-run truncates the file.
        self.resume = resume
        # require_cold_tree refuses to run over a substrate already on disk,
        # so one cell never mixes trees built under different stacks.
        self.require_cold_tree = require_cold_tree

    def _existing_query_ids(self) -> set[str]:
        """Return the query_ids already in the output file when resuming."""
        if not self.resume or not self.output_path.exists():
            return set()
        done: set[str] = set()
        # A torn final line is skipped, so its query is answered again.
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return done

    def _answer_unit(
        self,
        system: BaseSystem,
        queries: list[EvalQuery],
    ) -> Iterator[tuple[EvalQuery, "AnswerResult", float]]:
        """Yield (query, AnswerResult, latency_s) for one unit in order."""
        for q in queries:
            t_q = time.perf_counter()
            ar = system.answer(q.question_text)
            yield q, ar, time.perf_counter() - t_q

    def _cold_tree_preflight(
        self, system, units: list, already_done: set
    ) -> None:
        """Abort before indexing if any unit left to answer is already warm."""
        if not self.require_cold_tree:
            return
        if not getattr(system, "has_cacheable_substrate", False):
            return

        # Check every unit with outstanding queries, so all warm substrates
        # are listed in one go. substrate_warm_path only hashes the corpus
        # layout: no embedder, no clustering, no GPU.
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

        # Name every warm directory so they can all be deleted at once.
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
        """Index and answer each unit with unbanked queries; yield each row."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # n_units_processed lives on the instance so main() can check the
        # population after the pass; skipped units are counted apart.
        self.n_units_processed = 0
        n_units = 0
        n_units_skipped = 0
        n_queries = 0
        t_start = time.perf_counter()

        # Work out what is already banked and open the file accordingly.
        already_done = self._existing_query_ids()
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

                # Filter the unit's queries before indexing: a unit with
                # nothing left to answer is skipped without building anything.
                unit_queries: list[EvalQuery] = [
                    q for q in unit.queries if q.query_id not in already_done
                ]

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

                # Backstop to the preflight: catch a substrate that turned
                # warm mid-run or a system whose warm check disagrees with
                # what index_items did.
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

                    # Score retrieval: set-F1 over the reader context,
                    # rank-aware metrics over the scoring ranking; then the
                    # answer.
                    # harness choice: one scoring depth for every system (METHODS §D)
                    retr = benchmark.score_retrieval(
                        ar.retrieved, q,
                        scoring_ranking=getattr(
                            ar, "scoring_ranking", None) or None,
                    )
                    ans = benchmark.score_answer(ar.answer, q)

                    # Count source unit types over retrieved and packed sets.
                    retrieved_unit_types: dict[str, int] = {}
                    for r in ar.retrieved:
                        ut = getattr(r, "source_unit_type", "chunk")
                        retrieved_unit_types[ut] = retrieved_unit_types.get(ut, 0) + 1
                    packed_unit_types: dict[str, int] = {}
                    for r in ar.packed:
                        ut = getattr(r, "source_unit_type", "chunk")
                        packed_unit_types[ut] = packed_unit_types.get(ut, 0) + 1

                    # Write the row and flush, so a killed session loses at
                    # most the in-flight query.
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
                        # metadata = loader query metadata + the system's
                        # per-query diagnostics (namespaced, e.g. "m4_*") +
                        # packed_ids, the packed chunk ids in prompt order.
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

        # Final summary line for the log.
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
