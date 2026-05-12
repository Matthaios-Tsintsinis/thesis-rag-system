"""Benchmark harness skeleton.

Runs one or more systems on one or more benchmarks and writes per-question
records to JSONL for downstream scoring. Evaluation (RAGAS, dataset-native
metrics) lives in evaluation.py and consumes these JSONL files.

A Benchmark is anything with `name`, `corpus_path`, and an iterable of
QuestionRecord(question_id, question, references). Concrete benchmark
loaders for NarrativeQA, QASPER, QuALITY, MultiHop-RAG, CRAG land
under benchmarks/ in later steps.

Output destination is resolved via src.paths so the same code writes
to Drive on Colab when mounted, /content/thesis_rag/outputs on Colab
without Drive, and <repo>/local_runs/outputs locally.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from . import paths
from .config import DEFAULT_CONFIG, HarnessConfig
from .retrievers.base import AnswerResult, BaseSystem


@dataclass
class QuestionRecord:
    question_id: str
    question: str
    references: list[str] = field(default_factory=list)
    question_type: str | None = None
    metadata: dict = field(default_factory=dict)


class Benchmark(Protocol):
    name: str
    corpus_path: Path

    def questions(self) -> Iterable[QuestionRecord]: ...


@dataclass
class RunRecord:
    system_id: str
    benchmark: str
    question_id: str
    question: str
    answer: str
    references: list[str]
    question_type: str | None
    retrieved_chunk_ids: list[str]
    retrieved_texts: list[str]
    retrieval_scores: list[float]
    latency_s: float
    n_retrieval_calls: int
    extra: dict


def _record(
    system_id: str,
    benchmark: str,
    q: QuestionRecord,
    ar: AnswerResult,
) -> RunRecord:
    return RunRecord(
        system_id=system_id,
        benchmark=benchmark,
        question_id=q.question_id,
        question=q.question,
        answer=ar.answer,
        references=q.references,
        question_type=q.question_type,
        retrieved_chunk_ids=[r.chunk.chunk_id for r in ar.retrieved],
        retrieved_texts=[r.chunk.text for r in ar.retrieved],
        retrieval_scores=[r.score for r in ar.retrieved],
        latency_s=ar.latency_s,
        n_retrieval_calls=ar.n_retrieval_calls,
        extra=ar.extra,
    )


def run(
    system: BaseSystem,
    benchmark: Benchmark,
    output_dir: Path | None = None,
    limit: int | None = None,
    config: HarnessConfig = DEFAULT_CONFIG,
) -> Path:
    out_root = Path(output_dir) if output_dir is not None else paths.output_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_root / f"{benchmark.name}__{system.system_id}__{stamp}.jsonl"

    print(f"[harness] indexing {system.system_id} on {benchmark.corpus_path}")
    system.index(benchmark.corpus_path)

    print(f"[harness] answering questions → {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for n, q in enumerate(benchmark.questions()):
            if limit is not None and n >= limit:
                break
            ar = system.answer(q.question)
            rec = _record(system.system_id, benchmark.name, q, ar)
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"  [{system.system_id}|{benchmark.name}] q={q.question_id} "
                f"lat={ar.latency_s:.2f}s retrieved={len(ar.retrieved)}"
            )

    return out_path


def run_matrix(
    systems: list[BaseSystem],
    benchmarks: list[Benchmark],
    output_dir: Path | None = None,
    limit: int | None = None,
    config: HarnessConfig = DEFAULT_CONFIG,
) -> list[Path]:
    paths: list[Path] = []
    for bench in benchmarks:
        for system in systems:
            paths.append(run(system, bench, output_dir, limit, config))
    return paths


@dataclass
class JSONLBenchmark:
    """Minimal benchmark loader: corpus folder + questions.json file."""

    name: str
    corpus_path: Path
    questions_path: Path

    def questions(self) -> Iterable[QuestionRecord]:
        data = json.loads(Path(self.questions_path).read_text(encoding="utf-8"))
        for q in data:
            yield QuestionRecord(
                question_id=q["question_id"],
                question=q["question"],
                references=q.get("references", []),
                question_type=q.get("question_type"),
                metadata=q.get("metadata", {}),
            )
