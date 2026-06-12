"""QuALITY benchmark loader + multiple-choice accuracy scoring.

QuALITY (Pang et al., NAACL 2022): multiple-choice reading
comprehension over long articles (~5k tokens; Project Gutenberg, Open
American National Corpus and others), four options per question,
`gold_label` = 1-indexed majority vote of untimed annotators. Loaded
from the OFFICIAL nyu-mll release zip, version-pinned (v1.0.1,
htmlstripped variant) — no third-party HF mirror drift, no
dataset-script deprecation risk.

SPLIT MAPPING (goes verbatim into the thesis methodology):

    harness --split validation  ->  QuALITY train  (~2,523 q / ~300 articles)
    harness --split test        ->  QuALITY dev    (~2,086 q / ~230 articles)

QuALITY's own test split has NO public labels (leaderboard-held) and
the loader refuses it. RESERVE DISCIPLINE: harness-test (= QuALITY
dev) is the reserved final-numbers set; ALL development and the
20-article small-sample gate happen on harness-validation (= QuALITY
train); no peeking — same discipline as QASPER test. dev is the
largest fully-labeled held-out set, so it takes the test role; train
questions are written and validated identically, so developing on a
train slice loses nothing. Each query records its underlying QuALITY
split in metadata["quality_split"] so the JSONL is self-describing.

CORPUS SHAPE: one EvalUnit per article (QASPER pattern). Articles can
carry multiple question sets (set_unique_id); records are grouped by
article_id and the questions of every set merge into one unit so no
article indexes twice. One CorpusItem per article (span_id "<whole>",
MultiHop convention); the chunker splits the long article downstream.

NO GOLD PASSAGES -> answer-only benchmark: score_retrieval returns
RetrievalScore(skipped=True) for every query (the analyser already
excludes skipped rows from retrieval means and counts them); no CK-2,
no rank-aware metrics.

QUERY FORMAT: the four options plus a letter instruction are embedded
in the query string — the only channel that reaches every system
identically through the shared answer path. Consequences: retrieval
queries include option text (standard for MC-RAG; options carry
retrieval signal; uniform across systems = fair), and M1 closed-book
receives question+options with no article = the generator prior, as
intended.

M7 NOTE-FOR-LATER: M7's intent decomposition will see an MC-shaped
query (four options + instruction line) — a materially different
input shape than the bare questions of other benchmarks. Fair
(identical input for all systems), but if M7 looks anomalous on
QuALITY specifically, check its decomposition logs before concluding
anything.

HARD SUBSET: question_type = "hard" | "easy" from the per-question
`difficult` flag (1 = fewer than half of the timed speed-validation
annotators answered correctly), so the existing --by-type slicing
works unchanged.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable

from .. import paths
from ..retrievers.base import RetrievedChunk
from .scorers import score_multiple_choice
from .types import (
    ANSWER_TYPE_MULTIPLE_CHOICE,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)


QUALITY_VERSION = "v1.0.1"
# Official release zip in the nyu-mll/quality GitHub repo. ~18 MB.
ZIP_URL = (
    "https://github.com/nyu-mll/quality/raw/main/data/"
    f"{QUALITY_VERSION}/QuALITY.{QUALITY_VERSION}.zip"
)

# harness split -> QuALITY split. QuALITY's own test split has no
# public labels (leaderboard-held) and is NOT mapped.
SPLIT_MAP = {"validation": "train", "test": "dev"}

LETTERS = "ABCD"

QUERY_TEMPLATE = (
    "{question}\n"
    "\n"
    "Options:\n"
    "A) {opt_a}\n"
    "B) {opt_b}\n"
    "C) {opt_c}\n"
    "D) {opt_d}\n"
    "\n"
    "Answer with the letter (A, B, C, or D) of the correct option."
)


def _data_dir() -> Path:
    return paths.input_dir() / f"quality_{QUALITY_VERSION}"


def _ensure_downloaded() -> Path:
    """Download + extract the official zip once; idempotent thereafter.

    The extracted files persist under the (Drive-backed on Colab)
    input dir, so later sessions skip the download entirely.
    """
    data_dir = _data_dir()
    if list(data_dir.glob("*htmlstripped*")):
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"QuALITY.{QUALITY_VERSION}.zip"
    if not zip_path.exists():
        import urllib.request

        print(f"[quality] downloading {ZIP_URL} -> {zip_path}")
        urllib.request.urlretrieve(ZIP_URL, zip_path)  # noqa: S310 — pinned https URL
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    if not list(data_dir.glob("*htmlstripped*")):
        raise RuntimeError(
            f"[quality] zip extracted but no htmlstripped files found under "
            f"{data_dir} — release layout changed? Contents: "
            f"{[p.name for p in data_dir.iterdir()]}"
        )
    return data_dir


def _split_file(quality_split: str) -> Path:
    data_dir = _ensure_downloaded()
    matches = sorted(data_dir.glob(f"*htmlstripped.{quality_split}"))
    if not matches:
        raise FileNotFoundError(
            f"[quality] no htmlstripped.{quality_split} file under {data_dir}; "
            f"contents: {[p.name for p in data_dir.iterdir()]}"
        )
    return matches[0]


def format_query(question: str, options: list[str]) -> str:
    if len(options) != 4:
        raise ValueError(f"QuALITY question must have 4 options; got {len(options)}")
    return QUERY_TEMPLATE.format(
        question=question.strip(),
        opt_a=options[0],
        opt_b=options[1],
        opt_c=options[2],
        opt_d=options[3],
    )


class QualityBenchmark:
    """Iterable over QuALITY EvalUnits + multiple-choice accuracy scorer."""

    name = "quality"

    def __init__(self) -> None:
        self._records_cache: dict[str, list[dict]] = {}
        self.stats: dict[str, int] = {
            "n_articles": 0,
            "n_question_sets": 0,
            "n_queries": 0,
            "n_hard": 0,
        }

    def _load_records(self, quality_split: str) -> list[dict]:
        if quality_split not in self._records_cache:
            path = _split_file(quality_split)
            records: list[dict] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            self._records_cache[quality_split] = records
        return self._records_cache[quality_split]

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        quality_split = SPLIT_MAP.get(split)
        if quality_split is None:
            raise ValueError(
                f"QuALITY split must be one of {sorted(SPLIT_MAP)} "
                f"(harness validation -> QuALITY train, harness test -> "
                f"QuALITY dev); got {split!r}. QuALITY's own test split has "
                "no public labels (leaderboard-held) and cannot be scored."
            )
        records = self._load_records(quality_split)

        # Group records by article_id, merging question sets so each
        # article indexes exactly once.
        by_article: dict[str, list[dict]] = {}
        order: list[str] = []
        for rec in records:
            aid = rec["article_id"]
            if aid not in by_article:
                by_article[aid] = []
                order.append(aid)
            by_article[aid].append(rec)

        n_questions = sum(
            len(rec.get("questions") or ()) for rec in records
        )
        print(
            f"[quality] split mapping: harness {split!r} -> QuALITY "
            f"{quality_split!r}; {len(order)} articles, "
            f"{len(records)} question sets, {n_questions} questions"
        )

        for n_articles, aid in enumerate(order):
            if max_units is not None and n_articles >= max_units:
                break
            sets = by_article[aid]
            first = sets[0]

            corpus_items = (
                CorpusItem(
                    item_id=f"{aid}::<whole>",
                    parent_id=aid,
                    span_id="<whole>",
                    text=first.get("article") or "",
                    metadata={
                        "title": first.get("title") or "",
                        "source": first.get("source") or "",
                        "year": first.get("year") or "",
                    },
                ),
            )

            queries: list[EvalQuery] = []
            for rec in sets:
                set_id = rec.get("set_unique_id") or "set0"
                self.stats["n_question_sets"] += 1
                for q_idx, q in enumerate(rec.get("questions") or ()):
                    options = list(q.get("options") or ())
                    gold_label = int(q.get("gold_label") or 0)
                    difficult = int(q.get("difficult") or 0)
                    if difficult:
                        self.stats["n_hard"] += 1
                    gold_text = (
                        options[gold_label - 1]
                        if 1 <= gold_label <= len(options)
                        else ""
                    )
                    queries.append(
                        EvalQuery(
                            query_id=f"{aid}::{set_id}::q{q_idx}",
                            question_text=format_query(q.get("question") or "", options),
                            parent_scope=aid,
                            gold_answers=(
                                GoldAnswer(
                                    answer_type=ANSWER_TYPE_MULTIPLE_CHOICE,
                                    free_form=gold_text,
                                ),
                            ),
                            gold_passage_sets=(),  # no gold passages in QuALITY
                            question_type="hard" if difficult else "easy",
                            metadata={
                                "quality_split": quality_split,
                                "gold_label": gold_label,
                                "options": options,
                                "writer_label": q.get("writer_label"),
                                "difficult": difficult,
                                "set_unique_id": set_id,
                            },
                        )
                    )
                    self.stats["n_queries"] += 1

            self.stats["n_articles"] += 1
            yield EvalUnit(
                corpus_id=aid,
                corpus=corpus_items,
                queries=tuple(queries),
            )

    def score_retrieval(
        self,
        retrieved: list[RetrievedChunk],
        query: EvalQuery,
    ) -> RetrievalScore:
        """QuALITY has no gold passages — retrieval is never scored.

        skipped=True rows are excluded from the analyser's retrieval
        means and surface in its retr_n_skipped counter (same handling
        as QASPER's empty-gold queries).
        """
        del retrieved, query
        return RetrievalScore(skipped=True)

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """Accuracy via the multiple-choice extractor (single annotator)."""
        options = list(query.metadata.get("options") or ())
        gold_label = int(query.metadata.get("gold_label") or 0)
        value, method, md = score_multiple_choice(predicted, options, gold_label)
        return AnswerScore(
            value=value,
            method=method,
            per_annotator=(value,),
            metadata=md,
        )


__all__ = ["QualityBenchmark", "SPLIT_MAP", "ZIP_URL", "format_query"]
