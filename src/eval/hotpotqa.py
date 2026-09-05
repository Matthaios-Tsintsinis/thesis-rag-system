"""HotpotQA loaders: the distractor setting (one corpus per question) and
a pooled variant (the paragraphs of a shard of questions in one corpus).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace as dc_replace
from typing import Any, Iterable

from ..retrievers.base import RetrievedChunk
from .alignment import score_retrieval_ck2, score_retrieval_rank_aware
from .sampling import SUBSAMPLE_SEED, subsample_indices
from .scorers.extractive import (
    assert_gold_not_empty,
    normalize_qasper_answer,
    token_f1,
)
from .scorers.unanswerable import is_abstention
from .types import (
    ANSWER_TYPE_FREE_FORM,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)


# dataset: hotpotqa/hotpot_qa distractor, validation (7,405); seeded 1,000
# The repo id must be namespace/name. The default revision is native
# Parquet and loads without a loader script; do not pin another one.
HF_REPO = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"

# Test labels are not public, so every harness split name maps onto the
# validation split.
VALID_SPLITS = ("validation", "test", "dev")
_HF_SPLIT = "validation"

# Pooled shard size in questions; ~700-900 distinct paragraphs per shard
# after title dedup, enough for a multi-layer tree.
# harness choice: our construction, not comparable to published HotpotQA (METHODS §B.4)
SHARD_QUESTIONS = 100

# Sample size; the constructor default, so a bare run gets the seeded draw.
# A random draw, not the head of the file: dev rows can be grouped by type
# and level. Seed and sampler come from src/eval/sampling.py, shared with
# NarrativeQA. max_questions=None loads the full 7,405-question split.
# harness choice: preregistered seed (METHODS §B)
PREREGISTERED_Q = 1000

# Rank-aware K grid at title level.
# harness choice: two gold titles, Hit@2 is the headline (METHODS §B.3)
RANK_K_VALUES = (1, 2, 5, 10)

# Span id for title-level projection in rank-aware scoring; never a real
# sentence id.
_TITLE_SPAN = "<title>"

# The official yes/no/noanswer sentinels. They live here, not in the
# shared scorer module, so MultiHop and NarrativeQA (which import the
# shared token_f1) cannot reach the guard.
# official: hotpot_evaluate_v1.py::f1_score @ 36358534 (both early returns)
_HOTPOT_SENTINELS = ("yes", "no", "noanswer")


def hotpot_token_f1(predicted: str, gold: str) -> float:
    """Token-F1 with the official yes/no/noanswer guard; F1 only."""
    return hotpot_token_f1_prf(predicted, gold)[0]


def hotpot_token_f1_prf(predicted: str, gold: str) -> tuple[float, float, float]:
    """Return (f1, precision, recall) with the official yes/no guard."""
    # A sentinel on either side that does not match zeroes all three.
    # official: hotpot_evaluate_v1.py::normalize_answer @ 36358534
    # official: hotpot_evaluate_v1.py::f1_score @ 36358534 (both early returns)
    np_ = normalize_qasper_answer(predicted)
    ng_ = normalize_qasper_answer(gold)
    if np_ != ng_ and (np_ in _HOTPOT_SENTINELS or ng_ in _HOTPOT_SENTINELS):
        return 0.0, 0.0, 0.0

    # Empty side: agree only when both are empty.
    # SQuAD 2.0 evaluate-v2.0.py rule; unreachable, loaders refuse empty gold
    pred_tokens = np_.split()
    gold_tokens = ng_.split()
    if not pred_tokens or not gold_tokens:
        agree = float(pred_tokens == gold_tokens)
        return agree, agree, agree
    # P = shared/pred, R = shared/gold, F1 = 2PR/(P+R).
    # official: hotpot_evaluate_v1.py::f1_score @ 36358534
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0, 0.0, 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1, precision, recall


def _sentence_items(context: Any) -> list[CorpusItem]:
    """One CorpusItem per non-empty sentence, parented by its title."""
    # Sentence items carry provenance; index_items reassembles each title's
    # sentences into the paragraph ahead of chunking, so chunks stay
    # paragraph-sized.
    # harness choice: supporting facts are (title, sentence) pairs (METHODS §B.3)
    titles = list(context["title"])
    sentence_lists = list(context["sentences"])
    items: list[CorpusItem] = []
    for title, sentences in zip(titles, sentence_lists):
        for idx, sentence in enumerate(sentences):
            text = (sentence or "").strip()
            if not text:
                continue
            items.append(CorpusItem(
                item_id=f"{title}::sent{idx}",
                parent_id=str(title),
                span_id=f"sent{idx}",
                text=text,
                metadata={"title": str(title), "sent_id": idx},
            ))
    return items


def _gold_atoms(row: Any) -> frozenset[tuple[str, str]]:
    """Gold (title, sentN) atoms from the row's supporting facts."""
    sf = row["supporting_facts"]
    return frozenset(
        (str(t), f"sent{int(i)}")
        for t, i in zip(sf["title"], sf["sent_id"])
    )


def _query(row: Any, variant: str, parent_scope: str | None) -> EvalQuery:
    """Build the EvalQuery for one row."""
    answer = (row.get("answer") or "").strip()
    # HotpotQA has no unanswerable questions, so every gold must survive
    # normalisation.
    assert_gold_not_empty(query_id=str(row["id"]), gold=answer,
                          benchmark="hotpotqa")
    atoms = _gold_atoms(row)
    return EvalQuery(
        query_id=str(row["id"]),
        question_text=(row.get("question") or "").strip(),
        parent_scope=parent_scope,
        gold_answers=(GoldAnswer(
            answer_type=ANSWER_TYPE_FREE_FORM, free_form=answer),),
        gold_passage_sets=(atoms,),
        # bridge | comparison is the benchmark's own split, the one slice
        # reported separately.
        question_type=str(row.get("type") or "unknown"),
        metadata={
            "level": str(row.get("level") or "unknown"),
            "variant": variant,
            "n_supporting_sentences": len(atoms),
            "n_gold_titles": len({t for t, _ in atoms}),
        },
    )


def _project_to_titles(
    retrieved: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Copy each chunk with provenance rewritten to (title, <title>) atoms."""
    # The document unit is the paragraph title. Copies, not in-place: the
    # chunks are live index state that set-F1 scores on the same objects.
    # harness choice: document-level metrics (METHODS §C.5)
    out: list[RetrievedChunk] = []
    for r in retrieved:
        titles = {parent for parent, _ in (r.chunk.gold_provenance or ())}
        projected = tuple((t, _TITLE_SPAN) for t in sorted(titles))
        out.append(dc_replace(
            r, chunk=dc_replace(r.chunk, gold_provenance=projected)))
    return out


class HotpotQABenchmark:
    """Distractor setting: one corpus of ten paragraphs per question."""

    name = "hotpotqa"

    @property
    def cell_units(self) -> int | None:
        """One corpus per question, so units equal questions."""
        return self.max_questions
    variant = "distractor"

    def __init__(self, max_questions: int | None = PREREGISTERED_Q) -> None:
        """Set the sample size and the empty stats block."""
        self._rows: Any = None
        self.max_questions = max_questions
        self.stats: dict[str, Any] = {
            "n_questions": 0,
            # Every loader records "n_queries"; the run summary reads it as
            # expected_n_queries.
            "n_queries": 0,
            "n_units": 0,
            "n_bridge": 0,
            "n_comparison": 0,
            # Measured on first load; both feed the tree-size arithmetic.
            "mean_paragraph_tokens": None,
            "n_distinct_titles": 0,
        }

    # --- preflight -------------------------------------------------------

    def preflight(self) -> None:
        """Resolve the repo id and config by metadata call; no rows load."""
        from datasets import get_dataset_config_names

        try:
            configs = get_dataset_config_names(HF_REPO)
        except Exception as e:
            raise RuntimeError(
                f"HotpotQA preflight FAILED: cannot resolve dataset "
                f"{HF_REPO!r} ({type(e).__name__}: {e}). "
                "The repo id must be namespace/name — the bare legacy name "
                "'hotpot_qa' is rejected by modern huggingface_hub. Do NOT "
                "pin an older revision as a workaround: pre-2025 revisions "
                "are loader-script based and fetch from curtis.ml.cmu.edu, "
                "which is offline."
            ) from e
        if HF_CONFIG not in configs:
            raise RuntimeError(
                f"HotpotQA preflight FAILED: config {HF_CONFIG!r} not in "
                f"{sorted(configs)} for {HF_REPO!r}."
            )
        print(f"[hotpotqa/{self.variant}] preflight OK: {HF_REPO} "
              f"config={HF_CONFIG}")

    # --- loading ---------------------------------------------------------

    def _load(self) -> Any:
        """Load the validation split once, draw the sample and measure it."""
        if self._rows is None:
            from datasets import load_dataset

            # Seeded draw of max_questions rows from the default revision.
            # harness choice: preregistered seed (METHODS §B)
            ds = load_dataset(HF_REPO, HF_CONFIG, split=_HF_SPLIT)
            if self.max_questions is not None and self.max_questions < len(ds):
                ds = ds.select(subsample_indices(len(ds), self.max_questions))
            self._rows = ds
            self._measure(ds)
        return self._rows

    def _measure(self, ds: Any) -> None:
        """Measure paragraph tokens, distinct titles and the type split."""
        from ..prompt_packing import count_tokens

        # Token mean over the first 200 questions; titles over every row.
        titles: set[str] = set()
        tok_total = n_paras = 0
        for row in ds.select(range(min(200, len(ds)))):
            ctx = row["context"]
            for title, sentences in zip(ctx["title"], ctx["sentences"]):
                titles.add(str(title))
                tok_total += count_tokens(" ".join(sentences))
                n_paras += 1
        for row in ds:
            titles.update(str(t) for t in row["context"]["title"])
        self.stats["mean_paragraph_tokens"] = (
            round(tok_total / n_paras, 1) if n_paras else None)
        self.stats["n_distinct_titles"] = len(titles)
        self.stats["n_questions"] = len(ds)
        self.stats["n_queries"] = len(ds)

        # Realised type/level split. The draw is random, not stratified, so
        # a lopsided draw shows up here. type (bridge / comparison) is the
        # axis that matters; level is all hard on the dev split and is
        # printed only to show that.
        from collections import Counter

        types = Counter(str(r.get("type") or "unknown") for r in ds)
        levels = Counter(str(r.get("level") or "unknown") for r in ds)
        self.stats["type_distribution"] = dict(types)
        self.stats["level_distribution"] = dict(levels)
        self.stats["subsample_seed"] = (
            SUBSAMPLE_SEED if self.max_questions is not None else None)

        # Print the measurements once.
        n = max(1, len(ds))
        fmt = lambda c: ", ".join(  # noqa: E731
            f"{k}={v} ({v / n:.1%})" for k, v in sorted(c.items()))
        print(
            f"[hotpotqa/{self.variant}] {len(ds)} questions, "
            f"{len(titles)} distinct titles, mean paragraph "
            f"{self.stats['mean_paragraph_tokens']} tokens "
            f"(token mean sampled over 200 questions)"
        )
        if self.max_questions is not None:
            print(f"[hotpotqa/{self.variant}] SEEDED RANDOM subsample, "
                  f"seed={SUBSAMPLE_SEED} (not the head of the file)")
        print(f"[hotpotqa/{self.variant}] type:  {fmt(types)}")
        print(f"[hotpotqa/{self.variant}] level: {fmt(levels)}")

    # --- units -----------------------------------------------------------

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        """Yield one EvalUnit per question, capped by max_units."""
        if split not in VALID_SPLITS:
            raise ValueError(
                f"HotpotQA split must be one of {VALID_SPLITS} (test labels "
                f"are not public, so all map to {_HF_SPLIT!r}); got {split!r}"
            )
        ds = self._load()
        # The corpus is the question's own ten paragraphs; skip a row with
        # no non-empty sentence.
        # HotpotQA paper §2: 2 gold + 8 TF-IDF distractor paragraphs
        for n_done, row in enumerate(ds):
            if max_units is not None and n_done >= max_units:
                break
            items = _sentence_items(row["context"])
            if not items:
                continue
            self._count(row)
            self.stats["n_units"] += 1
            yield EvalUnit(
                corpus_id=f"hotpot::{row['id']}",
                corpus=tuple(items),
                # parent_scope stays None: the corpus is already this
                # question's candidate set.
                queries=(_query(row, self.variant, None),),
            )

    def _count(self, row: Any) -> None:
        """Tally the row's question type."""
        t = str(row.get("type") or "")
        if t == "bridge":
            self.stats["n_bridge"] += 1
        elif t == "comparison":
            self.stats["n_comparison"] += 1

    # --- scoring ---------------------------------------------------------

    def score_retrieval(
        self,
        retrieved: list[RetrievedChunk],
        query: EvalQuery,
        scoring_ranking: list[RetrievedChunk] | None = None,
    ) -> RetrievalScore:
        """Set-F1 over sentence atoms plus rank-aware metrics by title."""
        # Set-F1 over the reader context at sentence level; precision is
        # capped by the chunk size, so it is secondary.
        # harness choice: chunker-independent recall (METHODS §C.4)
        gold = query.gold_passage_sets[0] if query.gold_passage_sets else frozenset()
        base = score_retrieval_ck2(retrieved, query.gold_passage_sets)
        if not gold:
            return base

        # Rank-aware over the depth-50 scoring ranking, projected to titles;
        # Hit@2 is the headline.
        # harness choice: one scoring depth for every system (METHODS §D)
        title_gold = frozenset((t, _TITLE_SPAN) for t, _ in gold)
        ranked = scoring_ranking if scoring_ranking is not None else retrieved
        rank = score_retrieval_rank_aware(
            _project_to_titles(ranked), title_gold, k_values=RANK_K_VALUES)
        return dc_replace(
            base,
            hit_at_k=rank.get("hit_at_k", {}),
            map_at_k=rank.get("map_at_k", {}),
            mrr=float(rank.get("mrr", 0.0)),
        )

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """Guarded token-F1 as the score; EM, P and R in metadata."""
        gold = query.gold_answers[0].free_form if query.gold_answers else ""
        # official: hotpot_evaluate_v1.py::f1_score @ 36358534
        f1, precision, recall = hotpot_token_f1_prf(predicted, gold)
        # official: hotpot_evaluate_v1.py::exact_match_score @ 36358534
        em = float(
            normalize_qasper_answer(predicted) == normalize_qasper_answer(gold)
        )
        return AnswerScore(
            value=f1,
            method="token_f1",
            per_annotator=(f1,),
            metadata={
                # abstained is recorded for parity with the other loaders
                # and changes no score.
                "abstained": is_abstention(predicted),
                "exact_match": em,
                # The official report carries em, f1, prec and recall, so
                # all four are recorded.
                "answer_precision": precision,
                "answer_recall": recall,
                "gold": gold,
            },
        )


class HotpotQAPooledBenchmark(HotpotQABenchmark):
    """Pooled variant: one corpus per shard of questions, titles deduped."""

    name = "hotpotqa_pooled"

    @property
    def cell_units(self) -> int | None:
        """Questions pooled into shards, so units equal ceil(q / shard)."""
        if self.max_questions is None:
            return None
        return -(-self.max_questions // self.shard_questions)
    variant = "pooled"

    def __init__(
        self,
        max_questions: int | None = PREREGISTERED_Q,
        shard_questions: int = SHARD_QUESTIONS,
    ) -> None:
        """Set the shard size and record it in stats."""
        super().__init__(max_questions=max_questions)
        self.shard_questions = shard_questions
        self.stats["shard_questions"] = shard_questions
        self.stats["shard_sizes"] = []

    def iter_eval_units(
        self,
        *,
        split: str,
        max_units: int | None = None,
    ) -> Iterable[EvalUnit]:
        """Yield one EvalUnit per shard of shard_questions questions."""
        if split not in VALID_SPLITS:
            raise ValueError(
                f"HotpotQA split must be one of {VALID_SPLITS}; got {split!r}")
        ds = self._load()
        n_shards = 0
        for start in range(0, len(ds), self.shard_questions):
            if max_units is not None and n_shards >= max_units:
                break
            rows = [ds[i] for i in range(
                start, min(start + self.shard_questions, len(ds)))]
            if not rows:
                continue

            # Dedup by title across the shard, first occurrence wins: one
            # paragraph is a distractor for many questions.
            by_title: dict[str, list[CorpusItem]] = {}
            for row in rows:
                for item in _sentence_items(row["context"]):
                    by_title.setdefault(item.parent_id, [])
                    if len(by_title[item.parent_id]) <= item.metadata["sent_id"]:
                        by_title[item.parent_id].append(item)
            corpus = [it for items in by_title.values() for it in items]
            if not corpus:
                continue

            # One query per row; parent_scope stays None because the whole
            # shard is the haystack.
            queries = []
            for row in rows:
                self._count(row)
                queries.append(_query(row, self.variant, None))

            # Record the shard and emit it.
            self.stats["n_units"] += 1
            self.stats["shard_sizes"].append(len(by_title))
            n_shards += 1
            print(
                f"[hotpotqa/pooled] shard {n_shards}: {len(rows)} questions, "
                f"{len(by_title)} distinct paragraphs, {len(corpus)} sentences"
            )
            yield EvalUnit(
                corpus_id=f"hotpot_pooled::shard{n_shards:03d}",
                corpus=tuple(corpus),
                queries=tuple(queries),
            )


__all__ = [
    "HotpotQABenchmark",
    "HotpotQAPooledBenchmark",
    "hotpot_token_f1",
    "hotpot_token_f1_prf",
    "SHARD_QUESTIONS",
    "SUBSAMPLE_SEED",
    "PREREGISTERED_Q",
    "RANK_K_VALUES",
    "subsample_indices",
]
