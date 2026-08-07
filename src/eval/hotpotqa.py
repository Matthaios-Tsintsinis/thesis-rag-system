"""HotpotQA — TWO variants, run side by side (Sevetlidis, 2026-08-05).

The dataset stays and both ideas get built: the standard distractor
setting as the COMPARABLE headline, and a pooled setting as the variant
where a retrieval hierarchy actually exists. Keep or discard on results.

VARIANT A — `hotpotqa` — standard distractor.
    One EvalUnit per question, corpus = that question's 10 paragraphs
    (2 gold + 8 TF-IDF distractors). The published setting.

    ⚠ M4 HAS NO TREE HERE, AND THAT IS STRUCTURAL. Ten paragraphs is
    ~9-14 leaves, at or under `reduction_dimension + 1` = 11, so
    `build_paper_tree` halts at layer 0 and M4 degenerates to flat dense
    retrieval — effectively M2 with mpnet. MEASURED: a 40-corpus probe
    at this shape produced NO tree in 40/40 cases. M4's rows here must
    NEVER be presented as a RAPTOR result; the runner enforces this
    (`m4_tree_degenerate` per row, a banner in `analyse`). M7 inherits
    it when it joins — with no hierarchy, its multi-branch axis is inert
    and it is a two-axis system on this variant.

VARIANT B — `hotpotqa_pooled` — pooled shards.
    Paragraphs pooled across `SHARD_QUESTIONS` questions into one shared
    corpus per shard, deduplicated by title, one EvalUnit per shard.
    ~700-900 paragraphs per shard gives a real multi-layer tree.

    ⚠ NOT COMPARABLE TO PUBLISHED HOTPOTQA. Pooling changes the task
    from multi-hop REASONING over given candidates to multi-hop
    RETRIEVAL plus reasoning. Declared, not discovered.

# === DEVIATIONS AND LIMITS — thesis footnote ===
#
# 1. RESIDUAL COMPARABILITY GAP, even on variant A. Our generator is a
#    non-fine-tuned Qwen2.5-7B and our retrieval unit is a 100-token
#    chunk rather than a paragraph-selection pipeline. Published numbers
#    come from models trained on HotpotQA. Variant A is the CLOSER of
#    the two, not a like-for-like comparison, and must be reported that
#    way rather than implying full comparability.
#
# 2. OFFICIAL SUPPORTING-FACT METRICS ARE NOT REPORTABLE. `sp_em` /
#    `sp_f1` score a SENTENCE SET that a supporting-fact classifier
#    asserts. We have no such classifier: our predicted set would be
#    every sentence any retrieved chunk touches (~45-75 sentences at
#    top-15 against 2-3 gold), so precision would be ~0.03 and the
#    number would measure the absence of a selection step rather than
#    retrieval quality. A coverage proxy is exposed as
#    `sp_coverage_f1` in metadata under a DELIBERATELY DIFFERENT NAME.
#    Never report it as `sp_f1`.
#
# 3. SET-F1 IS PRECISION-CAPPED BY CONSTRUCTION, so it is SECONDARY.
#    With 2-3 gold sentences and top-15 chunks covering 45-75
#    sentences, precision is bounded near 0.05 and F1 near 0.1; on
#    variant B, with 2 gold paragraphs in a ~800-paragraph haystack,
#    F1 is capped near 0.23. A low number here is the metric's ceiling,
#    not a system failure. RANK-AWARE AT TITLE LEVEL IS PRIMARY.
#
# 4. Gold is SENTENCE-LEVEL (ruling (ii)), for both variants. Each
#    sentence is a CorpusItem under its title; M4's per-parent
#    `index_items` reassembles the paragraph and derives provenance by
#    char-span intersection, so a chunk spanning three sentences
#    correctly carries three atoms.
#
# 5. Dev split only — HotpotQA test labels are not public. Reported as
#    a single-split asymmetry, the same one MultiHop carries.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any, Iterable

from ..retrievers.base import RetrievedChunk
from .alignment import score_retrieval_ck2, score_retrieval_rank_aware
from .scorers.extractive import normalize_qasper_answer, token_f1
from .types import (
    ANSWER_TYPE_FREE_FORM,
    AnswerScore,
    CorpusItem,
    EvalQuery,
    EvalUnit,
    GoldAnswer,
    RetrievalScore,
)


# NAMESPACED REPO ID, and both halves of this matter.
#
# `hotpot_qa` (bare) is the legacy name and modern huggingface_hub
# REJECTS it: "Repository id must be 'namespace/name'". That was the
# first failure.
#
# DO NOT PIN AN OLDER REVISION to work around anything here. The repo was
# converted to Parquet in early 2025; before that it was loader-script
# based and the script fetched from curtis.ml.cmu.edu, which has been
# OFFLINE SINCE MAY 2025. A script-era revision therefore cannot load at
# all, with or without the right repo id — the Parquet conversion is what
# makes this dataset usable, so track the default revision.
HF_REPO = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"

# HotpotQA ships train + validation; the test labels are not public, so
# `validation` is the only usable split and both harness split names map
# onto it. Stated rather than silently aliased.
VALID_SPLITS = ("validation", "test", "dev")
_HF_SPLIT = "validation"

# Variant B shard size, in QUESTIONS. 100 gives ~700-900 distinct
# paragraphs per shard after title dedup, which clears the ~74-leaf
# threshold for a second summary layer by a wide margin.
SHARD_QUESTIONS = 100

# SEEDED RANDOM SUBSAMPLE, not the head of the file.
#
# HotpotQA dev is not guaranteed to be randomly ordered — it can be
# grouped by `type` and `level` — so taking the first N rows risks a
# sample skewed on exactly the dimension that justifies including the
# benchmark at all, the bridge/comparison split. A head slice would be
# indefensible precisely where the benchmark is most interesting.
#
# Same seed convention as the M7 dev/test partition, so every
# pre-registered split in this project traces to one dated constant.
#
# `random.Random(seed).sample` rather than `Dataset.shuffle(seed=...)`:
# the datasets library's shuffle implementation is free to change
# between versions, and this project has already been bitten by a
# version-sensitive algorithm (UMAP) whose output a seed does not fully
# pin. Python's Mersenne Twister is specified and stable.
#
# Indices are SORTED after sampling, so the subsample keeps dataset
# order. That makes variant B's shard boundaries a function of the
# sample alone, not of the order `sample()` happened to emit.
SUBSAMPLE_SEED = 20260805

# Rank-aware K grid. HotpotQA has exactly 2 gold paragraphs, so Hit@2 is
# the directly interpretable "did it find both", and MAP@10 is
# comparable in shape to what MultiHop reports.
RANK_K_VALUES = (1, 2, 5, 10)

# The atom span_id used for TITLE-level projection in rank-aware
# scoring. Distinct from any real sentence id.
_TITLE_SPAN = "<title>"


def subsample_indices(n_total: int, k: int, seed: int = SUBSAMPLE_SEED) -> list[int]:
    """Seeded random indices, sorted. Pure, so it is testable and quotable.

    BOTH VARIANTS MUST GET THE SAME SAMPLE. `seed` is a module constant
    and both classes share this function through `_load`, so variant A
    and variant B answer the SAME 1,000 questions. If they diverged, any
    A-vs-B comparison would confound the pooling change with a change of
    question set — and comparing the two variants is the entire reason
    both are being built.
    """
    import random

    if k >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), k))


def _sentence_items(context: Any) -> list[CorpusItem]:
    """One CorpusItem per SENTENCE, parented by its paragraph title.

    Sentence granularity is ruling (ii): HotpotQA's supporting facts are
    (title, sent_id) pairs, so anything coarser discards the annotation
    the benchmark is distinctive for. M4's per-parent `index_items`
    concatenates a title's sentences back into the paragraph before
    chunking, so nothing is fragmented at retrieval time — the fine
    granularity buys provenance, not smaller chunks.
    """
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
    sf = row["supporting_facts"]
    return frozenset(
        (str(t), f"sent{int(i)}")
        for t, i in zip(sf["title"], sf["sent_id"])
    )


def _query(row: Any, variant: str, parent_scope: str | None) -> EvalQuery:
    answer = (row.get("answer") or "").strip()
    atoms = _gold_atoms(row)
    return EvalQuery(
        query_id=str(row["id"]),
        question_text=(row.get("question") or "").strip(),
        parent_scope=parent_scope,
        gold_answers=(GoldAnswer(
            answer_type=ANSWER_TYPE_FREE_FORM, free_form=answer),),
        gold_passage_sets=(atoms,),
        # bridge | comparison — the benchmark's own split, and the one
        # slice worth reporting separately.
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
    """Rewrite each chunk's provenance to (title, <title>) atoms.

    Rank-aware scoring is DOCUMENT-level by design (see
    `score_retrieval_rank_aware`), and for HotpotQA the document is the
    paragraph/title. Our gold is sentence-level, so passing it unprojected
    would collapse to a SENTENCE ranking instead — a different metric
    wearing the same name.

    Copies rather than mutates: these Chunk objects are the system's live
    index state, and rewriting provenance in place would corrupt the CK-2
    scoring that runs on the same objects.
    """
    out: list[RetrievedChunk] = []
    for r in retrieved:
        titles = {parent for parent, _ in (r.chunk.gold_provenance or ())}
        projected = tuple((t, _TITLE_SPAN) for t in sorted(titles))
        out.append(dc_replace(
            r, chunk=dc_replace(r.chunk, gold_provenance=projected)))
    return out


class HotpotQABenchmark:
    """Variant A — standard distractor, one corpus per question."""

    name = "hotpotqa"
    variant = "distractor"

    def __init__(self, max_questions: int | None = None) -> None:
        self._rows: Any = None
        self.max_questions = max_questions
        self.stats: dict[str, Any] = {
            "n_questions": 0,
            "n_units": 0,
            "n_bridge": 0,
            "n_comparison": 0,
            # Both feed the tree arithmetic and BOTH were assumptions
            # until now, so the loader measures them on first use rather
            # than leaving them estimated.
            "mean_paragraph_tokens": None,
            "n_distinct_titles": 0,
        }

    # --- preflight -------------------------------------------------------

    def preflight(self) -> None:
        """Resolve the dataset BEFORE anything expensive is loaded.

        The bare-repo-id failure surfaced at the first `iter_eval_units`
        call — AFTER `--prewarm` had already pulled 15 GB of Qwen into
        VRAM. A two-second metadata check would have failed in two
        seconds instead of two minutes. Same class as the API-key guard:
        validate the cheap precondition before paying the expensive one.

        Deliberately a METADATA call, not a load. It resolves the repo id
        and confirms the config exists, which is exactly the pair of
        things that were wrong, without downloading rows.
        """
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
        if self._rows is None:
            from datasets import load_dataset

            # Native Parquet: the default revision is the right one, and
            # refs/convert/parquet does not apply (see the HF_REPO note).
            ds = load_dataset(HF_REPO, HF_CONFIG, split=_HF_SPLIT)
            if self.max_questions is not None and self.max_questions < len(ds):
                ds = ds.select(subsample_indices(len(ds), self.max_questions))
            self._rows = ds
            self._measure(ds)
        return self._rows

    def _measure(self, ds: Any) -> None:
        """Print the two quantities the tree arithmetic rested on.

        Mean paragraph tokens was assumed ~70 and distinct-title count
        was never counted; both set variant B's realised shard size and
        variant A's leaf count. Measured on load, printed once.
        """
        from ..prompt_packing import count_tokens

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

        # REALISED distribution of the two stratifying variables. The
        # subsample is random rather than stratified, so nothing
        # GUARANTEES it is balanced — printing the realised split is what
        # makes a lopsided draw visible before it is spent, rather than
        # discovered in a results table. bridge/comparison is the axis
        # HotpotQA is distinctive for; level is its own difficulty label.
        from collections import Counter

        types = Counter(str(r.get("type") or "unknown") for r in ds)
        levels = Counter(str(r.get("level") or "unknown") for r in ds)
        self.stats["type_distribution"] = dict(types)
        self.stats["level_distribution"] = dict(levels)
        self.stats["subsample_seed"] = (
            SUBSAMPLE_SEED if self.max_questions is not None else None)

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
        if split not in VALID_SPLITS:
            raise ValueError(
                f"HotpotQA split must be one of {VALID_SPLITS} (test labels "
                f"are not public, so all map to {_HF_SPLIT!r}); got {split!r}"
            )
        ds = self._load()
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
                # parent_scope stays None: the corpus IS this question's
                # candidate set, so retrieval is already scoped and a
                # further restriction would be a second, silent filter.
                queries=(_query(row, self.variant, None),),
            )

    def _count(self, row: Any) -> None:
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
    ) -> RetrievalScore:
        """Set-F1 at SENTENCE level + rank-aware at TITLE level.

        The two units are deliberate. Set-F1 over sentence atoms is the
        finest-grained thing the annotation supports, and is
        PRECISION-CAPPED by construction (deviations note 3) — reported
        for completeness, read as secondary. Rank-aware is projected to
        TITLE level because that is HotpotQA's document unit and because
        rank-aware metrics are document-level by contract; with exactly
        two gold titles, Hit@2 is the directly interpretable number and
        is the PRIMARY retrieval metric for this benchmark.
        """
        gold = query.gold_passage_sets[0] if query.gold_passage_sets else frozenset()
        base = score_retrieval_ck2(retrieved, query.gold_passage_sets)
        if not gold:
            return base

        title_gold = frozenset((t, _TITLE_SPAN) for t, _ in gold)
        rank = score_retrieval_rank_aware(
            _project_to_titles(retrieved), title_gold, k_values=RANK_K_VALUES)
        return dc_replace(
            base,
            hit_at_k=rank.get("hit_at_k", {}),
            map_at_k=rank.get("map_at_k", {}),
            mrr=float(rank.get("mrr", 0.0)),
        )

    def score_answer(self, predicted: str, query: EvalQuery) -> AnswerScore:
        """Token-F1 primary, exact match alongside — HotpotQA's own pair.

        Both use the SQuAD-style normalisation the official evaluator
        uses (lowercase, strip articles, drop punctuation, collapse
        whitespace), which `normalize_qasper_answer` already implements;
        reusing it keeps one normaliser in the codebase rather than two
        that could drift.
        """
        gold = query.gold_answers[0].free_form if query.gold_answers else ""
        f1 = token_f1(predicted, gold)
        em = float(
            normalize_qasper_answer(predicted) == normalize_qasper_answer(gold)
        )
        return AnswerScore(
            value=f1,
            method="token_f1",
            per_annotator=(f1,),
            metadata={"exact_match": em, "gold": gold},
        )


class HotpotQAPooledBenchmark(HotpotQABenchmark):
    """Variant B — paragraphs pooled across a shard of questions.

    One EvalUnit per shard, so `--max-units 1` is a single-shard gate and
    `--max-units 10` is the Q=1000 run. Titles are deduplicated within a
    shard: the same Wikipedia paragraph is a distractor for many
    questions, and indexing it twice would both waste the build and put
    exact-duplicate vectors into the clustering.
    """

    name = "hotpotqa_pooled"
    variant = "pooled"

    def __init__(
        self,
        max_questions: int | None = None,
        shard_questions: int = SHARD_QUESTIONS,
    ) -> None:
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

            # Dedup by TITLE across the shard, first occurrence wins.
            by_title: dict[str, list[CorpusItem]] = {}
            for row in rows:
                for item in _sentence_items(row["context"]):
                    by_title.setdefault(item.parent_id, [])
                    if len(by_title[item.parent_id]) <= item.metadata["sent_id"]:
                        by_title[item.parent_id].append(item)
            corpus = [it for items in by_title.values() for it in items]
            if not corpus:
                continue

            queries = []
            for row in rows:
                self._count(row)
                # parent_scope stays None: the whole shard is the
                # haystack, which is the point of the variant.
                queries.append(_query(row, self.variant, None))

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
    "SHARD_QUESTIONS",
    "SUBSAMPLE_SEED",
    "RANK_K_VALUES",
    "subsample_indices",
]
