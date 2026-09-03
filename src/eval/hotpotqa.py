"""HotpotQA — TWO variants, run side by side (Sevetlidis, 2026-08-05).

The dataset stays and both ideas get built: the standard distractor
setting as the COMPARABLE headline, and a pooled setting as the variant
where a retrieval hierarchy actually exists. Keep or discard on results.

VARIANT A — `hotpotqa` — standard distractor.
    One EvalUnit per question, corpus = that question's 10 paragraphs
    (2 gold + 8 TF-IDF distractors). The published setting.

    ⚠ CORRECTED 2026-08-05 — M4 DOES BUILD A TREE HERE. An earlier note
    in this file predicted a flat index, on the assumption that a
    HotpotQA paragraph is ~70 tokens. MEASURED: **127.7 tokens**, so ten
    paragraphs give **15-20 leaves**, above the stop condition of 11.
    Observed on the first smoke: 15 leaves -> 3 layer-1 nodes, 17 -> 4,
    20 -> 4. Variant A therefore has ONE summary layer, the same depth
    as a QASPER-scale corpus, and the flat-index banner correctly never
    fired.

    ⚠ WHAT IS STILL TRUE, and it is the reason variant A carries no
    primary comparison: a 15-20 leaf corpus against top-15 retrieval
    means EVERY retrieval system returns essentially the whole corpus.
    Variant A cannot discriminate retrievers — it is a READER benchmark,
    which is what the standard distractor setting was designed to be.
    What it CAN separate is M1 (no evidence) from the rest, and M4 (whose
    context carries summary nodes) from the flat systems.

VARIANT B — `hotpotqa_pooled` — pooled shards.
    Paragraphs pooled across `SHARD_QUESTIONS` questions into one shared
    corpus per shard, deduplicated by title, one EvalUnit per shard.
    ~700-900 paragraphs per shard gives a real multi-layer tree.

    ⚠ NOT COMPARABLE TO PUBLISHED HOTPOTQA. Pooling changes the task
    from multi-hop REASONING over given candidates to multi-hop
    RETRIEVAL plus reasoning. Declared, not discovered.

# === DEVIATIONS AND LIMITS — thesis footnote ===
#
# 0. M4 ON VARIANT A IS A REAL RAPTOR RESULT, with an 8.3% flat tail.
#    MEASURED FROM THE BANKED CELL by `analyse` counting
#    `metadata.m4_tree_degenerate` (2026-08-22); leaf population 17,443,
#    median 17, max 37. RAPTOR stops when a layer holds
#    <= reduction_dimension + 1 = 11 nodes, so:
#
#      917/1000 units (91.7%) build a 2-layer hierarchy
#       83/1000 units  (8.3%) fall at or below the threshold and are
#                             scored on flat dense retrieval with M4's
#                             own components
#
#    THE EARLIER 964/36 (3.6%) FIGURE IS DEAD — it predates the
#    single-item-rule corpus layout in `BaseSystem.index_items` and
#    describes a population this code no longer produces. It must not
#    appear in the thesis or in any caption.
#
#    The results caption states the 8.3%. Reported rather than dropped:
#    an empty cell in the matrix invites the question the label answers
#    in place, and RAPTOR's behaviour at corpus sizes near its own stop
#    condition is a regime the paper never tests.
#
# 0b. THE APP. I NON-LEAF GATE FAILS ON THIS CELL, and the failure is
#    reported rather than resolved. Measured over all 1,000 units:
#    16.4% micro / 15.6% macro against the paper's 18.5-57.0% band.
#    The 83 degenerate units contribute leaves and zero summary nodes, so
#    they mechanically depress a micro-average over a MIXED population,
#    and `analyse` therefore reports the gate twice — over all rows, and
#    over the 917 tree-building rows alone. Whichever way the second
#    figure lands, BOTH are reported and the caption names the population
#    each describes. If the tree-building figure is in band, the all-rows
#    number is a reporting artifact of mixing populations; if it is not,
#    RAPTOR's node distribution on ~18-leaf corpora sits outside what the
#    paper observed, which is a FINDING for the discussion and not
#    something to explain away. The split exists to tell those two cases
#    apart, not to rescue one of them.
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
# 3. RETRIEVAL IS MEASURED AT DEPTH 50, GENERATION AT TOP-15 / BUDGET.
#    Rank-aware metrics run over a fixed-depth scoring ranking so every
#    system is measured over the same number of candidate retrieval
#    UNITS (50 chunks or nodes); the DOCUMENT ranking is derived from
#    them by first occurrence, so the number of distinct documents
#    inside the window is system-dependent (M4's summary nodes rank no
#    document at all) -- corrected wording, second fidelity audit;
#    set-level R/P/F1 stay over the reader context, because they measure
#    what the generator actually saw. Two depths in one table, stated
#    here and in the results caption.
#
# 3b. SET-F1 IS PRECISION-CAPPED BY CONSTRUCTION, so it is SECONDARY.
#    With 2-3 gold sentences and top-15 chunks covering 45-75
#    sentences, precision is bounded near 0.05 and F1 near 0.1; on
#    variant B, with 2 gold paragraphs in a ~800-paragraph haystack,
#    F1 is capped near 0.23. A low number here is the metric's ceiling,
#    not a system failure. RANK-AWARE AT TITLE LEVEL IS PRIMARY.
#
# 4. Gold is SENTENCE-LEVEL (ruling (ii)), for both variants. Each
#    sentence is a CorpusItem under its title; the shared per-parent
#    `index_items` reassembles the paragraph and derives provenance by
#    char-span intersection, so a chunk spanning three sentences
#    correctly carries three atoms.
#
#    THE RETRIEVAL UNIT IS THE PARAGRAPH FOR EVERY SYSTEM (2026-08-12).
#    The per-parent layout was M4-local until it was promoted to
#    BaseSystem; before that, M2/M3/M7/M9 indexed one SENTENCE per file
#    and `walk_corpus(min_chars_per_doc=200)` dropped most of them
#    (~124 chars each), which crashed M2 at unit 41 and M3 at unit 42.
#    Recorded as a dated amendment in docs/PREREGISTRATION.md
#    (ADDENDUM 4) because it is a benchmark-definition change; it landed
#    before any HotpotQA cell was reported.
#
# 5. Dev split only — HotpotQA test labels are not public. Reported as
#    a single-split asymmetry, the same one MultiHop carries.
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
# benchmark at all, the bridge/comparison split.
#
# The seed and the sampler now live in `src/eval/sampling.py` and are
# SHARED with NarrativeQA, so the two benchmarks cannot drift into two
# conventions. Re-exported here because this module's public surface
# already carried them.

# THE PRE-REGISTERED SAMPLE SIZE, as a DEFAULT rather than a CLI flag.
#
# The runner constructs benchmarks with no arguments, so a subsample that
# only happens when `max_questions` is passed would never happen in a
# real run -- and `--max-units 1000` would silently take the FIRST 1,000
# of 7,405 rows, which is precisely the head slice the seeding exists to
# avoid. Defaulting here makes the registered sample the thing you get
# unless you deliberately ask for otherwise.
#
# The loader's `max_units` parameter caps units WITHIN the subsample (a
# two-unit smoke is two questions of the registered sample, never a
# different sample); the runner passes no cap since the repo reduction,
# so it is reached only from tests.
#
# Pass max_questions=None explicitly for the full 7,405-question split.
PREREGISTERED_Q = 1000

# Rank-aware K grid. HotpotQA has exactly 2 gold paragraphs, so Hit@2 is
# the directly interpretable "did it find both", and MAP@10 is
# comparable in shape to what MultiHop reports.
RANK_K_VALUES = (1, 2, 5, 10)

# The atom span_id used for TITLE-level projection in rank-aware
# scoring. Distinct from any real sentence id.
_TITLE_SPAN = "<title>"

# The official evaluator's yes/no/noanswer sentinels, verbatim from
# `hotpot_evaluate_v1.f1_score`. HOTPOTQA-LOCAL BY PLACEMENT: this lives
# here rather than in `scorers/extractive.py` so it is structurally
# unreachable from MultiHop and NarrativeQA, which import the shared
# `token_f1` directly. Unreachable beats unused — an "official" guard
# sitting in the shared module is one import away from silently
# rewriting two other benchmarks' scores.
_HOTPOT_SENTINELS = ("yes", "no", "noanswer")


def hotpot_token_f1(predicted: str, gold: str) -> float:
    """Shared token-F1 plus the official yes/no/noanswer guard.

    TRANSCRIBED FROM THE OFFICIAL EVALUATOR, `hotpot_evaluate_v1.py`,
    whose `f1_score` opens with two early returns before any token
    counting happens:

        ZERO_METRIC = (0, 0, 0)

        if normalized_prediction in ['yes', 'no', 'noanswer'] and
                normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and
                normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

    WHAT IT MEANS. On a yes/no question the official scorer demands the
    normalised prediction be EXACTLY 'yes' or 'no'. Partial credit is
    refused outright: gold 'yes' against "Yes, both films were directed
    by the same person." shares one token of eight and would otherwise
    earn F1 0.2222, but the official answer is 0.0. A yes/no answer is
    right or it is wrong; there is no partial version of it.

    WHY THIS EXISTS AS ITS OWN FUNCTION rather than a flag on the shared
    scorer: `token_f1` is the harness-wide contract and MUST keep
    behaving identically for MultiHop and NarrativeQA, whose golds are
    free-form and where a prediction of 'no' against a real answer is an
    ordinary wrong answer rather than a sentinel mismatch.

    BOTH OFFICIAL BRANCHES ARE IMPLEMENTED. The guard fires when EITHER
    side is a sentinel and the two differ -- a sentinel PREDICTION
    against a real gold, and a sentinel GOLD against a non-matching
    prediction. `hotpot_token_f1_prf` expresses the reference's two
    early returns as one equivalent condition,
    `np_ != ng_ and (np_ in _HOTPOT_SENTINELS or ng_ in _HOTPOT_SENTINELS)`,
    which is the same predicate the reference reaches in two statements.

    DIRECTION OF EFFECT, and it is one-way -- WHICH IS NOT THE SAME AS
    ONE-SIDED, and the phrasing matters because "one-way" read as
    "one branch" would describe a defect this code does not have. The
    guard is two-sided in its CONDITION and one-way in its EFFECT: it
    can only ever ZERO a score the fall-through would have granted, and
    can never raise one. So it can lower a HotpotQA answer column and
    can never inflate it.

    VERIFIED, not asserted (2026-08-22, docs/SCORER_COMPARISON.md): a
    three-way battery -- a freshly transcribed `hotpot_evaluate_v1.f1_score`,
    this function, and hand-computed values -- agrees 12/12 to 1e-9, and
    the battery includes one case per branch: "no" vs "November 2016"
    (prediction-side) and "Yes, both films were directed by the same
    person." vs "yes" (gold-side, where the fall-through would have paid
    0.2222). `TestYesNoGuard` in tests/test_hotpotqa.py pins both against
    its own transcription of the reference.

    MEASURED EFFECT ON RUN DATA -- `mean_hedged = 0.000` over 369 rows of
    banked cell 6 (M4/hotpotqa, 2026-08-22). This is the guard working,
    observed rather than argued: where the gold is `yes` or `no`, exact
    match is the only way to score, so a hedged answer collects NOTHING.
    Not one of 369 hedged rows scored above zero.

    THE CONTRAST IS THE POINT, and it belongs beside this number whenever
    it is quoted. On MultiHop the same class of answer takes the 0.5
    credited-refusal payment, because that benchmark's golds include the
    bare token "no" and the canonical refusal shares exactly one token
    with it -- and MultiHop's own scorer would pay 1.0. Same harness,
    same hedge detector, opposite payoffs, and the difference is entirely
    the presence or absence of an official sentinel guard. It is also why
    the two answer columns must never be pooled or compared row-for-row.
    Adopting the guard at `68f6056` was a fidelity fix; 369 rows scoring
    zero where they would otherwise have earned partial credit is what
    that fix cost, stated rather than absorbed.

    MEASURED REACH on the preregistered sample (seed 20260805, 2026-08-18,
    counted with this module's own normaliser): 61 of 1,000 questions
    (6.1%) carry a yes/no gold — yes 38 / no 23 — and all 61 are
    comparison questions, i.e. 32.6% of that slice. The full 7,405-question
    dev split is 6.19%, so the draw is representative on this axis.

    Exact match is NOT routed through here. The official
    `exact_match_score` is an unguarded normalised string comparison and
    this harness already matched it; wrapping it would introduce a
    divergence where none existed.
    """
    return hotpot_token_f1_prf(predicted, gold)[0]


def hotpot_token_f1_prf(predicted: str, gold: str) -> tuple[float, float, float]:
    """(f1, precision, recall) — the full triple the official scorer returns.

    `hotpot_evaluate_v1.f1_score` returns `f1, precision, recall`, and
    `update_answer` accumulates ALL THREE alongside EM:

        metrics['em'] += float(em)
        metrics['f1'] += f1
        metrics['prec'] += prec
        metrics['recall'] += recall

    so the official report carries FOUR answer numbers, not two. This
    harness computed precision and recall inside `token_f1` and threw
    them away — a value that was correct and inert, which is the defect
    class this project keeps finding. They are surfaced now because a
    reader comparing against the official script will look for them and
    they cost nothing: the arithmetic is already being done.

    `ZERO_METRIC` is `(0, 0, 0)` in the reference, so the yes/no guard
    zeroes all three together rather than F1 alone.

    The both-empty branch mirrors `token_f1`'s adopted SQuAD rule and is
    unreachable in this pipeline (`assert_gold_not_empty`).
    """
    np_ = normalize_qasper_answer(predicted)
    ng_ = normalize_qasper_answer(gold)
    if np_ != ng_ and (np_ in _HOTPOT_SENTINELS or ng_ in _HOTPOT_SENTINELS):
        return 0.0, 0.0, 0.0

    pred_tokens = np_.split()
    gold_tokens = ng_.split()
    if not pred_tokens or not gold_tokens:
        agree = float(pred_tokens == gold_tokens)
        return agree, agree, agree
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0, 0.0, 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return f1, precision, recall


def _sentence_items(context: Any) -> list[CorpusItem]:
    """One CorpusItem per SENTENCE, parented by its paragraph title.

    Sentence granularity is ruling (ii): HotpotQA's supporting facts are
    (title, sent_id) pairs, so anything coarser discards the annotation
    the benchmark is distinctive for. `BaseSystem.index_items`
    concatenates a title's sentences back into the paragraph before
    chunking, so nothing is fragmented at retrieval time — the fine
    granularity buys provenance, not smaller chunks. That reassembly is
    shared by every system as of 2026-08-12; see deviation 4.
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
    # HotpotQA has no unanswerable questions, so EVERY gold must survive
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

    @property
    def cell_units(self) -> int | None:
        """One corpus per question, so units == questions.

        Derived from `max_questions` rather than hardcoded: that value is
        already a CONSTRUCTOR DEFAULT, which is what makes this
        benchmark's population a property of the code. NarrativeQA had
        the same intent expressed as a flag and lost it.
        """
        return self.max_questions
    variant = "distractor"

    def __init__(self, max_questions: int | None = PREREGISTERED_Q) -> None:
        self._rows: Any = None
        self.max_questions = max_questions
        self.stats: dict[str, Any] = {
            "n_questions": 0,
            # SAME KEY AS EVERY OTHER LOADER. The run summary's
            # expected_n_queries reads "n_queries", and this loader
            # recorded only "n_questions" — so P8's short-cell guard had
            # nothing to compare against on all ten HotpotQA cells, and
            # the null was written in silence.
            "n_queries": 0,
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
        self.stats["n_queries"] = len(ds)

        # REALISED distribution of the stratifying variables. The
        # subsample is random rather than stratified, so nothing
        # GUARANTEES it is balanced — printing the realised split is what
        # makes a lopsided draw visible before it is spent, rather than
        # discovered in a results table.
        #
        # `type` (bridge / comparison) is the real axis: MEASURED 79.9% /
        # 20.1% across the full 7,405-question dev split, and it is the
        # dimension HotpotQA is distinctive for.
        #
        # `level` IS NOT A USABLE SLICE. Measured 100% "hard" across the
        # entire dev-distractor split — the easy/medium examples live in
        # train only. It is still printed, because a printed 100% is what
        # tells the next reader the slice is unavailable rather than
        # merely unused, but no analysis may condition on it.
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
        scoring_ranking: list[RetrievedChunk] | None = None,
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

        # Set-level over the reader context (above); rank-aware over the
        # fixed-depth scoring ranking: 50 retrieval UNITS for every
        # system, from which the document ranking is derived (the
        # document count inside the window is system-dependent).
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
        """Token-F1 primary, exact match alongside — HotpotQA's own pair.

        Both use the SQuAD-style normalisation the official evaluator
        uses (lowercase, strip articles, drop punctuation, collapse
        whitespace), which `normalize_qasper_answer` already implements;
        reusing it keeps one normaliser in the codebase rather than two
        that could drift.

        F1 additionally carries the official yes/no/noanswer guard — see
        `hotpot_token_f1`. Without it this benchmark awarded token overlap
        on 6.1% of the sample where the published evaluator awards zero,
        which was the one place the deviations table claimed an
        unqualified match it did not have.
        """
        gold = query.gold_answers[0].free_form if query.gold_answers else ""
        f1, precision, recall = hotpot_token_f1_prf(predicted, gold)
        em = float(
            normalize_qasper_answer(predicted) == normalize_qasper_answer(gold)
        )
        return AnswerScore(
            value=f1,
            method="token_f1",
            per_annotator=(f1,),
            metadata={
                # Recorded for parity with the other two benchmarks under
                # the shared contract. HotpotQA never had an abstention
                # gate, so this field changes no score here; it exists so
                # abstention RATES are comparable across all three.
                "abstained": is_abstention(predicted),
                "exact_match": em,
                # The official report carries FOUR answer numbers -- em,
                # f1, prec, recall (`update_answer`). We surfaced two and
                # discarded two that were already being computed. A
                # reader checking against the official script looks for
                # these, so they are recorded rather than re-derived.
                "answer_precision": precision,
                "answer_recall": recall,
                "gold": gold,
            },
        )


class HotpotQAPooledBenchmark(HotpotQABenchmark):
    """Variant B — paragraphs pooled across a shard of questions.

    One EvalUnit per shard (ten shards for the Q=1000 run). Titles are
    deduplicated within a
    shard: the same Wikipedia paragraph is a distractor for many
    questions, and indexing it twice would both waste the build and put
    exact-duplicate vectors into the clustering.
    """

    name = "hotpotqa_pooled"

    @property
    def cell_units(self) -> int | None:
        """Questions pooled into shards, so units == ceil(q / shard)."""
        if self.max_questions is None:
            return None
        return -(-self.max_questions // self.shard_questions)
    variant = "pooled"

    def __init__(
        self,
        max_questions: int | None = PREREGISTERED_Q,
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
    "hotpot_token_f1",
    "hotpot_token_f1_prf",
    "SHARD_QUESTIONS",
    "SUBSAMPLE_SEED",
    "PREREGISTERED_Q",
    "RANK_K_VALUES",
    "subsample_indices",
]
