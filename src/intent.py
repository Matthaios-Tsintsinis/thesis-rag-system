"""Axis 3 — intent decomposition (PIPELINE_DESIGN.md §4.1–4.3).

Query → simple/multi-aspect classification → ≤3 aspects (importance
≥ MIN_ASPECT_IMPORTANCE, low ones dropped, not min-quota'd) → two
retrieval views per aspect (paraphrase + HyDE) → aspect scoring
(0.5·importance + 0.5·retrieval_confidence) → quota-preserving budget
allocation.

All LLM calls are injected callables (defaulting to the summarization
helpers) so this module is fully unit-testable offline and so the
orchestrator can route them through gpt-4o-mini or a local model
without this module knowing. Retrieval/rerank is also injected: the
preliminary confidence callback is owned by the orchestrator (it has
the indexes); this module only consumes a score.

Ablation hooks (config flags, read by the orchestrator and passed in):
  * A2 use_intent_decomposition=False → one `main` aspect = full query.
  * A3 view_types=("paraphrase","paraphrase2") → second paraphrase
    instead of HyDE (paraphrase-of-paraphrase, deterministic, distinct
    surface form, no summarization.py change).
  * A8 always_include_global_query_view handled by the orchestrator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .config import M7Config
from .summarization import (
    Aspect,
    extract_aspects as _extract_aspects,
    hyde_view as _hyde_view,
    paraphrase_view as _paraphrase_view,
)

ExtractFn = Callable[[str], "object"]
ViewFn = Callable[[str], str]
ConfidenceFn = Callable[[str], float]


GLOBAL_VIEW_NAME = "__global__"


@dataclass
class ViewSpec:
    aspect_name: str
    view_type: str          # "paraphrase" | "hyde" | "paraphrase2"
    text: str


@dataclass
class AspectPlan:
    name: str          # short label for the [Aspect: <name>] prompt header
    text: str          # retrieval string (views/probe/rerank/bias use this)
    importance: float
    views: list[ViewSpec] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    score: float = 0.0
    budget: int = 0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --- Step 1: decomposition + views (§4.1) ---------------------------------


def decompose(
    query: str,
    cfg: M7Config,
    *,
    extract_fn: ExtractFn | None = None,
    paraphrase_fn: ViewFn | None = None,
    hyde_fn: ViewFn | None = None,
) -> list[AspectPlan]:
    """Classify, extract aspects, drop low-importance, cap, build views.

    Never raises on aspect-extractor failure: a parse error or the A2
    ablation collapses to a single protected `main` aspect equal to the
    full query (PIPELINE_DESIGN §4.1 "safety net for aspect-extractor
    failure").
    """
    extract_fn = extract_fn or _extract_aspects
    paraphrase_fn = paraphrase_fn or _paraphrase_view
    hyde_fn = hyde_fn or _hyde_view

    # The "main" fallback aspect's retrieval text is the original query,
    # never the literal token "main" (finding #1).
    main = Aspect(name="main", text=query.strip(), importance=1.0)

    if not cfg.use_intent_decomposition:
        aspects = [main]
    else:
        try:
            result = extract_fn(query)
            aspects = list(result.aspects)  # type: ignore[attr-defined]
        except Exception:
            aspects = [main]

    if not aspects:
        aspects = [main]

    # Drop sub-threshold aspects entirely (NOT min-quota'd) — spurious
    # aspects must not consume protected budget (§4.1).
    if cfg.aspects.drop_low_importance_aspects and len(aspects) > 1:
        kept = [
            a for a in aspects
            if a.importance >= cfg.aspects.min_aspect_importance
        ]
        if kept:
            aspects = kept

    # Cap to MAX_ASPECTS, keeping the most important. Stable on ties.
    if len(aspects) > cfg.aspects.max_aspects:
        aspects = sorted(aspects, key=lambda a: a.importance, reverse=True)
        aspects = aspects[: cfg.aspects.max_aspects]

    plans: list[AspectPlan] = []
    for a in aspects:
        # `text` is the retrieval string; older/mocked extractors that
        # omit it fall back to the label so behaviour stays defined.
        a_text = (getattr(a, "text", "") or a.name).strip()
        plan = AspectPlan(
            name=a.name, text=a_text, importance=float(a.importance)
        )
        for vt in cfg.view_types:
            if vt == "paraphrase":
                txt = paraphrase_fn(a_text)
            elif vt == "hyde":
                txt = hyde_fn(a_text)
            elif vt == "paraphrase2":
                # A3 ablation: a second, distinct paraphrase. Paraphrase
                # the paraphrase so the surface form differs from view 1
                # without a temperature change or summarization.py edit.
                txt = paraphrase_fn(paraphrase_fn(a_text))
            else:
                raise ValueError(f"unknown view_type {vt!r}")
            txt = (txt or "").strip()
            if not txt:
                # View generation returned empty — fall back to the raw
                # aspect text so the view still retrieves something.
                txt = a_text
            plan.views.append(ViewSpec(a.name, vt, txt))
        plans.append(plan)
    return plans


def global_view_spec(query: str) -> ViewSpec:
    """The untouched original query as the protected global view (§4.1).
    Unconditional at the call site; ablation A8 is enforced by the
    orchestrator simply not calling this."""
    return ViewSpec(GLOBAL_VIEW_NAME, "global", query.strip())


# --- Step 2: aspect scoring (§4.2) ----------------------------------------


def score_aspects(
    plans: list[AspectPlan],
    cfg: M7Config,
    confidence_fn: ConfidenceFn,
) -> list[AspectPlan]:
    """aspect_score = w_i·importance + w_c·retrieval_confidence.

    retrieval_confidence = sigmoid(top-1 cross-encoder logit) over the
    preliminary first-stage hits of the aspect's PARAPHRASE view (§4.2).
    `confidence_fn(view_text)` is supplied by the orchestrator and must
    already return a [0,1] confidence (sigmoid applied there or here —
    we sigmoid defensively only if the value is outside [0,1])."""
    w_i = cfg.scoring.importance_weight
    w_c = cfg.scoring.retrieval_confidence_weight
    for p in plans:
        para = next(
            (v for v in p.views if v.view_type in ("paraphrase", "paraphrase2")),
            p.views[0] if p.views else None,
        )
        probe = para.text if para is not None else p.text
        conf = float(confidence_fn(probe))
        if conf < 0.0 or conf > 1.0:
            conf = _sigmoid(conf)
        p.retrieval_confidence = conf
        p.score = w_i * p.importance + w_c * conf
    return plans


def global_confidence(
    query: str,
    confidence_fn: ConfidenceFn,
) -> float:
    """Confidence for the global view — abstention signal only (§4.9),
    never enters budget math (the global quota is fixed)."""
    conf = float(confidence_fn(query))
    if conf < 0.0 or conf > 1.0:
        conf = _sigmoid(conf)
    return conf


# --- Step 3: quota-preserving budget allocation (§4.3) --------------------


def allocate_budget(plans: list[AspectPlan], cfg: M7Config) -> list[AspectPlan]:
    """Distribute the aspect budget ∝ aspect_score, clamped to
    [MIN_CHUNKS_PER_ASPECT, MAX_CHUNKS_PER_ASPECT], largest-remainder
    rounding, leftover redistributed to non-maxed aspects by score.

    Edge cases (flagged in the design proposal):
      * single aspect → min(MAX, total); any clamp slack is left unused
        (the global quota is fixed; we never pad aspects past MAX).
      * n·MIN > total (too many aspects for the budget) → keep the
        highest-scoring aspects that fit at MIN each, drop the rest
        (their budget would be below the useful minimum anyway).
      * all scores zero → equal split.
    """
    b = cfg.budget
    total = b.final_context_chunks
    if cfg.always_include_global_query_view:
        total -= b.global_view_quota
    total = max(total, 0)

    if not plans or total <= 0:
        for p in plans:
            p.budget = 0
        return plans

    # Too many aspects to give each the minimum: keep the top ones.
    max_fit = total // max(b.min_chunks_per_aspect, 1)
    if max_fit < len(plans):
        plans_sorted = sorted(plans, key=lambda p: p.score, reverse=True)
        keep = set(id(p) for p in plans_sorted[: max(max_fit, 1)])
        for p in plans:
            if id(p) not in keep:
                p.budget = 0
        active = [p for p in plans if id(p) in keep]
    else:
        active = list(plans)

    s = sum(max(p.score, 0.0) for p in active)
    n = len(active)
    if s <= 0.0:
        raw = {id(p): total / n for p in active}
    else:
        raw = {id(p): (max(p.score, 0.0) / s) * total for p in active}

    # floor + clamp to [min, max]
    alloc: dict[int, int] = {}
    for p in active:
        v = int(math.floor(raw[id(p)]))
        v = max(b.min_chunks_per_aspect, min(b.max_chunks_per_aspect, v))
        alloc[id(p)] = v

    assigned = sum(alloc.values())

    # Surplus: hand out 1 at a time to non-maxed aspects, highest
    # fractional remainder first (largest-remainder), then by score.
    def _rank(p: AspectPlan) -> tuple[float, float]:
        return (raw[id(p)] - math.floor(raw[id(p)]), p.score)

    order = sorted(active, key=_rank, reverse=True)
    while assigned < total:
        progressed = False
        for p in order:
            if assigned >= total:
                break
            if alloc[id(p)] < b.max_chunks_per_aspect:
                alloc[id(p)] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break  # everyone at MAX; remaining budget left unused

    # Deficit: reclaim from lowest-score aspects above MIN.
    order_low = sorted(active, key=lambda p: p.score)
    while assigned > total:
        progressed = False
        for p in order_low:
            if assigned <= total:
                break
            if alloc[id(p)] > b.min_chunks_per_aspect:
                alloc[id(p)] -= 1
                assigned -= 1
                progressed = True
        if not progressed:
            break

    for p in active:
        p.budget = alloc[id(p)]
    return plans


__all__ = [
    "GLOBAL_VIEW_NAME",
    "ViewSpec",
    "AspectPlan",
    "decompose",
    "global_view_spec",
    "score_aspects",
    "global_confidence",
    "allocate_budget",
]
