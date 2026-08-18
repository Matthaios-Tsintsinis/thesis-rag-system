"""ROUGE-L exactly as the de facto official NarrativeQA scorer computes it.

WHY THIS FILE EXISTS, and why it is configured rather than convenient.
ROUGE-L is in this thesis for ONE purpose: it is the bridge to published
NarrativeQA numbers. token-F1 is the primary answer metric and our own
convention is the defensible choice there. ROUGE-L is the opposite case
-- matching the de facto standard IS the entire point, and a bridge
computed under different settings bridges nothing.

THE SOURCE. `deepmind/narrativeqa` ships NO evaluation code at all --
`documents.csv`, `qaps.csv`, `download_stories.sh`, `compare.sh` and
`third_party/wikipedia/`, data and fetch scripts only, with no metrics
file and no evaluation procedure in its README. The de facto standard
implementation is AllenNLP's `allennlp_models/rc/tools/narrativeqa.py`,
and that is what the constants below are transcribed from:

    rouge_l_evaluator = rouge.Rouge(
        metrics=["rouge-l"],
        max_n=4,
        limit_length=True,
        length_limit=100,
        length_limit_type="words",
        apply_avg=True,
        apply_best=True,
        alpha=0.5,
        weight_factor=1.2,
        stemming=True,
    )

    def rouge_l(p, g):
        return rouge_l_evaluator.get_scores(p, g)

`length_limit=100` TRUNCATES THE PREDICTION TO 100 WORDS and
`stemming=True` applies Porter stemming. A bare `rouge-l` call does
neither, and would be a different number wearing the benchmark's label.
`max_n=4` is passed by the reference even though ROUGE-L uses no n-gram
order; it is reproduced for exactness rather than reasoned about.

MAX OVER REFERENCES, per `metric_max_over_ground_truths`: each reference
is scored separately and the maximum is taken. Note the reference takes
the max of f, p and r INDEPENDENTLY, so the returned triple need not come
from a single reference. That is reproduced, not corrected -- it is the
published behaviour and this module exists to match it.

# === DECLARED DEPARTURE FROM THE ALLENNLP SCRIPT: NO ROUNDING ===
#
# `metric_max_over_ground_truths` wraps every per-example score in
# `round(..., 2)`, quantising to a 0.01 grid BEFORE the caller averages.
# WE DO NOT REPRODUCE THAT, by ruling:
#
#   * it is a REPORTING artifact, not part of the metric's definition --
#     ROUGE-L is defined by its LCS and its alpha, not by a display
#     precision;
#   * quantising per-example scores before aggregation degrades the mean
#     for no benefit: it injects up to 0.005 of rounding error per row
#     that does not cancel in general;
#   * every other number in this harness is carried at full precision,
#     and one column silently on a coarser grid is the kind of thing that
#     is discovered later by someone recomputing a mean and failing to
#     reproduce it.
#
# So: SAME METRIC, FULL PRECISION. Declared in the deviations table
# rather than silently differing.

# === BLEU IS DECLINED, NOT APPROXIMATED ===
#
# The same AllenNLP script computes BLEU-4 as
# `sentence_bleu(g, p, weights=(0, 0, 0, 1))` -- 100% of the weight on
# 4-grams, per sentence, rather than the standard geometric mean over
# 1..4-grams. Published NarrativeQA "BLEU-4" is therefore NOT standard
# BLEU-4, and reproducing it would mean reproducing a non-standard metric
# in order to be comparable to it.
#
# We report token-F1 as primary and ROUGE-L as the bridge. BLEU is
# DECLINED rather than approximated -- the same posture as HotpotQA's
# supporting facts, and for the same reason: a number that is neither the
# official quantity nor our own is worse than an honest absence.
"""

from __future__ import annotations

from typing import Any, Callable


# Transcribed verbatim from the AllenNLP script's `rouge.Rouge(...)`
# construction. Pinned as a literal because it is a COMMITMENT -- the
# whole value of this column is that it matches the de facto standard, so
# a change here must break a test rather than pass quietly.
ALLENNLP_ROUGE_L_CONFIG: dict[str, Any] = {
    "metrics": ["rouge-l"],
    "max_n": 4,
    "limit_length": True,
    "length_limit": 100,
    "length_limit_type": "words",
    "apply_avg": True,
    "apply_best": True,
    "alpha": 0.5,
    "weight_factor": 1.2,
    "stemming": True,
}

# Bumped if the produced number changes for identical input. ROUGE-L is
# post-hoc and lands in no cache key, so this is provenance only.
ROUGE_L_IMPL_VERSION = "allennlp_narrativeqa_rouge_l_v1_unrounded"


def build_evaluator(rouge_module: Any | None = None) -> Any:
    """Construct the `rouge.Rouge` evaluator with the pinned config.

    LAZY IMPORT, DELIBERATELY. `rouge` is NOT in `requirements.txt` and
    MUST NOT be added to `requirements.lock` while the matrix is running:
    `pin_environment.lockfile_hash` hashes the lockfile's requirement
    lines, cell 1 is banked at `17878bc8740173be`, and regenerating the
    lock mid-matrix would move that hash and fire the environment gate on
    every remaining cell.

    That costs nothing, because ROUGE-L is computed POST-HOC over the
    stored `predicted_answer` field: no cell needs this package while it
    runs. Install it on whatever host computes the column.

    `rouge_module` is an INJECTION SEAM so the pinned kwargs can be
    asserted by driving this function for real, on a host without the
    package. It is not a configuration knob and production never passes
    it. The seam exists because the alternative — a test that greps this
    function's source for the call — proves the call is written, never
    that it runs, which is the defect class this project keeps finding.
    """
    if rouge_module is None:
        try:
            import rouge as rouge_module  # type: ignore[no-redef]
        except ImportError as e:  # pragma: no cover - exercised by absence
            raise ImportError(
                "ROUGE-L needs the `rouge` package (pip install rouge). It "
                "is deliberately absent from requirements.txt/"
                "requirements.lock: the lockfile hash is pinned mid-matrix "
                "and ROUGE-L is a post-hoc column, so no cell run requires "
                "it."
            ) from e
    return rouge_module.Rouge(**ALLENNLP_ROUGE_L_CONFIG)


def rouge_l_max_over_references(
    prediction: str,
    references: tuple[str, ...] | list[str],
    *,
    evaluator_factory: Callable[[], Any] = build_evaluator,
) -> dict[str, float]:
    """ROUGE-L f/p/r, max over references, at full precision.

    Mirrors `metric_max_over_ground_truths(rouge_l, prediction,
    ground_truths, tokenize=False)`: each reference is passed to
    `get_scores(p, [g])` on its own and the maximum of each field is
    taken independently, exactly as the reference does.

    `evaluator_factory` is a SEAM, not a knob. It exists so the pinned
    configuration can be asserted end to end without the `rouge` package
    installed; production always uses the default.

    Returns {"f": .., "p": .., "r": ..}. Empty references -> zeros.
    """
    refs = [r for r in references if r and r.strip()]
    if not prediction or not prediction.strip() or not refs:
        return {"f": 0.0, "p": 0.0, "r": 0.0}

    evaluator = evaluator_factory()
    best = {"f": 0.0, "p": 0.0, "r": 0.0}
    for ref in refs:
        scores = evaluator.get_scores(prediction, [ref])
        rl = scores["rouge-l"]
        for field in ("f", "p", "r"):
            value = float(rl[field])
            if value > best[field]:
                best[field] = value
    # NO round(..., 2) -- see the declared departure in the module
    # docstring. Same metric, full precision.
    return best


__all__ = [
    "ALLENNLP_ROUGE_L_CONFIG",
    "ROUGE_L_IMPL_VERSION",
    "build_evaluator",
    "rouge_l_max_over_references",
]
