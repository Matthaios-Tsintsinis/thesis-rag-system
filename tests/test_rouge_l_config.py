"""The ROUGE-L configuration is a COMMITMENT and is pinned as one.

ROUGE-L exists in this thesis only as the bridge to published NarrativeQA
numbers, so matching the de facto standard IS the point — unlike
token-F1, where our own convention is the defensible choice. A bridge
computed under different settings bridges nothing.

These tests do TWO different jobs, deliberately, because the project has
repeatedly shipped constants nothing read:

  * one pins the literal against the AllenNLP source, transcribed below
    so the assertion carries its own provenance;
  * the others drive `rouge_l_max_over_references` end to end through an
    injected evaluator and assert what the PIPELINE actually constructed
    and how it combined references. "Does it exist" and "does the code
    read it" are separate questions and both are asked here.

The `rouge` package is NOT required to run any of this — that is the
reason for the `evaluator_factory` seam. `rouge` is deliberately absent
from requirements.lock while the matrix runs.
"""

from __future__ import annotations

import unittest

from src.eval.scorers.rouge_l import (
    ALLENNLP_ROUGE_L_CONFIG,
    build_evaluator,
    rouge_l_max_over_references,
)


# Transcribed from allennlp_models/rc/tools/narrativeqa.py:
#
#     rouge_l_evaluator = rouge.Rouge(
#         metrics=["rouge-l"],
#         max_n=4,
#         limit_length=True,
#         length_limit=100,
#         length_limit_type="words",
#         apply_avg=True,
#         apply_best=True,
#         alpha=0.5,
#         weight_factor=1.2,
#         stemming=True,
#     )
ALLENNLP_SOURCE_KWARGS = {
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


class _RecordingEvaluator:
    """Stands in for `rouge.Rouge`, recording how it was called."""

    def __init__(self, scores_by_ref: dict[str, dict[str, float]]) -> None:
        self.scores_by_ref = scores_by_ref
        self.calls: list[tuple[str, list[str]]] = []

    def get_scores(self, prediction, references):
        self.calls.append((prediction, list(references)))
        (ref,) = references
        return {"rouge-l": dict(self.scores_by_ref[ref])}


class TestTheConfigMatchesTheSource(unittest.TestCase):
    def test_every_kwarg_matches_the_allennlp_construction(self):
        self.assertEqual(ALLENNLP_ROUGE_L_CONFIG, ALLENNLP_SOURCE_KWARGS)

    def test_the_two_settings_that_actually_change_the_number(self):
        """A bare `rouge-l` call does neither of these, and the result
        would be a different metric under the same name."""
        self.assertTrue(ALLENNLP_ROUGE_L_CONFIG["limit_length"])
        self.assertEqual(ALLENNLP_ROUGE_L_CONFIG["length_limit"], 100)
        self.assertEqual(ALLENNLP_ROUGE_L_CONFIG["length_limit_type"], "words")
        self.assertTrue(ALLENNLP_ROUGE_L_CONFIG["stemming"])


class _FakeRougeModule:
    """Stands in for the `rouge` package, recording construction kwargs."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def Rouge(self, **kwargs):  # noqa: N802 - mirrors the real class name
        self.kwargs = kwargs
        return _RecordingEvaluator({"a": {"f": 0.5, "p": 0.5, "r": 0.5}})


class TestThePipelineReadsIt(unittest.TestCase):
    """The behavioural half — a constant nothing consumes is the defect
    class this project keeps rediscovering.

    This drives the REAL `build_evaluator` and asserts what it actually
    passed to `Rouge(...)`. An earlier draft grepped the function's
    source for the call instead; that was caught in review of its own
    diff, and it is the fourteenth instance of the same reflex. A grep
    proves the call is written, never that it runs.
    """

    def test_build_evaluator_passes_the_pinned_kwargs(self):
        fake = _FakeRougeModule()
        build_evaluator(rouge_module=fake)
        self.assertEqual(fake.kwargs, ALLENNLP_SOURCE_KWARGS)

    def test_the_scorer_reaches_the_evaluator_it_built(self):
        fake = _FakeRougeModule()
        out = rouge_l_max_over_references(
            "pred", ("a",),
            evaluator_factory=lambda: build_evaluator(rouge_module=fake),
        )
        self.assertEqual(fake.kwargs, ALLENNLP_SOURCE_KWARGS)
        self.assertEqual(out, {"f": 0.5, "p": 0.5, "r": 0.5})


class TestMaxOverReferences(unittest.TestCase):
    """Mirrors `metric_max_over_ground_truths`."""

    def test_each_reference_is_scored_separately(self):
        ev = _RecordingEvaluator({
            "ref one": {"f": 0.2, "p": 0.9, "r": 0.1},
            "ref two": {"f": 0.7, "p": 0.3, "r": 0.8},
        })
        rouge_l_max_over_references("p", ("ref one", "ref two"),
                                    evaluator_factory=lambda: ev)
        self.assertEqual([c[1] for c in ev.calls], [["ref one"], ["ref two"]])

    def test_f_p_and_r_are_maxed_INDEPENDENTLY(self):
        """The reference takes the max of each field on its own, so the
        triple need not come from one reference. Reproduced, not fixed —
        this module exists to match published behaviour."""
        ev = _RecordingEvaluator({
            "ref one": {"f": 0.2, "p": 0.9, "r": 0.1},
            "ref two": {"f": 0.7, "p": 0.3, "r": 0.8},
        })
        out = rouge_l_max_over_references("p", ("ref one", "ref two"),
                                          evaluator_factory=lambda: ev)
        self.assertEqual(out, {"f": 0.7, "p": 0.9, "r": 0.8})

    def test_empty_prediction_or_references_scores_zero(self):
        ev = _RecordingEvaluator({})
        for pred, refs in (("", ("a",)), ("   ", ("a",)), ("p", ()), ("p", ("",))):
            self.assertEqual(
                rouge_l_max_over_references(pred, refs,
                                            evaluator_factory=lambda: ev),
                {"f": 0.0, "p": 0.0, "r": 0.0}, (pred, refs))


class TestTheRoundingDepartureIsRealAndDeliberate(unittest.TestCase):
    """AllenNLP wraps every per-example score in `round(..., 2)`. We do
    not, by ruling: it is a reporting artifact rather than part of the
    metric's definition, and quantising before aggregation degrades the
    mean for no benefit."""

    def test_scores_are_not_quantised_to_two_decimals(self):
        ev = _RecordingEvaluator({"a": {"f": 0.123456, "p": 0.234567,
                                        "r": 0.345678}})
        out = rouge_l_max_over_references("p", ("a",),
                                          evaluator_factory=lambda: ev)
        self.assertAlmostEqual(out["f"], 0.123456, places=6)
        self.assertNotEqual(out["f"], round(0.123456, 2))


if __name__ == "__main__":
    unittest.main()
