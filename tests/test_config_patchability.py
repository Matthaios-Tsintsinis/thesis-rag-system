"""Which config constants a monkeypatch can reach — and the cap that must.

WHY THIS FILE EXISTS. A probe was run with
`src.config.GEN_MAX_NEW_TOKENS = 1` set in-process before importing the
runner. It printed 1. It generated full-length answers anyway, scored
0.211 answer-F1 where a one-token answer cannot score above ~0.05, and
the resulting timings were used as a measurement. Nothing raised.

The mechanism: `GenerationConfig.max_new_tokens` takes the constant as a
dataclass field DEFAULT, and Python evaluates field defaults ONCE at
class-definition time. Rebinding the module attribute afterwards changes
nothing any dataclass reads.

This is not specific to that one constant — it is how every constant in
src/config.py is consumed EXCEPT `EVIDENCE_TOKEN_BUDGET`, which is read
through the module object at call time on purpose. These tests pin both
halves so the trap is documented in executable form rather than in a
comment somebody can miss, and so a future refactor that quietly makes
`--evidence-budget` inert gets caught.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import src.config as C
from src.config import GenerationConfig, HarnessConfig
from src.prompt_packing import _resolve_budget


class TestRebindingIsSilentlyIgnored(unittest.TestCase):
    """The trap. Every one of these is a rebind that does NOT take."""

    def _rebind_does_not_reach(self, name, value, read):
        old = getattr(C, name)
        setattr(C, name, value)
        try:
            got = read(HarnessConfig())
        finally:
            setattr(C, name, old)
        self.assertNotEqual(
            got, value,
            f"src.config.{name} became patchable. That is a WELCOME change, "
            "but it means this test and the --max-new-tokens flag's "
            "justification are stale — update both rather than deleting "
            "the assertion.",
        )

    def test_generation_constants_do_not_reach_a_fresh_config(self):
        for name, value, read in [
            ("GEN_MAX_NEW_TOKENS", 12345, lambda c: c.generation.max_new_tokens),
            ("GEN_TEMPERATURE", 0.77, lambda c: c.generation.temperature),
            ("GEN_TOP_P", 0.33, lambda c: c.generation.top_p),
            ("GENERATOR_MODEL", "X/Y", lambda c: c.generation.model),
            ("LOAD_GENERATOR_IN_4BIT", True, lambda c: c.generation.load_in_4bit),
        ]:
            with self.subTest(constant=name):
                self._rebind_does_not_reach(name, value, read)

    def test_retrieval_constants_do_not_reach_a_fresh_config(self):
        for name, value, read in [
            ("FINAL_CONTEXT_CHUNKS", 99, lambda c: c.retrieval.top_k),
            ("FIRST_STAGE_TOP_K", 98, lambda c: c.retrieval.first_stage_top_k),
            ("RRF_K", 97, lambda c: c.retrieval.rrf_k),
            ("JUDGE_MODEL", "J/M", lambda c: c.m4.summary_model),
        ]:
            with self.subTest(constant=name):
                self._rebind_does_not_reach(name, value, read)


class TestEvidenceBudgetIsTheOneThatWorks(unittest.TestCase):
    """`--evidence-budget` monkey-patches, and that path must keep working.

    It is the ONE constant read through the module object at call time
    (`prompt_packing._resolve_budget` late-imports `config`), which is why
    the CK-4 ablation CLI is not affected by the trap above.
    """

    def test_rebinding_reaches_the_packer(self):
        old = C.EVIDENCE_TOKEN_BUDGET
        C.EVIDENCE_TOKEN_BUDGET = 3000
        try:
            self.assertEqual(_resolve_budget(None), 3000)
        finally:
            C.EVIDENCE_TOKEN_BUDGET = old

    def test_an_explicit_argument_still_wins(self):
        self.assertEqual(_resolve_budget(1234), 1234)


class TestExplicitConstructionIsTheLever(unittest.TestCase):
    """What the runner's --max-new-tokens actually does."""

    def test_replace_reaches_generation(self):
        cfg = HarnessConfig()
        capped = replace(
            cfg, generation=replace(cfg.generation, max_new_tokens=1)
        )
        self.assertEqual(capped.generation.max_new_tokens, 1)

    def test_the_system_reads_the_overridden_config(self):
        """The property that matters: a system built from the overridden
        config carries it to the object generation actually consumes."""
        from src.retrievers.m2_flat_dense import FlatDenseSystem

        cfg = HarnessConfig()
        capped = replace(
            cfg, generation=replace(cfg.generation, max_new_tokens=7)
        )
        sysm = FlatDenseSystem(config=capped)
        self.assertEqual(sysm.config.generation.max_new_tokens, 7)

    def test_default_is_unchanged_by_an_override_elsewhere(self):
        self.assertEqual(
            GenerationConfig().max_new_tokens, C.GEN_MAX_NEW_TOKENS
        )


class TestCapVerification(unittest.TestCase):
    """The runner must ABORT on an unapplied cap, not report a number."""

    def _runner(self, cap):
        from pathlib import Path
        import tempfile

        from src.eval.base import BenchmarkRunner

        self._td = tempfile.TemporaryDirectory()
        return BenchmarkRunner(
            output_path=Path(self._td.name) / "o.jsonl",
            verbose=False,
            verify_max_new_tokens=cap,
        )

    def tearDown(self):
        td = getattr(self, "_td", None)
        if td is not None:
            td.cleanup()

    def test_an_overlong_answer_raises(self):
        r = self._runner(1)
        r._verify_tok = lambda text, add_special_tokens=False: {
            "input_ids": list(range(len(text.split())))
        }
        with self.assertRaises(RuntimeError) as cm:
            r._check_output_length(
                "this is a full length answer that ignored the cap entirely",
                "q1", "model",
            )
        self.assertIn("NOT APPLIED", str(cm.exception))

    def test_one_token_of_slack_is_tolerated(self):
        """decode -> strip -> re-encode is not an exact inverse of
        generation, so an off-by-one must not abort a real run."""
        r = self._runner(1)
        r._verify_tok = lambda text, add_special_tokens=False: {
            "input_ids": list(range(len(text.split())))
        }
        r._check_output_length("two words", "q1", "model")  # 2 <= 1+1

    def test_disabled_by_default(self):
        r = self._runner(None)
        r._verify_tok = None
        r._check_output_length("anything at all, arbitrarily long", "q", "m")


if __name__ == "__main__":
    unittest.main()
