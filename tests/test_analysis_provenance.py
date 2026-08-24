"""Two analysis-layer provenance guards, both earned on the 16/16 bank.

1. `significance_diagnostic._generator_note` — the answer-delta provenance
   line is READ from the sibling summaries, never asserted. The previous
   version printed a hardcoded "gpt-4o-mini-era, must be re-measured" over
   the local-Qwen P10 bank: instance 15 of the recurring lesson — the
   summaries recorded the generator and the note consumed nothing.

2. `score_rouge_l._punkt_fix_message` — the ROUGE preflight's refusal text
   must name BOTH fix paths, because the obvious advice
   (nltk.download(..., quiet=True)) fails SILENTLY on proxied hosts and a
   verification one-liner printed "nltk ok" over a refused download.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_rouge_l import _punkt_fix_message
from scripts.significance_diagnostic import _generator_note


def _summary(path: Path, generator: str | None) -> None:
    body: dict = {"n_queries_scored": 1}
    if generator is not None:
        body["model_revisions"] = {"generator": generator}
    path.write_text(json.dumps(body), encoding="utf-8")


class TestGeneratorNote(unittest.TestCase):
    def test_one_generator_is_named_verbatim(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _summary(root / "a.summary.json", "Qwen/Qwen2.5-7B-Instruct")
            _summary(root / "b.summary.json", "Qwen/Qwen2.5-7B-Instruct")
            note = _generator_note(root)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", note)
        self.assertNotIn("gpt-4o-mini", note)

    def test_disagreeing_generators_warn_and_name_both(self):
        """Two generators in one bank is a finding, not a formatting
        problem — the note must surface it, not pick one."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _summary(root / "a.summary.json", "Qwen/Qwen2.5-7B-Instruct")
            _summary(root / "b.summary.json", "meta-llama/Llama-3.1-8B")
            note = _generator_note(root)
        self.assertIn("MULTIPLE", note)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", note)
        self.assertIn("meta-llama/Llama-3.1-8B", note)

    def test_unrecorded_generator_says_unrecorded_not_an_era(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _summary(root / "a.summary.json", None)
            note = _generator_note(root)
        self.assertIn("unrecorded", note)
        self.assertNotIn("gpt-4o-mini", note)

    def test_no_summaries_at_all_is_stated(self):
        with tempfile.TemporaryDirectory() as td:
            note = _generator_note(Path(td))
        self.assertIn("NO summaries", note)

    def test_the_stale_string_is_gone_from_the_module(self):
        """The one place a source assertion is the right tool: the defect
        WAS a string literal, so its absence is the fix — paired with the
        behavioural tests above that prove what replaced it."""
        src = Path("scripts/significance_diagnostic.py").read_text(
            encoding="utf-8")
        self.assertNotIn(
            "answer deltas are "
            + chr(34),  # avoid matching THIS test file's own docstring
            src)
        self.assertEqual(
            src.count("gpt-4o-mini-era"), 2,
            "gpt-4o-mini-era may appear only in the two comments that "
            "RECORD the defect, never in a printed string")


class TestPunktMessage(unittest.TestCase):
    def test_both_fix_paths_are_named(self):
        msg = _punkt_fix_message()
        self.assertIn("NLTK_ALLOW_PROXIED_URLOPEN=1", msg)
        self.assertIn("punkt_tab.zip", msg)
        self.assertIn("nltk_data/tokenizers/", msg)

    def test_the_silent_failure_trap_is_warned_about(self):
        self.assertIn("SILENTLY", _punkt_fix_message())


if __name__ == "__main__":
    unittest.main()
