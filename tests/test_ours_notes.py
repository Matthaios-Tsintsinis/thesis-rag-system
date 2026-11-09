"""Pins v20: a final answer of several lines is the reader's notes gone on whole, and goes
out as the refusal, read last, after the cut, so no call is added, dropped or changed."""

from __future__ import annotations

import unittest
from unittest import mock

from ours import config as settings
from ours import model as om
from tests.test_ours_reasoning import ReasoningReader, run

NOTES = ("Facts that bear on the question:\n"
         "- Chunk 1 says topic0x0 travelled with a friend.\n"
         "- Chunk 2 says they reached the river.")


class TestIsNotes(unittest.TestCase):
    def test_several_lines_are_notes(self):
        self.assertTrue(om.is_notes(NOTES))

    def test_one_line_is_an_answer(self):
        for final in ("the river", "The evidence does not mention the river.", "No"):
            self.assertFalse(om.is_notes(final), final)

    def test_surrounding_space_is_not_a_line(self):
        self.assertFalse(om.is_notes("\n  the river  \n"))

    def test_a_refusal_is_left_alone(self):
        self.assertFalse(om.is_notes(settings.REFUSAL))
        self.assertFalse(om.is_notes(None))

    def test_its_last_line_is_not_read(self):
        # Unlike v19, a right answer at the end of the notes does not save them.
        self.assertTrue(om.is_notes("Chunk 1 names the castle.\nIt is the castle on the hill."))

    def test_shipped_on(self):
        self.assertTrue(settings.NOTES_ARE_REFUSALS)


class TestTheAnswerPath(unittest.TestCase):
    def test_notes_that_go_on_whole_are_the_refusal(self):
        # A cut that refuses leaves them whole, as one that is not shorter or adds a word.
        reader = ReasoningReader(reasoned=NOTES, cuts={NOTES: settings.REFUSAL})
        result = run(reader)
        self.assertEqual(result.answer, settings.REFUSAL)
        self.assertEqual(result.extra["ours_reader_answer"], NOTES)
        self.assertTrue(any(s == settings.SHORT_ANSWER_PROMPT for s, _ in reader.prompts))

    def test_switched_off_they_go_out(self):
        with mock.patch.object(settings, "NOTES_ARE_REFUSALS", False):
            result = run(ReasoningReader(reasoned=NOTES, cuts={NOTES: settings.REFUSAL}))
        self.assertEqual(result.answer, NOTES)

    def test_a_cut_to_one_line_is_kept(self):
        reader = ReasoningReader(reasoned=NOTES, cuts={NOTES: "the river"})
        self.assertEqual(run(reader).answer, "the river")

    def test_no_call_changes(self):
        on_reader = ReasoningReader(reasoned=NOTES, cuts={NOTES: settings.REFUSAL})
        off_reader = ReasoningReader(reasoned=NOTES, cuts={NOTES: settings.REFUSAL})
        on = run(on_reader)
        with mock.patch.object(settings, "NOTES_ARE_REFUSALS", False):
            off = run(off_reader)
        self.assertEqual(on_reader.prompts, off_reader.prompts)
        self.assertEqual(on.extra, off.extra)
        self.assertNotEqual(on.answer, off.answer)

    def test_a_why_question_answered_in_several_lines_is_refused(self):
        # Declared in the version log: a why question is not cut, so its notes go out whole.
        why = "Why did topic0x0 travel with topic0x1?"
        result = run(ReasoningReader(reasoned=NOTES), why)
        self.assertEqual(result.answer, settings.REFUSAL)

    def test_an_answer_line_goes_on_as_before(self):
        reader = ReasoningReader(cuts={"the river in the north": "the river"})
        self.assertEqual(run(reader).answer, "the river")


if __name__ == "__main__":
    unittest.main()
