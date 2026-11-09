"""Pins v19: a reader that has nothing but does not say it in the refusal's words is
read as the refusal, last, after the cut, so no call is added, dropped or changed."""

from __future__ import annotations

import unittest
from unittest import mock

from ours import config as settings
from ours import model as om
from tests.test_ours_reasoning import ReasoningReader, run

EMPTY = ("Chunk 1 says topic0x0 travelled with a friend.\n"
         "The evidence does not mention where they went.")


class TestUnreadRefusal(unittest.TestCase):
    """What the test takes, on a last reply and the final answer."""

    def test_no_answer_line_and_a_last_line_that_has_nothing(self):
        self.assertTrue(om.unread_refusal(EMPTY, EMPTY))

    def test_its_words(self):
        for last in ("No information on this.", "It is not stated in the evidence.", "The chunks do not say.",
                     "The year cannot be determined.", "Her birthplace is unknown.", "The evidence is insufficient."):
            reply = "Chunk 1 names a castle.\n" + last
            self.assertTrue(om.unread_refusal(reply, reply), last)

    def test_an_answer_line_is_read_as_it_stands(self):
        reply = "The evidence does not mention a river, but Chunk 2 names the castle.\nAnswer: the castle on the hill"
        self.assertFalse(om.unread_refusal(reply, "the castle on the hill"))

    def test_only_the_last_line_counts_when_there_is_no_answer_line(self):
        reply = "Chunk 1 does not mention the year.\nChunk 2 gives it: they reached the river in 1840."
        self.assertFalse(om.unread_refusal(reply, reply))

    def test_a_short_final_answer_that_has_nothing(self):
        reply = "Facts.\nAnswer: No information."
        self.assertTrue(om.unread_refusal(reply, "No information."))

    def test_a_long_final_answer_is_not_read_for_its_words(self):
        final = "the river that is not mentioned by name in Chunk 1 but which Chunk 2 places north of the castle walls"
        self.assertGreater(len(final.split()), settings.SHORT_NOTHING_WORDS)
        self.assertFalse(om.unread_refusal("Facts.\nAnswer: " + final, final))

    def test_an_answer_is_no_refusal(self):
        self.assertFalse(om.unread_refusal("Facts.\nAnswer: the river", "the river"))

    def test_a_refusal_is_left_alone(self):
        self.assertFalse(om.unread_refusal(f"Nothing here.\nAnswer: {settings.REFUSAL}", settings.REFUSAL))

    def test_a_title_that_reads_as_nothing_is_taken(self):
        # Declared in the version log: the words are the 6 Nov replay's, unchanged.
        self.assertTrue(om.unread_refusal("Facts.\nAnswer: Unknown Pleasures", "Unknown Pleasures"))

    def test_shipped_on(self):
        self.assertTrue(settings.READ_UNREAD_REFUSALS)


class TestTheAnswerPath(unittest.TestCase):
    def test_a_reply_that_has_nothing_is_the_refusal(self):
        result = run(ReasoningReader(reasoned=EMPTY))
        self.assertEqual(result.answer, settings.REFUSAL)
        self.assertFalse(result.extra["ours_reasked"])
        self.assertEqual(result.extra["ours_reader_answer"], EMPTY)
        self.assertEqual(result.extra["ours_reasoning"], EMPTY)

    def test_switched_off_it_goes_on_to_the_cut(self):
        # v18: the whole reply goes to the cut, whose first line here is the reasoning's.
        with mock.patch.object(settings, "READ_UNREAD_REFUSALS", False):
            result = run(ReasoningReader(reasoned=EMPTY))
        self.assertEqual(result.extra["ours_reader_answer"], EMPTY)
        self.assertEqual(result.answer, EMPTY.splitlines()[0])

    def test_no_call_changes(self):
        on_reader, off_reader = ReasoningReader(reasoned=EMPTY), ReasoningReader(reasoned=EMPTY)
        on = run(on_reader)
        with mock.patch.object(settings, "READ_UNREAD_REFUSALS", False):
            off = run(off_reader)
        self.assertEqual(on_reader.prompts, off_reader.prompts)
        self.assertEqual(on.extra, off.extra)
        self.assertNotEqual(on.answer, off.answer)

    def test_read_after_the_cut(self):
        # A cut that leaves only "No information." is taken; the cut itself was still asked.
        reader = ReasoningReader(reasoned="Facts.\nAnswer: No information about the river.",
                                 cuts={"No information about the river.": "No information."})
        result = run(reader)
        self.assertEqual(result.answer, settings.REFUSAL)
        self.assertTrue(any(s == settings.SHORT_ANSWER_PROMPT for s, _ in reader.prompts))

    def test_an_answer_goes_on_as_before(self):
        reader = ReasoningReader(cuts={"the river in the north": "the river"})
        self.assertEqual(run(reader).answer, "the river")


if __name__ == "__main__":
    unittest.main()
