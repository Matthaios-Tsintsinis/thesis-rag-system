"""Pins that the reader reasons before it answers: only its last "Answer:" line goes on
to step 10, a refusal is read from that line, and the baselines' plain prompt is asked
once on the same evidence, saved for the report, never the answer and never counted."""

from __future__ import annotations

import tempfile
import unittest

from ours import config as settings
from ours import model as om
from tests.test_ours_smoke import DECOMPOSITIONS, FakeReader, Q0, hotpot_unit, llama_config, ours_fakes

PLAIN = "They travelled to the river."
REASONED = ("Chunk 1 says topic0x0 travelled with a friend. Chunk 2 says they reached the river.\n"
            "Answer: the river in the north")


class ReasoningReader(FakeReader):
    """Replies by prompt: the plain prompt, the reasoned one, and the reasoned second asking."""

    def __init__(self, reasoned=REASONED, reasked=settings.REFUSAL, plain=PLAIN, cuts=None):
        super().__init__(DECOMPOSITIONS, {}, cuts=cuts)
        self.by_prompt = {settings.ANSWER_PROMPT: plain, settings.READER_PROMPT: reasoned,
                          settings.READER_REASK_PROMPT: reasked}

    def __call__(self, system_prompt, user_prompt, cfg=None):
        if system_prompt in self.by_prompt:
            self.prompts.append((system_prompt, user_prompt))
            return self.by_prompt[system_prompt]
        return super().__call__(system_prompt, user_prompt, cfg)


def run(reader, question=Q0):
    with tempfile.TemporaryDirectory() as td, ours_fakes(td, reader):
        system = om.OursSystem(config=llama_config())
        system.index_items(hotpot_unit(0).corpus)
        return system.answer(question)


class TestFinalAnswer(unittest.TestCase):
    """What is read out of the reader's reply."""

    def test_the_last_answer_line(self):
        self.assertEqual(om.final_answer("Facts here.\nAnswer: the river"), "the river")

    def test_the_last_of_several(self):
        self.assertEqual(om.final_answer("Answer: a guess\nMore facts.\nAnswer: the river"), "the river")

    def test_its_forms(self):
        for reply in ("Facts.\n**Answer:** the river", "Facts.\nFinal answer: the river",
                      "Facts.\nanswer:   the river  ", "Facts.\nAnswer:\nthe river\nand a note"):
            self.assertEqual(om.final_answer(reply), "the river", reply)

    def test_no_line_leaves_the_reply(self):
        for reply in ("the river", settings.REFUSAL, "The answer: it is somewhere in the text."):
            self.assertEqual(om.final_answer(reply), reply)

    def test_an_empty_answer_line_leaves_the_reply(self):
        self.assertEqual(om.final_answer("Facts.\nAnswer:"), "Facts.\nAnswer:")


class TestPrompts(unittest.TestCase):
    def test_the_reader_keeps_the_baselines_prompt_and_adds_the_reasoning(self):
        self.assertEqual(settings.READER_PROMPT, settings.ANSWER_PROMPT + " " + settings.REASON_LINE)
        self.assertEqual(settings.READER_REASK_PROMPT, settings.REASK_PROMPT + " " + settings.REASON_LINE)

    def test_the_example_is_invented(self):
        self.assertIn("Anna Reed", settings.REASON_LINE)
        self.assertIn("Answer:", settings.REASON_LINE)


class TestTheAnswerPath(unittest.TestCase):
    def test_only_the_answer_line_goes_on(self):
        reader = ReasoningReader(cuts={"the river in the north": "the river"})
        result = run(reader)
        self.assertEqual(result.answer, "the river")
        self.assertEqual(result.extra["ours_reader_answer"], "the river in the north")
        self.assertEqual(result.extra["ours_first_answer"], "the river in the north")
        self.assertEqual(result.extra["ours_reasoning"], REASONED)
        cuts = [u for s, u in reader.prompts if s == settings.SHORT_ANSWER_PROMPT]
        self.assertEqual(cuts, [f"Question: {Q0}\nAnswer: the river in the north"])

    def test_the_plain_answer_is_saved_and_is_not_the_answer(self):
        result = run(ReasoningReader())
        self.assertEqual(result.extra["ours_plain_answer"], PLAIN)
        self.assertNotEqual(result.answer, PLAIN)

    def test_the_plain_call_reads_the_same_evidence_and_is_not_counted(self):
        reader = ReasoningReader()
        result = run(reader)
        prompts = dict(reader.prompts)
        self.assertEqual(prompts[settings.ANSWER_PROMPT], prompts[settings.READER_PROMPT])
        plain = [s for s, _ in reader.prompts if s == settings.ANSWER_PROMPT]
        self.assertEqual(len(plain), 1)
        self.assertEqual(result.extra["ours_n_llm_calls"], len(reader.prompts) - 1)

    def test_a_refusal_is_read_from_the_answer_line(self):
        reader = ReasoningReader(reasoned=f"Chunk 1 does not say.\nAnswer: {settings.REFUSAL}",
                                 reasked="Chunk 2 says the castle.\nAnswer: the castle")
        result = run(reader)
        self.assertTrue(result.extra["ours_reasked"])
        self.assertEqual(result.extra["ours_first_answer"], settings.REFUSAL)
        self.assertEqual(result.extra["ours_reader_answer"], "the castle")
        self.assertEqual(result.extra["ours_reasoning"], "Chunk 2 says the castle.\nAnswer: the castle")
        self.assertEqual(result.answer, "the castle")

    def test_reasoning_that_doubts_but_answers_is_no_refusal(self):
        reader = ReasoningReader(reasoned=f"{settings.REFUSAL} might be expected, but Chunk 1 says it.\nAnswer: the river")
        result = run(reader)
        self.assertFalse(result.extra["ours_reasked"])
        self.assertEqual(result.extra["ours_reader_answer"], "the river")

    def test_a_refusal_twice_stays_a_refusal_and_is_not_cut(self):
        reader = ReasoningReader(reasoned=f"Answer: {settings.REFUSAL}", reasked=f"Nothing.\nAnswer: {settings.REFUSAL}")
        result = run(reader)
        self.assertEqual(result.answer, settings.REFUSAL)
        self.assertFalse(any(s == settings.SHORT_ANSWER_PROMPT for s, _ in reader.prompts))

    def test_a_reply_with_no_answer_line_goes_on_whole(self):
        reader = ReasoningReader(reasoned="the river in the north")
        result = run(reader)
        self.assertEqual(result.extra["ours_reader_answer"], "the river in the north")

    def test_a_why_question_keeps_its_answer_line_uncut(self):
        why = "Why did topic0x0 travel with topic0x1?"
        reader = ReasoningReader(reasoned="Facts.\nAnswer: Because they wanted to see the river.",
                                 cuts={"Because they wanted to see the river.": "the river"})
        result = run(reader, why)
        self.assertEqual(result.answer, "Because they wanted to see the river.")
        self.assertFalse(any(s == settings.SHORT_ANSWER_PROMPT for s, _ in reader.prompts))


if __name__ == "__main__":
    unittest.main()
