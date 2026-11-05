"""Pins ours' LLM steps: reading the decomposition JSON, falling back to one
aspect on a parse failure, the importance floor and the cap of three, the
views, and the answer prompt's exact refusal."""

from __future__ import annotations

import unittest
from unittest import mock

from ours import config as settings
from ours import model as om

Q = "Who directed Sinister and where was he born?"


class _Ask:
    """A scripted reader: one reply per prompt kind, every call recorded."""

    def __init__(self, decomposition: str, paraphrase="re: {x}",
                 hyde="passage on {x}") -> None:
        self.replies = {settings.DECOMPOSE_PROMPT: lambda x: decomposition,
                        settings.PARAPHRASE_PROMPT: lambda x: paraphrase.format(x=x),
                        settings.HYDE_PROMPT: lambda x: hyde.format(x=x)}
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, system_prompt, user_prompt, max_tokens):
        self.calls.append((system_prompt, user_prompt, max_tokens))
        return self.replies[system_prompt](user_prompt.split(": ", 1)[1])


class TestParse(unittest.TestCase):
    """parse_aspects reads the reply or returns None."""

    def test_simple_is_the_question_itself(self):
        got = om.parse_aspects('{"type": "simple", "aspects": '
                               '[{"name": "main", "importance": 1.0}]}', Q)
        self.assertEqual([(a.name, a.text, a.importance) for a in got],
                         [("main", Q, 1.0)])

    def test_multi_aspect(self):
        got = om.parse_aspects(
            '{"type": "multi_aspect", "aspects": ['
            '{"name": "Who directed Sinister?", "importance": 0.6},'
            '{"name": "Where was that director born?", "importance": 0.4}]}', Q)
        self.assertEqual([(a.text, a.importance) for a in got],
                         [("Who directed Sinister?", 0.6),
                          ("Where was that director born?", 0.4)])

    def test_code_fences_and_prose_around_the_json(self):
        body = '{"type": "simple", "aspects": [{"name": "main", "importance": 1}]}'
        for raw in (f"```json\n{body}\n```", f"Here it is: {body} Done."):
            self.assertIsNotNone(om.parse_aspects(raw, Q), raw)

    def test_a_reply_one_brace_short_is_read(self):
        """MultiHop's scored replies (v2): the reply ends after the aspects' ]."""
        body = ('{"type": "multi_aspect", "aspects": ['
                '{"name": "What is Twitch\'s revenue split?", "importance": 0.6}, '
                '{"name": "Does it use \\"tiers\\" {or not}?", "importance": 0.4}]')
        for raw in (body, f"```json\n{body}\n```"):
            got = om.parse_aspects(raw, Q)
            self.assertEqual([(a.text, a.importance) for a in got],
                             [("What is Twitch's revenue split?", 0.6),
                              ('Does it use "tiers" {or not}?', 0.4)], raw)

    def test_close_brackets(self):
        self.assertEqual(om.close_brackets('Here: {"a": [1, {"b": "]"}'), '{"a": [1, {"b": "]"}]}')
        for text in ("no brace", '{"a": 1}', '{"a": "cut mid-str', '{"a": [1}', '{"a": 1}}'):
            self.assertEqual(om.close_brackets(text), "", text)

    def test_unusable_replies_are_none(self):
        for raw in ("no json here", '{"type": "other"}',
                    '{"type": "multi_aspect", "aspects": [{"name": "cut mid-na',
                    '{"type": "multi_aspect", "aspects": [{"name": "x", "importance": 0.5},',
                    '{"type": "multi_aspect", "aspects": [{"name": "Did "Fortune" say so?", '
                    '"importance": 0.6}]',
                    '{"type": "multi_aspect", "aspects": []}',
                    '{"type": "multi_aspect", "aspects": [{"name": "x"}]}',
                    '{"type": "multi_aspect", "aspects": '
                    '[{"name": "x", "importance": "high"}]}',
                    '{"type": "multi_aspect", "aspects": '
                    '[{"name": "", "importance": 0.5}]}'):
            self.assertIsNone(om.parse_aspects(raw, Q), raw)


class TestChoose(unittest.TestCase):
    """The importance floor and the cap of three."""

    def _aspects(self, *imps):
        return [om.Aspect(name=f"a{i}", text=f"a{i}", importance=v)
                for i, v in enumerate(imps)]

    def test_low_importance_is_dropped(self):
        got = om.choose_aspects(self._aspects(0.6, 0.2, 0.3), Q)
        self.assertEqual([a.name for a in got], ["a0", "a2"])

    def test_at_most_three_most_important(self):
        got = om.choose_aspects(self._aspects(0.3, 0.9, 0.5, 0.4), Q)
        self.assertEqual([a.name for a in got], ["a1", "a2", "a3"])

    def test_all_dropped_falls_back_to_the_question(self):
        got = om.choose_aspects(self._aspects(0.1, 0.2), Q)
        self.assertEqual([(a.name, a.text) for a in got], [("main", Q)])

    def test_a_lone_aspect_is_kept_whatever_its_importance(self):
        got = om.choose_aspects(self._aspects(0.1), Q)
        self.assertEqual([a.name for a in got], ["a0"])


class TestDecompose(unittest.TestCase):
    """decompose and add_views call the reader as the design says."""

    def test_parse_failure_is_one_aspect_and_flagged(self):
        ask = _Ask("I cannot answer in JSON.")
        aspects, failed = om.decompose(Q, ask)
        self.assertTrue(failed)
        self.assertEqual([(a.name, a.text) for a in aspects], [("main", Q)])

    def test_two_views_per_aspect_with_their_caps(self):
        ask = _Ask('{"type": "multi_aspect", "aspects": ['
                   '{"name": "Who directed Sinister?", "importance": 0.6},'
                   '{"name": "Where was he born?", "importance": 0.4}]}')
        aspects, failed = om.decompose(Q, ask)
        om.add_views(aspects, ask)
        self.assertFalse(failed)
        self.assertEqual(aspects[0].views, ["re: Who directed Sinister?",
                                            "passage on Who directed Sinister?"])
        caps = [c[2] for c in ask.calls]
        self.assertEqual(caps, [settings.DECOMPOSE_MAX_TOKENS]
                         + [settings.PARAPHRASE_MAX_TOKENS,
                            settings.HYDE_MAX_TOKENS] * 2)

    def test_empty_view_falls_back_to_the_aspect_text(self):
        ask = _Ask('{"type": "simple", "aspects": []}', paraphrase="  ",
                   hyde="")
        aspects, _ = om.decompose(Q, ask)
        om.add_views(aspects, ask)
        self.assertEqual(aspects[0].views, [Q, Q])


class TestAnswerPrompt(unittest.TestCase):
    """The answer prompt keeps the exact refusal the scorers recognise."""

    def test_refusal_is_the_scorers_string(self):
        from src.config import ABSTENTION_RESPONSE
        self.assertEqual(settings.REFUSAL, ABSTENTION_RESPONSE)

    def test_prompt_is_the_baselines_plus_the_orientation_line(self):
        """Ours asks for the same answer form as M1-M4, plus one line on orientation."""
        from src.config import BASE_ANSWER_SYSTEM_PROMPT
        self.assertEqual(settings.ANSWER_PROMPT,
                         BASE_ANSWER_SYSTEM_PROMPT + " " + settings.ORIENTATION_LINE)

    def test_second_asking_keeps_the_exact_refusal_and_the_orientation_line(self):
        self.assertIn(f"'{settings.REFUSAL}'", settings.REASK_PROMPT)
        self.assertTrue(settings.REASK_PROMPT.endswith(settings.ORIENTATION_LINE))

    def _contexts(self, **switches):
        high = om.Aspect(name="main", text=Q, importance=1.0)
        high.confidence = 0.9
        low = om.Aspect(name="main", text=Q, importance=1.0)
        low.confidence = 0.1
        with mock.patch.multiple(settings, **switches):
            return (om.build_context([high], [["Paris."]], [[]], []),
                    om.build_context([low], [["Paris."]], [[]], []))

    def test_signal_off_shows_neither_score_nor_marker(self):
        for ctx in self._contexts(PASS_RETRIEVAL_CONFIDENCE_TO_LLM=False,
                                  MARK_LOW_CONFIDENCE=False):
            self.assertEqual(ctx.splitlines()[0], "[Aspect: main]")

    def test_score_alone_can_be_shown(self):
        high, low = self._contexts(PASS_RETRIEVAL_CONFIDENCE_TO_LLM=True,
                                   MARK_LOW_CONFIDENCE=False)
        self.assertEqual(low.splitlines()[0],
                         "[Aspect: main] (retrieval_confidence: 0.10)")

    def test_marker_follows_the_threshold(self):
        high, low = self._contexts(PASS_RETRIEVAL_CONFIDENCE_TO_LLM=True,
                                   MARK_LOW_CONFIDENCE=True)
        self.assertNotIn("LOW CONFIDENCE", high)
        self.assertIn("[LOW CONFIDENCE]", low)


class TestShortAnswer(unittest.TestCase):
    """shorten_answer keeps the answer's own words, or yes / no, and nothing else."""

    LONG = "The film Sinister was directed by Scott Derrickson."

    def _cut(self, reply, answer=LONG):
        calls = []

        def ask(system_prompt, user_prompt, max_tokens):
            calls.append((system_prompt, user_prompt, max_tokens))
            return reply
        return om.shorten_answer(Q, answer, ask), calls

    def test_the_cut_is_asked_with_its_prompt_and_cap(self):
        _, calls = self._cut("Scott Derrickson")
        self.assertEqual(calls, [(settings.SHORT_ANSWER_PROMPT,
                                  f"Question: {Q}\nAnswer: {self.LONG}",
                                  settings.SHORT_ANSWER_MAX_TOKENS)])

    def test_a_phrase_from_the_answer_replaces_it(self):
        self.assertEqual(self._cut("Scott Derrickson")[0], "Scott Derrickson")

    def test_a_label_quotes_and_later_lines_are_dropped(self):
        self.assertEqual(self._cut('Answer: "Scott Derrickson"\n\nHe directed it.')[0],
                         "Scott Derrickson")

    def test_yes_or_no_stands_though_the_answer_lacks_it(self):
        self.assertEqual(self._cut("Yes", "Both were born in the United States.")[0], "Yes")

    def test_a_cut_that_adds_a_word_leaves_the_answer(self):
        self.assertEqual(self._cut("Scott Derrickson Jr")[0], self.LONG)

    def test_a_cut_no_shorter_leaves_the_answer(self):
        self.assertEqual(self._cut("the Derrickson", "Derrickson")[0], "Derrickson")

    def test_an_empty_or_refusing_cut_leaves_the_answer(self):
        for reply in ("", "   ", settings.REFUSAL):
            self.assertEqual(self._cut(reply)[0], self.LONG)

    def _on(self, question, answer, reply):
        return om.shorten_answer(question, answer, lambda *_: reply)

    def test_an_either_or_cut_to_a_supporting_date_takes_the_side_named_first(self):
        question = "Which was formed first, the duo Big & Rich or Lonestar?"
        answer = "Lonestar was formed in 1992, and Big & Rich was formed by 2003."
        self.assertEqual(self._on(question, answer, "1992"), "Lonestar")

    def test_an_either_or_cut_that_names_a_side_stands(self):
        question = "Which was formed first, the duo Big & Rich or Lonestar?"
        answer = "Lonestar was formed in 1992, and Big & Rich was formed by 2003."
        self.assertEqual(self._on(question, answer, "Big & Rich"), "Big & Rich")

    def test_only_one_side_named_is_enough(self):
        question = "Does Erodium or Cymbidium include more species?"
        self.assertEqual(self._on(question, "Erodium includes about 60 species.", "60"), "Erodium")

    def test_either_or_with_neither_side_named_keeps_the_cut(self):
        question = "Who wrote the novel Foo or the play Bar?"
        self.assertEqual(self._on(question, "Anna Reed wrote them both.", "Anna Reed"), "Anna Reed")

    def test_a_side_is_never_a_lone_stopword(self):
        question = "Was it the Battle of Foo or the Battle of Bar?"
        answer = "It was the first of many in the war of 1931."
        self.assertEqual(self._on(question, answer, "1931"), answer)

    def test_a_side_matches_whole_words(self):
        question = "Is the winner Rich or Poor?"
        answer = "He enriched many people in 1931."
        self.assertEqual(self._on(question, answer, "1931"), answer)

    def test_an_or_that_is_no_choice_leaves_a_name_alone(self):
        for question, answer, cut in (
                ("What concert hall is on the northern edge or South Kensington, London?",
                 "The Royal Albert Hall is on the northern edge of South Kensington.", "Royal Albert Hall"),
                ('Len Wiseman directed "Live Free or Die Hard", whose hero is who?',
                 'The hero of "Live Free or Die Hard" is John McClane.', "John McClane")):
            self.assertEqual(self._on(question, answer, cut), cut)

    def test_a_date_or_number_word_gives_way_to_a_side(self):
        question = "Who was born first, Ann Reed or Bo Lane?"
        answer = "Ann Reed was born on February 9, 1954, two years before Bo Lane."
        for cut in ("February 9, 1954", "1954", "two"):
            self.assertEqual(self._on(question, answer, cut), "Ann Reed", cut)

    def test_a_number_nobody_asked_for_leaves_the_answer(self):
        question = "Who founded the Lode Company?"
        answer = "The Lode Company was founded in 1931."
        self.assertEqual(self._on(question, answer, "1931"), answer)

    def test_a_number_the_question_asks_for_stands(self):
        for question in ("How many moons does it have, in all?", "In what year was the Lode Company founded?",
                         "When was the Lode Company founded?"):
            self.assertEqual(self._on(question, "It happened in 1931 after a long wait.", "1931"), "1931")

    def test_the_prompts_examples_are_invented(self):
        self.assertIn("Short answer: The Ember", settings.SHORT_ANSWER_PROMPT)
        self.assertNotIn("Lonestar", settings.SHORT_ANSWER_PROMPT)


class TestWhatTheQuestionAsks(unittest.TestCase):
    """What a question asks for and what a cut is, read from their own words."""

    def test_a_number_or_date_is_nothing_but_one(self):
        for cut in ("1992", "19 April 1832", "February 9, 1954", "74th", "two", "1,500"):
            self.assertTrue(om.is_a_number(cut), cut)

    def test_a_name_holding_a_number_is_not(self):
        for cut in ("Apollo 11", "2002 Winter Olympics", "875 acres", "Lonestar", ""):
            self.assertFalse(om.is_a_number(cut), cut)

    def test_how_big_asks_for_a_number(self):
        self.assertTrue(om.asks_for_a_number("How big is the campus?"))
        self.assertFalse(om.asks_for_a_number("Who founded the company?"))

    def test_why_and_how_want_a_reason(self):
        for question in ("Why did Foo leave?", "how does Foo escape?", "Why did it take so many years?"):
            self.assertTrue(om.asks_for_a_reason(question), question)

    def test_how_many_and_others_do_not(self):
        for question in ("How many days did it take?", "How old is Foo?", "Who left?", "What happened?", ""):
            self.assertFalse(om.asks_for_a_reason(question), question)


if __name__ == "__main__":
    unittest.main()
