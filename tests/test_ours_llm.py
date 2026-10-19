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

    def test_unusable_replies_are_none(self):
        for raw in ("no json here", '{"type": "other"}',
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

    def _contexts(self, **switches):
        high = om.Aspect(name="main", text=Q, importance=1.0)
        high.confidence = 0.9
        low = om.Aspect(name="main", text=Q, importance=1.0)
        low.confidence = 0.1
        with mock.patch.multiple(settings, **switches):
            return (om.build_context([high], [["Paris."]], [[]], []),
                    om.build_context([low], [["Paris."]], [[]], []))

    def test_confidence_signal_is_off_by_default(self):
        """v1 shows neither the score nor the low-confidence marker."""
        for ctx in self._contexts(PASS_RETRIEVAL_CONFIDENCE_TO_LLM=False,
                                  MARK_LOW_CONFIDENCE=False):
            self.assertEqual(ctx.splitlines()[0], "[Aspect: main]")
        self.assertFalse(settings.PASS_RETRIEVAL_CONFIDENCE_TO_LLM)
        self.assertFalse(settings.MARK_LOW_CONFIDENCE)

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


if __name__ == "__main__":
    unittest.main()
