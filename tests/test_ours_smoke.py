"""Smoke test for ours through the real runner loop and the real HotpotQA
scorers, on two synthetic HotpotQA-shaped units. M4's tree build, ours'
whole pipeline and the row writer are real; the embedder, the reader, the
tree summariser and the cross-encoder are faked so no model loads."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from ours import config as settings
from src.config import DEFAULT_CONFIG
from src.eval.types import ANSWER_TYPE_FREE_FORM, CorpusItem, EvalQuery, EvalUnit, GoldAnswer

DIM = 64


# --- fakes shared with test_ours_key ----------------------------------------


def fake_embed(texts, model_name=None):
    """Hashed bag of words, unit length: similar words, similar vectors."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for r, text in enumerate(texts):
        for w in re.findall(r"\w+", text.lower()):
            out[r, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = float(np.linalg.norm(out[r]))
        if norm:
            out[r] /= norm
        else:
            out[r, 0] = 1.0
    return out


def fake_generate_batch(system_prompts, user_prompts, cfg=None, **_):
    """Tree summaries: the first twenty words of each cluster's text."""
    return [" ".join(u.split()[:20]) for u in user_prompts]


def fake_cross_encoder(query, passages):
    """Shared words between query and passage, through a sigmoid."""
    q = set(re.findall(r"\w+", query.lower()))
    return [1.0 / (1.0 + math.exp(-(len(q & set(re.findall(r"\w+", p.lower()))) - 2)))
            for p in passages]


class FakeReader:
    """Replaces src.models.generate: scripted replies, every config kept."""

    def __init__(self, decompositions, answers, die_on_answer=None):
        self.decompositions = decompositions
        self.answers = answers
        self.die_on_answer = die_on_answer
        self.cfgs = []
        self.prompts = []

    def __call__(self, system_prompt, user_prompt, cfg=None):
        self.cfgs.append(cfg)
        self.prompts.append((system_prompt, user_prompt))
        if system_prompt == settings.DECOMPOSE_PROMPT:
            question = user_prompt.split("Question: ", 1)[1]
            return self.decompositions.get(question, "not json")
        if system_prompt == settings.PARAPHRASE_PROMPT:
            return "Rephrased " + user_prompt.split(": ", 1)[1]
        if system_prompt == settings.HYDE_PROMPT:
            return "A passage says " + user_prompt.split(": ", 1)[1]
        question = user_prompt.rsplit("Question: ", 1)[1]
        if question == self.die_on_answer:
            raise RuntimeError("session died")
        return self.answers[question]


def llama_config():
    """The runner's config under --generator Llama."""
    return replace(
        DEFAULT_CONFIG,
        generation=replace(DEFAULT_CONFIG.generation, model=settings.READER),
        m4=replace(DEFAULT_CONFIG.m4, summary_model=settings.READER))


@contextlib.contextmanager
def ours_fakes(cache_dir, reader):
    """Every model faked, the cache in a temp dir, the tree env pinned."""
    with mock.patch("src.models.embed_texts", fake_embed), \
            mock.patch("src.retrievers.m4_raptor.embed_texts", fake_embed), \
            mock.patch("src.models.generate", reader), \
            mock.patch("src.models.generate_batch", fake_generate_batch), \
            mock.patch("ours.model.cross_encoder_scores", fake_cross_encoder), \
            mock.patch("ours.model.check_model_revisions", lambda: None), \
            mock.patch.object(settings, "TREE_BUILD_ENV", "test-env"), \
            mock.patch("src.paths.cache_dir", return_value=Path(cache_dir)):
        yield


def _paragraph(title: str, topic: str) -> list[str]:
    """Eight sentences about one topic, long enough for two leaves."""
    return [f"The {topic} record number {i} says that {topic} travelled to "
            f"the {['river', 'castle', 'harbour', 'forest'][i % 4]} in the "
            f"year {1800 + 7 * i} with a friend called {title.split()[0]}."
            for i in range(8)]


def hotpot_unit(k: int) -> EvalUnit:
    """Ten titled paragraphs, sentence items, one bridge-style question."""
    titles = [f"Name{k}x{j} Place" for j in range(10)]
    items = []
    for j, title in enumerate(titles):
        for i, sentence in enumerate(_paragraph(title, f"topic{k}x{j}")):
            items.append(CorpusItem(item_id=f"{title}::sent{i}", parent_id=title,
                                    span_id=f"sent{i}", text=sentence,
                                    metadata={"title": title, "sent_id": i}))
    question = f"Where did topic{k}x0 travel with topic{k}x1?"
    query = EvalQuery(
        query_id=f"q{k}", question_text=question, parent_scope=None,
        gold_answers=(GoldAnswer(answer_type=ANSWER_TYPE_FREE_FORM,
                                 free_form="the river"),),
        gold_passage_sets=(frozenset({(titles[0], "sent0"),
                                      (titles[1], "sent0")}),),
        question_type="bridge")
    return EvalUnit(corpus_id=f"hotpot::{k}", corpus=tuple(items),
                    queries=(query,))


class TwoHotpotUnits:
    """Two synthetic units scored by the real HotpotQA scorers."""

    name = "hotpotqa"

    def __init__(self) -> None:
        from src.eval.hotpotqa import HotpotQABenchmark
        self._scorer = HotpotQABenchmark()
        self.stats = {"n_queries": 2}
        self.units = [hotpot_unit(0), hotpot_unit(1)]

    def iter_eval_units(self, *, split, max_units=None):
        yield from self.units

    def score_answer(self, predicted, query):
        return self._scorer.score_answer(predicted, query)

    def score_retrieval(self, retrieved, query, scoring_ranking=None):
        return self._scorer.score_retrieval(retrieved, query,
                                            scoring_ranking=scoring_ranking)


Q0 = "Where did topic0x0 travel with topic0x1?"
Q1 = "Where did topic1x0 travel with topic1x1?"
DECOMPOSITIONS = {Q0: json.dumps({"type": "multi_aspect", "aspects": [
    {"name": "Where did topic0x0 travel?", "importance": 0.6},
    {"name": "Who is topic0x1?", "importance": 0.4}]})}
ANSWERS = {Q0: "the river", Q1: settings.REFUSAL}


def run_cell(cache_dir, out_path, *, reader=None, resume=False):
    """One pass of the runner over the two units; returns the rows."""
    from ours.model import OursSystem
    from src.eval.base import BenchmarkRunner

    reader = reader or FakeReader(DECOMPOSITIONS, ANSWERS)
    with ours_fakes(cache_dir, reader):
        system = OursSystem(config=llama_config())
        runner = BenchmarkRunner(output_path=Path(out_path), verbose=False,
                                 resume=resume)
        list(runner.run(system, TwoHotpotUnits(), split="validation"))
    return [json.loads(x) for x in
            Path(out_path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _without_timing(row: dict) -> dict:
    row = dict(row)
    row.pop("latency_s", None)
    return row


class TestOursSmoke(unittest.TestCase):
    """Two units end to end: cold, warm, and cold again in a second cache."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        root = Path(cls.td.name)
        cls.reader = FakeReader(DECOMPOSITIONS, ANSWERS)
        cls.cold = run_cell(root / "cacheA", root / "a.jsonl", reader=cls.reader)
        cls.warm = run_cell(root / "cacheA", root / "b.jsonl")
        cls.cold_again = run_cell(root / "cacheB", root / "c.jsonl")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_two_rows_with_the_expected_calls(self):
        """Two aspects make six LLM calls; a parse failure makes four."""
        r0, r1 = self.cold
        self.assertEqual((r0["metadata"]["ours_n_llm_calls"],
                          r1["metadata"]["ours_n_llm_calls"]), (6, 4))
        self.assertEqual((r0["metadata"]["ours_parse_failed"],
                          r1["metadata"]["ours_parse_failed"]), (False, True))

    def test_every_reader_call_is_greedy_with_the_shared_model(self):
        """Temperature 0 and the Llama id on every call, the answer included."""
        self.assertEqual({c.temperature for c in self.reader.cfgs}, {0.0})
        self.assertEqual({c.model for c in self.reader.cfgs}, {settings.READER})

    def test_one_aspect_takes_thirteen_slots(self):
        aspects = self.cold[1]["metadata"]["ours_aspects"]
        self.assertEqual([a["budget"] for a in aspects], [13])

    def test_context_holds_at_most_fifteen_chunks(self):
        for row in self.cold:
            self.assertLessEqual(row["n_retrieved"], 15)
            self.assertGreater(row["n_retrieved"], 0)

    def test_retrieval_is_evidence_only(self):
        """Orientation summaries are packed but never scored as retrieval."""
        for row in self.cold:
            self.assertEqual(set(row["retrieved_unit_types"]), {"chunk"})
            n_orient = row["metadata"]["ours_n_orientation_summaries"]
            self.assertEqual(row["n_packed"], row["n_retrieved"] + n_orient)

    def test_answer_prompt_and_refusal(self):
        """The reader sees the approved prompt; the refusal is detected."""
        answer_calls = [p for p in self.reader.prompts
                        if p[0] == settings.ANSWER_PROMPT]
        self.assertEqual(len(answer_calls), 2)
        self.assertTrue(self.cold[1]["answer"]["metadata"]["abstained"])
        self.assertEqual(self.cold[0]["answer"]["value"], 1.0)

    def test_doc_ranking_reproduces_the_scorers_rank_metrics(self):
        from scripts.replay_retrieval import rank_stats_from_ranking
        from src.eval.hotpotqa import _TITLE_SPAN

        for row, unit in zip(self.cold, TwoHotpotUnits().units):
            gold = frozenset((t, _TITLE_SPAN)
                             for t, _ in unit.queries[0].gold_passage_sets[0])
            stats = rank_stats_from_ranking(
                [(p, _TITLE_SPAN) for p in row["metadata"]["ours_doc_ranking"]],
                gold, (1, 2, 5, 10))
            banked = {int(k): v for k, v in row["retrieval"]["hit_at_k"].items()}
            self.assertEqual(stats["hit_at_k"], banked)
            self.assertEqual(stats["mrr"], row["retrieval"]["mrr"])

    def test_second_pass_hits_the_trees_the_first_built(self):
        self.assertEqual([r["metadata"]["ours_tree_cache_hit"] for r in self.cold],
                         [False, False])
        self.assertEqual([r["metadata"]["ours_tree_cache_hit"] for r in self.warm],
                         [True, True])

    def test_rows_are_deterministic(self):
        """A cold run in a fresh cache writes the same rows, timing aside."""
        self.assertEqual([_without_timing(r) for r in self.cold],
                         [_without_timing(r) for r in self.cold_again])

    def test_warm_rows_equal_cold_rows(self):
        def strip(r):
            r = _without_timing(r)
            r["metadata"] = {k: v for k, v in r["metadata"].items()
                             if k != "ours_tree_cache_hit"}
            return r
        self.assertEqual([strip(r) for r in self.cold],
                         [strip(r) for r in self.warm])

    def test_one_code_fingerprint(self):
        codes = {r["metadata"]["ours_code"] for r in self.cold + self.warm}
        self.assertEqual(len(codes), 1)

    def test_section_detectors_are_recorded(self):
        for row in self.cold:
            self.assertEqual(set(row["metadata"]["ours_section_detectors"]),
                             {"fixed"})

    def test_resume_after_a_torn_line_gives_the_same_rows(self):
        """Die on the second answer, tear the file, resume: same rows."""
        from src.eval.runner import summarise_rows

        out = self.root / "resumed.jsonl"
        dying = FakeReader(DECOMPOSITIONS, ANSWERS, die_on_answer=Q1)
        with self.assertRaises(RuntimeError):
            run_cell(self.root / "cacheA", out, reader=dying)
        with out.open("a", encoding="utf-8") as f:
            f.write('{"query_id": "q1", "pred')
        rows = run_cell(self.root / "cacheA", out, resume=True)
        self.assertEqual([_without_timing(r)["query_id"] for r in rows],
                         ["q0", "q1"])
        self.assertEqual(summarise_rows(out),
                         summarise_rows(self.root / "b.jsonl"))


if __name__ == "__main__":
    unittest.main()
