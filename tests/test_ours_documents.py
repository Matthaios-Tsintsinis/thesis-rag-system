"""Pins that every chunk reaches the reader, the reranker and BM25 under its document's
name (development v15): source, date and title as the corpus gives them. A one-document
unit has nothing to tell apart and stays as it was; M4's tree and vectors never see a name."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import numpy as np

from ours import config as settings
from ours import model as om
from src.eval.types import ANSWER_TYPE_FREE_FORM, CorpusItem, EvalQuery, EvalUnit, GoldAnswer
from tests.test_ours_smoke import (
    DECOMPOSITIONS,
    FakeReader,
    Q0,
    fake_cross_encoder,
    fake_embed,
    hotpot_unit,
    llama_config,
    ours_fakes,
)

ARTICLE = " ".join(f"Sentence {i} of the report says the merger of the firms went ahead in week {i}."
                   for i in range(40))


def article(url, source, date, title):
    """One news article as MultiHop's loader gives it."""
    return CorpusItem(item_id=url, parent_id=url, span_id="<whole>", text=f"{title}. {ARTICLE}",
                      metadata={"title": title, "author": "A Writer", "source": source,
                                "published_at": date, "category": "news"})


def news_unit():
    return EvalUnit(corpus_id="news", corpus=(
        article("u0", "The Verge", "2023-10-07T08:45:59+00:00", "Firms merge"),
        article("u1", "TechCrunch", "2023-11-27T21:31:00+00:00", "A second report"),
        article("u2", "Fortune", "2023-12-24T23:34:39+00:00", "A third report")), queries=())


def story_unit():
    """One story, as NarrativeQA's loader gives it."""
    return EvalUnit(corpus_id="s1", corpus=(CorpusItem(
        item_id="s1::<whole>", parent_id="s1", span_id="<whole>", text=ARTICLE,
        metadata={"kind": "book", "title": "The Long Story", "url": ""}),), queries=())


def prepared(unit, question=Q0, **flags):
    """prepare() on a unit, with every model faked: (the prepared query, the unit's index)."""
    reader = FakeReader(DECOMPOSITIONS, {})
    with tempfile.TemporaryDirectory() as td, ours_fakes(td, reader), \
            mock.patch.multiple(settings, **(flags or {"DOCUMENT_HEADERS": True})):
        system = om.OursSystem(config=llama_config())
        system.index_items(unit.corpus)
        return system.prepare(question), system._tree_index


def chunk_lines(prepared_query):
    return [ln.strip() for ln in prepared_query.user_prompt.splitlines() if ln.strip().startswith("Chunk ")]


class TestPublishedDate(unittest.TestCase):
    def test_in_the_words_questions_use(self):
        self.assertEqual(om.published_date("2023-10-07T08:45:59+00:00"), "October 7, 2023")

    def test_nothing_readable_is_nothing(self):
        for value in ("", None, "soon"):
            self.assertEqual(om.published_date(value), "")


class TestDocumentHeaders(unittest.TestCase):
    def _headers(self, unit):
        chunks = [mock.Mock(doc_id=f"{item.parent_id}.txt", gold_provenance=((item.parent_id, "x"),))
                  for item in unit.corpus]
        return om.document_headers(unit.corpus, chunks), chunks

    def test_an_article_gives_its_source_date_and_title(self):
        headers, chunks = self._headers(news_unit())
        self.assertEqual(headers[chunks[0].doc_id], "The Verge | October 7, 2023 | Firms merge")

    def test_a_paragraph_gives_its_title(self):
        headers, chunks = self._headers(hotpot_unit(0))
        self.assertEqual(headers[chunks[0].doc_id], chunks[0].gold_provenance[0][0])

    def test_one_document_has_nothing_to_tell_apart(self):
        headers, _ = self._headers(story_unit())
        self.assertEqual(headers, {})

    def test_switched_off(self):
        with mock.patch.object(settings, "DOCUMENT_HEADERS", False):
            headers, _ = self._headers(news_unit())
        self.assertEqual(headers, {})


class TestWhereTheNameGoes(unittest.TestCase):
    def test_every_chunk_the_reader_sees_names_its_document(self):
        query, index = prepared(hotpot_unit(0))
        lines = chunk_lines(query)
        self.assertTrue(lines)
        titles = {str(i.parent_id) for i in hotpot_unit(0).corpus}
        for line in lines:
            self.assertRegex(line, r"^Chunk \d+: \[")
            self.assertIn(line.split("[", 1)[1].split("]", 1)[0], titles)

    def test_each_chunk_gets_its_own_document(self):
        query, index = prepared(hotpot_unit(0))
        for chunk, line in zip(query.retrieved, chunk_lines(query)):
            title = str(chunk.chunk.gold_provenance[0][0])
            self.assertTrue(line.split(": ", 1)[1].startswith(f"[{title}] {chunk.chunk.text[:20]}"), line)

    def test_an_article_shows_its_source_and_date(self):
        query, _ = prepared(news_unit(), question="What did the report say about the merger?")
        self.assertRegex(chunk_lines(query)[0], r"^Chunk 1: \[(The Verge|TechCrunch|Fortune) \| \w+ \d+, 2023 \| ")

    def test_bm25_finds_a_chunk_by_its_document_name(self):
        _, index = prepared(news_unit(), question="What did the report say about the merger?")
        scores = om.bm25_scores(index, "Fortune")
        fortune = [i for i, c in enumerate(index.chunks) if str(c.gold_provenance[0][0]) == "u2"]
        others = [i for i in range(len(index.chunks)) if i not in fortune]
        self.assertTrue((scores[fortune] > 0).all())
        self.assertTrue((scores[others] == 0).all())

    def test_the_reranker_reads_the_name(self):
        seen = []

        def spy(query, passages):
            seen.extend(passages)
            return fake_cross_encoder(query, passages)

        _, index = prepared(hotpot_unit(0))
        with mock.patch("ours.model.cross_encoder_scores", spy):
            om.rerank(index, "a question", [0, 1])
        self.assertEqual(seen, [om.headed_text(index, 0), om.headed_text(index, 1)])
        self.assertTrue(seen[0].startswith("["))

    def test_the_tree_and_its_vectors_never_see_a_name(self):
        _, index = prepared(hotpot_unit(0))
        for i, chunk in enumerate(index.chunks):
            self.assertFalse(chunk.text.startswith("["))
            self.assertTrue(np.allclose(index.leaf_vectors[i], fake_embed([chunk.text])[0]))

    def test_the_rows_keep_the_chunks_as_they_are(self):
        query, _ = prepared(hotpot_unit(0))
        self.assertFalse(any(r.chunk.text.startswith("[") for r in query.retrieved))


class TestNothingChangesWithoutANameToGive(unittest.TestCase):
    def test_one_document_reads_as_it_did(self):
        on, _ = prepared(story_unit(), question="What did the report say about the merger?",
                         DOCUMENT_HEADERS=True)
        off, _ = prepared(story_unit(), question="What did the report say about the merger?",
                          DOCUMENT_HEADERS=False)
        self.assertEqual(on.user_prompt, off.user_prompt)
        self.assertEqual([r.chunk.chunk_id for r in on.packed], [r.chunk.chunk_id for r in off.packed])
        self.assertFalse(any(line.startswith("Chunk 1: [") for line in chunk_lines(on)))

    def test_switched_off_reads_as_v14_did(self):
        off, _ = prepared(hotpot_unit(0), DOCUMENT_HEADERS=False)
        self.assertRegex(chunk_lines(off)[0], r"^Chunk 1: The ")


if __name__ == "__main__":
    unittest.main()
