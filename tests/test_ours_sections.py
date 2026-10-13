"""Pins ours' section detection on chunks cut by M4's own chunker: headings
in a book and a film script, fixed-size blocks on long unstructured text, and
one section for documents as short as HotpotQA paragraphs and news articles."""

from __future__ import annotations

import unittest
from unittest import mock

from ours import config as settings
from ours.model import find_sections
from src.parsing import clean_text
from src.raptor_paper import split_text_raptor


def _chunks(text: str) -> list[str]:
    """Chunk a document the way M4 does: cleaned, then 100-token pieces."""
    return [s.text for s in split_text_raptor(clean_text(text), max_tokens=100)]


def _prose(tag: str, n_sentences: int) -> str:
    """Plain sentences of about fifteen tokens each, tagged for locating."""
    return " ".join(
        f"In part {tag} the walker number {i} crossed the long grey hill "
        f"before the evening rain began to fall." for i in range(n_sentences))


def _book(n_chapters: int = 5, with_contents: bool = False) -> str:
    """A small book: a title page, optional contents, then CHAPTER sections."""
    names = ["I", "II", "III", "IV", "V", "VI"][:n_chapters]
    parts = ["THE LONG GREY HILL", "", "A Story", ""]
    if with_contents:
        parts += ["CONTENTS", ""] + [f"CHAPTER {n}" for n in names] + [""]
    parts += [_prose("front", 20), ""]
    for n in names:
        parts += [f"CHAPTER {n}", "", _prose(f"ch{n}", 70), ""]
    return "\n".join(parts)


def _script(n_scenes: int = 10) -> str:
    """A film script: scene headings, speaker cues in capitals, dialogue."""
    parts = []
    for s in range(n_scenes):
        parts += [f"INT. HOUSE ROOM {s} - NIGHT", ""]
        for line in range(24):
            speaker = "JOHN" if line % 2 else "MARY"
            parts += [f"                    {speaker}",
                      f"          Line {line} of scene {s} says that the long "
                      "grey hill is far away and the rain is coming soon.", ""]
    return "\n".join(parts)


def _section_first_chunks(sections: list[int]) -> list[int]:
    """Index of the first chunk of every section after the first."""
    return [i for i in range(1, len(sections)) if sections[i] != sections[i - 1]]


class TestHeadings(unittest.TestCase):
    """Headings in the document text start sections, titled by the heading."""

    def test_book_chapters_become_sections(self):
        """Each CHAPTER line starts a section whose chunks all mention it."""
        text = _book()
        chunks = _chunks(text)
        self.assertGreater(len(chunks), settings.SECTION_SIZE_CHUNKS)
        sections, titles, detector = find_sections(chunks, text, "Hill")
        self.assertEqual(detector, "headings")
        self.assertEqual(titles, ["Hill", "CHAPTER I", "CHAPTER II",
                                  "CHAPTER III", "CHAPTER IV", "CHAPTER V"])
        # Every chunk that carries only one chapter's prose sits in that
        # chapter's section.
        for i, chunk in enumerate(chunks):
            if "CHAPTER" in chunk:
                continue
            for k, n in enumerate(["I", "II", "III", "IV", "V"], start=1):
                if f"part ch{n} " in chunk:
                    self.assertEqual(sections[i], k, (i, n))

    def test_contents_page_does_not_hide_the_chapters(self):
        """With a table of contents, the body's chapters still start sections."""
        text = _book(with_contents=True)
        chunks = _chunks(text)
        sections, titles, detector = find_sections(chunks, text, "Hill")
        self.assertEqual(detector, "headings")
        starts = _section_first_chunks(sections)
        body = [i for i, c in enumerate(chunks)
                if "CHAPTER" in c and "part ch" in c]
        for i in body:
            self.assertIn(i, starts, i)

    def test_script_scenes_win_over_speaker_cues(self):
        """Scene headings start sections; capitalised speaker cues do not."""
        text = _script()
        chunks = _chunks(text)
        self.assertGreater(len(chunks), settings.SECTION_SIZE_CHUNKS)
        sections, titles, detector = find_sections(chunks, text, "Play")
        self.assertEqual(detector, "headings")
        self.assertTrue(all(t.startswith("INT.") for t in titles[1:]), titles)
        self.assertNotIn("JOHN", titles)

    def test_one_heading_is_not_a_structure(self):
        """A single heading line falls back to fixed-size blocks."""
        text = "CHAPTER I\n\n" + _prose("solo", 300)
        chunks = _chunks(text)
        _, _, detector = find_sections(chunks, text, "Solo")
        self.assertEqual(detector, "fixed")


class TestFixedBlocks(unittest.TestCase):
    """A long document without headings is cut into fixed-size blocks."""

    def test_long_unstructured_text_gets_numbered_blocks(self):
        """Blocks of SECTION_SIZE_CHUNKS chunks, titled with a block number."""
        text = _prose("flat", 500)
        chunks = _chunks(text)
        size = settings.SECTION_SIZE_CHUNKS
        self.assertGreater(len(chunks), 2 * size)
        sections, titles, detector = find_sections(chunks, text, "Flat")
        self.assertEqual(detector, "fixed")
        self.assertEqual(sections, [i // size for i in range(len(chunks))])
        self.assertEqual(titles[:2], ["Flat [1]", "Flat [2]"])


class TestShortDocuments(unittest.TestCase):
    """Documents no longer than a block stay one section titled by the document."""

    def test_hotpotqa_paragraph_is_one_section(self):
        """A paragraph of a few sentences is one section."""
        text = "\n\n".join([
            "Scott Derrickson (born July 16, 1966) is an American director.",
            "He lives in Los Angeles, California.",
            "He is best known for directing horror films."])
        chunks = _chunks(text)
        self.assertEqual(find_sections(chunks, text, "Scott Derrickson"),
                         ([0] * len(chunks), ["Scott Derrickson"], "fixed"))

    def test_news_article_with_capital_lines_is_one_section(self):
        """A news-length article stays whole even with capitalised lines."""
        text = "\n".join(["ADVERTISEMENT", _prose("news", 60), "ADVERTISEMENT",
                          _prose("more", 60)])
        chunks = _chunks(text)
        self.assertLessEqual(len(chunks), settings.SECTION_SIZE_CHUNKS)
        sections, titles, detector = find_sections(chunks, text, "Headline")
        self.assertEqual((set(sections), titles, detector),
                         ({0}, ["Headline"], "fixed"))


class TestDocling(unittest.TestCase):
    """Docling is an extension point; switching it on says so."""

    def test_docling_switch_is_not_implemented(self):
        """USE_DOCLING=True raises instead of silently falling back."""
        with mock.patch.object(settings, "USE_DOCLING", True):
            with self.assertRaises(NotImplementedError):
                find_sections(["a"], "a", "A")


if __name__ == "__main__":
    unittest.main()
