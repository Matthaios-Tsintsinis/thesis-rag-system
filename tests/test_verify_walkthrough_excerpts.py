"""Tests for scripts/verify_walkthrough_excerpts: every code excerpt in
docs/CODE_WALKTHROUGH.md must be a verbatim copy of its cited line range
at the document's pinned revision, and every fidelity line must name a
METHODS section that exists. Runs on a throwaway git repository.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.verify_walkthrough_excerpts import (
    check_document,
    main,
    methods_sections,
)

FILE_LINES = [f"line {i}: x = {i}" for i in range(1, 31)]
METHODS_TEXT = (
    "# METHODS\n\n## A. Systems\n\n### A.1 M1 — closed-book\n\n"
    "#### A.4.2 Parameter table\n\n## D. The harness\n"
)


def _git(root: Path, *args: str) -> str:
    """Run git in the throwaway repo with a fixed identity."""
    proc = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root, capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def _make_repo(td: str, lines: list[str] = FILE_LINES) -> tuple[Path, str]:
    """Create a repo holding src/mod.py; return (root, commit sha)."""
    root = Path(td)
    _git(root, "init", "-q")
    (root / "src").mkdir()
    (root / "src" / "mod.py").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    _git(root, "add", "src/mod.py")
    _git(root, "commit", "-q", "-m", "one")
    return root, _git(root, "rev-parse", "HEAD")


def _doc(
    rev: str,
    start: int,
    end: int,
    *,
    lines: list[str] = FILE_LINES,
    header: bool = True,
    fence: bool = True,
    cite_rev: str | None = None,
    extra: str = "",
    path: str = "src/mod.py",
) -> str:
    """Build a walkthrough document with one excerpt."""
    body: list[str] = []
    if header:
        body.append(f"**Code revision:** `{rev}`")
    body.append("")
    body.append(f"`{path}:{start}-{end} @ {cite_rev or rev}`")
    if fence:
        body.append("```python")
    body.extend(lines[start - 1:end])
    if fence:
        body.append("```")
    body.append("")
    body.append(extra)
    return "\n".join(body) + "\n"


class TestMethodsSections(unittest.TestCase):
    """The section ids come from the METHODS headers, at every depth."""

    def test_headers_become_section_ids(self):
        self.assertEqual(
            methods_sections(METHODS_TEXT), {"A", "A.1", "A.4.2", "D"})


class TestExcerpts(unittest.TestCase):
    """An excerpt passes only as an exact copy of the cited lines."""

    def _check(self, root: Path, text: str, rev: str | None = None):
        return check_document(
            text, root=root, sections=methods_sections(METHODS_TEXT),
            rev_override=rev)

    def test_exact_excerpt_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            report = self._check(root, _doc(sha, 2, 8))
            self.assertEqual(report.problems, [])
            self.assertEqual(report.n_excerpts, 1)
            self.assertEqual(report.pin, sha)

    def test_altered_line_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            lines = list(FILE_LINES)
            lines[4] = "line 5: x = 55"
            report = self._check(root, _doc(sha, 2, 8, lines=lines))
            self.assertEqual(len(report.problems), 1)
            self.assertIn("DIFFERS", report.problems[0])
            self.assertIn("src/mod.py:5", report.problems[0])

    def test_shifted_range_refuses(self):
        """The right text cited under the wrong line numbers is a mismatch."""
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            text = _doc(sha, 2, 8).replace("src/mod.py:2-8", "src/mod.py:3-9")
            report = self._check(root, text)
            self.assertTrue(any("DIFFERS" in p for p in report.problems))

    def test_missing_file_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            report = self._check(root, _doc(sha, 2, 8, path="src/absent.py"))
            self.assertTrue(any("MISSING FILE" in p for p in report.problems))

    def test_range_past_end_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            text = _doc(sha, 26, 30).replace("src/mod.py:26-30", "src/mod.py:28-32")
            report = self._check(root, text)
            self.assertTrue(any("OUT OF RANGE" in p for p in report.problems))

    def test_too_short_and_too_long_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            short = self._check(root, _doc(sha, 2, 5))
            self.assertTrue(any("LENGTH" in p for p in short.problems))
            long = self._check(root, _doc(sha, 2, 22))
            self.assertTrue(any("LENGTH" in p for p in long.problems))
            ok5 = self._check(root, _doc(sha, 2, 6))
            ok20 = self._check(root, _doc(sha, 2, 21))
            self.assertEqual(ok5.problems + ok20.problems, [])

    def test_crlf_document_matches_lf_file(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            text = _doc(sha, 2, 8).replace("\n", "\r\n")
            self.assertEqual(self._check(root, text).problems, [])

    def test_header_without_fence_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            report = self._check(root, _doc(sha, 2, 8, fence=False))
            self.assertTrue(any("NO FENCE" in p for p in report.problems))

    def test_missing_pin_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            report = self._check(root, _doc(sha, 2, 8, header=False))
            self.assertTrue(any("NO PIN" in p for p in report.problems))

    def test_excerpt_revision_must_equal_the_pin(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            other = "0" * 40
            report = self._check(root, _doc(sha, 2, 8, cite_rev=other))
            self.assertTrue(
                any("REVISION" in p for p in report.problems), report.problems)

    def test_rev_override_reads_the_other_revision(self):
        """--rev re-anchors: the same doc fails where the file changed."""
        with tempfile.TemporaryDirectory() as td:
            root, sha_a = _make_repo(td)
            lines = list(FILE_LINES)
            lines[4] = "line 5: x = 55"
            (root / "src" / "mod.py").write_text(
                "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            _git(root, "commit", "-q", "-am", "two")
            sha_b = _git(root, "rev-parse", "HEAD")
            doc = _doc(sha_a, 2, 8)
            self.assertEqual(self._check(root, doc).problems, [])
            moved = self._check(root, doc, rev=sha_b)
            self.assertTrue(any("DIFFERS" in p for p in moved.problems))
            self.assertEqual(moved.rev, sha_b)
            unchanged = self._check(root, _doc(sha_a, 10, 15), rev=sha_b)
            self.assertEqual(unchanged.problems, [])


class TestFidelityLines(unittest.TestCase):
    """Every fidelity line names a status and an existing METHODS section."""

    def _check(self, root: Path, text: str):
        return check_document(
            text, root=root, sections=methods_sections(METHODS_TEXT))

    def test_valid_fidelity_line_passes_and_is_counted(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            line = ("**Fidelity:** MATCHES the paper: the leaf size is 100 "
                    "tokens (METHODS §A.4.2, row \"leaf size\").")
            report = self._check(root, _doc(sha, 2, 8, extra=line))
            self.assertEqual(report.problems, [])
            self.assertEqual(report.n_fidelity, 1)

    def test_unknown_section_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            line = "**Fidelity:** MATCHES the paper (METHODS §Z.9)."
            report = self._check(root, _doc(sha, 2, 8, extra=line))
            self.assertTrue(any("UNKNOWN SECTION" in p for p in report.problems))

    def test_missing_section_reference_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            line = "**Fidelity:** MATCHES the paper, trust me."
            report = self._check(root, _doc(sha, 2, 8, extra=line))
            self.assertTrue(any("NO SECTION" in p for p in report.problems))

    def test_missing_status_word_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            line = "**Fidelity:** roughly like the paper (METHODS §A.4.2)."
            report = self._check(root, _doc(sha, 2, 8, extra=line))
            self.assertTrue(any("NO STATUS" in p for p in report.problems))


class TestMain(unittest.TestCase):
    """The command refuses a missing document, an unresolvable revision
    and any problem; a clean document returns normally."""

    def _files(self, td: str, sha: str, doc_text: str) -> tuple[Path, Path]:
        doc = Path(td) / "CODE_WALKTHROUGH.md"
        methods = Path(td) / "METHODS.md"
        doc.write_text(doc_text, encoding="utf-8")
        methods.write_text(METHODS_TEXT, encoding="utf-8")
        return doc, methods

    def test_clean_document_returns(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            doc, methods = self._files(td, sha, _doc(sha, 2, 8))
            main(["--doc", str(doc), "--methods", str(methods),
                  "--root", str(root)])

    def test_problem_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            lines = list(FILE_LINES)
            lines[4] = "line 5: x = 55"
            doc, methods = self._files(td, sha, _doc(sha, 2, 8, lines=lines))
            with self.assertRaises(SystemExit) as ctx:
                main(["--doc", str(doc), "--methods", str(methods),
                      "--root", str(root)])
            self.assertEqual(ctx.exception.code, 1)

    def test_missing_document_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            _, methods = self._files(td, sha, _doc(sha, 2, 8))
            with self.assertRaises(SystemExit) as ctx:
                main(["--doc", str(Path(td) / "absent.md"),
                      "--methods", str(methods), "--root", str(root)])
            self.assertNotEqual(ctx.exception.code, 0)

    def test_unresolvable_rev_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            root, sha = _make_repo(td)
            doc, methods = self._files(td, sha, _doc(sha, 2, 8))
            with self.assertRaises(SystemExit) as ctx:
                main(["--doc", str(doc), "--methods", str(methods),
                      "--root", str(root), "--rev", "deadbeef"])
            self.assertIn("does not resolve", str(ctx.exception.code))


if __name__ == "__main__":
    unittest.main()
