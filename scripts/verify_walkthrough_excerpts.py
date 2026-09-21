"""Check every code excerpt in docs/CODE_WALKTHROUGH.md against the tree at
the document's pinned revision, and every fidelity line against the
METHODS section headers. Documentation tooling off the output path; no
runner, replay or export uses it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "CODE_WALKTHROUGH.md"
METHODS = ROOT / "docs" / "METHODS_AND_FIDELITY.md"

# The status words METHODS uses in its tables; a fidelity line carries one.
STATUS_WORDS = (
    "MATCHES",
    "REFERENCE-DERIVED",
    "HARNESS-CHOICE",
    "DECLARED-DEVIATION",
    "DECLINED",
    "ADDITION",
)
# An excerpt is 5 to 20 lines, the spec's bound for a readable block.
MIN_LINES, MAX_LINES = 5, 20

# The document pins one revision in its header; every excerpt cites it.
PIN = re.compile(r"^\*\*Code revision:\*\* `([0-9a-f]{7,40})`")
EXCERPT = re.compile(
    r"^`((?:src|scripts|tests)/[A-Za-z0-9_/]+\.py):(\d+)-(\d+) @ ([0-9a-f]{7,40})`\s*$"
)
FENCE_OPEN = re.compile(r"^```")
FENCE_CLOSE = re.compile(r"^```\s*$")
FIDELITY = re.compile(r"^\*\*Fidelity:\*\*")
# Any letter-led id is a reference; membership in the headers decides it.
SECTION_REF = re.compile(r"METHODS §([A-Z](?:\.\d+)*)")
# METHODS headers: "## A. Systems", "### A.1 ...", "#### A.4.2 ...".
HEADER = re.compile(r"^#{2,4}\s+([A-E](?:\.\d+)*)\.?(?:\s|$)")


@dataclass
class Report:
    """What one check found: the pin, the revision read, counts, problems."""

    pin: str | None
    rev: str | None
    n_excerpts: int = 0
    n_fidelity: int = 0
    problems: list[str] = field(default_factory=list)


def read_at_rev(path: str, rev: str, root: Path) -> list[str] | None:
    """The file's lines at `rev`, or None if absent there."""
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=root,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()


def methods_sections(text: str) -> set[str]:
    """Section ids (A, A.1, A.4.2, ...) from the METHODS headers."""
    out: set[str] = set()
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            out.add(m.group(1))
    return out


def _find_pin(lines: list[str]) -> str | None:
    """The revision the document header pins, or None."""
    for line in lines:
        m = PIN.match(line)
        if m:
            return m.group(1)
    return None


def check_document(
    text: str,
    *,
    root: Path,
    sections: set[str],
    rev_override: str | None = None,
) -> Report:
    """Check every excerpt and fidelity line; return the report."""
    lines = text.splitlines()
    pin = _find_pin(lines)
    rev = rev_override or pin
    report = Report(pin=pin, rev=rev)
    if pin is None:
        report.problems.append(
            "NO PIN: the document names no `**Code revision:** `<sha>`` line")

    files: dict[str, list[str] | None] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = EXCERPT.match(line)
        if m:
            path, start, end, cited = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            where = f"{path}:{start}-{end}"
            report.n_excerpts += 1
            if rev_override is None and pin is not None and cited != pin:
                report.problems.append(
                    f"REVISION MISMATCH {where} cites {cited}, document pinned to {pin}")

            # The fenced block must follow the header line directly.
            if i + 1 >= len(lines) or not FENCE_OPEN.match(lines[i + 1]):
                report.problems.append(f"NO FENCE after {where}")
                i += 1
                continue
            body: list[str] = []
            j = i + 2
            while j < len(lines) and not FENCE_CLOSE.match(lines[j]):
                body.append(lines[j])
                j += 1
            if j >= len(lines):
                report.problems.append(f"UNCLOSED FENCE after {where}")
                break
            i = j + 1

            n = end - start + 1
            if start < 1 or end < start:
                report.problems.append(f"OUT OF RANGE {where}")
                continue
            if not MIN_LINES <= n <= MAX_LINES:
                report.problems.append(
                    f"LENGTH {where} has {n} lines ({MIN_LINES}-{MAX_LINES} required)")
            if rev is None:
                continue
            if path not in files:
                files[path] = read_at_rev(path, rev, root)
            file_lines = files[path]
            if file_lines is None:
                report.problems.append(f"MISSING FILE {path} at {rev}")
                continue
            if end > len(file_lines):
                report.problems.append(
                    f"OUT OF RANGE {where} ({len(file_lines)} lines in the file)")
                continue
            expected = file_lines[start - 1:end]
            if len(body) != len(expected):
                report.problems.append(
                    f"DIFFERS {where}: the block holds {len(body)} lines, the range {len(expected)}")
                continue
            for k, (have, want) in enumerate(zip(body, expected)):
                if have != want:
                    report.problems.append(
                        f"DIFFERS {path}:{start + k}: document {have!r} vs file {want!r}")
                    break
            else:
                print(f"  {where:<44} {n:>2} lines  OK")
            continue

        if FIDELITY.match(line):
            report.n_fidelity += 1
            refs = SECTION_REF.findall(line)
            if not refs:
                report.problems.append(
                    f"FIDELITY NO SECTION at line {i + 1}: {line[:80]!r}")
            for ref in refs:
                if ref not in sections:
                    report.problems.append(
                        f"FIDELITY UNKNOWN SECTION §{ref} at line {i + 1}")
            if not any(w in line for w in STATUS_WORDS):
                report.problems.append(
                    f"FIDELITY NO STATUS at line {i + 1}: {line[:80]!r}")
        i += 1
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", type=Path, default=DOC,
                    help="the walkthrough document (default: docs/CODE_WALKTHROUGH.md)")
    ap.add_argument("--methods", type=Path, default=METHODS,
                    help="the METHODS document whose headers the fidelity "
                         "lines must cite")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="the git repository the excerpts are read from")
    ap.add_argument("--rev", default=None,
                    help="read the cited files at this revision instead of "
                         "the document's own pin (a deliberate re-anchoring)")
    args = ap.parse_args(argv)

    # A missing document is a refusal, never a silent skip.
    missing = [p for p in (args.doc, args.methods) if not p.exists()]
    if missing:
        raise SystemExit(
            "[walk] document(s) not found - run where docs/ lives: "
            + ", ".join(str(p) for p in missing))
    doc_text = args.doc.read_text(encoding="utf-8")
    sections = methods_sections(args.methods.read_text(encoding="utf-8"))

    # The revision must resolve in this clone before any file is read.
    rev = args.rev or _find_pin(doc_text.splitlines())
    if rev is not None:
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
            cwd=args.root, capture_output=True, text=True)
        if probe.returncode != 0:
            raise SystemExit(
                f"[walk] revision {rev!r} does not resolve in this clone "
                "(fetch the branch first)")
        print(f"[walk] reading cited files from {rev} "
              f"({probe.stdout.strip()[:12]}), not the working tree")

    print(f"\n[walk] {args.doc.name}")
    report = check_document(
        doc_text, root=args.root, sections=sections, rev_override=args.rev)

    # Print the tally; any problem exits 1.
    print(f"\n[walk] {report.n_excerpts} excerpts and {report.n_fidelity} "
          f"fidelity lines checked at {report.rev}")
    if report.problems:
        print("[walk] PROBLEMS:")
        for p in report.problems:
            print(f"  - {p}")
        sys.exit(1)
    print("[walk] every excerpt is verbatim at the pinned revision and "
          "every fidelity line names an existing METHODS section.")


if __name__ == "__main__":
    main()
