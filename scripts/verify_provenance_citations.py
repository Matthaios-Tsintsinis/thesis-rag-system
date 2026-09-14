"""Check every `path:line` citation in docs/PROVENANCE_TABLE.md and
docs/METHODS_AND_FIDELITY.md against the tree at the pinned tag.
Documentation tooling off the output path; no runner, replay or export uses it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "docs" / "PROVENANCE_TABLE.md",
    ROOT / "docs" / "METHODS_AND_FIDELITY.md",
)
# The tag every code citation in the fidelity documents is pinned to.
CITATION_TAG = "thesis-full-2026-09-03"

CITE = re.compile(
    r"`((?:src|tests|scripts)/[A-Za-z0-9_/]+\.py):(\d+(?:[,\-]\d+)*)`"
)
# A backticked identifier right after the citation: at most one space and
# an optional "(" before the backtick, and no "/" inside it (that would be
# a path, so another citation). Prose in between means the backtick is
# not this citation's anchor.
ANCHOR = re.compile(r"^ ?\(?`([A-Za-z_][A-Za-z0-9_]*)[^`/]*`")

ANCHOR_WINDOW = 2  # lines either side of the cited line


def read_at_rev(path: str, rev: str) -> list[str] | None:
    """The file's lines at `rev` (a tag, sha or ref), or None if absent."""
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.splitlines()


def check_document(
    doc: Path, files: dict[str, list[str] | None], rev: str
) -> tuple[int, int, list[str]]:
    """Check one document at `rev`: (n_checked, n_anchored, problems)."""
    text = doc.read_text(encoding="utf-8")
    problems: list[str] = []
    n_checked = n_anchored = 0
    print(f"\n[cite] {doc.name} @ {rev}")

    # Each citation, plus the symbol anchored beside it when there is one.
    for m in CITE.finditer(text):
        path, spec = m.group(1), m.group(2)
        tail = text[m.end():m.end() + 40]
        am = ANCHOR.match(tail)
        anchor = am.group(1) if am else None

        # Read each cited file once from the pinned revision; the cache is
        # shared across both documents.
        if path not in files:
            files[path] = read_at_rev(path, rev)
            if files[path] is None:
                problems.append(f"MISSING FILE {path} at {rev}")
        lines = files[path]
        if not lines:
            continue

        # Every cited line must exist; an anchored one must hold its symbol
        # within the window. Resolved lines are printed for a human to read.
        for part in re.split(r"[,\-]", spec):
            line_no = int(part)
            n_checked += 1
            if line_no < 1 or line_no > len(lines):
                problems.append(
                    f"OUT OF RANGE {path}:{line_no} ({len(lines)} lines)")
                continue
            shown = lines[line_no - 1].strip()
            if anchor:
                n_anchored += 1
                lo = max(0, line_no - 1 - ANCHOR_WINDOW)
                hi = min(len(lines), line_no + ANCHOR_WINDOW)
                window = "\n".join(lines[lo:hi])
                if anchor not in window:
                    problems.append(
                        f"DRIFTED {path}:{line_no} - expected `{anchor}` "
                        f"within +/-{ANCHOR_WINDOW} lines; line reads: "
                        f"{shown[:70]!r}")
                    continue
            print(f"  {path}:{line_no:<5} {('[' + anchor + ']').ljust(28) if anchor else ' ' * 28} {shown[:70]}")

    print(f"[cite] {doc.name}: {n_checked} citations checked, "
          f"{n_anchored} symbol-anchored")
    return n_checked, n_anchored, [f"{doc.name}: {p}" for p in problems]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev", default=CITATION_TAG,
                    help="git revision whose tree the citations are read "
                         f"from (default: the pinned tag {CITATION_TAG})")
    args = ap.parse_args(argv)

    # The revision must resolve in this clone before any file is read.
    probe = subprocess.run(["git", "rev-parse", "--verify", "--quiet",
                            f"{args.rev}^{{commit}}"], cwd=ROOT,
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(f"[cite] revision {args.rev!r} does not resolve in "
                         f"this clone - fetch the tag first (git fetch --tags)")
    print(f"[cite] reading cited files from {args.rev} "
          f"({probe.stdout.strip()[:12]}), not the working tree")

    # A missing document is a refusal, never a silent skip.
    missing = [d for d in DOCS if not d.exists()]
    if missing:
        raise SystemExit(
            "[cite] document(s) not found - run where docs/ lives, and "
            "never skip a missing one silently: "
            + ", ".join(str(d) for d in missing)
        )

    # Check both documents against one shared file cache.
    files: dict[str, list[str] | None] = {}
    problems: list[str] = []
    n_checked = n_anchored = 0
    for doc in DOCS:
        c, a, p = check_document(doc, files, args.rev)
        n_checked += c
        n_anchored += a
        problems.extend(p)

    # Print the tally; any problem exits 1.
    print(f"\n[cite] {n_checked} citations checked across {len(DOCS)} "
          f"documents at {args.rev}, {n_anchored} symbol-anchored")
    if problems:
        print("[cite] PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("[cite] all citations resolve; anchored ones match their symbol. "
          "READ the printed lines - unanchored citations are verified by "
          "that reading, not by this exit code.")


if __name__ == "__main__":
    main()
