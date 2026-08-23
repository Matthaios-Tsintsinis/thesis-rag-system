"""Verify every `path:line` citation in docs/PROVENANCE_TABLE.md.

WHY THIS EXISTS, and why it is stricter than "does the line number exist".
The living fidelity record cites our code as `path:line` at a named HEAD.
The first checker only tested that each line number was in range - and a
three-line edit near the top of `raptor_paper.py` shifted every citation
below it by +3 while that checker kept passing, because line 488 still held
SOME text. A check that cannot fail for the right reason has not passed;
this is the project's recurring lesson applied to its own tooling.

Two layers of verification:

1. RANGE + CONTENT PRINT. Every citation must resolve to a real line, and
   the line is PRINTED so the final check is a human reading it. The number
   is verified by the content shown, not by the exit code alone.

2. SYMBOL ANCHORING, where the table provides it. Most rows put a backticked
   symbol right beside the citation - `` `src/raptor_paper.py:300`
   `split_text_raptor` `` or `` `src/config.py:93` (`RRF_K = 60`, ...) ``.
   When a backticked identifier directly follows the citation in the same
   table cell, the cited line (+/- 2 lines, since a def's decorator or a
   dataclass field's comment can sit adjacent) must CONTAIN that
   identifier's first token. A uniform shift now fails loudly instead of
   printing plausible-looking wrong lines.

Run on the agent host after ANY commit that edits a cited file:

    python -m scripts.verify_provenance_citations

Exit 0 = all resolve and every anchored citation matches. Exit 1 = a broken
or drifted citation, listed. The doc lives in `docs/`, which is disk-only;
this script therefore runs where the doc lives, not on Colab.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "PROVENANCE_TABLE.md"

CITE = re.compile(
    r"`((?:src|tests|scripts)/[A-Za-z0-9_/]+\.py):(\d+(?:[,\-]\d+)*)`"
)
# A backticked identifier IMMEDIATELY following the citation - at most one
# space and an optional "(" before the backtick, and no "/" inside the
# backticks (which would make it a path, i.e. some OTHER citation). Prose
# between citation and backtick means the backtick is not this citation's
# anchor, and greedily adopting it produced eleven false DRIFTED reports
# on the first run of this checker.
ANCHOR = re.compile(r"^ ?\(?`([A-Za-z_][A-Za-z0-9_]*)[^`/]*`")

ANCHOR_WINDOW = 2  # lines either side of the cited line


def main() -> None:
    if not DOC.exists():
        raise SystemExit(f"[cite] {DOC} not found - run where docs/ lives")
    text = DOC.read_text(encoding="utf-8")

    files: dict[str, list[str]] = {}
    problems: list[str] = []
    n_checked = n_anchored = 0

    for m in CITE.finditer(text):
        path, spec = m.group(1), m.group(2)
        tail = text[m.end():m.end() + 40]
        am = ANCHOR.match(tail)
        anchor = am.group(1) if am else None

        if path not in files:
            f = ROOT / path
            if not f.exists():
                problems.append(f"MISSING FILE {path}")
                files[path] = []
                continue
            files[path] = f.read_text(encoding="utf-8").splitlines()
        lines = files[path]
        if not lines:
            continue

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

    print(f"\n[cite] {n_checked} citations checked, "
          f"{n_anchored} symbol-anchored")
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
