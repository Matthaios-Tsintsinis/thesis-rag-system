"""List substrate manifests whose created_at falls inside a date window.

THE CP313-WINDOW SCREEN for the final six cells. M2/M3 substrates have no
cold-tree gate and their cache key carries no interpreter input, so a
slice-era substrate resolves under current keys and a warm hit is
legitimate BY RULE — but a substrate whose embeddings were computed while
the run host sat on CPython 3.13.15 carries cp313 torch artifacts, and
inheriting it would quietly break the "all under 3.12.13" provenance
claim the restored interpreter bought.

This script ENUMERATES; the operator DELETES exactly what it prints. Same
philosophy as the cold-tree preflight: a hand-maintained list of warm
substrates was wrong once already, so the set is computed, never
remembered. No deletion code here on purpose — a lister that can also
delete is one flag away from deleting on a mistyped date.

    python -m scripts.list_substrates_by_date \\
        --cache-root /content/drive/MyDrive/thesis_rag/cache \\
        --after 2026-08-19T00:00 --before 2026-08-23T12:00

Namespaces default to M2 and M3 (the two systems the screen exists for);
`--namespace` overrides, repeatable. Manifests without a parseable
created_at are listed separately as UNDATED — an undated substrate cannot
prove it predates the window, so treat it as suspect rather than clean.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_when(raw: str | None) -> datetime | None:
    """ISO date/datetime -> aware UTC datetime; None when unparseable."""
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def in_window(created_raw: str | None, after: datetime,
              before: datetime) -> bool | None:
    """True in-window, False out, None undated/unparseable."""
    ts = parse_when(created_raw)
    if ts is None:
        return None
    return after <= ts <= before


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--after", required=True,
                    help="window start, ISO (e.g. 2026-08-19T00:00)")
    ap.add_argument("--before", required=True,
                    help="window end, ISO")
    ap.add_argument("--namespace", action="append", default=None,
                    help="cache namespace(s); default M2 and M3")
    args = ap.parse_args()

    after = parse_when(args.after)
    before = parse_when(args.before)
    if after is None or before is None or after >= before:
        raise SystemExit("[substrates] bad window: --after must parse and "
                         "precede --before")

    root = Path(args.cache_root)
    namespaces = args.namespace or ["M2", "M3"]
    hits: list[tuple[str, str, str]] = []
    undated: list[str] = []
    n_scanned = 0

    for ns in namespaces:
        nsdir = root / ns
        if not nsdir.is_dir():
            print(f"[substrates] namespace {ns}: nothing under {nsdir}")
            continue
        for mp in sorted(nsdir.glob("*/manifest.json")):
            n_scanned += 1
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"[substrates] WARN unreadable {mp}: {e}")
                continue
            verdict = in_window(m.get("created_at"), after, before)
            if verdict is None:
                undated.append(str(mp.parent))
            elif verdict:
                hits.append((str(mp.parent), str(m.get("created_at")),
                             f"{m.get('embedder_model')}  "
                             f"n_chunks={m.get('n_chunks')}"))

    print(f"\n[substrates] scanned {n_scanned} manifest(s) in "
          f"{namespaces}; window {after.isoformat()} .. "
          f"{before.isoformat()}")
    if hits:
        print(f"[substrates] {len(hits)} IN-WINDOW — cp313-era embeddings; "
              "DELETE these exact directories before the cell runs:")
        for d, created, extra in hits:
            print(f"  {d}\n      created={created}  {extra}")
    else:
        print("[substrates] none in window — every existing substrate "
              "predates or postdates it.")
    if undated:
        print(f"[substrates] {len(undated)} UNDATED manifest(s) — cannot "
              "prove they predate the window; treat as suspect:")
        for d in undated:
            print(f"  {d}")


if __name__ == "__main__":
    main()
