"""Rebuild ONE banked M4 unit under the locked interpreter and diff its tree.

THE AUTHORITY STEP of the interpreter-drift question (SESSION_HANDOFF §1b).
The topology probe cleared the clustering stack across 3.12/3.13 with fixed
synthetic embeddings; what it holds fixed by construction — the mpnet leaf
embeddings under torch, whose cp312/cp313 wheels differ — is exactly what
this rebuild exercises. One ~18-leaf HotpotQA unit is rebuilt cold under
CPython 3.12.13 and its tree shape is diffed against the banked manifest:

    n_nodes  ·  layer_sizes  ·  n_summary_nodes  ·  n_chunks

  IDENTICAL -> cell 6 STANDS. Declare the interpreter note in the living
               record and continue the matrix on 3.12.13.
  DIFFERENT -> cell 6 RE-RUNS under 3.12.13 (~2.67 h, one session).

GUARDS, because a comparison that cannot fail for the right reason has not
passed:

  * REFUSES to run unless the ACTIVE interpreter equals the lockfile's
    `# python=` line exactly. A rebuild under 3.13 would diff a tree
    against itself and prove nothing.
  * REFUSES to run unless THESIS_CACHE_DIR points somewhere OTHER than the
    baseline cache root — the rebuild must never write into, or warm-hit
    from, the banked namespace.
  * REFUSES a confounded comparison: the banked manifest's build_env
    package tokens must equal the current stack's (python tokens excluded
    — the interpreter is the variable under test; banked cell-6 manifests
    predate the python token anyway).
  * The unit is matched by corpus_hash, never by cache path — the
    substrate key moved at e907d68 by design.

USAGE (run host, GPU, inside the 3.12.13 environment):

    THESIS_CACHE_DIR=/content/rebuild312 \\
    /content/py312/bin/python -m scripts.rebuild_unit_diff \\
        --benchmark hotpotqa \\
        --baseline-cache-root /content/drive/MyDrive/thesis_rag/cache \\
        --pick-env umap-learn

`--dry-run` stops after unit selection and baseline read — no GPU, no
model, no build — so the selection can be previewed before spending GPU
time. `--corpus-id PREFIX` overrides the automatic ~18-leaf choice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths
from src.cache import corpus_content_hash
from src.retrievers.m4_raptor import RaptorSystem
from scripts.report_children_per_parent import (
    BENCHMARKS,
    _benchmark,
    _index_manifests,
    _stats_of,
)

COMPARED_FIELDS = (
    # (banked index_stats key, rebuilt index_stats key, label)
    ("tree_n_nodes", "tree_n_nodes", "n_nodes"),
    ("tree_depth_counts", "tree_depth_counts", "layer_sizes"),
    ("n_summary_nodes_at_index", "n_summary_nodes_at_index", "n_summary_nodes"),
    ("flat_n_chunks", "flat_n_chunks", "n_chunks"),
)


def _env_package_tokens(env: str | None) -> tuple[str, ...]:
    """build_env tokens minus any python= token — the interpreter is the
    variable under test, and banked cell-6 manifests predate the token."""
    if not env:
        return ()
    toks = [t.strip() for t in str(env).replace(";", " ").split() if t.strip()]
    return tuple(sorted(t for t in toks if not t.startswith("python=")))


def _assert_locked_interpreter(lockfile: Path) -> str:
    from scripts.pin_environment import locked_python

    if not lockfile.exists():
        raise SystemExit(
            f"[rebuild] no lockfile at {lockfile} — the rebuild is only "
            "meaningful against the locked interpreter; copy "
            "requirements.lock from Drive first"
        )
    want = locked_python(lockfile.read_text(encoding="utf-8"))
    have = sys.version.split()[0]
    if want is None:
        raise SystemExit("[rebuild] lockfile records no interpreter")
    if have != want:
        raise SystemExit(
            f"[rebuild] REFUSING: running python {have}, locked {want}. "
            "A rebuild under the wrong interpreter diffs a tree against "
            "itself and proves nothing. Enter the 3.12.13 environment."
        )
    return have


def compare_stats(baseline: dict, rebuilt: dict) -> list[str]:
    """Field-by-field differences; empty list = tree shape identical."""
    diffs: list[str] = []
    for bk, rk, label in COMPARED_FIELDS:
        bv, rv = baseline.get(bk), rebuilt.get(rk)
        # layer_sizes serialises dict keys as strings in JSON; normalise.
        if isinstance(bv, dict):
            bv = {str(k): int(v) for k, v in bv.items()}
        if isinstance(rv, dict):
            rv = {str(k): int(v) for k, v in rv.items()}
        if bv != rv:
            diffs.append(f"{label}: banked {bv!r}  vs  rebuilt {rv!r}")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="hotpotqa", choices=BENCHMARKS)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--baseline-cache-root", required=True,
                    help="cache root holding the BANKED manifests (Drive)")
    ap.add_argument("--pick-env", default=None,
                    help="build_env substring filter for the banked era")
    ap.add_argument("--corpus-id", default=None,
                    help="corpus_id prefix override for unit choice")
    ap.add_argument("--target-leaves", type=int, default=18,
                    help="auto-select the first unit with this many leaves")
    ap.add_argument("--lockfile", default="requirements.lock")
    ap.add_argument("--dry-run", action="store_true",
                    help="stop after selection + baseline read (no GPU)")
    args = ap.parse_args()

    baseline_root = Path(args.baseline_cache_root).resolve()
    active_root = paths.cache_dir().resolve()
    if baseline_root == active_root:
        raise SystemExit(
            "[rebuild] REFUSING: THESIS_CACHE_DIR resolves to the baseline "
            f"cache root ({active_root}). The rebuild must write to a "
            "THROWAWAY dir — set THESIS_CACHE_DIR=/content/rebuild312."
        )
    if not os.environ.get("THESIS_CACHE_DIR"):
        print("[rebuild] WARN: THESIS_CACHE_DIR is unset; active cache is "
              f"{active_root} — confirm this is a throwaway location.")

    have_py = _assert_locked_interpreter(Path(args.lockfile))
    print(f"[rebuild] interpreter {have_py} == locked; active cache "
          f"{active_root}")

    by_hash = _index_manifests(baseline_root)

    # ---- select the unit -------------------------------------------------
    system_for_hash = RaptorSystem()  # config only until .index_items
    chosen = None  # (unit, chash, manifest)
    for u in _benchmark(args.benchmark).iter_eval_units(split=args.split):
        cid = str(u.corpus_id)
        if args.corpus_id and not cid.startswith(args.corpus_id):
            continue
        with tempfile.TemporaryDirectory(prefix="rbd_") as td:
            system_for_hash._write_corpus_layout(list(u.corpus), Path(td))
            chash = corpus_content_hash(Path(td))
        cands = by_hash.get(chash, [])
        if args.pick_env:
            cands = [m for m in cands if args.pick_env in
                     str((m.get("extra") or {}).get("build_env", ""))]
        if not cands:
            if args.corpus_id:
                raise SystemExit(
                    f"[rebuild] no banked manifest for {cid!r} under "
                    f"{baseline_root} (env filter {args.pick_env!r})")
            continue
        if len(cands) > 1:
            raise SystemExit(
                f"[rebuild] {cid!r}: {len(cands)} manifests share its "
                "corpus_hash — disambiguate with --pick-env")
        st = _stats_of(cands[0]) or {}
        if args.corpus_id or int(st.get("flat_n_chunks") or 0) == args.target_leaves:
            chosen = (u, chash, cands[0])
            break
    if chosen is None:
        raise SystemExit(
            f"[rebuild] no unit with exactly {args.target_leaves} leaves "
            "matched a banked manifest — pass --corpus-id or a different "
            "--target-leaves")

    unit, chash, manifest = chosen
    baseline_stats = _stats_of(manifest) or {}
    banked_env = (manifest.get("extra") or {}).get("build_env")
    print(f"[rebuild] unit {unit.corpus_id!r}  corpus_hash {chash[:16]}…")
    print(f"[rebuild] banked: dir={manifest['_dir'][:16]}… "
          f"created={manifest.get('created_at')} env={banked_env}")
    print("[rebuild] banked tree: " + "; ".join(
        f"{label}={baseline_stats.get(bk)!r}"
        for bk, _, label in COMPARED_FIELDS))

    # ---- refuse a confounded stack --------------------------------------
    from src.raptor_paper import PAPER_TREE_BUILD_ENV

    banked_toks = _env_package_tokens(banked_env)
    ours_toks = _env_package_tokens(PAPER_TREE_BUILD_ENV)
    if banked_toks and banked_toks != ours_toks:
        raise SystemExit(
            "[rebuild] REFUSING: topology package stacks differ —\n"
            f"  banked : {banked_toks}\n  current: {ours_toks}\n"
            "A diff across a package drift answers a different question "
            "than the interpreter one.")
    print(f"[rebuild] package stack matches banked: {ours_toks}")

    if args.dry_run:
        print("[rebuild] DRY RUN — stopping before any model loads.")
        return

    # ---- the build -------------------------------------------------------
    system = RaptorSystem()
    print("[rebuild] building cold under the locked interpreter…")
    system.index_items(list(unit.corpus))
    if getattr(system, "tree_cache_hit", None):
        raise SystemExit(
            "[rebuild] the rebuild was a CACHE HIT — the throwaway dir "
            "already held this substrate, so nothing was rebuilt and this "
            "run proves nothing. Empty the dir and re-run.")
    rebuilt_stats = system.index_stats

    print("[rebuild] rebuilt tree: " + "; ".join(
        f"{label}={rebuilt_stats.get(rk)!r}"
        for _, rk, label in COMPARED_FIELDS))

    diffs = compare_stats(baseline_stats, rebuilt_stats)
    print()
    if diffs:
        print("[rebuild] VERDICT: DIFFERENT — the interpreter moves the "
              "tree through the embedder path. Cell 6 RE-RUNS under "
              "3.12.13.")
        for d in diffs:
            print(f"  {d}")
        sys.exit(1)
    print("[rebuild] VERDICT: IDENTICAL on n_nodes, layer_sizes, "
          "n_summary_nodes and n_chunks. Cell 6 STANDS — declare the "
          "interpreter note in the living record and continue on 3.12.13.")


if __name__ == "__main__":
    main()
