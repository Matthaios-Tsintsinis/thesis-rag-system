"""Retrieval replay: bank the depth-50 document rankings, gated, forever.

WHY. The depth-50 scoring ranking was consumed at run time (hit@K /
MAP@K / MRR) and never written to any row, so recall@5 was not
derivable post-hoc. This tool re-runs RETRIEVAL ONLY over the warm
substrates for the 18 ranked cells (M2/M3/M4 x MultiHop / HotpotQA /
pooled x both generators), writes each cell's per-row document ranking
to a SIDECAR beside the banked cell (`rankings.<stem>.jsonl` +
`rankings.<stem>.json`; the banked JSONL is never opened for writing),
and computes recall@{1,5,10}. THE SIDECAR NAMES ARE GLOB-SAFE BY
AUDIT: they begin with `rankings.` and end without `.summary.json`, so
they match NONE of the bank's discovery patterns -- not the bank
gates' `*.summary.json` (runner.py), not aggregate's rglob, and not
significance_diagnostic's `{bench}_{sys}_*.jsonl` fallback. A
`<stem>.rankings.summary.json` name would have been swept into the
bank-gate population; found and renamed in the pre-flight audit. Once the sidecars exist, recall
at any K is derivable forever and this replay never needs repeating.

THE GATE — the new artifact earns trust by re-deriving the old one
(the R6 discipline). Per row, exactly, never on average:

  1. the replayed retrieval is scored through the FROZEN
     `benchmark.score_retrieval` and must reproduce the banked
     set-F1/recall/precision AND the banked hit_at_k, map_at_k and mrr
     bit-for-bit (JSON round-trips float64 exactly);
  2. the sidecar's collapsed document ranking must itself re-derive the
     scorer's hit@K and MRR per row, binding recall to the very ranking
     that reproduced the bank.

Any row disagreeing REFUSES the cell and names the row. A refused cell
writes no sidecar.

READ-ONLY, GUARANTEED BY CONSTRUCTION, three mechanisms:
  * every unit is probed with `substrate_warm_path` (itself read-only:
    key computation + completeness check, no embedding, no clustering)
    BEFORE `index_items`; a MISS refuses the cell -- the build path is
    never reachable, so no substrate can be written;
  * for M4, `tree_cache_hit` must be True after the load (the inverse
    of the run-time cold-tree gate): a replay that built anything is a
    defect and refuses;
  * no generation path is touched -- `prepare()` is Phase A by design
    (retrieve + rank + pack, no LLM), and the generator cache is
    asserted EMPTY after every cell (`load_generator.cache_info()`).

EXPECTED BOUNDS, asserted per cell (a violation is a bug, never a
finding): per-row recall@5 <= hit@5, so mean recall@5 <= mean hit@5
everywhere; on HotpotQA (exactly two gold titles per question) a hit
implies recall >= 1/2, so mean recall@5 sits in [hit@5 / 2, hit@5].

HARDWARE. Run on the SAME GPU class as the bank (L4). Not for speed --
~9,500 query encodings are minutes either way -- but for the GATE:
query embeddings recomputed under a different BLAS (CPU) differ at
epsilon, and the measured near-tie channel (living record §5c, the
9-row classification) would flip knife-edge rankings and refuse rows
that are not wrong. Same wheel + same hardware class reproduces
bit-exact (the M1 identity and the re-diff demonstrated it end to end).

Run-host invocation (all 18 cells; ~an hour class, M4 tree I/O
dominating):

    python -m scripts.replay_retrieval \\
      --p10 /content/drive/MyDrive/thesis_rag/outputs/p10 \\
      --p11 /content/drive/MyDrive/thesis_rag/outputs/p11

An existing sidecar means DONE and refuses, so an interrupted session
resumes by re-running the same command; to regenerate a cell
deliberately, delete its two sidecar files by hand first. All 18 ranked
cells are replayed in one invocation (the per-cell selector, the
overwrite flag and the key dry-run left in the repo reduction).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_comparison import LLAMA, QWEN, _fail, _generating_commit

RANKED_BENCHMARKS = ("multihop_rag", "hotpotqa", "hotpotqa_pooled")
RANKED_SYSTEMS = ("M2", "M3", "M4")
RECALL_KS = (1, 5, 10)
DOC_RANKING_DEPTH = 50


def collapse_to_doc_ranking(retrieved) -> list[tuple[str, str]]:
    """First-occurrence document collapse — a verbatim mirror of the
    frozen scorer's own collapse (`score_retrieval_rank_aware`,
    src/eval/alignment.py). Two guarantees keep this copy honest: the
    oracle test re-derives the scorer's outputs through it on shared
    fixtures, and at replay time every row's hit@K/MRR are re-derived
    from THIS collapse and asserted equal to the scorer's."""
    doc_ranking: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in retrieved:
        for atom in (r.chunk.gold_provenance or ()):
            try:
                parent, span = atom
            except (TypeError, ValueError):
                continue
            key = (str(parent), str(span))
            if key not in seen:
                seen.add(key)
                doc_ranking.append(key)
    return doc_ranking


def rank_stats_from_ranking(doc_ranking, gold, k_values) -> dict:
    """hit@K / MRR / recall@K from a collapsed doc ranking + gold set."""
    relevance = [d in gold for d in doc_ranking]
    mrr = 0.0
    for i, rel in enumerate(relevance):
        if rel:
            mrr = 1.0 / (i + 1)
            break
    out = {"mrr": mrr, "hit_at_k": {}, "recall_at_k": {}}
    n_gold = len(gold)
    for k in k_values:
        top = relevance[:k]
        out["hit_at_k"][k] = 1.0 if any(top) else 0.0
        out["recall_at_k"][k] = (sum(top) / n_gold) if n_gold else 0.0
    return out


def gold_and_ranked(benchmark_name: str, q, scoring_ranking):
    """Per-benchmark (gold document set, projected ranking) — using the
    benchmark's OWN projection, never a reimplementation."""
    atoms = q.gold_passage_sets[0] if q.gold_passage_sets else frozenset()
    if benchmark_name == "multihop_rag":
        return frozenset(atoms), scoring_ranking
    # HotpotQA family: document = title; project exactly as the scorer does
    from src.eval.hotpotqa import _TITLE_SPAN, _project_to_titles
    gold = frozenset((t, _TITLE_SPAN) for t, _ in atoms)
    return gold, _project_to_titles(list(scoring_ranking))


def _parse_env(env: str) -> dict[str, str]:
    out = {}
    for part in (env or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _floats_equal(a, b) -> bool:
    return float(a) == float(b)


def _norm_kdict(d) -> dict[int, float]:
    return {int(k): float(v) for k, v in (d or {}).items()}


def compare_row(banked_retr: dict, replayed) -> list[str]:
    """Field-by-field banked-vs-replayed comparison; returns mismatch
    descriptions (empty = row reproduces)."""
    bad = []
    if bool(banked_retr.get("skipped")) != bool(replayed.skipped):
        return [f"skipped {banked_retr.get('skipped')} vs {replayed.skipped}"]
    if banked_retr.get("skipped"):
        return []
    for f in ("f1", "recall", "precision"):
        if not _floats_equal(banked_retr.get(f, 0.0), getattr(replayed, f)):
            bad.append(f"set {f} {banked_retr.get(f)} vs {getattr(replayed, f)}")
    if not _floats_equal(banked_retr.get("mrr", 0.0), replayed.mrr):
        bad.append(f"mrr {banked_retr.get('mrr')} vs {replayed.mrr}")
    for name in ("hit_at_k", "map_at_k"):
        b = _norm_kdict(banked_retr.get(name))
        r = _norm_kdict(getattr(replayed, name))
        if b != r:
            bad.append(f"{name} {b} vs {r}")
    return bad


def _load_banked_rows(jpath: Path) -> dict[str, dict]:
    rows = {}
    with jpath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[str(r["query_id"])] = r
    return rows


def assemble_cdir(system, system_id: str, corpus_hash: str):
    """(cdir, required_files, inputs) via the SYSTEM'S OWN index-path
    key assembly -- pinned to the banked keys by the manifest-oracle
    tests (M2 reproduces 51a2e3f9..., M4 reproduces cc068144... from
    the recorded corpus hashes).

    WHY THIS EXISTS (the three-refusal session, 2026-08-31):
    `BaseSystem.substrate_warm_path` defaults to an UNCONDITIONAL None
    meaning "this system has no probe" -- its own docstring warns None
    must not be read as checked-and-cold -- and the first replay read
    exactly that None as a cache miss and refused the most-proven
    substrate in the project. Only M4 overrides the probe. So the
    replay now assembles M2/M3 keys EXACTLY as their index() does
    (M2: no extra; M3: the sparse/fusion/rrf_k extra) and uses M4's
    own override-aware `_cache_dir`. The key function is pure given
    (chunking config, embedder, corpus_hash, extra), which is what lets
    the key oracles in tests/test_key_oracles.py feed a recorded corpus
    hash directly."""
    from src import paths as _paths
    from src.cache import CacheDir, compute_cache_key

    if system_id == "M4":
        from src.retrievers.m4_raptor import (
            REQUIRED_FILES as M4_REQ,
            resolve_components as m4_resolve,
        )
        if getattr(system, "_resolved", None) is None:
            system._resolved = m4_resolve(
                system.config.m4, system.config, default_reranker=None)
        cdir = system._cache_dir(corpus_hash)
        inputs = {
            "embedder": system._resolved.embedder_id,
            "chunking": "raptor paper chunker (in extra)",
            "env_override": system.topology_env_override,
            "summary_model": system.config.m4.summary_model,
        }
        return cdir, M4_REQ, inputs

    from dataclasses import asdict
    from src.components import resolve_components
    res = resolve_components(None, system.config)
    if system_id == "M2":
        from src.retrievers.m2_flat_dense import REQUIRED_FILES as REQ
        extra = None
    elif system_id == "M3":
        from src.retrievers.m3_hybrid import REQUIRED_FILES as REQ
        extra = {"sparse": "bm25okapi", "fusion": "rrf",
                 "rrf_k": system.config.retrieval.rrf_k}
    else:
        _fail(f"assemble_cdir: unranked system {system_id!r}")
    key = compute_cache_key(chunking_config=res.chunker_config,
                            embedder_model=res.embedder_id,
                            corpus_hash=corpus_hash, extra=extra)
    cdir = CacheDir(_paths.cache_dir(), system_id, key)
    inputs = {"embedder": res.embedder_id,
              "chunking": asdict(res.chunker_config), "extra": extra}
    return cdir, REQ, inputs


def resolve_substrate(system, system_id: str, items):
    """(warm_dir_or_None, corpus_hash, expected_dir) for one unit --
    read-only: corpus layout into a TemporaryDirectory, hash, the
    system's own key assembly, completeness check. Never builds."""
    import tempfile
    from src.cache import corpus_content_hash

    with tempfile.TemporaryDirectory(prefix="replay_warm_") as td:
        tdp = Path(td)
        system._write_corpus_layout(list(items), tdp)
        chash = corpus_content_hash(tdp)
    cdir, req, _ = assemble_cdir(system, system_id, chash)
    expected = Path(str(cdir.manifest_path)).parent
    warm = expected if cdir.is_complete(req) else None
    return warm, chash, expected


def _assert_generator_never_loaded() -> None:
    from src.models import load_generator
    info = load_generator.cache_info()
    if info.currsize != 0:
        _fail(f"a generator was loaded during replay ({info}) -- the "
              "replay touched the generation path; nothing it wrote can "
              "be trusted")


def replay_cell(bank: Path, generator: str, benchmark_name: str,
                system_id: str) -> dict:
    from src.config import DEFAULT_CONFIG
    from src.eval.runner import BENCHMARK_REGISTRY, SYSTEM_REGISTRY

    stem = f"{benchmark_name}_{system_id}_validation"
    spath = bank / f"{stem}.summary.json"
    jpath = bank / f"{stem}.jsonl"
    side_rows = bank / f"rankings.{stem}.jsonl"
    side_sum = bank / f"rankings.{stem}.json"
    if not spath.is_file() or not jpath.is_file():
        _fail(f"missing banked cell {stem} in {bank}")
    if side_rows.exists():
        _fail(f"{side_rows} already exists -- the replay never needs "
              "repeating; delete the sidecar by hand to deliberately "
              "regenerate this cell")
    summary = json.loads(spath.read_text(encoding="utf-8"))
    if summary.get("partial_run"):
        _fail(f"{stem}: partial_run")
    if summary.get("generator") != generator:
        _fail(f"{stem}: generator {summary.get('generator')!r} != bank's "
              f"{generator!r}")
    banked = _load_banked_rows(jpath)

    # KEY IDENTITY, scoped per system (corrected after the first-run
    # refusal). M2/M3 keys carry NO topology component -- their
    # substrates warm-hit across eras BY DESIGN (the six bit-identity
    # proofs), so for them the recorded tree_build_env is informational
    # and key identity IS the warm probe (chunker + embedder + corpus
    # all fold into compute_cache_key). M4's key DOES fold the env, and
    # three P10 cells predate the python token, so string equality is
    # wrong even in principle: instead (i) assert the HOST is
    # COMPATIBLE with the banked tree -- identical umap/sklearn/numpy
    # versions, and the summary's python major.minor equal to the
    # host's -- then (ii) reconstruct the banked key by injecting the
    # RECORDED env string verbatim (token-less stays token-less) via
    # the replay-only topology_env_override lever, which the runner
    # never sets and which is byte-identical to the old key when None.
    env_override = None
    if system_id == "M4":
        banked_env = summary.get("tree_build_env")
        if not banked_env:
            _fail(f"{stem}: M4 summary records no tree_build_env -- the "
                  "banked substrate key cannot be reconstructed")
        from src.raptor_paper import PAPER_TREE_BUILD_ENV
        host = _parse_env(PAPER_TREE_BUILD_ENV)
        rec = _parse_env(banked_env)
        for pkg in ("umap-learn", "scikit-learn", "numpy"):
            if pkg not in rec:
                _fail(f"{stem}: banked tree_build_env lacks {pkg!r}: "
                      f"{banked_env!r}")
            if rec[pkg] != host.get(pkg):
                _fail(f"{stem}: HOST INCOMPATIBLE with the banked tree: "
                      f"{pkg} host {host.get(pkg)!r} != banked "
                      f"{rec[pkg]!r} -- run under the pinned stack "
                      "(blocks E/F)")
        if "python" in rec and rec["python"] != host.get("python"):
            _fail(f"{stem}: HOST INCOMPATIBLE: python {host.get('python')!r}"
                  f" != banked {rec['python']!r}")
        sum_py = ".".join(str((summary.get("environment") or {})
                              .get("python", "")).split(".")[:2])
        if sum_py and host.get("python") and sum_py != host["python"]:
            _fail(f"{stem}: HOST INCOMPATIBLE: summary python {sum_py} != "
                  f"host {host['python']}")
        env_override = banked_env

    cfg = replace(
        DEFAULT_CONFIG,
        generation=replace(DEFAULT_CONFIG.generation, model=generator),
        m4=replace(DEFAULT_CONFIG.m4, summary_model=generator),
    )
    system = SYSTEM_REGISTRY[system_id](config=cfg)
    if env_override is not None:
        system.topology_env_override = env_override
    benchmark = BENCHMARK_REGISTRY[benchmark_name]()

    t0 = time.time()
    out_rows: list[dict] = []
    mismatches: list[str] = []
    n_rank = 0
    recall_sums = {k: 0.0 for k in RECALL_KS}
    hit5_sum = 0.0

    n_units = 0
    for unit in benchmark.iter_eval_units(split="validation"):
        warm, chash, expected = resolve_substrate(
            system, system_id, unit.corpus)
        if warm is None:
            _fail(f"{stem}: unit {unit.corpus_id!r} has NO COMPLETE "
                  f"substrate at the expected directory {expected} -- a "
                  "replay must never build; refusing before the build "
                  "path is reachable")
        # KEY IDENTITY SEEN, NOT INFERRED: the resolved directory's own
        # manifest must record the corpus hash this replay derived.
        mpath = warm / "manifest.json"
        if mpath.is_file():
            man = json.loads(mpath.read_text(encoding="utf-8"))
            if str(man.get("corpus_hash")) != chash:
                _fail(f"{stem}: unit {unit.corpus_id!r}: resolved "
                      f"substrate {warm.name} records corpus_hash "
                      f"{man.get('corpus_hash')!r} but the replay derived "
                      f"{chash!r} -- key collision or corpus drift")
        else:
            _fail(f"{stem}: unit {unit.corpus_id!r}: {warm} has no "
                  "manifest.json -- cannot verify key identity")
        n_units += 1
        if n_units <= 3:
            print(f"[replay] {stem}: unit {unit.corpus_id} substrate "
                  f"{warm.name} (manifest corpus_hash verified)")
        system.index_items(list(unit.corpus))
        if system_id == "M4" and getattr(system, "tree_cache_hit", None) is not True:
            _fail(f"{stem}: unit {unit.corpus_id!r} loaded without a warm "
                  "tree hit -- the replay built or rebuilt something; "
                  "refusing")
        for q in unit.queries:
            row = banked.get(str(q.query_id))
            if row is None:
                _fail(f"{stem}: banked row missing for {q.query_id!r}")
            prepared = system.prepare(q.question_text)
            replayed = benchmark.score_retrieval(
                prepared.retrieved, q,
                scoring_ranking=prepared.scoring_ranking)
            bad = compare_row(row.get("retrieval") or {}, replayed)
            if bad:
                mismatches.append(f"{q.query_id}: " + "; ".join(bad))
                if len(mismatches) >= 5:
                    break
                continue
            if replayed.skipped:
                continue
            gold, ranked = gold_and_ranked(benchmark_name, q,
                                           prepared.scoring_ranking)
            doc_ranking = collapse_to_doc_ranking(ranked)
            stats = rank_stats_from_ranking(doc_ranking, gold, RECALL_KS)
            rep_hit = _norm_kdict(replayed.hit_at_k)
            for k in RECALL_KS:
                if k in rep_hit and stats["hit_at_k"][k] != rep_hit[k]:
                    _fail(f"{stem}: {q.query_id}: sidecar collapse "
                          f"disagrees with the scorer at hit@{k} -- the "
                          "collapse mirror has drifted; refusing")
            if not _floats_equal(stats["mrr"], replayed.mrr):
                _fail(f"{stem}: {q.query_id}: sidecar collapse disagrees "
                      "with the scorer on MRR; refusing")
            n_rank += 1
            hit5_sum += stats["hit_at_k"][5]
            for k in RECALL_KS:
                recall_sums[k] += stats["recall_at_k"][k]
            out_rows.append({
                "query_id": str(q.query_id),
                "n_gold": len(gold),
                "gold": sorted(list(g) for g in gold),
                "doc_ranking": [list(d) for d in
                                doc_ranking[:DOC_RANKING_DEPTH]],
                "recall_at_k": {str(k): stats["recall_at_k"][k]
                                for k in RECALL_KS},
            })
        if len(mismatches) >= 5:
            break
    _assert_generator_never_loaded()
    if mismatches:
        _fail(f"{stem}: GATE FAILED -- replayed retrieval does not "
              "reproduce the bank on these rows (first "
              f"{len(mismatches)}):\n  " + "\n  ".join(mismatches))

    expected = (int(summary["n_answerable"])
                if benchmark_name == "multihop_rag"
                else int(summary["n_queries_scored"]))
    if n_rank != expected:
        _fail(f"{stem}: rank-scored population {n_rank} != expected "
              f"{expected}")

    recall = {k: recall_sums[k] / n_rank for k in RECALL_KS}
    hit5 = hit5_sum / n_rank
    if recall[5] > hit5 + 1e-12:
        _fail(f"{stem}: recall@5 {recall[5]} > hit@5 {hit5} -- "
              "impossible; a bug, not a finding")
    if benchmark_name != "multihop_rag" and recall[5] < hit5 / 2 - 1e-12:
        _fail(f"{stem}: recall@5 {recall[5]} < hit@5/2 {hit5 / 2} on a "
              "two-gold benchmark -- impossible; a bug, not a finding")

    with side_rows.open("w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    side_summary = {
        "generator": generator, "benchmark": benchmark_name,
        "system": system_id, "n_rows": n_rank,
        "recall_at_1": recall[1], "recall_at_5": recall[5],
        "recall_at_10": recall[10],
        "hit_at_5_replayed": hit5,
        "gate": "PASS: every row reproduced banked set-F1/recall/"
                "precision, hit_at_k, map_at_k and mrr; sidecar collapse "
                "re-derived hit@K and MRR per row",
        "elapsed_s": round(time.time() - t0, 2),
        "generated_by": _generating_commit(),
        "timestamp": time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
    }
    side_sum.write_text(json.dumps(side_summary, indent=2),
                        encoding="utf-8")
    print(f"[replay] {generator.split('/')[-1]} {benchmark_name} "
          f"{system_id}: n={n_rank} recall@5={recall[5]:.6f} "
          f"hit@5={hit5:.6f} substrates={n_units} (manifest corpus_hash "
          f"verified on all) ({side_summary['elapsed_s']}s) GATE PASS")
    return side_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p10", required=True)
    ap.add_argument("--p11", required=True)
    args = ap.parse_args()
    banks = {"p10": (QWEN, Path(args.p10)), "p11": (LLAMA, Path(args.p11))}
    targets = []
    for tag in ("p10", "p11"):
        gen, bank = banks[tag]
        for bench in RANKED_BENCHMARKS:
            for sysid in RANKED_SYSTEMS:
                targets.append((gen, bank, bench, sysid))
    for gen, bank, bench, sysid in targets:
        replay_cell(bank, gen, bench, sysid)
    print(f"[replay] done: {len(targets)} cell(s)")


if __name__ == "__main__":
    main()
