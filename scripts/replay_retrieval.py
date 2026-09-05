"""Replay retrieval over the warm substrates of the 18 ranked cells, check
every row against the banked scores, and write a rankings sidecar with
recall@{1,5,10} beside each cell. Run it on the bank's GPU class (L4)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_comparison import LLAMA, QWEN, _fail, _generating_commit

# NarrativeQA is left out: it has no retrieval gold to rank against.
# dataset: no passage annotation, retrieval never scored
RANKED_BENCHMARKS = ("multihop_rag", "hotpotqa", "hotpotqa_pooled")
RANKED_SYSTEMS = ("M2", "M3", "M4")
RECALL_KS = (1, 5, 10)
# harness choice: one scoring depth for every system (METHODS §D)
DOC_RANKING_DEPTH = 50


def collapse_to_doc_ranking(retrieved) -> list[tuple[str, str]]:
    """Collapse retrieved chunks to a first-occurrence document ranking."""
    # Mirrors score_retrieval_rank_aware in src/eval/alignment.py; each
    # replayed row checks that both collapses give the same hit@K and MRR.
    # harness choice: document-level metrics (METHODS §C.5)
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
    """Compute hit@K, MRR and recall@K from a doc ranking and its gold set."""
    relevance = [d in gold for d in doc_ranking]
    # MRR is 1/rank of the first gold document, 0 when none appears.
    mrr = 0.0
    for i, rel in enumerate(relevance):
        if rel:
            mrr = 1.0 / (i + 1)
            break
    out = {"mrr": mrr, "hit_at_k": {}, "recall_at_k": {}}
    n_gold = len(gold)
    # hit@K is any gold in the top K; recall@K is gold found over gold total.
    for k in k_values:
        top = relevance[:k]
        out["hit_at_k"][k] = 1.0 if any(top) else 0.0
        out["recall_at_k"][k] = (sum(top) / n_gold) if n_gold else 0.0
    return out


def gold_and_ranked(benchmark_name: str, q, scoring_ranking):
    """Return gold documents and the ranking in the benchmark's own units."""
    atoms = q.gold_passage_sets[0] if q.gold_passage_sets else frozenset()
    # MultiHop's unit is the whole document, so its atoms are already docs.
    # deviation from official (retrieval_evaluate.py matches gold-fact substrings): see METHODS §B.1
    if benchmark_name == "multihop_rag":
        return frozenset(atoms), scoring_ranking
    # HotpotQA family: the document is the title; project as the scorer does.
    # harness choice: supporting facts are (title, sentence) pairs (METHODS §B.3)
    from src.eval.hotpotqa import _TITLE_SPAN, _project_to_titles
    gold = frozenset((t, _TITLE_SPAN) for t, _ in atoms)
    return gold, _project_to_titles(list(scoring_ranking))


def _parse_env(env: str) -> dict[str, str]:
    """Parse a 'pkg=version;pkg=version' env string into a dict."""
    out = {}
    for part in (env or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _floats_equal(a, b) -> bool:
    """Compare two values as float64, exactly."""
    return float(a) == float(b)


def _norm_kdict(d) -> dict[int, float]:
    """Normalise a per-K dict to int keys and float values."""
    return {int(k): float(v) for k, v in (d or {}).items()}


def compare_row(banked_retr: dict, replayed) -> list[str]:
    """Compare a banked retrieval row to its replay; return the mismatches."""
    bad = []
    # The skipped flag must agree; a skipped row has nothing else to compare.
    if bool(banked_retr.get("skipped")) != bool(replayed.skipped):
        return [f"skipped {banked_retr.get('skipped')} vs {replayed.skipped}"]
    if banked_retr.get("skipped"):
        return []
    # Set metrics, MRR and the per-K dicts must match float-exact.
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
    """Load a banked JSONL into a dict keyed by query_id."""
    rows = {}
    with jpath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[str(r["query_id"])] = r
    return rows


def assemble_cdir(system, system_id: str, corpus_hash: str):
    """Build the substrate cache dir with the system's own key assembly."""
    from src import paths as _paths
    from src.cache import CacheDir, compute_cache_key

    # M4 owns its key through _cache_dir, which honours the env override.
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

    # M2 and M3 assemble their keys exactly as their index() does: M2 folds
    # no extra, M3 folds its sparse/fusion/rrf_k extra.
    # harness choice: content-addressed substrates (METHODS §D)
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
    """Find a unit's warm substrate without building; None when incomplete."""
    import tempfile
    from src.cache import corpus_content_hash

    # Lay the corpus out in a temp dir only to hash it.
    with tempfile.TemporaryDirectory(prefix="replay_warm_") as td:
        tdp = Path(td)
        system._write_corpus_layout(list(items), tdp)
        chash = corpus_content_hash(tdp)
    # Warm means every required file is present under the expected dir.
    cdir, req, _ = assemble_cdir(system, system_id, chash)
    expected = Path(str(cdir.manifest_path)).parent
    warm = expected if cdir.is_complete(req) else None
    return warm, chash, expected


def _assert_generator_never_loaded() -> None:
    """Refuse if a generator is loaded; the replay never generates."""
    from src.models import load_generator
    info = load_generator.cache_info()
    if info.currsize != 0:
        _fail(f"a generator was loaded during replay ({info}) -- the "
              "replay touched the generation path; nothing it wrote can "
              "be trusted")


def _rows_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_present_sidecar(stem: str, generator: str, benchmark_name: str,
                            system_id: str, side_rows: Path,
                            side_sum: Path) -> None:
    """Refuse unless a present sidecar is this cell's own and intact."""
    # The summary must exist, parse, and name this cell.
    if not side_sum.is_file():
        _fail(f"{stem}: {side_rows.name} is present but {side_sum.name} is "
              "missing -- a half-written sidecar; delete the rows file by "
              "hand and re-run")
    try:
        meta = json.loads(side_sum.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _fail(f"{stem}: {side_sum.name} is not valid JSON ({e}); delete "
              "both sidecar files by hand and re-run")
    ident = (meta.get("generator"), meta.get("benchmark"), meta.get("system"))
    if ident != (generator, benchmark_name, system_id):
        _fail(f"{stem}: {side_sum.name} names {ident}, not this cell "
              f"({generator!r}, {benchmark_name!r}, {system_id!r}) -- a "
              "foreign sidecar; delete both files by hand and re-run")
    # Every row must parse and carry query_id and recall_at_k.
    n = 0
    with side_rows.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                _fail(f"{stem}: {side_rows.name} row {n + 1} is not valid "
                      f"JSON ({e}); delete both sidecar files by hand and "
                      "re-run")
            if "query_id" not in row or "recall_at_k" not in row:
                _fail(f"{stem}: {side_rows.name} row {n + 1} lacks "
                      "query_id / recall_at_k; delete both sidecar files by "
                      "hand and re-run")
            n += 1
    # The row count, and the rows hash when the summary embeds one, must
    # match the summary.
    if n != int(meta.get("n_rows", -1)):
        _fail(f"{stem}: {side_rows.name} holds {n} rows but {side_sum.name} "
              f"records n_rows={meta.get('n_rows')}; delete both sidecar "
              "files by hand and re-run")
    embedded = meta.get("rows_sha256")
    if embedded is not None:
        actual = _rows_sha256(side_rows)
        if actual != embedded:
            _fail(f"{stem}: {side_rows.name} sha256 {actual} != the embedded "
                  f"rows_sha256 {embedded} -- the sidecar was altered after "
                  "it was written; delete both files by hand and re-run")
        how = f"rows_sha256 {embedded[:16]}... verified"
    else:
        how = ("no rows_sha256 field (written before the reduction); "
               "identity and row count verified")
    print(f"[replay] {generator.split('/')[-1]} {benchmark_name} "
          f"{system_id}: sidecar present, DONE -- {n} rows, {how}; the "
          f"replay never repeats a cell (delete {side_rows.name} and "
          f"{side_sum.name} by hand to regenerate it deliberately)")


def replay_cell(bank: Path, generator: str, benchmark_name: str,
                system_id: str) -> dict | None:
    """Replay one cell behind the row gate; None if its sidecar verifies."""
    from src.config import DEFAULT_CONFIG
    from src.eval.runner import BENCHMARK_REGISTRY, SYSTEM_REGISTRY

    # A verified sidecar means the cell is done; a partial or foreign bank
    # refuses. Sidecar names start with "rankings." so no bank glob sees them.
    stem = f"{benchmark_name}_{system_id}_validation"
    spath = bank / f"{stem}.summary.json"
    jpath = bank / f"{stem}.jsonl"
    side_rows = bank / f"rankings.{stem}.jsonl"
    side_sum = bank / f"rankings.{stem}.json"
    if not spath.is_file() or not jpath.is_file():
        _fail(f"missing banked cell {stem} in {bank}")
    if side_rows.exists():
        _verify_present_sidecar(stem, generator, benchmark_name, system_id,
                                side_rows, side_sum)
        return None
    summary = json.loads(spath.read_text(encoding="utf-8"))
    if summary.get("partial_run"):
        _fail(f"{stem}: partial_run")
    if summary.get("generator") != generator:
        _fail(f"{stem}: generator {summary.get('generator')!r} != bank's "
              f"{generator!r}")
    banked = _load_banked_rows(jpath)

    # Key identity per system. M2/M3 keys fold no topology component, so
    # the warm check alone identifies them. M4's key folds the tree-build
    # env, so first check the host runs the banked umap/sklearn/numpy and
    # python major.minor, then inject the recorded env string verbatim
    # through topology_env_override so the key rebuilds the banked one.
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

    # Build the system for this bank's generator; M4 also summarises with it.
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

    # Per unit: check the substrate is warm ahead of index_items so the
    # build path is never reachable, then replay every query against the bank.
    n_units = 0
    for unit in benchmark.iter_eval_units(split="validation"):
        warm, chash, expected = resolve_substrate(
            system, system_id, unit.corpus)
        if warm is None:
            _fail(f"{stem}: unit {unit.corpus_id!r} has NO COMPLETE "
                  f"substrate at the expected directory {expected} -- a "
                  "replay must never build; refusing before the build "
                  "path is reachable")
        # The resolved directory's manifest must record the corpus hash
        # this replay derived.
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
        # M4 must load a warm tree; a build of any kind refuses the cell.
        system.index_items(list(unit.corpus))
        if system_id == "M4" and getattr(system, "tree_cache_hit", None) is not True:
            _fail(f"{stem}: unit {unit.corpus_id!r} loaded without a warm "
                  "tree hit -- the replay built or rebuilt something; "
                  "refusing")
        for q in unit.queries:
            row = banked.get(str(q.query_id))
            if row is None:
                _fail(f"{stem}: banked row missing for {q.query_id!r}")
            # prepare() retrieves, ranks and packs without an LLM; score it
            # through the frozen scorer and compare to the banked row.
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
            # The sidecar's own collapse must re-derive the scorer's hit@K
            # and MRR before its recall@K counts.
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

    # MultiHop ranks only its answerable rows; the others rank every
    # scored query.
    expected = (int(summary["n_answerable"])
                if benchmark_name == "multihop_rag"
                else int(summary["n_queries_scored"]))
    if n_rank != expected:
        _fail(f"{stem}: rank-scored population {n_rank} != expected "
              f"{expected}")

    # Bounds: recall@5 <= hit@5 always; with two gold titles per question
    # a hit gives recall >= 1/2, so recall@5 >= hit@5 / 2 on HotpotQA.
    recall = {k: recall_sums[k] / n_rank for k in RECALL_KS}
    hit5 = hit5_sum / n_rank
    if recall[5] > hit5 + 1e-12:
        _fail(f"{stem}: recall@5 {recall[5]} > hit@5 {hit5} -- "
              "impossible; a bug, not a finding")
    if benchmark_name != "multihop_rag" and recall[5] < hit5 / 2 - 1e-12:
        _fail(f"{stem}: recall@5 {recall[5]} < hit@5/2 {hit5 / 2} on a "
              "two-gold benchmark -- impossible; a bug, not a finding")

    # Write the rows first, then the summary with the rows hash embedded so
    # a later run can verify the rows file instead of trusting it.
    with side_rows.open("w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    side_summary = {
        "generator": generator, "benchmark": benchmark_name,
        "system": system_id, "n_rows": n_rank,
        "rows_sha256": _rows_sha256(side_rows),
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
    """Visit all 18 ranked cells of both banks in one invocation."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p10", required=True)
    ap.add_argument("--p11", required=True)
    args = ap.parse_args()
    # Both banks, three ranked benchmarks, three ranked systems.
    banks = {"p10": (QWEN, Path(args.p10)), "p11": (LLAMA, Path(args.p11))}
    targets = []
    for tag in ("p10", "p11"):
        gen, bank = banks[tag]
        for bench in RANKED_BENCHMARKS:
            for sysid in RANKED_SYSTEMS:
                targets.append((gen, bank, bench, sysid))
    n_done = n_present = 0
    for gen, bank, bench, sysid in targets:
        if replay_cell(bank, gen, bench, sysid) is None:
            n_present += 1
        else:
            n_done += 1
    print(f"[replay] done: {n_done} cell(s) replayed, {n_present} already "
          f"present of {len(targets)}")


if __name__ == "__main__":
    main()
