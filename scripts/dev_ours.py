"""Development tooling for ours, kept away from every scored cell.

  census  read-only: how many scored units already have their Llama tree,
          the topology string to pin in ours/config.py, the pinned revisions
  smoke   two held-out HotpotQA units through the runner loop on a CPU:
          real loader and embedder, reader and cross-encoder faked
  run     one held-out development cell; appends its scores to DEV_LOG.jsonl

Held-out data: HotpotQA validation questions outside the registered 1,000
(both variants) and NarrativeQA validation stories outside the registered 40.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ours import config as settings
from src.eval.hotpotqa import (
    HF_CONFIG,
    HF_REPO,
    PREREGISTERED_Q,
    _HF_SPLIT,
    HotpotQABenchmark,
    HotpotQAPooledBenchmark,
)
from src.eval.narrativeqa import NarrativeQABenchmark, select_units
from src.eval.sampling import subsample_indices

# Development draw, dated like the registered seed.
DEV_SEED = 20261007
N_DEV_QUESTIONS = 300
N_DEV_STORIES = 10
SCORED_BENCHMARKS = ("multihop_rag", "narrativeqa", "hotpotqa", "hotpotqa_pooled")


def llama_config():
    """The runner's config under --generator Llama."""
    from src.config import DEFAULT_CONFIG

    return replace(
        DEFAULT_CONFIG,
        generation=replace(DEFAULT_CONFIG.generation, model=settings.READER),
        m4=replace(DEFAULT_CONFIG.m4, summary_model=settings.READER))


# --- held-out loaders -----------------------------------------------------


def held_out_indices(n_total: int) -> list[int]:
    """A seeded draw of dev questions from those outside the registered 1,000."""
    scored = set(subsample_indices(n_total, PREREGISTERED_Q))
    held = [i for i in range(n_total) if i not in scored]
    return [held[j] for j in subsample_indices(len(held), N_DEV_QUESTIONS, seed=DEV_SEED)]


class _HeldOutRows:
    """Loads the held-out draw in place of the registered sample."""

    def _load(self):
        if self._rows is None:
            from datasets import load_dataset

            ds = load_dataset(HF_REPO, HF_CONFIG, split=_HF_SPLIT)
            ds = ds.select(held_out_indices(len(ds)))
            self._rows = ds
            self._measure(ds)
            self.stats["subsample_seed"] = DEV_SEED
            self.stats["held_out"] = True
        return self._rows


class HeldOutHotpotQA(_HeldOutRows, HotpotQABenchmark):
    """HotpotQA distractor over the held-out questions."""

    def __init__(self) -> None:
        super().__init__(max_questions=N_DEV_QUESTIONS)


class HeldOutHotpotQAPooled(_HeldOutRows, HotpotQAPooledBenchmark):
    """HotpotQA pooled over the same held-out questions."""

    def __init__(self) -> None:
        super().__init__(max_questions=N_DEV_QUESTIONS)


class HeldOutNarrativeQA(NarrativeQABenchmark):
    """NarrativeQA over a seeded draw of the stories outside the registered 40."""

    cell_units = N_DEV_STORIES

    def iter_eval_units(self, *, split, max_units=None):
        units = list(super().iter_eval_units(split=split, max_units=10 ** 6))
        scored = set(select_units([u.corpus_id for u in units], None))
        held = [u for u in units if u.corpus_id not in scored]
        picked = [held[i] for i in subsample_indices(len(held), N_DEV_STORIES,
                                                     seed=DEV_SEED)]
        self.stats.update(n_queries=sum(len(u.queries) for u in picked),
                          n_stories=len(picked), subsample_seed=DEV_SEED,
                          sampled_story_ids=[u.corpus_id for u in picked],
                          held_out=True)
        yield from picked


HELD_OUT = {"hotpotqa": HeldOutHotpotQA,
            "hotpotqa_pooled": HeldOutHotpotQAPooled,
            "narrativeqa": HeldOutNarrativeQA}


# --- numbers from a cell's rows (also used by scripts/report_ours.py) -----


def recall_at(ranking, gold, k: int) -> float:
    """Share of gold documents among the first k ranked; a short ranking is not padded."""
    return len(set(ranking[:k]) & set(gold)) / len(gold) if gold else 0.0


def gold_documents(query) -> frozenset:
    """A question's gold documents: the parents of its gold atoms."""
    atoms = query.gold_passage_sets[0] if query.gold_passage_sets else frozenset()
    return frozenset(str(parent) for parent, _ in atoms)


def ours_numbers(rows, golds, gold_docs) -> dict:
    """F1, EM, R@1/5/10 and ours' mechanics, from one cell's rows."""
    from src.eval.scorers.extractive import normalize_qasper_answer

    answerable = [r for r in rows if r["answer"]["method"] != "unanswerable_rule"]
    em = [max((int(normalize_qasper_answer(r.get("predicted_answer") or "")
                   == normalize_qasper_answer(g)) for g in golds[str(r["query_id"])]),
              default=0)
          for r in answerable]
    ranked = [r for r in rows if gold_docs.get(str(r["query_id"]))]
    meta = [r["metadata"] for r in rows]
    aspects = [a for m in meta for a in m["ours_aspects"]]
    detectors: dict = {}
    for m in meta:
        for name, count in m["ours_section_detectors"].items():
            detectors[name] = detectors.get(name, 0) + count
    n_docs = sum(detectors.values())

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    return {
        "n_rows": len(rows),
        "f1": mean(r["answer"]["value"] for r in answerable),
        "em": mean(em),
        **{f"recall_at_{k}": mean(recall_at(r["metadata"]["ours_doc_ranking"],
                                            gold_docs[str(r["query_id"])], k)
                                  for r in ranked) for k in (1, 5, 10)},
        "parse_failure_rate": mean(int(m["ours_parse_failed"]) for m in meta),
        "mean_llm_calls": mean(m["ours_n_llm_calls"] for m in meta),
        "mean_aspects": mean(len(m["ours_aspects"]) for m in meta),
        "low_confidence_share": mean(int(a["low_confidence"]) for a in aspects),
        "mean_evidence_chunks": mean(r["n_retrieved"] for r in rows),
        "mean_evidence_tokens": mean(r["evidence_tokens"] for r in rows),
        "tree_cache_hit_share": mean(int(bool(m["ours_tree_cache_hit"])) for m in meta),
        "section_detector_share": {k: v / n_docs for k, v in sorted(detectors.items())},
        "s_per_query": mean(r["latency_s"] for r in rows),
        "code": sorted({m["ours_code"] for m in meta}),
    }


def golds_of(benchmark) -> tuple[dict, dict]:
    """Gold answers and gold documents per query id, from a loader's units."""
    from scripts.export_comparison import gold_texts

    golds, docs = {}, {}
    for unit in benchmark.iter_eval_units(split="validation"):
        for q in unit.queries:
            golds[str(q.query_id)] = gold_texts(q)
            docs[str(q.query_id)] = gold_documents(q)
    return golds, docs


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
            if x.strip()]


# --- census -----------------------------------------------------------------


def census(p11: Path, benchmarks=SCORED_BENCHMARKS) -> str:
    """Count warm Llama trees for the scored units; read only. Returns the topology string."""
    from src.eval.runner import BENCHMARK_REGISTRY
    from src.retrievers.m4_raptor import RaptorSystem

    summaries = {b: json.loads((p11 / f"{b}_M4_validation.summary.json")
                               .read_text(encoding="utf-8")) for b in benchmarks}
    envs = {s.get("tree_build_env") for s in summaries.values()}
    if len(envs) != 1 or None in envs:
        raise SystemExit(f"[census] REFUSED: the p11 M4 summaries record {sorted(map(str, envs))}; "
                         "one topology string is required")
    env = envs.pop()
    print(f"[census] p11 tree_build_env: {env!r}")
    print(f"[census] ours/config.py TREE_BUILD_ENV: {settings.TREE_BUILD_ENV!r} "
          f"({'matches' if settings.TREE_BUILD_ENV == env else 'PIN THE VALUE ABOVE'})")
    for b, s in summaries.items():
        revisions = (s.get("model_revisions") or {}).get("revisions") or {}
        for role, pinned in (("generator", settings.READER_REVISION),
                             ("embedder", settings.EMBEDDER_REVISION)):
            got = revisions.get(role)
            print(f"[census] {b} p11 {role} revision {got} "
                  f"({'matches' if got == pinned else 'DIFFERS from ' + pinned})")

    m4 = RaptorSystem(config=llama_config())
    m4.topology_env_override = env
    for b in benchmarks:
        warm, cold, first = 0, 0, None
        for unit in BENCHMARK_REGISTRY[b]().iter_eval_units(split="validation"):
            path = m4.substrate_warm_path(unit.corpus)
            if path:
                warm += 1
                first = first or path
            else:
                cold += 1
        print(f"[census] {b}: {warm} units warm, {cold} to build")
        if first:
            manifest = json.loads((Path(first) / "manifest.json").read_text())
            print(f"[census] {b}: e.g. {first} summary_model="
                  f"{manifest['extra']['m4']['summary_model']} "
                  f"build_env={manifest['extra'].get('build_env')}")
    return env


# --- smoke ------------------------------------------------------------------


def smoke() -> None:
    """Two held-out HotpotQA units, cold then warm: real loader and embedder, reader and cross-encoder faked."""
    from unittest import mock

    from ours.model import OursSystem
    from scripts.replay_retrieval import rank_stats_from_ranking
    from src.eval.base import BenchmarkRunner
    from src.eval.hotpotqa import _TITLE_SPAN
    from src.retrievers.m4_raptor import RaptorSystem
    from tests.test_ours_smoke import FakeReader, fake_cross_encoder, fake_generate_batch

    loader = HeldOutHotpotQA()
    units = list(loader.iter_eval_units(split="validation"))[:2]

    class _TwoUnits:
        name = "hotpotqa"
        stats = {"n_queries": 2}

        def iter_eval_units(self, *, split, max_units=None):
            yield from units

        def score_answer(self, predicted, query):
            return loader.score_answer(predicted, query)

        def score_retrieval(self, retrieved, query, scoring_ranking=None):
            return loader.score_retrieval(retrieved, query, scoring_ranking=scoring_ranking)

    q0, q1 = (u.queries[0].question_text for u in units)
    half = len(q0) // 2
    reader = FakeReader(
        {q0: json.dumps({"type": "multi_aspect", "aspects": [
            {"name": q0[:half], "importance": 0.6},
            {"name": q0[half:], "importance": 0.4}]})},
        {q0: "a short answer", q1: settings.REFUSAL})
    env = settings.TREE_BUILD_ENV or "smoke-only"
    root = Path(tempfile.mkdtemp(prefix="ours_smoke_"))
    passes = []
    for name in ("cold", "warm"):
        out = root / f"{name}.jsonl"
        with mock.patch("src.models.generate", reader), \
                mock.patch("src.models.generate_batch", fake_generate_batch), \
                mock.patch("ours.model.cross_encoder_scores", fake_cross_encoder), \
                mock.patch.object(settings, "TREE_BUILD_ENV", env), \
                mock.patch("src.paths.cache_dir", return_value=root / "cache"):
            system = OursSystem(config=llama_config())
            list(BenchmarkRunner(output_path=out, verbose=False).run(
                system, _TwoUnits(), split="validation"))
            m4 = RaptorSystem(config=llama_config())
            m4.topology_env_override = env
            warm_paths = [m4.substrate_warm_path(u.corpus) for u in units]
        passes.append(read_rows(out))

    cold, warm = passes
    assert [r["metadata"]["ours_n_llm_calls"] for r in cold] == [6, 4], cold
    assert [r["metadata"]["ours_parse_failed"] for r in cold] == [False, True]
    assert cold[1]["metadata"]["ours_aspects"][0]["budget"] == 13
    assert all(0 < r["n_retrieved"] <= 15 for r in cold)
    assert all(set(r["retrieved_unit_types"]) == {"chunk"} for r in cold)
    assert cold[1]["answer"]["metadata"]["abstained"] is True
    assert all(warm_paths), "M4 does not find the trees ours built"
    assert [r["metadata"]["ours_tree_cache_hit"] for r in cold] == [False, False]
    assert [r["metadata"]["ours_tree_cache_hit"] for r in warm] == [True, True]
    for row, unit in zip(cold, units):
        gold = frozenset((t, _TITLE_SPAN) for t in gold_documents(unit.queries[0]))
        stats = rank_stats_from_ranking(
            [(p, _TITLE_SPAN) for p in row["metadata"]["ours_doc_ranking"]], gold,
            (1, 2, 5, 10))
        assert stats["hit_at_k"] == {int(k): v for k, v in row["retrieval"]["hit_at_k"].items()}
        assert stats["mrr"] == row["retrieval"]["mrr"]

    def strip(r):
        r = dict(r, metadata={k: v for k, v in r["metadata"].items()
                              if k != "ours_tree_cache_hit"})
        r.pop("latency_s")
        return r
    assert [strip(r) for r in cold] == [strip(r) for r in warm], "cold and warm rows differ"
    for row in cold:
        print(f"SMOKE OK {row['query_id']}: calls={row['metadata']['ours_n_llm_calls']} "
              f"chunks={row['n_retrieved']} docs={row['metadata']['ours_doc_ranking'][:3]}")
    print(f"[smoke] rows and cache under {root}")


# --- held-out development runs ---------------------------------------------


def dev_run(benchmark: str, version: str, lockfile: Path, resume: bool) -> dict:
    """One held-out cell; its numbers are appended to outputs/ours_dev/DEV_LOG.jsonl."""
    from ours.model import CODE_FINGERPRINT, OursSystem
    from src import paths
    from src.eval.base import BenchmarkRunner
    from src.eval.runner import (
        _git_commit_short,
        assert_bank_generator_consistent,
        assert_bank_gpu_consistent,
        assert_environment_pinned,
    )

    # Development trees never go into the banked cache.
    if not os.environ.get("THESIS_CACHE_DIR"):
        raise SystemExit("[dev] REFUSED: set THESIS_CACHE_DIR to a development cache "
                         "(for example .../thesis_rag/cache_ours_dev); the banked "
                         "cache holds only scored-unit trees")
    assert_environment_pinned(lockfile)
    out = paths.output_dir() / "ours_dev" / version / f"{benchmark}_OURS_dev.jsonl"
    assert_bank_generator_consistent(out.parent, settings.READER)
    assert_bank_gpu_consistent(out.parent)

    loader = HELD_OUT[benchmark]()
    runner = BenchmarkRunner(output_path=out, resume=resume)
    t0 = time.perf_counter()
    n_this = sum(1 for _ in runner.run(OursSystem(config=llama_config()), loader,
                                       split="validation"))
    golds, docs = golds_of(loader)
    numbers = ours_numbers(read_rows(out), golds, docs)
    line = {"version": version, "benchmark": benchmark,
            "git_commit": _git_commit_short(), "ours_code": CODE_FINGERPRINT,
            "rows_this_process": n_this, "elapsed_s": round(time.perf_counter() - t0, 1),
            "timestamp": time.strftime("%Y%m%d-%H%M%S"), **numbers}
    log = paths.output_dir() / "ours_dev" / "DEV_LOG.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line, indent=2))
    return line


def main() -> None:
    """Parse the subcommand and run it."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("census", help="Count warm Llama trees; print the pins.")
    c.add_argument("--p11", type=Path, required=True)
    sub.add_parser("smoke", help="Two held-out HotpotQA units on a CPU, reader faked.")
    r = sub.add_parser("run", help="One held-out development cell.")
    r.add_argument("--benchmark", required=True, choices=sorted(HELD_OUT))
    r.add_argument("--version", required=True, help="Label for this development version.")
    r.add_argument("--lockfile", type=Path, default=Path("requirements.lock"))
    r.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.command == "census":
        census(args.p11)
    elif args.command == "smoke":
        smoke()
    else:
        dev_run(args.benchmark, args.version, args.lockfile, args.resume)


if __name__ == "__main__":
    main()
