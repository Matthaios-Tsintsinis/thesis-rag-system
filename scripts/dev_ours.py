"""Development tooling for ours, kept away from every scored cell.

  census  read-only: how many scored units already have their Llama tree,
          the topology string to pin in ours/config.py, the pinned revisions
  smoke   two held-out HotpotQA units through the runner loop on a CPU:
          real loader and embedder, reader and cross-encoder faked
  run     one held-out development cell; appends its scores to DEV_LOG.jsonl;
          --no-gold takes each unit's gold paragraphs out, so nothing answers
          the questions: what a refusal is worth, rows kept apart
  bench   retrieval alone under each option, on the same questions, no reader's
          answer: which change is worth a full run. On MultiHop label-free only
  replay  the decomposition call alone over a scored benchmark's questions,
          with more room than the scored cap, raw replies saved: why the
          aspect parse fails (reads questions only, never gold or scores)

Held-out data: HotpotQA validation questions outside the registered 1,000
(both variants) and NarrativeQA validation stories outside the registered 40.
Three draws, which never meet: a, the first; b, larger, for development once a
had tested ten versions; c, kept for the last check before a freeze.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
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
# Later draws, from what the registered sample and the first draw left. b has
# four times a's HotpotQA questions, so a paired interval about half as wide, and
# NarrativeQA's 40 stories match the scored cell's. c takes the 25 stories left.
LATER_SEED = 20261106
N_LATER_QUESTIONS = {"b": 1200, "c": 1000}
N_LATER_STORIES = {"b": 40, "c": 25}
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


def later_slice(keys, sizes: dict, draw: str) -> list:
    """A later draw's share of what is left: a seeded order, cut so the draws never meet."""
    order = list(keys)
    random.Random(LATER_SEED).shuffle(order)
    start = sizes["b"] if draw == "c" else 0
    return order[start: start + sizes[draw]]


def later_indices(n_total: int, draw: str) -> list[int]:
    """A later draw's questions, outside the registered 1,000 and the first draw."""
    taken = set(subsample_indices(n_total, PREREGISTERED_Q)) | set(held_out_indices(n_total))
    rest = [i for i in range(n_total) if i not in taken]
    return sorted(later_slice(rest, N_LATER_QUESTIONS, draw))


def question_indices(n_total: int, draw: str) -> list[int]:
    """Which questions a draw holds."""
    return held_out_indices(n_total) if draw == "a" else later_indices(n_total, draw)


def draw_seed(draw: str) -> int:
    """The seed behind a draw."""
    return DEV_SEED if draw == "a" else LATER_SEED


class _HeldOutRows:
    """Loads a held-out draw in place of the registered sample."""

    draw = "a"

    def _load(self):
        if self._rows is None:
            from datasets import load_dataset

            ds = load_dataset(HF_REPO, HF_CONFIG, split=_HF_SPLIT)
            ds = ds.select(question_indices(len(ds), self.draw))
            if self.draw != "a":
                # The loader refuses a gold that normalises to empty ("!!!"); a later
                # draw goes without that question.
                from src.eval.scorers.extractive import normalize_qasper_answer

                ds = ds.select([i for i, answer in enumerate(ds["answer"])
                                if normalize_qasper_answer((answer or "").strip())])
            self._rows = ds
            self._measure(ds)
            self.stats["subsample_seed"] = draw_seed(self.draw)
            self.stats["held_out"] = True
            # _measure prints the registered seed; this is what was drawn.
            print(f"[dev] held-out draw {self.draw}: {len(ds)} questions outside the "
                  f"registered {PREREGISTERED_Q}, seed {draw_seed(self.draw)}")
        return self._rows


class HeldOutHotpotQA(_HeldOutRows, HotpotQABenchmark):
    """HotpotQA distractor over the held-out questions."""

    def __init__(self) -> None:
        super().__init__(max_questions=N_DEV_QUESTIONS if self.draw == "a"
                         else N_LATER_QUESTIONS[self.draw])


class HeldOutHotpotQAPooled(_HeldOutRows, HotpotQAPooledBenchmark):
    """HotpotQA pooled over the same held-out questions."""

    def __init__(self) -> None:
        super().__init__(max_questions=N_DEV_QUESTIONS if self.draw == "a"
                         else N_LATER_QUESTIONS[self.draw])


class HeldOutNarrativeQA(NarrativeQABenchmark):
    """NarrativeQA over a seeded draw of the stories outside the registered 40."""

    draw = "a"
    cell_units = N_DEV_STORIES

    def iter_eval_units(self, *, split, max_units=None):
        units = list(super().iter_eval_units(split=split, max_units=10 ** 6))
        scored = set(select_units([u.corpus_id for u in units], None))
        held = [u for u in units if u.corpus_id not in scored]
        picked = [held[i] for i in subsample_indices(len(held), N_DEV_STORIES,
                                                     seed=DEV_SEED)]
        if self.draw != "a":
            first = {u.corpus_id for u in picked}
            picked = later_slice([u for u in held if u.corpus_id not in first],
                                 N_LATER_STORIES, self.draw)
        self.stats.update(n_queries=sum(len(u.queries) for u in picked),
                          n_stories=len(picked), subsample_seed=draw_seed(self.draw),
                          sampled_story_ids=[u.corpus_id for u in picked],
                          held_out=True)
        yield from picked


HELD_OUT = {"hotpotqa": HeldOutHotpotQA,
            "hotpotqa_pooled": HeldOutHotpotQAPooled,
            "narrativeqa": HeldOutNarrativeQA}


def _later_draw(draw: str) -> dict:
    """The three loaders of a later draw."""
    return {"hotpotqa": type(f"HotpotQADraw{draw}", (HeldOutHotpotQA,), {"draw": draw}),
            "hotpotqa_pooled": type(f"HotpotQAPooledDraw{draw}", (HeldOutHotpotQAPooled,),
                                    {"draw": draw}),
            "narrativeqa": type(f"NarrativeQADraw{draw}", (HeldOutNarrativeQA,),
                                {"draw": draw, "cell_units": N_LATER_STORIES[draw]})}


DRAWS = {"a": HELD_OUT, "b": _later_draw("b"), "c": _later_draw("c")}


class _NoGold:
    """A held-out HotpotQA loader with each unit's gold paragraphs taken out."""

    def iter_eval_units(self, *, split, max_units=None):
        for unit in super().iter_eval_units(split=split, max_units=max_units):
            gold = {doc for query in unit.queries for doc in gold_documents(query)}
            corpus = tuple(item for item in unit.corpus if str(item.parent_id) not in gold)
            if corpus:
                yield replace(unit, corpus_id=f"{unit.corpus_id}::no_gold", corpus=corpus)


class NoGoldHotpotQA(_NoGold, HeldOutHotpotQA):
    """Each question over its own paragraphs, its gold ones gone."""


class NoGoldHotpotQAPooled(_NoGold, HeldOutHotpotQAPooled):
    """Each shard with the gold paragraphs of all its questions gone."""


# NarrativeQA's gold is an answer, not a document: there is nothing to take out.
NO_GOLD = {"hotpotqa": NoGoldHotpotQA, "hotpotqa_pooled": NoGoldHotpotQAPooled}


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
        "map_at_10": mean(float(r["retrieval"]["map_at_k"]["10"]) for r in ranked),
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


def inspect_rows(rows, golds, gold_docs, n_samples: int = 15) -> tuple[dict, list]:
    """Where the answers win or lose: refusals, lengths, splits, and sample rows."""
    from src.eval.scorers.extractive import normalize_qasper_answer

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    def f1(r):
        return float(r["answer"]["value"])

    def by(label_of, subset):
        groups: dict = {}
        for r in subset:
            groups.setdefault(label_of(r), []).append(f1(r))
        return {k: [len(v), mean(v)] for k, v in sorted(groups.items())}

    answerable = [r for r in rows if r["answer"]["method"] != "unanswerable_rule"]
    abstained = [r for r in answerable if (r["answer"].get("metadata") or {}).get("abstained")]
    answered = [r for r in answerable if r not in abstained]
    words = sorted(len((r.get("predicted_answer") or "").split()) for r in answerable)
    yes_no = [r for r in answerable
              if any(normalize_qasper_answer(g) in ("yes", "no") for g in golds[str(r["query_id"])])]
    ranked = [r for r in answerable if gold_docs.get(str(r["query_id"]))]
    found = [r for r in ranked if recall_at(r["metadata"]["ours_doc_ranking"],
                                            gold_docs[str(r["query_id"])], 5) == 1.0]

    def any_low(r):
        return "some aspect low" if any(a["low_confidence"] for a in
                                        r["metadata"]["ours_aspects"]) else "none low"

    report = {
        "n_rows": len(answerable),
        "abstained_share": len(abstained) / len(answerable),
        "exact_refusal_share": mean(int((r.get("predicted_answer") or "").strip()
                                        == "No answer available.") for r in answerable),
        "f1": mean(f1(r) for r in answerable),
        "f1_answered": mean(f1(r) for r in answered),
        "answer_words_mean": mean(words),
        "answer_words_median": words[len(words) // 2] if words else None,
        "answer_words_p90": words[int(0.9 * (len(words) - 1))] if words else None,
        "gold_words_mean": mean(min(len(g.split()) for g in golds[str(r["query_id"])])
                                for r in answerable if golds[str(r["query_id"])]),
        "f1_by_type": by(lambda r: r.get("question_type") or "?", answerable),
        "f1_yes_no_gold": [len(yes_no), mean(f1(r) for r in yes_no)],
        "f1_by_low_confidence": by(any_low, answerable),
        "abstained_by_low_confidence": {
            k: mean(int(r in abstained) for r in answerable if any_low(r) == k)
            for k in ("some aspect low", "none low")},
        "f1_by_n_aspects": by(lambda r: str(len(r["metadata"]["ours_aspects"])), answerable),
        "map_at_10": mean(float(r["retrieval"]["map_at_k"]["10"]) for r in ranked),
        "all_gold_in_top5_share": len(found) / len(ranked) if ranked else None,
        "f1_when_all_gold_in_top5": mean(f1(r) for r in found),
        "zero_f1_when_all_gold_in_top5": mean(int(f1(r) == 0.0) for r in found),
    }
    step = max(1, len(rows) // n_samples)
    samples = [{"query_id": r["query_id"], "question": r["question_text"],
                "gold": list(golds.get(str(r["query_id"]), ())),
                "predicted": (r.get("predicted_answer") or "")[:400],
                "f1": f1(r),
                "aspects": [(a["name"], round(a["confidence"], 2))
                            for a in r["metadata"]["ours_aspects"]]}
               for r in rows[::step][:n_samples]]
    return report, samples


def compare_versions(rows_a, rows_b, golds) -> dict:
    """Version b against version a on the same questions: F1, EM and refusals, paired."""
    import numpy as np

    from scripts.export_comparison import NULL_METHOD
    from scripts.report_ours import SEED, bootstrap_ci, permutation_p
    from src.eval.scorers.extractive import normalize_qasper_answer

    a = {str(r["query_id"]): r for r in rows_a if r["answer"]["method"] != NULL_METHOD}
    b = {str(r["query_id"]): r for r in rows_b if r["answer"]["method"] != NULL_METHOD}
    keys = sorted(a.keys() & b.keys())

    def f1(r):
        return float(r["answer"]["value"])

    def em(r):
        pred = normalize_qasper_answer(r.get("predicted_answer") or "")
        return float(max((pred == normalize_qasper_answer(g) for g in golds[str(r["query_id"])]),
                         default=False))

    def refused(r):
        return float(bool((r["answer"].get("metadata") or {}).get("abstained")))

    out = {"n": len(keys),
           "n_answers_changed": sum(a[k].get("predicted_answer") != b[k].get("predicted_answer")
                                    for k in keys),
           "refusal_a": sum(refused(a[k]) for k in keys) / len(keys),
           "refusal_b": sum(refused(b[k]) for k in keys) / len(keys)}
    # The grid's paired sign-flip test and bootstrap, seeded the same way.
    rng = np.random.default_rng(SEED)
    for name, value in (("f1", f1), ("em", em)):
        d = np.asarray([value(b[k]) - value(a[k]) for k in keys], dtype=float)
        lo, hi = bootstrap_ci(d, rng)
        out[name] = {"a": sum(value(a[k]) for k in keys) / len(keys),
                     "b": sum(value(b[k]) for k in keys) / len(keys),
                     "delta": float(d.mean()), "ci": [lo, hi], "p": permutation_p(d, rng)}
    return out


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
    """Two held-out HotpotQA units, one with a short document, cold then warm: real loader and embedder, reader and cross-encoder faked."""
    from unittest import mock

    from ours.model import OursSystem, short_documents
    from scripts.replay_retrieval import rank_stats_from_ranking
    from src.eval.base import BenchmarkRunner
    from src.eval.hotpotqa import _TITLE_SPAN
    from src.retrievers.m4_raptor import M4_SUBSTRATE_NAMESPACE, RaptorSystem
    from tests.test_ours_smoke import FakeReader, fake_cross_encoder, fake_generate_batch

    loader = HeldOutHotpotQA()
    every = list(loader.iter_eval_units(split="validation"))
    judge = RaptorSystem(config=llama_config())
    # One unit for each tree: ours' own first, then M4's.
    units = [next(u for u in every if short_documents(u.corpus, judge)),
             next(u for u in every if not short_documents(u.corpus, judge))]

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
            warm_paths = [system.tree_for(u.corpus)[0].substrate_warm_path(u.corpus)
                          for u in units]
            m4 = RaptorSystem(config=llama_config())
            m4.topology_env_override = env
            m4_paths = [m4.substrate_warm_path(u.corpus) for u in units]
        passes.append(read_rows(out))

    cold, warm = passes
    # Step 10 adds the cut of q0's answer and the second asking of q1's refusal.
    assert [r["metadata"]["ours_n_llm_calls"] for r in cold] == [7, 5], cold
    assert [r["metadata"]["ours_parse_failed"] for r in cold] == [False, True]
    assert cold[1]["metadata"]["ours_aspects"][0]["budget"] == 13
    assert all(0 < r["n_retrieved"] <= 15 for r in cold)
    assert all(set(r["retrieved_unit_types"]) == {"chunk"} for r in cold)
    assert cold[1]["answer"]["metadata"]["abstained"] is True
    assert all(warm_paths), "ours does not find the trees it built"
    assert Path(warm_paths[0]).parent.name == settings.TREE_NAMESPACE, warm_paths
    assert Path(warm_paths[1]).parent.name == M4_SUBSTRATE_NAMESPACE, warm_paths
    assert m4_paths[0] is None, "ours' own tree landed where M4 looks"
    assert m4_paths[1] == warm_paths[1], "M4 does not find the tree of a unit it reads whole"
    assert all(r["metadata"]["ours_plain_answer"] for r in cold), "no plain answer on a row"
    reader_prompts = [u for s, u in reader.prompts if s == settings.READER_PROMPT]
    assert all("Chunk 1: [" in u for u in reader_prompts), "a chunk under no document's name"
    assert cold[0]["metadata"]["ours_n_short_documents"] > 0, cold[0]["metadata"]
    assert cold[1]["metadata"]["ours_n_short_documents"] == 0, cold[1]["metadata"]
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
    print(f"[smoke] short documents per unit: "
          f"{[r['metadata']['ours_n_short_documents'] for r in cold]}")
    print(f"[smoke] rows and cache under {root}")


# --- held-out development runs ---------------------------------------------


def backup_of(path: Path) -> Path:
    """Where a running cell's copy sits on Drive: a name nothing reads as a finished cell."""
    return path.with_name(path.name + ".bak")


def _copy_whole(source: Path, target: Path) -> None:
    """Copy, then rename, so a reader never sees half a file."""
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


@contextlib.contextmanager
def kept_on_drive(drive_path: Path, local_dir: Path | None, every: float = 300.0):
    """Yield where a cell should write: the VM's disk, copied to Drive every few minutes.

    Drive saves a file only when its writer closes it, so a runtime that dies mid-cell
    would leave nothing there. A new session starts from the copy; only a finished
    cell reaches the final name. Without a local_dir the cell writes to Drive itself."""
    if local_dir is None:
        yield drive_path
        return
    local = Path(local_dir) / drive_path.parent.name / drive_path.name
    local.parent.mkdir(parents=True, exist_ok=True)
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_of(drive_path)
    if not local.exists():
        for found in (drive_path, backup):
            if found.exists():
                shutil.copyfile(found, local)
                break
    stop = threading.Event()

    def copy_now() -> None:
        if local.exists():
            _copy_whole(local, backup)

    def keep_copying() -> None:
        while not stop.wait(every):
            copy_now()

    copier = threading.Thread(target=keep_copying, daemon=True)
    copier.start()
    try:
        yield local
    except BaseException:
        stop.set()
        copier.join()
        copy_now()
        raise
    stop.set()
    copier.join()
    _copy_whole(local, drive_path)
    if backup.exists():
        backup.unlink()


def refuse_other_code(path: Path, code: str) -> None:
    """A cell holds one model state: rows of another code are never resumed over."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        first = f.readline()
    try:
        found = json.loads(first)["metadata"]["ours_code"]
    except (ValueError, KeyError, TypeError):
        return
    if found != code:
        raise SystemExit(f"[dev] REFUSED: {path} holds rows of code {found}, and this is {code}. "
                         "Check out that commit, or give this version its own name.")


def dev_run(benchmark: str, version: str, lockfile: Path, resume: bool,
            no_gold: bool = False, draw: str = "a", local_dir: Path | None = None) -> dict:
    """One held-out cell; its numbers are appended to outputs/ours_dev/DEV_LOG.jsonl."""
    if no_gold and benchmark not in NO_GOLD:
        raise SystemExit(f"[dev] REFUSED: {benchmark} has no gold documents to take out")
    if no_gold and draw != "a":
        raise SystemExit("[dev] REFUSED: the no-gold measure is on the first draw")
    label = f"{version}_no_gold" if no_gold else (version if draw == "a" else f"{version}_{draw}")
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
    out = paths.output_dir() / "ours_dev" / label / f"{benchmark}_OURS_dev.jsonl"
    assert_bank_generator_consistent(out.parent, settings.READER)
    assert_bank_gpu_consistent(out.parent)

    loader = (NO_GOLD if no_gold else DRAWS[draw])[benchmark]()
    t0 = time.perf_counter()
    with kept_on_drive(out, local_dir) as rows_path:
        refuse_other_code(rows_path, CODE_FINGERPRINT)
        runner = BenchmarkRunner(output_path=rows_path, resume=resume)
        n_this = sum(1 for _ in runner.run(OursSystem(config=llama_config()), loader,
                                           split="validation"))
    golds, docs = golds_of(loader)
    numbers = ours_numbers(read_rows(out), golds, docs)
    line = {"version": label, "benchmark": benchmark,
            "git_commit": _git_commit_short(), "ours_code": CODE_FINGERPRINT,
            "rows_this_process": n_this, "elapsed_s": round(time.perf_counter() - t0, 1),
            "timestamp": time.strftime("%Y%m%d-%H%M%S"), **numbers}
    log = paths.output_dir() / "ours_dev" / "DEV_LOG.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line, indent=2))
    return line


# --- bench: retrieval alone, option against option -------------------------

_SWITCHES_OFF = {"DOCUMENT_HEADERS": False, "SOURCE_FOCUS": False, "QUESTION_VIEW": False,
                 "SECOND_EMBEDDER_MODE": "off"}
_V15 = dict(_SWITCHES_OFF, DOCUMENT_HEADERS=True)
_V16 = dict(_V15, SOURCE_FOCUS=True)
_V17 = dict(_V16, QUESTION_VIEW=True)
# Each option is the settings of one state of the model, so a bench of one process
# shows each step's effect on retrieval.
BENCH_OPTIONS = {"as_is": {}, "v14": _SWITCHES_OFF, "v15": _V15, "v16": _V16, "v17": _V17,
                 "v18_both": dict(_V17, SECOND_EMBEDDER_MODE="both"),
                 "v18_only": dict(_V17, SECOND_EMBEDDER_MODE="only")}
# v16 changes nothing where a corpus has no sources; MultiHop has no labels to read,
# so all it can show there is what v16 is for.
BENCH_DEFAULTS = {"hotpotqa": ["v14", "v15", "v17", "v18_both", "v18_only"],
                  "hotpotqa_pooled": ["v14", "v15", "v17", "v18_both", "v18_only"],
                  "narrativeqa": ["v14", "v17", "v18_both", "v18_only"],
                  "multihop_rag": ["v15", "v16"]}
MULTIHOP = "multihop_rag"


class _Replies:
    """The reader's replies to the steps before retrieval: paid for once, kept on disk for a later session."""

    def __init__(self, path: Path, every: float = 600.0) -> None:
        self.path = path
        self.every = every
        self.seen = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.saved_at = time.monotonic()

    def save(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.seen), encoding="utf-8")
        temporary.replace(self.path)
        self.saved_at = time.monotonic()

    def wrapping(self, generate):
        def generate_once(system_prompt, user_prompt, cfg=None):
            key = json.dumps([system_prompt, user_prompt, cfg.max_new_tokens if cfg is not None else None])
            if key not in self.seen:
                self.seen[key] = generate(system_prompt=system_prompt, user_prompt=user_prompt, cfg=cfg)
                if time.monotonic() - self.saved_at >= self.every:
                    self.save()
            return self.seen[key]
        return generate_once


def best_phrase_f1(golds, text: str) -> float:
    """The best token F1 any run of the text reaches against a gold, of about the gold's length."""
    from src.eval.scorers.extractive import normalize_qasper_answer as norm

    tokens = norm(text).split()
    at: dict = {}
    for i, token in enumerate(tokens):
        at.setdefault(token, []).append(i)
    best = 0.0
    for gold in golds:
        want = Counter(norm(gold).split())
        size = sum(want.values())
        starts = {max(0, p - k) for token in want for p in at.get(token, ()) for k in range(size + 1)}
        for start in starts:
            for length in {max(1, size - 1), size, size + 1}:
                run = tokens[start: start + length]
                common = sum((Counter(run) & want).values())
                if common:
                    precision, recall = common / len(run), common / size
                    best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _named_numbers(prepared, index) -> dict:
    """Label-free: whether the evidence holds what a question names. Reads no gold and no score."""
    leaf_of = {chunk.chunk_id: leaf for leaf, chunk in enumerate(index.chunks)}
    leaves = [leaf_of[r.chunk.chunk_id] for r in prepared.retrieved]
    names = prepared.extra["ours_named_sources"]
    dates = prepared.extra["ours_named_dates"]
    read = [index.source[leaf] for leaf in leaves]
    days = {index.date[leaf] for leaf in leaves}
    return {"names_a_source": float(bool(names)),
            "every_source_read": float(all(n in read for n in names)) if names else None,
            "from_named": sum(s in names for s in read) / len(read) if names and read else None,
            "dated_read": sum(d in days for d in dates) / len(dates) if dates else None,
            "placed": float(prepared.extra["ours_n_chunks_placed"])}


def bench_numbers(benchmark: str, loader, query, prepared, index) -> dict:
    """One question's retrieval numbers under one option."""
    numbers = {"chunks": float(len(prepared.retrieved)), "tokens": float(prepared.evidence_tokens)}
    if benchmark == MULTIHOP:
        return {**numbers, **_named_numbers(prepared, index)}
    if benchmark == "narrativeqa":
        from scripts.export_comparison import gold_texts

        best = best_phrase_f1(gold_texts(query), " ".join(r.chunk.text for r in prepared.retrieved))
        return {**numbers, "best_phrase": best, "holds_answer": float(best >= 0.8)}
    score = loader.score_retrieval(prepared.retrieved, query, scoring_ranking=prepared.scoring_ranking)
    gold = gold_documents(query)
    ranking = prepared.extra["ours_doc_ranking"]
    return {**numbers, "r1": recall_at(ranking, gold, 1), "r5": recall_at(ranking, gold, 5),
            "r10": recall_at(ranking, gold, 10), "map10": float(score.map_at_k.get(10, 0.0)),
            "context_recall": float(score.recall)}


def bench_summary(results: dict) -> dict:
    """Each option's means, and its paired difference from the option before, by the grid's bootstrap."""
    import numpy as np

    from scripts.report_ours import SEED, bootstrap_ci

    rng = np.random.default_rng(SEED)
    out, before = {}, None
    for name, rows in results.items():
        metrics = sorted({m for numbers in rows.values() for m in numbers})
        entry = {"n": len(rows), "means": {}, "against_previous": {}}
        for metric in metrics:
            values = [r[metric] for r in rows.values() if r.get(metric) is not None]
            entry["means"][metric] = sum(values) / len(values) if values else None
            if before is None:
                continue
            pairs = [(rows[q][metric], results[before][q][metric]) for q in rows
                     if rows[q].get(metric) is not None and results[before].get(q, {}).get(metric) is not None]
            if pairs:
                d = np.asarray([a - b for a, b in pairs], dtype=float)
                lo, hi = bootstrap_ci(d, rng)
                entry["against_previous"][metric] = {"delta": float(d.mean()), "ci": [lo, hi]}
        out[name] = entry
        before = name
    return out


def bench(benchmark: str, draw: str, options: list, lockfile: Path, limit: int | None = None,
          check_rows: str | None = None) -> dict:
    """Retrieval alone under each option, one process, the same questions; no reader's answer."""
    from unittest import mock

    from ours import model as om
    from src import models, paths
    from src.eval.multihop import MultiHopBenchmark
    from src.eval.runner import assert_environment_pinned

    if benchmark != MULTIHOP and not os.environ.get("THESIS_CACHE_DIR"):
        raise SystemExit("[bench] REFUSED: set THESIS_CACHE_DIR to a development cache; "
                         "the banked cache holds only scored-unit trees")
    if benchmark == MULTIHOP and draw != "a":
        raise SystemExit("[bench] REFUSED: MultiHop has one set of questions and no draws")
    assert_environment_pinned(lockfile)
    out_dir = paths.output_dir() / "ours_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = benchmark if benchmark == MULTIHOP else f"{benchmark}_{draw}"
    # HotpotQA's two sets share their questions, so they share the replies too.
    replies = _Replies(out_dir / ("replies_multihop.json" if benchmark == MULTIHOP else f"replies_{draw}.json"))
    loader = MultiHopBenchmark() if benchmark == MULTIHOP else DRAWS[draw][benchmark]()
    results: dict = {name: {} for name in options}
    packed: dict = {}
    embedded: dict = {}
    relevance: dict = {}
    real_embed, real_encoder = models.embed_texts, om.cross_encoder_scores

    def embed_once(texts, model_name=None):
        key = (model_name, tuple(texts))
        if key not in embedded:
            embedded[key] = real_embed(texts, model_name=model_name)
        return embedded[key]

    def encoder_once(query, passages):
        missing = [p for p in dict.fromkeys(passages) if (query, p) not in relevance]
        if missing:
            relevance.update({(query, p): s for p, s in zip(missing, real_encoder(query, missing))})
        return [relevance[(query, p)] for p in passages]

    with mock.patch.object(models, "generate", replies.wrapping(models.generate)), \
            mock.patch.object(models, "embed_texts", embed_once), \
            mock.patch.object(om, "cross_encoder_scores", encoder_once):
        system = om.OursSystem(config=llama_config())
        for unit in loader.iter_eval_units(split="validation"):
            items = list(unit.corpus)
            queries = list(unit.queries)
            if benchmark == MULTIHOP:
                # MultiHop's tree is banked, and nothing here may add to that cache.
                if system.tree_for(items)[0].substrate_warm_path(items) is None:
                    raise SystemExit("[bench] REFUSED: no banked MultiHop tree under this cache")
            if limit is not None and len(queries) > limit:
                queries = random.Random(LATER_SEED).sample(queries, limit)
            system.index_items(items)
            for name in options:
                overrides = BENCH_OPTIONS[name]
                with (mock.patch.multiple(settings, **overrides) if overrides else contextlib.nullcontext()):
                    system.build_index(items)
                    for query in queries:
                        prepared = system.prepare(query.question_text)
                        qid = str(query.query_id)
                        results[name][qid] = bench_numbers(benchmark, loader, query, prepared,
                                                           system._tree_index)
                        if name == options[0]:
                            packed[qid] = [r.chunk.chunk_id for r in prepared.packed]
            embedded.clear()
            relevance.clear()
    replies.save()
    summary = bench_summary(results)
    report = {"benchmark": benchmark, "draw": draw, "options": options, "summary": summary, "per_question": results}
    if check_rows:
        label = check_rows if draw == "a" else f"{check_rows}_{draw}"
        rows = read_rows(paths.output_dir() / "ours_dev" / label / f"{benchmark}_OURS_dev.jsonl")
        same = sum(packed.get(str(r["query_id"])) == r["metadata"]["packed_ids"] for r in rows)
        report["same_evidence_as_rows"] = {"version": check_rows, "same": same, "rows": len(rows)}
        print(f"[bench] {options[0]} reads the evidence of {check_rows}'s rows on {same} of {len(rows)}")
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    for name, entry in summary.items():
        print(f"[bench] {stem} {name} n={entry['n']}: " + ", ".join(
            f"{m} {v:.4f}" for m, v in entry["means"].items() if v is not None))
        for metric, d in entry["against_previous"].items():
            print(f"[bench]     {metric}: {d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    print(f"[bench] saved {path}")
    return report


# --- decomposition replay -------------------------------------------------

REPLAY_CAP = 512


def reply_failure(raw: str, question: str) -> str | None:
    """Why parse_aspects refuses a decomposition reply; None when it reads it."""
    from ours.model import close_brackets, parse_aspects

    if parse_aspects(raw, question) is not None:
        return None
    # The same steps as parse_aspects, stopping at the first that fails.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    if "{" not in text:
        return "no_json"
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1],
                      close_brackets(text)):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            return "not_object"
        if obj.get("type") not in ("simple", "multi_aspect"):
            return "bad_type"
        return "bad_aspects"
    return "bad_json"


def replay_row(query, generate_ids, decode, stop, cap: int, banked: dict) -> dict:
    """One question's decomposition as the scored cell saw it, and with room to finish."""
    from ours.model import decompose, parse_aspects

    seen = {}

    def ask(system_prompt, user_prompt, max_tokens):
        # Greedy decoding: the first max_tokens of a longer reply are the
        # scored reply, so one generation gives both.
        seen["ids"] = generate_ids(system_prompt, user_prompt, cap)
        seen["scored_cap"] = max_tokens
        return decode(seen["ids"][:max_tokens])

    aspects, failed = decompose(query.question_text, ask)
    ids, scored_cap = seen["ids"], seen["scored_cap"]
    n_tokens = len(ids) - (1 if ids and ids[-1] in stop else 0)
    reply, full = decode(ids[:scored_cap]), decode(ids)
    return {"query_id": query.query_id, "question_type": query.question_type,
            "question": query.question_text, "n_tokens": n_tokens,
            "ended": n_tokens < len(ids), "cut": n_tokens > scored_cap,
            "parse_failed": failed,
            "failure": reply_failure(reply, query.question_text) if failed else None,
            "parses_with_room": parse_aspects(full, query.question_text) is not None,
            "n_aspects": len(aspects), "reply": reply,
            "reply_full": None if full == reply else full,
            "banked_parse_failed": banked.get(query.query_id)}


def replay_summary(rows) -> dict:
    """Counts over replayed decompositions: failure causes and the reproduction check."""
    failed = [r for r in rows if r["parse_failed"]]
    checked = [r for r in rows if r["banked_parse_failed"] is not None]
    tokens = sorted(r["n_tokens"] for r in rows)
    by_type: dict = {}
    for r in rows:
        n, k = by_type.get(r["question_type"], (0, 0))
        by_type[r["question_type"]] = (n + 1, k + int(r["parse_failed"]))
    return {
        "n": len(rows), "n_parse_failed": len(failed),
        "failure": dict(Counter(r["failure"] for r in failed).most_common()),
        "failed_and_cut": sum(r["cut"] for r in failed),
        "failed_but_parse_with_room": sum(r["parses_with_room"] for r in failed),
        "never_ended": sum(not r["ended"] for r in rows),
        "tokens": {q: tokens[min(len(tokens) - 1, int(f * len(tokens)))]
                   for q, f in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))}
                  if tokens else {},
        "failure_rate_by_type": {t: round(k / n, 4) for t, (n, k) in sorted(by_type.items())},
        "reproduction": {"checked": len(checked), "mismatched": sum(
            r["parse_failed"] != r["banked_parse_failed"] for r in checked)},
    }


def reader_ids():
    """(generate_ids, decode, stop tokens) for the shared reader, as the runner calls it."""
    import torch

    from src import models

    cfg = llama_config().generation
    tokenizer, model = models.load_generator(cfg.model)

    def generate_ids(system_prompt, user_prompt, max_new_tokens):
        # src/models.py::_generate_local, returning the generated ids.
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=cfg.temperature > 0,
                                 temperature=max(cfg.temperature, 1e-5),
                                 top_p=cfg.top_p, pad_token_id=tokenizer.eos_token_id)
        return out[0, inputs["input_ids"].shape[1]:].tolist()

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    eos = model.generation_config.eos_token_id
    stop = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tokenizer.eos_token_id}
    return generate_ids, decode, stop


def replay(benchmark: str, lockfile: Path, banked: Path | None, out: Path | None,
           resume: bool, cap: int = REPLAY_CAP, reader=reader_ids) -> dict:
    """Replay every decomposition of a scored benchmark; rows and a summary beside them."""
    from ours.model import CODE_FINGERPRINT
    from src import paths
    from src.eval.runner import BENCHMARK_REGISTRY, _git_commit_short, assert_environment_pinned

    assert_environment_pinned(lockfile)
    out = out or (paths.output_dir() / "ours_diag"
                  / f"{benchmark}_decompose_{CODE_FINGERPRINT}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not resume:
        raise SystemExit(f"[replay] REFUSED: {out} exists; pass --resume to continue it")
    if out.exists():
        # Cut a half-written last line before appending.
        data = out.read_bytes()
        if data and not data.endswith(b"\n"):
            with out.open("r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
    done = {r["query_id"] for r in read_rows(out)} if out.exists() else set()
    flags = ({str(r["query_id"]): bool(r["metadata"]["ours_parse_failed"])
              for r in read_rows(banked)} if banked else {})
    generate_ids, decode, stop = reader()
    for unit in BENCHMARK_REGISTRY[benchmark]().iter_eval_units(split="validation"):
        for q in unit.queries:
            if q.query_id in done:
                continue
            row = replay_row(q, generate_ids, decode, stop, cap, flags)
            # Open and close per row: a mounted Drive uploads a file on close.
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"benchmark": benchmark, "ours_code": CODE_FINGERPRINT, "cap": cap,
               "scored_cap": settings.DECOMPOSE_MAX_TOKENS,
               "git_commit": _git_commit_short(), **replay_summary(read_rows(out))}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


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
    r.add_argument("--no-gold", action="store_true",
                   help="HotpotQA only: take each unit's gold paragraphs out; rows go to <version>_no_gold.")
    r.add_argument("--draw", default="a", choices=sorted(DRAWS),
                   help="Which held-out questions; rows of b or c go to <version>_<draw>.")
    r.add_argument("--local-dir", type=Path, default=None,
                   help="Write rows here and copy them to Drive every 5 minutes (for a cell that may outlive a session).")
    i = sub.add_parser("inspect", help="Answer-side breakdown of a development cell.")
    i.add_argument("--benchmark", required=True, choices=sorted(HELD_OUT))
    i.add_argument("--version", required=True)
    v = sub.add_parser("compare", help="Two development versions, paired on the same questions.")
    v.add_argument("--benchmark", required=True, choices=sorted(HELD_OUT))
    v.add_argument("--a", required=True, help="The earlier version.")
    v.add_argument("--b", required=True, help="The later version.")
    v.add_argument("--draw", default="a", choices=sorted(DRAWS))
    b = sub.add_parser("bench", help="Retrieval alone under each option, on the same questions.")
    b.add_argument("--benchmark", required=True, choices=sorted(BENCH_DEFAULTS))
    b.add_argument("--draw", default="a", choices=sorted(DRAWS))
    b.add_argument("--options", default=None, help="Comma-separated, in order. Default by benchmark.")
    b.add_argument("--limit", type=int, default=None, help="A seeded sample of each unit's questions.")
    b.add_argument("--check-rows", default=None,
                   help="A version whose rows the first option should reproduce, evidence for evidence.")
    b.add_argument("--lockfile", type=Path, default=Path("requirements.lock"))
    p = sub.add_parser("replay", help="Decomposition replies over a scored benchmark's questions.")
    p.add_argument("--benchmark", required=True, choices=SCORED_BENCHMARKS)
    p.add_argument("--banked", type=Path, default=None,
                   help="The scored ours cell, to check the replay reproduces its parse flags.")
    p.add_argument("--out", type=Path, default=None,
                   help="Default: OUTPUT_DIR/ours_diag/<benchmark>_decompose_<code>.jsonl")
    p.add_argument("--cap", type=int, default=REPLAY_CAP)
    p.add_argument("--lockfile", type=Path, default=Path("requirements.lock"))
    p.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.command == "bench":
        chosen = args.options.split(",") if args.options else BENCH_DEFAULTS[args.benchmark]
        unknown = [name for name in chosen if name not in BENCH_OPTIONS]
        if unknown:
            raise SystemExit(f"[bench] REFUSED: unknown options {unknown}; there are {sorted(BENCH_OPTIONS)}")
        bench(args.benchmark, args.draw, chosen, args.lockfile, limit=args.limit, check_rows=args.check_rows)
        return
    if args.command == "replay":
        replay(args.benchmark, args.lockfile, args.banked, args.out, args.resume, args.cap)
        return
    if args.command == "compare":
        from src import paths

        def rows_of(version):
            label = version if args.draw == "a" else f"{version}_{args.draw}"
            return read_rows(paths.output_dir() / "ours_dev" / label
                             / f"{args.benchmark}_OURS_dev.jsonl")
        golds, _ = golds_of(DRAWS[args.draw][args.benchmark]())
        print(json.dumps(compare_versions(rows_of(args.a), rows_of(args.b), golds), indent=2))
        return
    if args.command == "census":
        census(args.p11)
    elif args.command == "smoke":
        smoke()
    elif args.command == "inspect":
        inspect(args.benchmark, args.version)
    else:
        dev_run(args.benchmark, args.version, args.lockfile, args.resume, no_gold=args.no_gold,
                draw=args.draw, local_dir=args.local_dir)


def inspect(benchmark: str, version: str) -> None:
    """Print the answer-side breakdown and sample rows of one development cell."""
    from src import paths

    rows = read_rows(paths.output_dir() / "ours_dev" / version
                     / f"{benchmark}_OURS_dev.jsonl")
    golds, docs = golds_of(HELD_OUT[benchmark]())
    report, samples = inspect_rows(rows, golds, docs)
    print(json.dumps(report, indent=2))
    for s in samples:
        print(f"\n--- {s['query_id']}  f1={s['f1']:.3f}  aspects={s['aspects']}")
        print(f"Q: {s['question']}")
        print(f"gold: {s['gold']}")
        print(f"answer: {s['predicted']}")


if __name__ == "__main__":
    main()
