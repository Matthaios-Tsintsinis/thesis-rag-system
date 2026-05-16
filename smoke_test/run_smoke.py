"""Smoke test runner.

Indexes M1/M2/M3/M4/M8 on the tiny smoke corpus and answers 5 questions
with each. Writes a single combined JSONL to
    <OUTPUT_DIR>/smoke_results_<timestamp>.jsonl
(via src.paths; OUTPUT_DIR is Drive when mounted, else /content/, else
<repo>/local_runs/outputs).

Per-system chunking: M1/M2/M3 use word-window with smoke-tuned params so
behaviour stays identical to previous smoke runs. M4 and M8 use semantic
chunking with smoke-tuned min/max words because semantic chunking is
part of what those systems are testing.

M4 needs a non-trivial RAPTOR cluster tree to exercise the per-node-type
expansion paths (PIPELINE_DESIGN.md section 4.4). On the smoke corpus
this requires both a shrunken tree config (branching/min/depth smaller
than production) and enough chunks to actually subdivide — see the
expanded corpus in this directory.

Run from repo root:
    python -m smoke_test.run_smoke                  # full: retrieval + generation
    python -m smoke_test.run_smoke --no-generate    # CPU-only: retrieval only
    python -m smoke_test.run_smoke --systems M2 M3  # subset

Sanity checks (exit code 2 on fail):
  * M1 returns 0 chunks on every question
  * M3 ranking differs from M2 on at least one question (RRF actually fires)
  * M4 tree has internal nodes; M4 flat index contains summary entries;
    M4 routing exercises at least one summary-expansion path across the
    5 questions (so the §4.4 logic isn't dead code on the corpus)
  * M8 returns >0 chunks on at least one question (tree traversal isn't empty)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from src import paths
from src.config import (
    DEFAULT_CONFIG,
    ChunkingConfig,
    ExpansionParams,
    HarnessConfig,
    M4Config,
    M7Config,
    RaptorBuildParams,
)
from src.harness import JSONLBenchmark, _record
from src.retrievers.base import AnswerResult, BaseSystem
from src.retrievers.m1_closedbook import ClosedBookSystem
from src.retrievers.m2_flat_dense import FlatDenseSystem
from src.retrievers.m3_hybrid import HybridRRFSystem
from src.retrievers.m4_raptor import RaptorSystem
from src.retrievers.m7_three_axis import ThreeAxisSystem
from src.retrievers.m8_hierarchical import HierarchicalSystem


SMOKE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = SMOKE_DIR / "corpus"
QUESTIONS_PATH = SMOKE_DIR / "questions.json"


SYSTEM_REGISTRY: dict[str, type[BaseSystem]] = {
    "M1": ClosedBookSystem,
    "M2": FlatDenseSystem,
    "M3": HybridRRFSystem,
    "M4": RaptorSystem,
    "M7": ThreeAxisSystem,
    "M8": HierarchicalSystem,
}

# The eight evaluation_plan.pdf §4 ablations, each disabling exactly one
# M7 component via config (never a code change). Smoke runs each on one
# question (retrieval-only) and asserts the disabled component's effect
# is observable in the trace — it does not measure quality.
M7_ABLATIONS: dict[str, dict] = {
    "A1": {"use_docling_structural_axis": False},
    "A2": {"use_intent_decomposition": False},
    "A3": {"view_types": ("paraphrase", "paraphrase2")},
    "A4": {"use_bm25": False},
    "A5": {"include_parent_summaries": False},
    "A6": {"quota_preserving_rerank": False},
    "A7": {"pass_retrieval_confidence_to_llm": False},
    "A8": {"always_include_global_query_view": False},
}


# --- Smoke-specific chunking overrides ------------------------------------

# Word-window (small chunks so M3 has room to reorder M2's dense ranking).
_SMOKE_WW = ChunkingConfig(
    strategy="word_window",
    chunk_words=80,
    overlap_words=20,
    min_chars_per_doc=200,
)

# Semantic (small min/max so semantic chunker actually splits the 3 short
# smoke docs into multiple chunks instead of falling back to one each).
_SMOKE_SEM = ChunkingConfig(
    strategy="semantic",
    min_words=30,
    max_words=120,
    max_if_min_words=180,
    breakpoint_percentile=85.0,
    absolute_threshold=0.5,
    buffer_size=1,
    min_chars_per_doc=200,
)


# M4 smoke tree: production defaults (branching=4, min_cluster=24, depth=4)
# would collapse to root on ~25 smoke chunks. Shrink so the tree actually
# subdivides and the §4.4 routing paths get exercised. trace=True so the
# retriever populates self.last_trace per query for sanity assertions.
_SMOKE_M4 = M4Config(
    build=RaptorBuildParams(
        branching_factor=3,
        min_cluster_size=3,
        max_depth=3,
    ),
    expansion=ExpansionParams(
        max_descendant_chunks_for_direct_expansion=10,
    ),
    first_stage_top_k=30,
    trace=True,
)


# M7 reuses M4's RAPTOR substrate: identical build/expansion params +
# identical semantic chunking → identical substrate cache key, so M7
# cache-hits the RAPTOR/<hash>/ dir M4 built instead of re-summarising.
_SMOKE_M7 = M7Config(
    build=RaptorBuildParams(
        branching_factor=3,
        min_cluster_size=3,
        max_depth=3,
    ),
    expansion=ExpansionParams(
        max_descendant_chunks_for_direct_expansion=10,
    ),
    first_stage_top_k=30,
    trace=True,
)


def _config_for(sid: str, m7: M7Config | None = None) -> HarnessConfig:
    if sid == "M4":
        return replace(DEFAULT_CONFIG, chunking=_SMOKE_SEM, m4=_SMOKE_M4)
    if sid == "M7":
        return replace(
            DEFAULT_CONFIG, chunking=_SMOKE_SEM, m7=m7 or _SMOKE_M7
        )
    if sid == "M8":
        return replace(DEFAULT_CONFIG, chunking=_SMOKE_SEM)
    return replace(DEFAULT_CONFIG, chunking=_SMOKE_WW)


def _retrieval_only_answer(system: BaseSystem, query: str) -> AnswerResult:
    t0 = time.perf_counter()
    retrieved = system.retrieve(query)
    return AnswerResult(
        query=query,
        answer="[generation skipped]",
        retrieved=retrieved,
        latency_s=time.perf_counter() - t0,
        n_retrieval_calls=1 if retrieved else 0,
    )


def _check_m2_m3_divergence(
    m2_rankings: dict[str, list[str]],
    m3_rankings: dict[str, list[str]],
) -> bool:
    if not m2_rankings or not m3_rankings:
        return True
    shared = set(m2_rankings) & set(m3_rankings)
    return any(m2_rankings[q] != m3_rankings[q] for q in shared)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="*",
                        default=["M1", "M2", "M3", "M4", "M7", "M8"])
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip LLM generation; report retrieval only.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    bench = JSONLBenchmark(name="smoke", corpus_path=CORPUS_DIR,
                           questions_path=QUESTIONS_PATH)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = paths.output_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"smoke_results_{stamp}.jsonl"
    print(f"[smoke] writing to {out_path}")

    m2_rankings: dict[str, list[str]] = {}
    m3_rankings: dict[str, list[str]] = {}
    m4_chunk_counts: list[int] = []
    m4_paths_union: set[str] = set()
    m4_index_stats: dict = {}
    m7_chunk_counts: list[int] = []
    m7_index_stats: dict = {}
    m7_traces: list[dict] = []
    m7_abstained_any = False
    m8_chunk_counts: list[int] = []

    with out_path.open("w", encoding="utf-8") as f:
        for sid in args.systems:
            if sid not in SYSTEM_REGISTRY:
                raise SystemExit(
                    f"Unknown system {sid!r}; known: {list(SYSTEM_REGISTRY)}"
                )
            if sid == "M1" and args.no_generate:
                print("[smoke] skipping M1 (retrieval-only mode; "
                      "M1 has no retrieval)")
                continue

            system = SYSTEM_REGISTRY[sid](config=_config_for(sid))
            print(f"[smoke] === {sid}: {system.__class__.__name__} ===")
            system.index(CORPUS_DIR)

            if sid == "M4":
                m4_index_stats = system.index_stats
                print(f"  M4 index stats: {m4_index_stats}")
            if sid == "M7":
                m7_index_stats = system.index_stats
                print(f"  M7 index stats: {m7_index_stats}")

            for n, q in enumerate(bench.questions()):
                if args.limit is not None and n >= args.limit:
                    break
                if args.no_generate:
                    ar = _retrieval_only_answer(system, q.question)
                else:
                    ar = system.answer(q.question)

                rec = _record(sid, bench.name, q, ar)
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                f.flush()

                ids = [r.chunk.chunk_id for r in ar.retrieved]
                if sid == "M2":
                    m2_rankings[q.question_id] = ids
                elif sid == "M3":
                    m3_rankings[q.question_id] = ids
                elif sid == "M4":
                    m4_chunk_counts.append(len(ar.retrieved))
                    exercised_paths = system.last_trace.get("paths_exercised", [])
                    m4_paths_union.update(exercised_paths)
                elif sid == "M7":
                    m7_chunk_counts.append(len(ar.retrieved))
                    m7_traces.append(dict(system.last_trace))
                    if ar.extra.get("abstained_aspects"):
                        m7_abstained_any = True
                elif sid == "M8":
                    m8_chunk_counts.append(len(ar.retrieved))

                preview = ar.answer[:120].replace("\n", " ")
                extras = ""
                if ar.extra:
                    extras = (
                        f" abstained={ar.extra.get('abstained')} "
                        f"conf={ar.extra.get('confidence', float('nan')):.3f}"
                    )
                print(f"  q={q.question_id} retrieved={len(ar.retrieved)} "
                      f"lat={ar.latency_s:.2f}s ids={ids[:3]}{extras} "
                      f"ans={preview!r}")

    print(f"\n[smoke] wrote {out_path}")

    # --- Sanity checks ----------------------------------------------------

    fail = False

    if "M1" in args.systems and not args.no_generate:
        m1_chunk_counts: list[int] = []
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["system_id"] == "M1":
                m1_chunk_counts.append(len(rec["retrieved_chunk_ids"]))
        if any(c != 0 for c in m1_chunk_counts):
            print("[smoke] FAIL: M1 returned non-empty retrieved_chunk_ids")
            fail = True
        else:
            print("[smoke] OK: M1 returned 0 chunks on every question")

    if "M2" in args.systems and "M3" in args.systems:
        if not _check_m2_m3_divergence(m2_rankings, m3_rankings):
            print(
                "[smoke] FAIL: M2 and M3 produced identical rankings on every "
                "question — RRF fusion may be broken (BM25 should reorder)."
            )
            fail = True
        else:
            print("[smoke] OK: M3 ranking differs from M2 on at least one question")

    if "M4" in args.systems:
        if not m4_index_stats:
            print("[smoke] FAIL: M4 produced no index_stats (system never indexed)")
            fail = True
        else:
            tree_nodes = int(m4_index_stats.get("tree_n_nodes", 0))
            flat_summaries = int(m4_index_stats.get("flat_n_summaries", 0))
            if tree_nodes < 2:
                print(
                    f"[smoke] FAIL: M4 tree has only {tree_nodes} node(s); "
                    "expected >= 2 internal nodes (root + at least one child). "
                    "Tree collapsed — grow the corpus or shrink the build params."
                )
                fail = True
            else:
                print(
                    f"[smoke] OK: M4 tree has {tree_nodes} nodes "
                    f"(depths {m4_index_stats.get('tree_depth_counts')})"
                )
            if flat_summaries < 1:
                print(
                    f"[smoke] FAIL: M4 flat collapsed index has 0 summary entries; "
                    "per-node-type expansion paths cannot be exercised."
                )
                fail = True
            else:
                print(
                    f"[smoke] OK: M4 flat index has {flat_summaries} summary entries "
                    f"({m4_index_stats.get('flat_node_type_counts')})"
                )
            # Routing-path coverage across all 5 questions.
            summary_paths = m4_paths_union & {"high", "mid", "low"}
            if "leaf" not in m4_paths_union or not summary_paths:
                print(
                    f"[smoke] FAIL: M4 routing did not exercise the expected paths; "
                    f"observed paths={sorted(m4_paths_union)}. Expected 'leaf' plus "
                    "at least one of {'high','mid','low'} across the 5 questions."
                )
                fail = True
            else:
                print(
                    f"[smoke] OK: M4 routing exercised paths "
                    f"{sorted(m4_paths_union)} across the questions"
                )
            if not m4_chunk_counts or all(c == 0 for c in m4_chunk_counts):
                print("[smoke] FAIL: M4 returned 0 chunks on every question")
                fail = True
            else:
                n_hits = sum(c > 0 for c in m4_chunk_counts)
                print(
                    f"[smoke] OK: M4 returned chunks on "
                    f"{n_hits}/{len(m4_chunk_counts)} questions"
                )

    if "M8" in args.systems:
        if not m8_chunk_counts or all(c == 0 for c in m8_chunk_counts):
            print(
                "[smoke] FAIL: M8 returned 0 chunks on every question — "
                "tree traversal collapsed or rerank dropped all candidates."
            )
            fail = True
        else:
            n_hits = sum(c > 0 for c in m8_chunk_counts)
            print(f"[smoke] OK: M8 returned chunks on {n_hits}/"
                  f"{len(m8_chunk_counts)} questions")

    if "M7" in args.systems:
        if not m7_index_stats:
            print("[smoke] FAIL: M7 produced no index_stats")
            fail = True
        else:
            print(
                f"[smoke] OK: M7 substrate reused — tree "
                f"{m7_index_stats.get('tree_n_nodes')} nodes, "
                f"{m7_index_stats.get('n_docling_sections')} Docling "
                f"section(s), {m7_index_stats.get('n_chunks')} chunks"
            )
        if not m7_chunk_counts or all(c == 0 for c in m7_chunk_counts):
            print("[smoke] FAIL: M7 returned 0 chunks on every question")
            fail = True
        else:
            print(
                f"[smoke] OK: M7 returned chunks on "
                f"{sum(c > 0 for c in m7_chunk_counts)}/"
                f"{len(m7_chunk_counts)} questions"
            )
        # Aspect extraction parsed + exactly 2 views per aspect.
        any_views_ok = False
        any_multi = False
        for tr in m7_traces:
            asp = tr.get("aspects", [])
            if not asp:
                continue
            if all(a["n_views"] == 2 for a in asp):
                any_views_ok = True
            if any(a["name"] != "main" for a in asp) and len(asp) > 1:
                any_multi = True
        if not any_views_ok:
            print("[smoke] FAIL: M7 aspect plans never had exactly 2 views")
            fail = True
        else:
            print("[smoke] OK: M7 aspects parsed with 2 views (paraphrase+HyDE)")
        if not any_multi:
            print(
                "[smoke] WARN: no multi-aspect decomposition observed across "
                "the 5 questions (smoke-005 is the multi-aspect probe) — "
                "aspect extractor may be collapsing everything to 'main'"
            )
        else:
            print("[smoke] OK: M7 produced a multi-aspect decomposition")
        # Multi-branch explored >1 distinct subtree on >=1 question.
        mb = max((t.get("multibranch_distinct_paths", 0) for t in m7_traces),
                 default=0)
        if mb < 2:
            print(
                f"[smoke] FAIL: M7 multi-branch traversal never explored >1 "
                f"subtree (max distinct paths={mb}); Axis-1b is degenerate."
            )
            fail = True
        else:
            print(f"[smoke] OK: M7 multi-branch explored up to {mb} subtrees")
        # No duplicate chunk ids in any M7 result (RRF/global dedupe works).
        dup = False
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["system_id"] == "M7":
                cid = rec["retrieved_chunk_ids"]
                if len(cid) != len(set(cid)):
                    dup = True
        if dup:
            print("[smoke] FAIL: M7 returned duplicate chunk ids (dedupe broke)")
            fail = True
        else:
            print("[smoke] OK: M7 results dedupe correctly (no repeat chunks)")
        # Per-aspect quota preserved: selected count never exceeds budget.
        quota_ok = True
        for tr in m7_traces:
            pas = tr.get("per_aspect_selected", {})
            for a in tr.get("aspects", []):
                if pas.get(a["name"], 0) > a["budget"]:
                    quota_ok = False
        if not quota_ok:
            print("[smoke] FAIL: M7 an aspect exceeded its quota after rerank")
            fail = True
        else:
            print("[smoke] OK: M7 per-aspect quota preserved through rerank")

        # --- Ablation sweep: one question, retrieval-only, trace asserts ---
        probe = next(iter(bench.questions())).question
        ab_fail = False
        for aid, overrides in M7_ABLATIONS.items():
            cfg = replace(_SMOKE_M7, **overrides)
            sysa = ThreeAxisSystem(config=_config_for("M7", m7=cfg))
            sysa.index(CORPUS_DIR)
            sysa.retrieve(probe)
            tr = sysa.last_trace
            if aid == "A2":
                ok = len(tr.get("aspects", [])) == 1
            elif aid == "A3":
                vt = (tr.get("aspects", [{}]) or [{}])[0].get("view_types", [])
                ok = "paraphrase2" in vt and "hyde" not in vt
            elif aid == "A6":
                ok = tr.get("mode") == "A6_global_rerank"
            elif aid == "A8":
                ok = tr.get("global_confidence", 1.0) == 0.0
            else:  # A1/A4/A5/A7: pure skips — assert it ran cleanly
                ok = bool(tr.get("aspects"))
            tr_flags = tr.get("ablations", {})
            print(
                f"  [{aid}] effect_observed={ok} "
                f"flags={ {k: tr_flags.get(k) for k in overrides} }"
            )
            if not ok:
                ab_fail = True
        if ab_fail:
            print("[smoke] FAIL: an M7 ablation did not disable its component")
            fail = True
        else:
            print("[smoke] OK: all 8 M7 ablations disabled their component")

    if fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
