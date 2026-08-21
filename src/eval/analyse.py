"""Read-only aggregator for ScoredQuery JSONL files.

Parses one or more JSONL files produced by `src.eval.runner` and prints
per-system aggregates: query count, chunk-count distribution, retrieval
F1 / recall / precision, answer score, abstention rate, plus a per-
(system, question_type) slice. Optionally dumps the aggregates to JSON.

USAGE:
    python -m src.eval.analyse <OUTPUT_DIR>/eval/qasper_*_validation_*.jsonl
    python -m src.eval.analyse --inputs file1.jsonl file2.jsonl --output aggregates.json
    python -m src.eval.analyse --inputs <dir>/*.jsonl --by-type

This script reads ONLY what the current ScoredQuery captures: per-query
chunk count (n_retrieved), retrieval/answer scores, predicted_answer
text, question_type. It does NOT have token volume or retrieved unit
types — those land in a follow-up instrumentation commit if the chunk-
count analysis is inconclusive (M4/M7 retrieve summary nodes mixed
with raw chunks, so count is a noisy proxy for token volume for those
two specifically).

ABSTENTION DETECTION uses the same `is_abstention` predicate the eval
scorers use, so the abstention-rate number here matches what the
scorers see at scoring time.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..raptor_paper import PAPER_NON_LEAF_SHARE_BAND
from .scorers import is_abstention


def _iter_records(paths: Iterable[Path]) -> list[dict]:
    """Load rows, DEDUPLICATED by query_id, last row wins.

    Without this, a duplicated query_id is counted twice and silently
    reweights every mean. `scripts/significance_diagnostic.py` has always
    deduplicated (it keys a dict by query_id), so the two tools could
    report different n and different means from the SAME file - measured
    at 0.6667 against 0.5000 on a three-row fixture holding one duplicate
    (docs/EVAL_AUDIT.md ISSUE-3). It is not acceptable for two tools in
    one repo to disagree about what a file says.

    Duplicates are NAMED, not just counted: an unexplained duplicate is a
    symptom, and the id is what makes it diagnosable.

    REFUSES a mix of systems or benchmarks in one invocation. Pooling two
    systems into one mean is not a report anyone wants, and it is exactly
    what a careless glob produces.
    """
    rows: dict[str, dict] = {}
    duplicates: list[str] = []
    n_lines = 0
    systems: set[str] = set()
    benchmarks: set[str] = set()

    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[analyse] WARN: bad JSONL line in {p}: {e}")
                    continue
                n_lines += 1
                systems.add(str(r.get("system_id", "?")))
                benchmarks.add(str(r.get("benchmark", "?")))
                qid = r.get("query_id")
                if qid is None:
                    print(f"[analyse] WARN: row without query_id in {p}")
                    continue
                qid = str(qid)
                if qid in rows:
                    duplicates.append(qid)
                rows[qid] = r

    if len(systems) > 1:
        raise ValueError(
            f"[analyse] REFUSING to pool {len(systems)} systems in one "
            f"invocation: {sorted(systems)}. One system per run."
        )
    if len(benchmarks) > 1:
        raise ValueError(
            f"[analyse] REFUSING to pool {len(benchmarks)} benchmarks in one "
            f"invocation: {sorted(benchmarks)}. One benchmark per run."
        )
    if duplicates:
        uniq = sorted(set(duplicates))
        print(
            f"[analyse] WARN: {len(uniq)} duplicated query_id(s) collapsed "
            f"(last row wins), {n_lines} lines -> {len(rows)} rows: "
            f"{uniq[:10]}{' ...' if len(uniq) > 10 else ''}"
        )
    return list(rows.values())


def _check_expected_n(paths: list[Path], n_rows: int) -> None:
    """Abort when the row count disagrees with the loader's own count.

    The expected count is read from the sibling `.summary.json`, where the
    runner wrote what the BENCHMARK LOADER emitted at run time. It is
    never a literal in this file: P7 re-draws NarrativeQA's sample, so any
    constant here would be wrong by construction.

    Skipped when the run was capped (`--max-units` / `--max-queries`), and
    skipped with a warning when no summary is present - a partial mean is
    worth reporting when it is ASKED for, and worth refusing when it is
    not.
    """
    for p in paths:
        summary_path = p.with_suffix("").with_suffix(".summary.json")
        if not summary_path.exists():
            summary_path = Path(str(p)[: -len(".jsonl")] + ".summary.json")
        if not summary_path.exists():
            print(f"[analyse] WARN: no summary beside {p.name}; cannot check "
                  "the expected query count")
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[analyse] WARN: unreadable summary {summary_path.name}: {e}")
            continue
        if summary.get("max_units") is not None or summary.get(
            "max_queries"
        ) is not None:
            continue  # deliberately capped run
        expected = summary.get("expected_n_queries")
        if expected is None:
            print(f"[analyse] WARN: {summary_path.name} predates "
                  "expected_n_queries; cannot check the count")
            continue
        if int(expected) != n_rows:
            raise ValueError(
                f"[analyse] REFUSING to report a partial mean: {p.name} has "
                f"{n_rows} distinct rows against the loader's "
                f"{int(expected)} queries. A cell missing rows is an "
                "incomplete run, not a smaller sample."
            )


def _expand_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for spec in inputs:
        if any(ch in spec for ch in "*?["):
            out.extend(Path(p) for p in glob.glob(spec))
        else:
            out.append(Path(spec))
    return [p for p in out if p.is_file()]


# M4's retrieval-unit TIER LABELS ARE INVERTED across the paper-fidelity
# rebuild, so pooling the two eras averages opposite meanings:
#
#   old top-down  (raptor.py:152)  depth 0-1  -> summary_high  (BROADEST)
#   new bottom-up (m4_raptor)      top layer  -> summary_high  (BROADEST)
#
# Both name the broadest tier "high", but top-down puts broad nodes at
# SMALL depth numbers and bottom-up at LARGE layer numbers. A merged
# unit-type distribution is therefore not merely noisy, it is wrong in a
# way no reader can detect from the table.
#
# Detected by the m4_* per-query diagnostics, which landed WITH the
# rebuild: an M4 row without them predates it. Non-M4 systems retrieve no
# summary units at all, so the question does not arise for them.
PRE_REBUILD_SUFFIX = "[pre-rebuild]"


def _row_bucket_key(record: dict) -> str:
    sid = record.get("system_id", "?")
    if sid != "M4":
        return sid
    md = record.get("metadata") or {}
    if any(k.startswith("m4_") for k in md):
        return sid
    return f"{sid}{PRE_REBUILD_SUFFIX}"


def _safe_stats(xs: list[float]) -> dict[str, float | int]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def _m9_discrimination(bucket: dict) -> dict[str, Any] | None:
    """Does M9's action label actually track retrieval quality?

    The pre-registered transfer check compares the realised action MIX
    against the derivation-time mix, but the mix alone is not diagnostic:
    corpus difficulty and miscalibration produce the identical signature.
    This reports the two things that ARE diagnostic.

    DISCRIMINATION — Cohen's d between CORRECT and AMBIGUOUS on
    retrieval F1, over ANSWERABLE queries only. Large d means the
    evaluator is tracking relevance and a mix shift is corpus difficulty.
    Small d means the threshold sits at an arbitrary point.

    ANSWERABILITY — the AMBIGUOUS rate among NULL (no-gold) queries
    against the rate among answerable ones. These are DIFFERENT
    abilities, and MultiHop separates them starkly: the evaluator put
    99% of no-gold queries in AMBIGUOUS while barely separating retrieval
    quality among the rest. An evaluator can be good at "nothing here"
    and useless at "how good is this".
    """
    by_action = bucket.get("m9_f1_by_action") or {}
    correct, ambiguous = by_action.get("correct", []), by_action.get("ambiguous", [])
    if len(correct) < 20 or len(ambiguous) < 20:
        return None
    mc, ma = statistics.mean(correct), statistics.mean(ambiguous)
    sd = statistics.pstdev(correct + ambiguous) or 1e-9
    d = (mc - ma) / sd
    nulls = bucket.get("m9_null_by_action") or {}
    n_null = sum(nulls.values())
    n_ans = len(correct) + len(ambiguous)
    return {
        "n_answerable": n_ans,
        "n_null": n_null,
        "f1_correct": mc,
        "f1_ambiguous": ma,
        "delta_f1": mc - ma,
        "cohens_d": d,
        "discriminates": bool(d >= 0.2),
        "ambiguous_rate_answerable": len(ambiguous) / n_ans,
        "ambiguous_rate_null": (
            nulls.get("ambiguous", 0) / n_null if n_null else None),
    }


def _non_leaf_summary(bucket: dict) -> dict[str, Any]:
    """Non-leaf share of retrieved units — the RAPTOR App. I gate.

    MICRO is the primary figure: total non-leaf units over total units,
    which is what the paper's "18.5% to 57% of retrieved nodes" means.
    It is computed from `retrieved_unit_types`, a field every eval row
    has carried since CK-4 — so this gate reads correctly on JSONLs
    banked long before the M4 diagnostics existed.

    MACRO (the mean of per-query shares) comes from the newer
    `m4_non_leaf_share` metadata and is reported alongside because the
    two diverge when queries return different numbers of units; neither
    is wrong, but only one is the paper's.

    `in_band` is None rather than False for a system that retrieves no
    summary units at all — a flat retriever is not failing a RAPTOR
    gate, it is outside its scope, and printing FAIL against M2 would be
    a category error.
    """
    counts = bucket["retrieved_unit_types_agg"]
    total = sum(counts.values())
    non_leaf = sum(v for k, v in counts.items() if k != "chunk")
    micro = (non_leaf / total) if total else None
    per_query = bucket["m4_non_leaf_share"]
    lo, hi = PAPER_NON_LEAF_SHARE_BAND
    return {
        "micro": micro,
        "macro": statistics.mean(per_query) if per_query else None,
        "n_units": total,
        "n_non_leaf": non_leaf,
        "in_band": (lo <= micro <= hi) if (micro is not None and non_leaf) else None,
        "band": [lo, hi],
        # .get() rather than [] so a partial bucket (a caller checking one
        # field, or a rollup built under an older aggregate shape) reads
        # as "no trips" instead of raising.
        "expansion_rows": bucket.get("m4_expansion_rows", 0),
        "degenerate_rows": bucket.get("m4_degenerate_rows", 0),
        "bic_failure_rows": bucket.get("m4_bic_failure_rows", 0),
    }


def _aggregate(records: Iterable[dict]) -> dict[str, Any]:
    """Per-system aggregates. Records read in any order; system_id keys the rollup."""
    by_system: dict[str, dict[str, list]] = defaultdict(
        lambda: {
            "chunk_counts": [],
            "packed_counts": [],
            "evidence_tokens": [],
            "input_tokens": [],
            "retr_f1": [],
            "retr_recall": [],
            "retr_precision": [],
            "retr_skipped": 0,
            # Rank-aware (MultiHop). hit_at_k / map_at_k are per-K lists
            # accumulated when present; mrr is a flat list of values.
            "hit_at_k": defaultdict(list),
            "map_at_k": defaultdict(list),
            "mrr": [],
            "ans_score": [],
            "abstained": 0,
            # THE ANSWER SCORE SPLIT BY ABSTENTION, over ANSWERABLE ROWS
            # ONLY. On a free-form benchmark a refusal scores 0.0 against
            # every reference by construction, so a low mean can be
            # measuring how often a system declined rather than how well
            # it answered.
            #
            # NULL ROWS ARE EXCLUDED, and that exclusion is the whole
            # correctness of this block. On a null query a refusal is the
            # CORRECT answer and scores 1.0 under `unanswerable_rule`, so
            # including nulls makes the abstained side a blend of
            # "correctly refused an unanswerable" and "declined an
            # answerable" — not an answer-quality figure at all. Measured
            # on M1 x MultiHop, which inverted the split it was supposed
            # to explain: mean_abs 0.303 against mean_ans 0.018, 17x the
            # wrong way, purely because 301 correct null refusals landed
            # on the abstained side.
            "ans_score_abstained": [],
            "ans_score_answered": [],
            "n_null_rows": 0,
            "n_answerable_rows": 0,
            "abstained_null": 0,
            "latency": [],
            "retrieved_unit_types_agg": defaultdict(int),
            "packed_unit_types_agg": defaultdict(int),
            "by_type_chunk_counts": defaultdict(list),
            "by_type_retr_f1": defaultdict(list),
            "by_type_ans_score": defaultdict(list),
            "by_type_n": defaultdict(int),
            # M9 corrective diagnostics (metadata m9_* keys, merged into
            # ScoredQuery.metadata from AnswerResult.extra by the runner).
            "m9_action_counts": defaultdict(int),
            "m9_rewrite_fired": 0,
            "m9_overlap": [],
            "m9_max_conf": [],
            "m9_strips_kept": 0,
            "m9_strips_total": 0,
            # DISCRIMINATION, per cell. The action MIX alone cannot
            # distinguish corpus difficulty from miscalibration -- both
            # predict more queries clearing tau_high. What separates them
            # is whether the action still tracks retrieval quality. And
            # the ANSWERABLE/NULL split is a second axis entirely: an
            # evaluator can be excellent at "there is nothing here" while
            # useless at ranking quality among queries that do have gold.
            "m9_f1_by_action": defaultdict(list),
            "m9_null_by_action": defaultdict(int),
            # Multiple-choice extraction stats (QuALITY; answer.metadata
            # "extraction" key). Per SYSTEM: one system landing
            # disproportionately in token_f1/unparseable means its
            # output format fights the extractor — that must be
            # immediately visible, not buried in zero answer scores.
            "mc_extraction_counts": defaultdict(int),
            "mc_abstained": 0,
            "mc_unparseable": 0,
            # M4 RAPTOR diagnostics (metadata m4_* keys). The App. I
            # non-leaf-share gate is derivable from retrieved_unit_types
            # alone, which every row has always carried — so the gate
            # also works on JSONLs banked BEFORE these keys existed. The
            # per-query list is the finer-grained version, present only
            # on newer rows.
            "m4_non_leaf_share": [],
            "m4_expansion_rows": 0,
            # Flat-index rows: the corpus fell at or below the layer stop
            # condition, so M4 ran as flat dense retrieval. Counted per
            # QUERY because with thousands of tiny corpora the index logs
            # scroll past and only this table survives.
            "m4_degenerate_rows": 0,
            "m4_bic_failure_rows": 0,
        }
    )

    n_total = 0
    for r in records:
        n_total += 1
        sid = _row_bucket_key(r)
        bucket = by_system[sid]

        bucket["chunk_counts"].append(int(r.get("n_retrieved", 0)))
        bucket["packed_counts"].append(int(r.get("n_packed", 0)))
        bucket["evidence_tokens"].append(int(r.get("evidence_tokens", 0)))
        bucket["input_tokens"].append(int(r.get("n_input_tokens", 0)))
        bucket["latency"].append(float(r.get("latency_s", 0.0)))

        for ut, n in (r.get("retrieved_unit_types") or {}).items():
            bucket["retrieved_unit_types_agg"][ut] += int(n)
        for ut, n in (r.get("packed_unit_types") or {}).items():
            bucket["packed_unit_types_agg"][ut] += int(n)

        retr = r.get("retrieval") or {}
        if retr.get("skipped"):
            bucket["retr_skipped"] += 1
        else:
            bucket["retr_f1"].append(float(retr.get("f1", 0.0)))
            bucket["retr_recall"].append(float(retr.get("recall", 0.0)))
            bucket["retr_precision"].append(float(retr.get("precision", 0.0)))

            # Rank-aware metrics (MultiHop): collect when present.
            # hit_at_k / map_at_k are dicts keyed by K (stringified
            # after JSON round-trip); mrr is a scalar.
            for k_str, v in (retr.get("hit_at_k") or {}).items():
                try:
                    bucket["hit_at_k"][int(k_str)].append(float(v))
                except (TypeError, ValueError):
                    continue
            for k_str, v in (retr.get("map_at_k") or {}).items():
                try:
                    bucket["map_at_k"][int(k_str)].append(float(v))
                except (TypeError, ValueError):
                    continue
            mrr_val = retr.get("mrr")
            if mrr_val is not None and float(mrr_val) > 0:
                # Only collect non-zero MRR to avoid flooding queries
                # where rank-aware was applied but no relevant chunk
                # was found (legitimate 0); keep the 0s too via the
                # separate counter.
                pass
            if "mrr" in retr:
                bucket["mrr"].append(float(retr.get("mrr") or 0.0))

        ans = r.get("answer") or {}
        bucket["ans_score"].append(float(ans.get("value", 0.0)))

        # Multiple-choice extraction stats (present on QuALITY rows).
        ans_md = ans.get("metadata") or {}
        extraction = ans_md.get("extraction")
        if extraction:
            bucket["mc_extraction_counts"][extraction] += 1
            if ans_md.get("abstained"):
                bucket["mc_abstained"] += 1
            if ans_md.get("unparseable"):
                bucket["mc_unparseable"] += 1

        predicted = r.get("predicted_answer", "") or ""
        # PREFER THE RECORDED FLAG. `answer.metadata.abstained` is what the
        # scorer wrote at run time; recomputing here would silently
        # re-classify banked rows if the detector ever moves, which is the
        # kind of drift that is invisible in a mean. Fall back to
        # recomputation only for rows that predate the field.
        recorded = ans_md.get("abstained")
        abstained = bool(recorded) if recorded is not None else is_abstention(predicted)
        if abstained:
            bucket["abstained"] += 1

        # A NULL ROW is one the loader marked unanswerable, which the P1
        # contract records as method == "unanswerable_rule". Its score is
        # a refusal-correctness figure on a different scale from token-F1,
        # so it is counted but kept OUT of the answer-quality split.
        if ans.get("method") == "unanswerable_rule":
            bucket["n_null_rows"] += 1
            if abstained:
                bucket["abstained_null"] += 1
        else:
            bucket["n_answerable_rows"] += 1
            if abstained:
                bucket["ans_score_abstained"].append(float(ans.get("value", 0.0)))
            else:
                bucket["ans_score_answered"].append(float(ans.get("value", 0.0)))

        # M9 corrective action logging (present only on M9 rows).
        md = r.get("metadata") or {}
        m9_action = md.get("m9_action")
        if m9_action:
            bucket["m9_action_counts"][m9_action] += 1
            if md.get("m9_rewrite_fired"):
                bucket["m9_rewrite_fired"] += 1
            overlap = md.get("m9_overlap_jaccard")
            if overlap is not None:
                bucket["m9_overlap"].append(float(overlap))
            max_conf = md.get("m9_max_conf")
            if max_conf is not None:
                bucket["m9_max_conf"].append(float(max_conf))
            bucket["m9_strips_kept"] += int(md.get("m9_n_strips_kept") or 0)
            bucket["m9_strips_total"] += int(md.get("m9_n_strips_total") or 0)
            if retr.get("skipped"):
                bucket["m9_null_by_action"][m9_action] += 1
            else:
                bucket["m9_f1_by_action"][m9_action].append(
                    float(retr.get("f1", 0.0)))

        # M4 RAPTOR diagnostics (present only on M4 rows from commit 5 on).
        m4_share = md.get("m4_non_leaf_share")
        if m4_share is not None:
            bucket["m4_non_leaf_share"].append(float(m4_share))
        if md.get("m4_summary_expansion"):
            bucket["m4_expansion_rows"] += 1
        if md.get("m4_tree_degenerate"):
            bucket["m4_degenerate_rows"] += 1
        if int(md.get("m4_bic_fit_failures") or 0) > 0:
            bucket["m4_bic_failure_rows"] += 1

        qtype = r.get("question_type", "?")
        bucket["by_type_n"][qtype] += 1
        bucket["by_type_chunk_counts"][qtype].append(int(r.get("n_retrieved", 0)))
        if not retr.get("skipped"):
            bucket["by_type_retr_f1"][qtype].append(float(retr.get("f1", 0.0)))
        bucket["by_type_ans_score"][qtype].append(float(ans.get("value", 0.0)))

    # Reshape to a serialisable rollup.
    out: dict[str, Any] = {"n_total_records": n_total, "systems": {}}
    for sid, b in sorted(by_system.items()):
        n_q = len(b["chunk_counts"])
        out["systems"][sid] = {
            "n_queries": n_q,
            "chunk_count": _safe_stats(b["chunk_counts"]),
            "packed_count": _safe_stats(b["packed_counts"]),
            "evidence_tokens": _safe_stats(b["evidence_tokens"]),
            "input_tokens": _safe_stats(b["input_tokens"]),
            "retrieved_unit_types": dict(b["retrieved_unit_types_agg"]),
            "packed_unit_types": dict(b["packed_unit_types_agg"]),
            "non_leaf": _non_leaf_summary(b),
            "m9_discrimination": _m9_discrimination(b),
            "retr_f1_mean": (statistics.mean(b["retr_f1"]) if b["retr_f1"] else None),
            "retr_recall_mean": (
                statistics.mean(b["retr_recall"]) if b["retr_recall"] else None
            ),
            "retr_precision_mean": (
                statistics.mean(b["retr_precision"]) if b["retr_precision"] else None
            ),
            "retr_n_scored": len(b["retr_f1"]),
            "retr_n_skipped": b["retr_skipped"],
            # Rank-aware aggregates (empty when only QASPER-style
            # records were aggregated; populated when MultiHop is in
            # the input set).
            "hit_at_k_mean": {
                k: (statistics.mean(vs) if vs else None)
                for k, vs in sorted(b["hit_at_k"].items())
            },
            "map_at_k_mean": {
                k: (statistics.mean(vs) if vs else None)
                for k, vs in sorted(b["map_at_k"].items())
            },
            "mrr_mean": (statistics.mean(b["mrr"]) if b["mrr"] else None),
            "ans_score_mean": (statistics.mean(b["ans_score"]) if b["ans_score"] else None),
            # M9 corrective rollup; None for non-M9 systems. The realized
            # action mix must roughly match the derivation-time mix from
            # scripts/derive_corrective_thresholds.py — large drift means
            # the thresholds are miscalibrated for this corpus; flag
            # BEFORE trusting the system's answer numbers.
            "m9_corrective": (
                {
                    "n": sum(b["m9_action_counts"].values()),
                    "action_mix": {
                        a: round(c / max(1, sum(b["m9_action_counts"].values())), 4)
                        for a, c in sorted(b["m9_action_counts"].items())
                    },
                    "rewrite_fired_rate": (
                        b["m9_rewrite_fired"]
                        / max(1, sum(b["m9_action_counts"].values()))
                    ),
                    "overlap_jaccard_mean": (
                        statistics.mean(b["m9_overlap"]) if b["m9_overlap"] else None
                    ),
                    "max_conf_mean": (
                        statistics.mean(b["m9_max_conf"]) if b["m9_max_conf"] else None
                    ),
                    "strips_kept_frac": (
                        b["m9_strips_kept"] / b["m9_strips_total"]
                        if b["m9_strips_total"]
                        else None
                    ),
                }
                if b["m9_action_counts"]
                else None
            ),
            # Multiple-choice extraction rollup; None when no MC rows.
            "mc_extraction": (
                {
                    "n": sum(b["mc_extraction_counts"].values()),
                    "methods": {
                        m: {
                            "count": c,
                            "frac": round(
                                c / max(1, sum(b["mc_extraction_counts"].values())), 4
                            ),
                        }
                        for m, c in sorted(b["mc_extraction_counts"].items())
                    },
                    "abstained_rate": (
                        b["mc_abstained"]
                        / max(1, sum(b["mc_extraction_counts"].values()))
                    ),
                    "unparseable_rate": (
                        b["mc_unparseable"]
                        / max(1, sum(b["mc_extraction_counts"].values()))
                    ),
                }
                if b["mc_extraction_counts"]
                else None
            ),
            "abstention_rate": b["abstained"] / max(1, n_q),
            # n AND mean for each side. A mean without its n is the
            # same defect as a guard without its comparison.
            #
            # NAMED FOR THEIR POPULATION: every field below is over
            # ANSWERABLE rows only. `abstention_rate` above is over ALL
            # rows and is a different quantity; both are reported because
            # both are asked for, and neither is allowed to borrow the
            # other's denominator.
            "n_null_rows": b["n_null_rows"],
            "n_answerable_rows": b["n_answerable_rows"],
            "n_abstained_null": b["abstained_null"],
            "abstention_rate_answerable": (
                len(b["ans_score_abstained"]) / b["n_answerable_rows"]
                if b["n_answerable_rows"] else None),
            "n_abstained": len(b["ans_score_abstained"]),
            "n_answered": len(b["ans_score_answered"]),
            "ans_score_abstained_mean": (
                statistics.mean(b["ans_score_abstained"])
                if b["ans_score_abstained"] else None),
            "ans_score_answered_mean": (
                statistics.mean(b["ans_score_answered"])
                if b["ans_score_answered"] else None),
            "latency_s_mean": (statistics.mean(b["latency"]) if b["latency"] else None),
            "by_question_type": {
                qt: {
                    "n": b["by_type_n"][qt],
                    "chunk_count_mean": (
                        statistics.mean(b["by_type_chunk_counts"][qt])
                        if b["by_type_chunk_counts"][qt]
                        else None
                    ),
                    "retr_f1_mean": (
                        statistics.mean(b["by_type_retr_f1"][qt])
                        if b["by_type_retr_f1"][qt]
                        else None
                    ),
                    "ans_score_mean": (
                        statistics.mean(b["by_type_ans_score"][qt])
                        if b["by_type_ans_score"][qt]
                        else None
                    ),
                }
                for qt in sorted(b["by_type_n"])
            },
        }
    return out


def _fmt(x: float | int | None, places: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.{places}f}"
    return str(x)


def _print_text(rollup: dict[str, Any], *, by_type: bool) -> None:
    print(f"[analyse] {rollup['n_total_records']} ScoredQuery records across "
          f"{len(rollup['systems'])} systems\n")

    # Top table.
    cols = [
        ("system", 8),
        ("n_q", 5),
        ("chunks", 8),
        ("packed", 8),
        ("ev_tok", 8),
        ("in_tok", 8),
        ("retr_f1", 8),
        ("ans", 7),
        ("abstain%", 9),
        ("skip", 5),
        ("lat_s", 7),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for sid in rollup["systems"]:
        s = rollup["systems"][sid]
        row = [
            (sid, 8),
            (str(s["n_queries"]), 5),
            (_fmt(s["chunk_count"].get("mean"), places=1), 8),
            (_fmt(s["packed_count"].get("mean"), places=1), 8),
            (_fmt(s["evidence_tokens"].get("mean"), places=0), 8),
            (_fmt(s["input_tokens"].get("mean"), places=0), 8),
            (_fmt(s["retr_f1_mean"]), 8),
            (_fmt(s["ans_score_mean"]), 7),
            (_fmt(s["abstention_rate"] * 100, places=1) + "%", 9),
            (str(s["retr_n_skipped"]), 5),
            (_fmt(s["latency_s_mean"], places=2), 7),
        ]
        print("  ".join(val.ljust(w) for val, w in row))

    # Rank-aware retrieval metrics (when present — MultiHop-only).
    any_rank_aware = any(
        rollup["systems"][sid].get("mrr_mean") is not None
        for sid in rollup["systems"]
    )
    if any_rank_aware:
        print("\n  --- rank-aware retrieval (MultiHop) ---")
        rank_cols = [("system", 8), ("mrr", 7), ("hit@1", 7), ("hit@5", 7),
                     ("hit@10", 8), ("map@1", 7), ("map@5", 7), ("map@10", 8)]
        rank_header = "  ".join(name.ljust(w) for name, w in rank_cols)
        print(rank_header)
        print("-" * len(rank_header))
        for sid in rollup["systems"]:
            s = rollup["systems"][sid]
            mrr_m = s.get("mrr_mean")
            if mrr_m is None:
                continue
            hits = s.get("hit_at_k_mean") or {}
            maps = s.get("map_at_k_mean") or {}
            row = [
                (sid, 8),
                (_fmt(mrr_m), 7),
                (_fmt(hits.get(1)), 7),
                (_fmt(hits.get(5)), 7),
                (_fmt(hits.get(10)), 8),
                (_fmt(maps.get(1)), 7),
                (_fmt(maps.get(5)), 7),
                (_fmt(maps.get(10)), 8),
            ]
            print("  ".join(val.ljust(w) for val, w in row))

    # ANSWER SCORE SPLIT BY ABSTENTION, over ANSWERABLE ROWS ONLY.
    #
    # Printed unconditionally, because the case it exists for is the case
    # where nobody thinks to ask for it. On a free-form benchmark a
    # refusal scores 0.0 against every reference BY CONSTRUCTION, so a
    # low mean answer score has two readings -- answered badly, or
    # declined often -- and the micro-mean cannot distinguish them.
    #
    # NULLS ARE EXCLUDED and the header says so. Including them inverted
    # the split on M1 x MultiHop (mean_abs 0.303 against mean_ans 0.018)
    # because a correct refusal on a null scores 1.0 and landed on the
    # abstained side. The guidance line below is only true once the null
    # rows are out.
    print("")
    print("answer score split by abstention  [ANSWERABLE ROWS ONLY]")
    ab_cols = [("system", 8), ("n_able", 7), ("n_abs", 7), ("mean_abs", 9),
               ("n_ans", 7), ("mean_ans", 9), ("abs%_able", 10),
               ("n_null", 7)]
    ab_header = "  ".join(name.ljust(w) for name, w in ab_cols)
    print(ab_header)
    print("-" * len(ab_header))
    any_null = False
    for sid in rollup["systems"]:
        st = rollup["systems"][sid]
        n_null = st.get("n_null_rows", 0)
        any_null = any_null or bool(n_null)
        rate = st.get("abstention_rate_answerable")
        row = [
            (sid, 8),
            (str(st.get("n_answerable_rows", 0)), 7),
            (str(st.get("n_abstained", 0)), 7),
            (_fmt(st.get("ans_score_abstained_mean")), 9),
            (str(st.get("n_answered", 0)), 7),
            (_fmt(st.get("ans_score_answered_mean")), 9),
            ("n/a" if rate is None else f"{rate:.1%}", 10),
            (str(n_null), 7),
        ]
        print("  ".join(val.ljust(w) for val, w in row))
    print("  a refusal scores 0.0 against free-form references by "
          "construction;")
    print("  read mean_ans, not the overall mean, as answer QUALITY.")
    if any_null:
        print("  NULL ROWS ARE EXCLUDED from every column except n_null: on a")
        print("  null query a refusal is CORRECT and scores 1.0, so including")
        print("  them would blend refusal-correctness into an answer-quality")
        print("  figure. Their own mean is mean_answer_score_null in the")
        print("  run summary.")
        print("  NOTE THE DENOMINATORS: the summary's")
        print("  mean_answer_score_answerable is over ALL answerable rows")
        print("  INCLUDING refusals; mean_ans above is over ATTEMPTED rows")
        print("  only. Both correct, different populations -- name whichever")
        print("  the results table uses.")
    print("")

    # M9 corrective action mix (when present). Compare against the
    # derivation-time mix from scripts/derive_corrective_thresholds.py:
    # large drift = threshold miscalibration on this corpus.
    any_m9 = any(
        rollup["systems"][sid].get("m9_corrective") for sid in rollup["systems"]
    )
    if any_m9:
        print("\n  --- M9 corrective action mix ---")
        for sid in rollup["systems"]:
            m9 = rollup["systems"][sid].get("m9_corrective")
            if not m9:
                continue
            mix = ", ".join(f"{a}={frac:.1%}" for a, frac in m9["action_mix"].items())
            print(f"  {sid}: n={m9['n']}  {mix}")
            print(
                f"  {sid}: rewrite_fired={m9['rewrite_fired_rate']:.1%}  "
                f"overlap_jaccard={_fmt(m9.get('overlap_jaccard_mean'))}  "
                f"max_conf={_fmt(m9.get('max_conf_mean'))}  "
                f"strips_kept={_fmt(m9.get('strips_kept_frac'))}"
            )

    # Multiple-choice extraction-method distribution (QuALITY rows),
    # PER SYSTEM: a system landing disproportionately in
    # token_f1/unparseable has an output format that fights the
    # extractor — a prompt/format problem, not a retrieval signal.
    any_mc = any(
        rollup["systems"][sid].get("mc_extraction") for sid in rollup["systems"]
    )
    if any_mc:
        print("\n  --- multiple-choice extraction (per system) ---")
        for sid in rollup["systems"]:
            mc = rollup["systems"][sid].get("mc_extraction")
            if not mc:
                continue
            dist = ", ".join(
                f"{m}={v['frac']:.1%}" for m, v in mc["methods"].items()
            )
            print(f"  {sid}: n={mc['n']}  {dist}")
            print(
                f"  {sid}: abstained={mc['abstained_rate']:.1%}  "
                f"unparseable={mc['unparseable_rate']:.1%}"
            )

    # ERA SPLIT WARNING. Reaching this means the caller passed JSONLs from
    # both sides of the M4 rebuild in one invocation. They are already
    # bucketed apart by _row_bucket_key; this says WHY, because a reader
    # who sees two M4 rows and assumes a duplicate would pool them by hand.
    if any(
        k + PRE_REBUILD_SUFFIX in rollup["systems"] for k in rollup["systems"]
    ):
        print("")
        print("  *** TWO M4 ERAS IN ONE INPUT SET - reported SEPARATELY, "
              "never pooled. ***")
        print("  Retrieval-unit tier labels are INVERTED across the rebuild "
              "(old top-down: depth 0-1 = summary_high, the BROADEST; new "
              "bottom-up: TOP layer = summary_high).")
        print("  Averaging them is WRONG, not merely noisy, and the error is "
              "invisible in the table.")
        print("  The pre-rebuild cells are STALE anyway - their substrate key "
              "moved. Stash them rather than reading both.")
        print("  See docs/FINDING_SHALLOW_HIERARCHY.md sections 0 and 7.")

    # M9 DISCRIMINATION — printed for every M9 cell, not once. Whether
    # the action label tracks retrieval quality is a property of the
    # CORPUS, so MultiHop's answer does not generalise to NarrativeQA or
    # HotpotQA and must be read per cell.
    for sid in rollup["systems"]:
        disc = rollup["systems"][sid].get("m9_discrimination")
        if not disc:
            continue
        print("")
        print(f"  --- {sid} corrective-action DISCRIMINATION ---")
        print(f"  answerable n={disc['n_answerable']}  "
              f"retr_f1 correct={disc['f1_correct']:.4f} "
              f"ambiguous={disc['f1_ambiguous']:.4f}  "
              f"delta={disc['delta_f1']:+.4f}  d={disc['cohens_d']:+.2f}")
        if disc["discriminates"]:
            print("  VERDICT: the action TRACKS retrieval quality (d >= 0.2). "
                  "A mix shift here is corpus difficulty, and an M9-M3 delta "
                  "may be read as detect-and-repair.")
        else:
            print("  VERDICT: the action does NOT track retrieval quality "
                  "(d < 0.2). An M9-M3 delta on this cell CANNOT be "
                  "attributed to detecting poor retrieval - only to running "
                  "a second pass on a set selected by something else.")
        if disc["ambiguous_rate_null"] is not None:
            print(f"  ANSWERABILITY: ambiguous rate {disc['ambiguous_rate_null']:.1%} "
                  f"on NULL (no-gold) queries vs "
                  f"{disc['ambiguous_rate_answerable']:.1%} on answerable "
                  f"(n_null={disc['n_null']})")
            if disc["ambiguous_rate_null"] > 0.8:
                print("  -> the evaluator IS discriminating, but on "
                      "ANSWERABILITY rather than on retrieval quality. "
                      "Those are different abilities and only the second "
                      "is what CRAG's corrective step claims.")

    # Unit-type distribution (per-system).
    print("\n  --- retrieved unit-type distribution ---")
    for sid in rollup["systems"]:
        ut = rollup["systems"][sid].get("retrieved_unit_types") or {}
        if ut:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(ut.items()))
            print(f"  {sid}: {parts}")
    print("  --- packed unit-type distribution ---")
    for sid in rollup["systems"]:
        ut = rollup["systems"][sid].get("packed_unit_types") or {}
        if ut:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(ut.items()))
            print(f"  {sid}: {parts}")

    # RAPTOR App. I gate: the non-leaf share of RETRIEVED nodes. Printed
    # only for systems that actually return summary units — a flat
    # retriever is outside the gate's scope, not failing it.
    lo, hi = PAPER_NON_LEAF_SHARE_BAND
    gated = [
        (sid, rollup["systems"][sid]["non_leaf"])
        for sid in rollup["systems"]
        if (rollup["systems"][sid].get("non_leaf") or {}).get("n_non_leaf")
    ]
    degenerate_only = [
        (sid, rollup["systems"][sid]["non_leaf"])
        for sid in rollup["systems"]
        if (rollup["systems"][sid].get("non_leaf") or {}).get("degenerate_rows")
        and not (rollup["systems"][sid].get("non_leaf") or {}).get("n_non_leaf")
    ]
    for sid, nl in degenerate_only:
        # Reached when EVERY corpus was flat: no summary units anywhere,
        # so the gate block below skips the system entirely and the
        # warning would have been lost with it.
        print("")
        print(
            f"  {sid}: *** NOT A RAPTOR RESULT on {nl['degenerate_rows']} "
            "rows - no tree was built on any corpus (all at or below the "
            "layer stop condition). M4 ran as flat dense retrieval "
            "throughout. ***"
        )

    if gated:
        print(
            f"\n  --- RAPTOR non-leaf share (paper App. I band "
            f"{lo:.1%}-{hi:.1%}) ---"
        )
        for sid, nl in gated:
            verdict = "IN BAND" if nl["in_band"] else "OUT OF BAND"
            macro = (
                f"  macro={nl['macro']:.1%}" if nl["macro"] is not None else ""
            )
            print(
                f"  {sid}: micro={nl['micro']:.1%} "
                f"({nl['n_non_leaf']}/{nl['n_units']} units){macro}  {verdict}"
            )
            if nl["degenerate_rows"]:
                # The loudest thing in this report, deliberately. A flat
                # M4 still retrieves, still answers, and still produces a
                # plausible row -- it just is not RAPTOR. Silent
                # degeneration in a results table is exactly the failure
                # this exists to prevent.
                print(
                    f"  {sid}: *** NOT A RAPTOR RESULT on "
                    f"{nl['degenerate_rows']} rows - the corpus was at or "
                    "below the layer stop condition, so there was no tree "
                    "and M4 ran as flat dense retrieval. ***"
                )
            if nl["bic_failure_rows"]:
                print(
                    f"  {sid}: {nl['bic_failure_rows']} rows came from a tree "
                    "whose BIC search skipped an unfittable k (guard v)"
                )
            if nl["expansion_rows"]:
                # Loud, because a leaf-expanded run is NOT a reportable
                # M4 cell: its evidence text is leaves, so its answers
                # are a different system's.
                print(
                    f"  {sid}: *** DIAGNOSTIC TWIN - {nl['expansion_rows']} "
                    "rows ran with summary expansion ON. Retrieval is "
                    "comparable to a leaf-only system; ANSWERS are not a "
                    "reportable M4 number. ***"
                )

    if not by_type:
        return

    # Per-(system, question_type) slice.
    print("\n  --- per question_type ---\n")
    qtype_cols = [
        ("system", 8),
        ("qtype", 16),
        ("n", 5),
        ("chunks_mean", 11),
        ("retr_f1", 8),
        ("ans", 7),
    ]
    header2 = "  ".join(name.ljust(w) for name, w in qtype_cols)
    print(header2)
    print("-" * len(header2))
    for sid in rollup["systems"]:
        s = rollup["systems"][sid]
        for qt, qs in s["by_question_type"].items():
            row = [
                (sid, 8),
                (qt, 16),
                (str(qs["n"]), 5),
                (_fmt(qs.get("chunk_count_mean")), 11),
                (_fmt(qs.get("retr_f1_mean")), 8),
                (_fmt(qs.get("ans_score_mean")), 7),
            ]
            print("  ".join(val.ljust(w) for val, w in row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate ScoredQuery JSONL files from src.eval.runner."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="JSONL paths or glob patterns. Combined with --inputs.",
    )
    parser.add_argument(
        "--inputs",
        dest="inputs_named",
        nargs="*",
        default=[],
        help="Alternate way to pass inputs (paths or glob patterns).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full rollup.",
    )
    parser.add_argument(
        "--by-type",
        action="store_true",
        help="Also print per-question-type rows under each system.",
    )
    parser.add_argument(
        "--check-budget-equality",
        action="store_true",
        help="Assert all systems' mean evidence_tokens fall within "
        "±5%% of the cross-system mean. CK-4 invariant check — the "
        "point of the whole change. Exits non-zero on violation.",
    )
    parser.add_argument(
        "--budget-equality-tolerance",
        type=float,
        default=0.05,
        help="Fractional tolerance for --check-budget-equality. Default 0.05 (±5%%).",
    )
    args = parser.parse_args()

    raw_inputs = (args.inputs or []) + (args.inputs_named or [])
    if not raw_inputs:
        parser.error("at least one input path or glob is required")

    paths = _expand_inputs(raw_inputs)
    if not paths:
        parser.error(f"no files matched the input(s): {raw_inputs!r}")

    print(f"[analyse] reading {len(paths)} JSONL file(s):")
    for p in paths:
        print(f"  {p}")

    records = _iter_records(paths)
    _check_expected_n(paths, len(records))
    rollup = _aggregate(records)
    _print_text(rollup, by_type=args.by_type)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(rollup, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[analyse] rollup -> {args.output}")

    if args.check_budget_equality:
        # Compare mean evidence_tokens across systems; flag outliers.
        # Skip systems with 0 mean (M1 closed-book legitimately has 0
        # evidence). Assertion: every non-zero system's mean falls
        # within (1 ± tol) × cross-system mean.
        means = []
        for sid, s in rollup["systems"].items():
            m = s["evidence_tokens"].get("mean")
            if m and m > 0:
                means.append((sid, float(m)))
        if not means:
            print("\n[analyse] --check-budget-equality: no system reported "
                  "non-zero evidence_tokens (CK-4 not applied?).")
            raise SystemExit(2)
        avg = sum(m for _, m in means) / len(means)
        tol = args.budget_equality_tolerance
        lo, hi = avg * (1.0 - tol), avg * (1.0 + tol)
        print(f"\n[analyse] --check-budget-equality (tol ±{tol*100:.1f}%):")
        print(f"  cross-system mean evidence_tokens = {avg:.0f}")
        print(f"  acceptable band: [{lo:.0f}, {hi:.0f}]")
        violations = []
        for sid, m in means:
            ok = lo <= m <= hi
            marker = "OK" if ok else "VIOLATION"
            print(f"    {sid}: {m:.0f}  [{marker}]")
            if not ok:
                violations.append((sid, m))
        if violations:
            print(f"\n[analyse] FAIL: {len(violations)} system(s) outside "
                  f"±{tol*100:.1f}% band.")
            raise SystemExit(2)
        print("\n[analyse] PASS: all systems within budget-equality band.")


if __name__ == "__main__":
    main()
