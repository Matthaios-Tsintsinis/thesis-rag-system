"""Re-derive a benchmark's M4 leaf population and the two chunker-divergence
incidences, through the pipeline's OWN corpus layout.

WHY THIS IS A SCRIPT AND NOT A TEST. It measures a property of the DATA,
not of the code, so there is no fixed value to assert. It exists because a
population description drifted silently: the HotpotQA figures recorded on
2026-08-16 (18,235 leaves, 36/1000 degenerate) predated the single-item-rule
layout promotion in `BaseSystem.index_items`, and nothing recomputed them.
The 2026-08-22 audit found the drift by re-running this measurement and
noticing that MultiHop reproduced EXACTLY (16,523 leaves, matching its
banked cell) while HotpotQA did not -- which is what localised the cause to
a benchmark-specific layout change rather than to the chunker.

WHAT IT REPORTS
  leaves            per-unit and total, with the quantiles the deviation
                    block quotes, plus the count at or below RAPTOR's stop
                    condition (reduction_dimension + 1 = 11)
  long sentences    sentences exceeding the 100-token budget -- the trigger
                    for ruling 1b's placement divergence (AF-2)
  duplicate chunks  byte-identical chunk texts within a unit -- the trigger
                    for micro-divergence (vi) (AF-3)

THIS IS AN ESTIMATE, NOT AN AUTHORITY. It re-runs the chunker; it does not
read a banked cell. For the degenerate count of a cell that has RUN, the
authority is `python -m src.eval.analyse <cell>.jsonl`, which counts
`metadata.m4_tree_degenerate` on the rows the cell actually produced. Quote
this script for corpora not yet run, and `analyse` for corpora already run;
never quote either as the other.

CPU ONLY -- no GPU, no model, no torch. It writes nothing outside a
temporary directory and touches no cache.

    python -m scripts.measure_chunk_population --benchmark narrativeqa
    python -m scripts.measure_chunk_population --benchmark all --json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.components import resolve_components
from src.parsing import walk_corpus
from src.raptor_paper import (
    _iter_sentences,
    count_tokens_reference,
    split_text_raptor,
)
from src.retrievers.m4_raptor import RaptorSystem


BENCHMARKS = ("multihop_rag", "narrativeqa", "hotpotqa", "hotpotqa_pooled")


def _benchmark(name: str):
    if name == "multihop_rag":
        from src.eval.multihop import MultiHopBenchmark

        return MultiHopBenchmark()
    if name == "narrativeqa":
        from src.eval.narrativeqa import NarrativeQABenchmark

        return NarrativeQABenchmark()
    if name == "hotpotqa":
        from src.eval.hotpotqa import HotpotQABenchmark

        return HotpotQABenchmark()
    if name == "hotpotqa_pooled":
        from src.eval.hotpotqa import HotpotQAPooledBenchmark

        return HotpotQAPooledBenchmark()
    raise SystemExit(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")


def measure(name: str, split: str, verbose: bool) -> dict:
    system = RaptorSystem()
    resolved = resolve_components(system.config.m4, system.config, default_reranker=None)
    chunker = resolved.chunker_config
    # ASSERT THE PROBE MEASURES WHAT IT CLAIMS. If M4's chunker is ever not
    # the paper chunker, these numbers describe a different system and must
    # not be reported as M4's.
    if chunker.strategy != "raptor_100tok":
        raise SystemExit(
            f"expected M4 chunker strategy 'raptor_100tok', got "
            f"{chunker.strategy!r} -- this script measures the PAPER chunker"
        )
    budget = chunker.chunk_words
    stop_at = system.config.m4.paper.reduction_dimension + 1

    per_unit_leaves: list[int] = []
    n_sentences = n_long = n_dup = 0
    units_with_long = units_with_dup = 0

    for u in _benchmark(name).iter_eval_units(split=split):
        chunk_texts: list[str] = []
        unit_long = 0
        with tempfile.TemporaryDirectory(prefix="chunkpop_") as td:
            # The REAL layout: one file per parent, single-item parents
            # byte-identical to the per-item form. Reading it back through
            # walk_corpus applies the same min_chars drop and the same
            # clean_text the index path applies.
            system._write_corpus_layout(list(u.corpus), Path(td))
            for doc in walk_corpus(Path(td), min_chars=chunker.min_chars_per_doc):
                for sentence, _, _ in _iter_sentences(doc.text):
                    n_sentences += 1
                    if count_tokens_reference(sentence) > budget:
                        unit_long += 1
                chunk_texts.extend(
                    span.text for span in split_text_raptor(doc.text, max_tokens=budget)
                )

        dup = sum(c - 1 for c in Counter(chunk_texts).values() if c > 1)
        per_unit_leaves.append(len(chunk_texts))
        n_long += unit_long
        n_dup += dup
        units_with_long += bool(unit_long)
        units_with_dup += bool(dup)
        if verbose:
            print(
                f"  {u.corpus_id[:48]:<48} leaves={len(chunk_texts):>6} "
                f"long={unit_long:>3} dup={dup:>3}"
            )

    leaves = np.array(per_unit_leaves, dtype=int)
    if leaves.size == 0:
        raise SystemExit(f"{name}: no units produced -- nothing measured")

    return {
        "benchmark": name,
        "split": split,
        "chunker": chunker.strategy,
        "max_tokens": budget,
        "stop_condition_leaves": stop_at,
        "n_units": int(leaves.size),
        "leaves_total": int(leaves.sum()),
        "leaves_min": int(leaves.min()),
        "leaves_p25": float(np.percentile(leaves, 25)),
        "leaves_median": float(np.median(leaves)),
        "leaves_p75": float(np.percentile(leaves, 75)),
        "leaves_max": int(leaves.max()),
        "units_at_or_below_stop": int((leaves <= stop_at).sum()),
        "n_sentences": n_sentences,
        "n_sentences_over_budget": n_long,
        "pct_sentences_over_budget": round(100 * n_long / max(1, n_sentences), 4),
        "units_with_over_budget_sentence": units_with_long,
        "n_duplicate_chunks": n_dup,
        "pct_duplicate_chunks": round(100 * n_dup / max(1, int(leaves.sum())), 4),
        "units_with_duplicate_chunks": units_with_dup,
    }


def _render(r: dict) -> str:
    deg = r["units_at_or_below_stop"]
    return (
        f"{r['benchmark']} ({r['split']}) -- ESTIMATE, not a banked-cell reading\n"
        f"  units                {r['n_units']}\n"
        f"  leaves               total {r['leaves_total']}  "
        f"min {r['leaves_min']} / p25 {r['leaves_p25']:.0f} / "
        f"median {r['leaves_median']:.0f} / p75 {r['leaves_p75']:.0f} / "
        f"max {r['leaves_max']}\n"
        f"  at/below stop ({r['stop_condition_leaves']} leaves)  "
        f"{deg} of {r['n_units']} "
        f"({100 * deg / r['n_units']:.1f}%)  <- degenerate, no RAPTOR tree\n"
        f"  AF-2 long sentences  {r['n_sentences_over_budget']} of "
        f"{r['n_sentences']} ({r['pct_sentences_over_budget']}%), "
        f"{r['units_with_over_budget_sentence']} units affected\n"
        f"  AF-3 duplicate chunks {r['n_duplicate_chunks']} of "
        f"{r['leaves_total']} ({r['pct_duplicate_chunks']}%), "
        f"{r['units_with_duplicate_chunks']} units affected"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, choices=(*BENCHMARKS, "all"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", action="store_true", help="one line per unit")
    args = ap.parse_args()

    names = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    results = [measure(n, args.split, args.verbose) for n in names]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(_render(r))
            print()
        print(
            "For a cell that has already RUN, the authoritative degenerate "
            "count is analyse over its JSONL, not this estimate."
        )


if __name__ == "__main__":
    main()
