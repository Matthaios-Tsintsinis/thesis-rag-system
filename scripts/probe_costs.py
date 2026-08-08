"""Mechanism probes, each asserting that it measured what it claims.

RESCOPED 2026-08-05. These began as COST probes, sizing a HotpotQA
allocation against a tight unit budget. That constraint is lifted, so
they are no longer here to shave GPU hours -- they are here to answer
MECHANISM questions that a thousand small builds would otherwise answer
the expensive way. The M9-vs-M2 probe is skipped by default for exactly
that reason: +-6 units is noise now.

WHY THE ASSERTIONS. Four cost forecasts in this project have been wrong,
and two of those were wrong because a probe silently did not measure what
it said: a synthetic benchmark run at a batch size that OOMs on real
prompts, and a 1-token probe whose cap never reached generation (it
printed the cap, generated full-length answers anyway, and the timings
were contaminated by a 15GB model download on top). Both produced numbers
rather than errors. So every probe here refuses to report unless its own
preconditions held.

WHAT IT MEASURES

  1. M1 batched vs sequential, SAME queries, SAME session. The ratio is
     the point: comparing against a 4.1 s/query figure taken under
     different conditions is what produced the last bad number. M1 is
     projected at ~4x from decode-dominance (measured 97% decode), but
     batching measurably LOST at 4k and that failure is still
     unexplained -- so if batched M1 comes out slower, that is the 4k
     mystery reproducing in a second regime, which is a finding worth
     having and NOT a reason to quietly keep the projection.

  2. Per-unit INDEX BUILD cost, the term that is ~27% of HotpotQA
     variant A and is currently a guess. Measured on QASPER per-paper
     corpora, which is the only existing many-small-corpora benchmark.
     PROXY, and a conservative one: a QASPER paper is bigger than a
     10-paragraph HotpotQA corpus, so this OVERSTATES. Good enough for a
     budget ceiling, not a point estimate. M2 gives the pure
     chunk+embed+FAISS+save cost; M4 adds tree construction.

  3. M9 vs M2 on identical queries. SKIPPED BY DEFAULT -- kept only
     because it is nearly free to re-enable with --skip (nothing).

  4. RETIRED 2026-08-06 — THE GMM CRASH BAND PROBE IS DELETED.

     It asked whether the GMM guard holds on 12-30 leaf corpora. That
     question is still live (HotpotQA variant A builds 15-20 leaf
     corpora), but a synthetic fixture is the wrong instrument and
     MISLED TWICE: its paragraph size was taken from an ASSUMED ~70
     tokens, the same assumption the prediction came from, so it could
     only ever agree. Real paragraphs are 127.7 tokens.

     The matrix answers it better, on real data: 1,000 variant-A builds
     with the guard in place and `m4_bic_fit_failures` carried per row
     into the results table. Removing the instrument is safer than
     trusting a third version of it.

     Historic description follows.

     ~~THE GMM CRASH BAND at variant-A corpus shape, and the one that
     matters most now. HotpotQA variant A builds ~10-paragraph corpora,
     i.e. ~10-14 leaves, which is inside the 12-30 band where the BIC
     sweep was MEASURED to raise -- and it is about to do that a
     thousand times. The QASPER build probe cannot answer this: a paper
     is 50+ chunks, well clear of the band. So this builds many tiny
     REAL corpora through the REAL M4 path and reports the guard trip
     rate and the no-tree rate. The failure is data-dependent and
     non-monotone in n, so passing at one size proves nothing about
     another.~~

USAGE (Colab, after `pip install -r requirements.txt` and Drive mount):

    python -m scripts.probe_costs      # writes to <OUTPUT_DIR>/probes

Add --skip m9 (etc.) to run a subset. Total ~15-20 min on an L4.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class ProbeFailure(RuntimeError):
    """A precondition did not hold, so the number is not usable."""


def _run(out_dir: Path, tag: str, *args: str) -> dict:
    """Invoke the real CLI in a subprocess and return its summary.

    A subprocess rather than an in-process call on purpose: it is the
    exact path a real run takes, including argument parsing and config
    construction, which is where the last two defects lived.
    """
    out = out_dir / f"{tag}.jsonl"
    cmd = [sys.executable, "-m", "src.eval.runner", "--output", str(out),
           "--prewarm", *args]
    print(f"\n=== {tag} ===\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise ProbeFailure(f"{tag}: runner exited {proc.returncode}")
    summary_path = out.with_suffix(".summary.json")
    if not summary_path.exists():
        raise ProbeFailure(f"{tag}: no summary written")
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    if s.get("s_per_query") is None:
        raise ProbeFailure(
            f"{tag}: summary has no s_per_query. This build predates the "
            "timing fields; pull and retry rather than timing by hand."
        )
    if s.get("prewarm_load_s") is None:
        raise ProbeFailure(
            f"{tag}: prewarm did not run, so the model load is INSIDE the "
            "timing. That is the contamination that broke the first "
            "1-token probe."
        )
    return s


def _rows(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["query_id"]] = r.get("predicted_answer", "")
    return out


# --- probe 1: M1 batched vs sequential -----------------------------------


def probe_m1(out_dir: Path, n: int, batch_size: int, padded: int) -> dict:
    common = ["--system", "M1", "--benchmark", "multihop_rag",
              "--split", "validation", "--max-queries", str(n)]
    seq = _run(out_dir, "m1_sequential", *common)
    bat = _run(out_dir, "m1_batched", *common,
               "--batch-size", str(batch_size),
               "--max-padded-tokens", str(padded))

    # ASSERTION 1 — the two passes must have answered the SAME queries.
    seq_rows = _rows(out_dir / "m1_sequential.jsonl")
    bat_rows = _rows(out_dir / "m1_batched.jsonl")
    if set(seq_rows) != set(bat_rows):
        raise ProbeFailure(
            "M1: the two passes answered different query sets "
            f"({len(seq_rows)} vs {len(bat_rows)}); the ratio would be "
            "meaningless."
        )
    if len(seq_rows) != n:
        raise ProbeFailure(f"M1: expected {n} queries, got {len(seq_rows)}")

    # ASSERTION 2 — batching must actually have batched. If the runner
    # fell back to sequential (supports_batched_answer False, say) both
    # numbers would be identical and the "win" would be a measurement of
    # nothing.
    if bat.get("batch_size") is None:
        raise ProbeFailure("M1: batched pass recorded no batch_size")
    if abs(bat["s_per_query"] - seq["s_per_query"]) < 1e-6:
        raise ProbeFailure(
            "M1: batched and sequential timings are identical to the "
            "microsecond — the batched path almost certainly did not run."
        )

    changed = sum(1 for q in seq_rows if seq_rows[q] != bat_rows[q])
    return {
        "sequential_s_per_query": seq["s_per_query"],
        "batched_s_per_query": bat["s_per_query"],
        "speedup": round(seq["s_per_query"] / bat["s_per_query"], 2),
        "n_queries": len(seq_rows),
        # Reported, NOT asserted: batch composition can legitimately
        # change generated text at temperature 0.
        "answers_changed": changed,
        "answers_changed_pct": round(100 * changed / max(1, len(seq_rows)), 1),
        "seq_answer_score": seq["mean_answer_score"],
        "bat_answer_score": bat["mean_answer_score"],
    }


# --- probe 2: per-unit index build cost ----------------------------------


def probe_build(out_dir: Path, units: int) -> dict:
    """Measure index_s per unit. QASPER = many small per-paper corpora.

    `--max-new-tokens 1` makes the answers nearly free so the run is
    dominated by indexing; the per-unit index_s values are parsed from
    the runner's own progress lines.
    """
    results: dict[str, dict] = {}
    for system in ("M2", "M4"):
        tag = f"build_{system}"
        out = out_dir / f"{tag}.jsonl"
        cmd = [sys.executable, "-m", "src.eval.runner",
               "--system", system, "--benchmark", "qasper",
               "--split", "validation", "--max-units", str(units),
               "--max-new-tokens", "1", "--prewarm",
               "--output", str(out)]
        print(f"\n=== {tag} ===\n$ {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        sys.stdout.write(proc.stdout[-4000:])
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-4000:])
            raise ProbeFailure(f"{tag}: runner exited {proc.returncode}")

        index_times = [
            float(line.split("index_s=")[1].split()[0])
            for line in proc.stdout.splitlines() if "index_s=" in line
        ]
        # ASSERTION — one index_s per unit, or we are averaging over
        # something other than what we think.
        if len(index_times) != units:
            raise ProbeFailure(
                f"{tag}: parsed {len(index_times)} index_s values for "
                f"{units} units. Cannot report a per-build cost."
            )
        results[system] = {
            "n_units": len(index_times),
            "mean_index_s": round(sum(index_times) / len(index_times), 2),
            "max_index_s": round(max(index_times), 2),
            "min_index_s": round(min(index_times), 2),
        }
    return results


# --- probe 3: M9 vs M2 ----------------------------------------------------


def probe_m9(out_dir: Path, n: int) -> dict:
    common = ["--benchmark", "multihop_rag", "--split", "validation",
              "--max-queries", str(n)]
    m2 = _run(out_dir, "cost_M2", "--system", "M2", *common)
    m9 = _run(out_dir, "cost_M9", "--system", "M9", *common)
    if m2["n_queries_scored"] != m9["n_queries_scored"]:
        raise ProbeFailure("M9: the two systems answered different counts")
    return {
        "m2_s_per_query": m2["s_per_query"],
        "m9_s_per_query": m9["s_per_query"],
        "ratio": round(m9["s_per_query"] / m2["s_per_query"], 2),
        "n_queries": m2["n_queries_scored"],
    }


# --- probe 5: the depth curve, the decisive test of the 74-leaf threshold -


# PRE-REGISTERED PREDICTION. Written before the sweep runs so it cannot
# be fitted afterwards. docs/FINDING_SHALLOW_HIERARCHY.md derives, from
# the reference stop condition (a layer of <= reduction_dimension + 1 =
# 11 halts the build) and the paper's ~6.7 children per parent:
#
#     first leaf count with TWO summary layers = 11 * 6.7 = ~74
#
# CONFIRMS  : the 1 -> 2 transition falls in [70, 85].
# REFUTES   : it falls at <= 60 or >= 100. The ACCOUNT (shallow trees on
#             document-scale corpora) would survive either way -- what
#             dies is the MECHANISM, i.e. that layer size and branching
#             factor are what set the depth. A jump at 50 would mean
#             something else governs it.
# INCONCLUSIVE: no transition anywhere in 40-140, or a non-monotone one
#             (depth rising then falling), which would mean the build is
#             data-dependent enough that a single sweep cannot answer it.
DEPTH_PREDICTION = (70, 85)

# Dense THROUGH the predicted threshold, sparse outside it. The claim is
# a step function, so the resolution has to be finer than the effect --
# sampling every 20 leaves could straddle the step and see nothing.
DEPTH_SIZES = (40, 50, 60, 65, 70, 75, 80, 85, 90, 100, 120, 140)


def probe_depth_curve(out_dir: Path, trials: int) -> dict:
    """Build real corpora at controlled leaf counts; read n_layers off each.

    Reports EVERY corpus, never a mean. Averaging across a step function
    describes neither side of it.

    Two trials per size by default because the clustering is
    data-dependent (the GMM crash band is non-monotone in n), so a single
    corpus per size cannot distinguish a threshold from a coincidence.
    """
    from src.eval.multihop import MultiHopBenchmark
    from src.eval.types import CorpusItem
    from src.retrievers.m4_raptor import RaptorSystem

    unit = next(iter(MultiHopBenchmark().iter_eval_units(
        split="validation", max_units=1)))
    # ~100-token paragraphs, so one item == one leaf and the leaf count
    # is controlled rather than inferred.
    paras: list[str] = []
    for item in unit.corpus:
        words = (item.text or "").split()
        for i in range(0, len(words), 75):
            block = " ".join(words[i:i + 75])
            if len(block.split()) >= 60:
                paras.append(block)
    need = max(DEPTH_SIZES) * trials
    if len(paras) < need:
        raise ProbeFailure(f"depth_curve: need {need} paragraphs, have {len(paras)}")

    print("")
    print("=== depth curve ===")
    print(f"PRE-REGISTERED: 1->2 transition in {DEPTH_PREDICTION} CONFIRMS "
          "the mechanism; <=60 or >=100 REFUTES it.")
    print(f"{'target':>7} {'trial':>6} {'built':>6} {'layer1':>7} {'b':>5} "
          f"{'layers':>7}  layer_sizes")
    print("  (VERDICT KEYS ON 'built' AND 'layer1', NEVER ON 'target' -- "
          "target is paragraphs fed in, and the chunker splits them)")
    rows: list[dict] = []
    cursor = 0
    for size in DEPTH_SIZES:
        for t in range(trials):
            block = paras[cursor:cursor + size]
            cursor = (cursor + size) % (len(paras) - max(DEPTH_SIZES))
            items = [
                CorpusItem(item_id=f"d{size}_{t}_{j}", parent_id=f"d{size}_{t}_{j}",
                           span_id="<whole>", text=x)
                for j, x in enumerate(block)
            ]
            sysm = RaptorSystem()
            try:
                sysm.index_items(items)
            except Exception as e:
                raise ProbeFailure(
                    f"depth_curve: {size} leaves RAISED {type(e).__name__}: {e}"
                ) from e
            st = sysm.index_stats
            built = int(st.get("n_leaves", 0))
            layers = int(st.get("n_layers", 1))
            sizes = st.get("layer_sizes", {})
            # layer_sizes keys survive a JSON round-trip as strings.
            l1 = sizes.get(1, sizes.get("1"))
            l1 = int(l1) if l1 is not None else None
            rows.append({"target": size, "trial": t, "leaves": built,
                         "n_layers": layers, "layer_sizes": sizes,
                         "layer1_size": l1,
                         "branching": (built / l1) if l1 else None,
                         "bic_fit_failures": int(st.get("bic_fit_failures", 0)),
                         "summary_layers": max(0, layers - 1)})
            b = f"{built / l1:.2f}" if l1 else "-"
            print(f"{size:>7} {t:>6} {built:>6} {str(l1):>7} {b:>5} "
                  f"{layers:>7}  {sizes}")

    # --- THE EXACT TEST, and the one that should have been primary ------
    #
    # UNIT BUG, fixed: this keyed on `target` (the number of paragraphs
    # fed in) while the prediction was about BUILT leaves. They diverge
    # badly -- the 100-token chunker splits a ~100-token paragraph in two
    # whenever a sentence would overflow, so target 40 built 51 and 61.
    # Scoring a leaf-count prediction against a paragraph count is the
    # same class of error as MAP@K over chunks instead of documents, and
    # as the inverted tier labels. Third of its kind; see standing rule 2.
    #
    # More importantly, the leaf count was ALWAYS the wrong variable to
    # test. `build_paper_tree` breaks when `len(current) <=
    # reduction_dimension + 1`, so the mechanism's real claim is EXACT
    # and deterministic:
    #
    #     a second summary layer exists  IFF  layer 1 holds > 11 nodes
    #
    # That is a hard invariant with no fitted parameter in it. The
    # ~74-leaf figure is a NOISY CONSEQUENCE of it, because leaves ->
    # layer-1 size goes through the branching factor, which is
    # data-dependent (measured 5.3 to 6.8 across this sweep). Testing the
    # invariant tests the mechanism; testing the leaf threshold tests the
    # mechanism AND an assumed branching factor at once, and cannot say
    # which failed.
    stop_at = 11  # PaperTreeParams.reduction_dimension + 1
    invariant_failures = [
        r for r in rows
        if r["layer1_size"] is not None
        and (r["summary_layers"] >= 2) != (r["layer1_size"] > stop_at)
    ]

    two_plus = [r["leaves"] for r in rows if r["summary_layers"] >= 2]
    one_only = [r["leaves"] for r in rows if r["summary_layers"] < 2]
    # The transition band in BUILT leaves: highest 1-layer corpus below
    # the lowest 2-layer one. Reported as a band because the branching
    # factor varies, so there is no single crossing point.
    transition_lo = max([n for n in one_only if n < min(two_plus)], default=None) \
        if two_plus else None
    transition_hi = min(two_plus) if two_plus else None

    branchings = [r["branching"] for r in rows if r["branching"]]
    b_mean = sum(branchings) / len(branchings) if branchings else None

    lo, hi = DEPTH_PREDICTION
    # PAPER-BAND prediction, and this is NOT post-hoc: App. C reports
    # 5.7-6.8 children per parent. The registered point prediction used
    # 6.7 alone, near the top of that band. Scoring against the FULL
    # PUBLISHED BAND is legitimate because the band was published before
    # any of this; re-deriving the threshold from OUR measured branching
    # factor would not be, and is deliberately not done here.
    band_lo, band_hi = 11 * 5.7, 11 * 6.8

    if not two_plus:
        verdict = "INCONCLUSIVE - no 1->2 transition anywhere in the sweep"
    elif invariant_failures:
        verdict = (
            f"REFUTES THE MECHANISM - {len(invariant_failures)} corpora "
            "violate the exact invariant (2nd summary layer iff layer 1 > "
            f"{stop_at} nodes). The stop condition is not what governs "
            "depth."
        )
    else:
        core = (f"MECHANISM CONFIRMED EXACTLY - all {len(rows)} corpora obey "
                f"'2nd summary layer iff layer1 > {stop_at}', with no fitted "
                "parameter.")
        band = (f"transition band {transition_lo}-{transition_hi} built "
                f"leaves; paper's own 5.7-6.8 branching implies "
                f"{band_lo:.0f}-{band_hi:.0f}")
        if transition_hi is not None and band_lo <= transition_hi <= band_hi:
            band += " -> INSIDE the paper band."
        else:
            band += " -> OUTSIDE the paper band."
        reg = (f"REGISTERED POINT PREDICTION ({lo}-{hi}, from b=6.7 alone): "
               + ("MET" if (transition_hi is not None and lo <= transition_hi <= hi)
                  else "NOT MET - the registered range was too narrow, having "
                       "used one value from a published band rather than the "
                       "band"))
        verdict = f"{core}\n  {band}\n  {reg}"

    print("")
    print(verdict)
    if b_mean:
        print(f"  measured branching factor: {b_mean:.2f} "
              f"(range {min(branchings):.2f}-{max(branchings):.2f}) — "
              "REPORTED, NOT used to re-derive the threshold")
    if invariant_failures:
        for r in invariant_failures:
            print(f"  VIOLATION: {r['leaves']} leaves, layer1="
                  f"{r['layer1_size']}, summary_layers={r['summary_layers']}")
    return {"prediction": list(DEPTH_PREDICTION),
            "paper_band_threshold": [band_lo, band_hi],
            "transition_lo": transition_lo, "transition_hi": transition_hi,
            "invariant_violations": len(invariant_failures),
            "branching_mean": b_mean, "verdict": verdict, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
        # DRIVE-BACKED BY DEFAULT. A runtime restart has now destroyed probe
    # output twice, because /content is ephemeral. paths.output_dir()
    # resolves to Drive when it is mounted and falls back to local only
    # when it is not, so the default survives a restart wherever it can.
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Where to write probe artifacts. Default: "
             "<OUTPUT_DIR>/probes, which is Drive-backed when mounted. "
             "Pass an explicit path only if you accept losing it.")
    ap.add_argument("--m1-queries", type=int, default=200)
    ap.add_argument("--m1-batch-size", type=int, default=32)
    ap.add_argument("--m1-padded-tokens", type=int, default=20000)
    ap.add_argument("--build-units", type=int, default=20)
    ap.add_argument("--m9-queries", type=int, default=50)
    ap.add_argument("--crash-corpora", type=int, default=40)
    ap.add_argument("--depth-trials", type=int, default=2)
    ap.add_argument("--crash-per-corpus", type=int, default=10)
    # m9 skipped by DEFAULT as of 2026-08-05: the budget constraint was
    # lifted, so +-6 units is noise and the probe is not worth the GPU.
    ap.add_argument("--skip", nargs="*", default=["m9"],
                    choices=["m1", "build", "m9", "depth_curve"])
    args = ap.parse_args()
    if args.out is None:
        from src import paths

        args.out = paths.output_dir() / "probes"
        print(f"[probe] writing to {args.out} (Drive-backed if mounted; "
              "a runtime restart has destroyed /content output twice)")
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {}
    failures: dict = {}
    for name, fn in [
        ("m1", lambda: probe_m1(args.out, args.m1_queries,
                                args.m1_batch_size, args.m1_padded_tokens)),
        ("build", lambda: probe_build(args.out, args.build_units)),
        ("m9", lambda: probe_m9(args.out, args.m9_queries)),
        ("depth_curve", lambda: probe_depth_curve(args.out, args.depth_trials)),
    ]:
        if name in args.skip:
            continue
        try:
            report[name] = fn()
        except ProbeFailure as e:
            # A failed probe must not take the others down with it, and
            # must NOT leave a plausible-looking partial number behind.
            failures[name] = str(e)
            print(f"\n!!! PROBE {name.upper()} FAILED: {e}", flush=True)

    print("\n" + "=" * 66)
    print("PROBE REPORT")
    print("=" * 66)
    print(json.dumps(report, indent=2))
    if failures:
        print("\nFAILED (numbers NOT usable):")
        print(json.dumps(failures, indent=2))

    if "m1" in report:
        r = report["m1"]
        print(f"\nM1: {r['sequential_s_per_query']} -> "
              f"{r['batched_s_per_query']} s/query  = {r['speedup']}x")
        if r["speedup"] < 1.0:
            print("  *** BATCHED M1 IS SLOWER. This is the unexplained 4k "
                  "batching failure reproducing in a decode-dominated "
                  "regime. Do NOT keep the 4x projection; report it. ***")
        elif r["speedup"] < 2.0:
            print("  *** Speedup well below the 4x projection. The forecast "
                  "built on it needs re-deriving before Q is allocated. ***")
        if r["answers_changed"]:
            print(f"  note: {r['answers_changed_pct']}% of answers differ "
                  "between batched and sequential (expected; batch "
                  "composition can move argmax on near-ties)")

    (args.out / "probe_report.json").write_text(
        json.dumps({"report": report, "failures": failures}, indent=2)
    )
    print(f"\nwritten -> {args.out / 'probe_report.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
