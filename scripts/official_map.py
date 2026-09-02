"""Official-numerator MAP@10 from the replay sidecars. CSV-ONLY, LABELLED,
NEVER TABLED.

WHY. The second fidelity audit (2026-09-02, AF2-1) executed the fetched
MultiHop-RAG `retrieval_evaluate.py` against `score_retrieval_rank_aware`
and found that the official MAP@10 numerator is NOT average precision:
at each relevant rank it adds (gold items newly matched at that rank) /
rank, i.e. 1/rank per newly-found gold, where standard AP adds
(relevant so far) / rank. The harness computes standard AP, which is
frozen mid-matrix and declared in the living record (section 3.1). This
tool derives the official-numerator figure POST-HOC from artifacts the
replay already banked -- every ranked cell's per-row document ranking
and gold set sit in `rankings.<stem>.jsonl` -- so a paper-comparable
MAP@10 costs no re-run.

THE GATE, before any official figure is written: for every sidecar row
the STANDARD MAP@10 is recomputed from the sidecar's document ranking
through the FROZEN scorer (`score_retrieval_rank_aware`, never a copy)
and asserted equal to the banked row's `retrieval.map_at_k["10"]`. That
binds the official column to the very ranking the bank was scored on;
the replay gate covered hit@K and MRR per row, and this closes MAP. Any
disagreement refuses the cell and names the row.

UNIT. The sidecars hold DOCUMENTS (MultiHop: article URL atoms; HotpotQA:
title atoms), so the official numerator is applied at the document unit:
a retrieved item is one document and `count` is 1 for a gold document
not yet found. The gold-unit deviation of the living record (AF2-2:
official gold is fact-counted, ours document-counted, 168 MultiHop
queries differ) is NOT undone here and the label says so.

OUTPUT. `<out>/MAP_OFFICIAL.csv` only -- no Markdown, by ruling: this
figure exists so the thesis never sets our MAP beside the paper's table
without it, not to be tabled itself. The name is glob-safe against every
bank-discovery pattern. A checksum line is printed.

    python -m scripts.official_map \\
      --p10 /content/drive/MyDrive/thesis_rag/outputs/p10 \\
      --p11 /content/drive/MyDrive/thesis_rag/outputs/p11 \\
      --out /content/drive/MyDrive/thesis_rag/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_matrix import LLAMA, QWEN, _fail, _generating_commit
from scripts.replay_retrieval import RANKED_BENCHMARKS, RANKED_SYSTEMS

K = 10
LABEL = (
    "official numerator: sum over newly-found gold documents of 1/rank "
    "within the top-10, divided by min(n_gold, 10); document unit "
    "(gold-unit deviation AF2-2 still applies); declared deviation AF2-1, "
    "living record 3.1; NOT tabled"
)
COLUMNS = [
    "generator", "benchmark", "system", "n_rows",
    "map_at_10_standard_banked", "map_at_10_standard_from_sidecar",
    "map_at_10_official_numerator", "standard_minus_official", "label",
]


def official_ap_at_k(doc_ranking, gold, k: int = K) -> float:
    """The official MultiHop-RAG MAP numerator, at the document unit.

    Transcribed from `yixuantt/MultiHop-RAG :: retrieval_evaluate.py ::
    calculate_metrics` at revision `cde8e844` (fetched 2026-09-02):

        for rank, retrieved_item in enumerate(retrieved[:11], start=1):
            if any(gold_item in retrieved_item for gold_item in gold):
                if rank <= 10:
                    ...
                    count = 0
                    for gold_item in gold:
                        if gold_item in retrieved_item and not gold_item in find_gold:
                            count = count + 1
                            find_gold.append(gold_item)
                    precision_at_rank = count / rank
                    average_precision_sum += precision_at_rank
        map_at_10_list.append(average_precision_sum / min(len(gold), 10))

    At the document unit a retrieved item IS one document, so `count` is
    1 when that document is gold and not yet found, else 0: each
    newly-found gold contributes 1/rank. `tests/test_official_map.py`
    executes a verbatim copy of the official function against this one
    on random document-level cases.
    """
    gold = set(gold)
    if not gold:
        return 0.0
    found: set = set()
    total = 0.0
    for rank, doc in enumerate(list(doc_ranking)[:k], start=1):
        if doc in gold and doc not in found:
            found.add(doc)
            total += 1.0 / rank
    return total / min(len(gold), k)


def standard_ap_at_k(doc_ranking, gold, k: int = K) -> float:
    """Standard AP@K through THE FROZEN SCORER, never a re-implementation.

    Wraps the document ranking in provenance-carrying chunks and calls
    `src.eval.alignment.score_retrieval_rank_aware`, so this is the exact
    quantity every banked row carries under `retrieval.map_at_k`.
    """
    from src.chunking import Chunk
    from src.eval.alignment import score_retrieval_rank_aware
    from src.retrievers.base import RetrievedChunk

    ranked = [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"d{i}", doc_id=str(d[0]), text="",
                        n_words=0, position=i,
                        gold_provenance=(tuple(d),)),
            score=1.0, rank=i)
        for i, d in enumerate(doc_ranking)
    ]
    out = score_retrieval_rank_aware(ranked, frozenset(gold), k_values=(k,))
    if out.get("skipped"):
        return 0.0
    return float(out["map_at_k"][k])


def _atom(x) -> tuple[str, str]:
    a, b = x
    return (str(a), str(b))


def _banked_rows(jpath: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with jpath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[str(r["query_id"])] = r
    return rows


def derive_cell(bank: Path, generator: str, benchmark: str,
                system: str) -> dict:
    """One CSV row for one ranked cell, gated per row against the bank."""
    stem = f"{benchmark}_{system}_validation"
    spath = bank / f"{stem}.summary.json"
    jpath = bank / f"{stem}.jsonl"
    side = bank / f"rankings.{stem}.jsonl"
    for p, what in ((spath, "summary"), (jpath, "banked rows"),
                    (side, "replay sidecar")):
        if not p.is_file():
            _fail(f"{stem}: {what} missing at {p} -- nothing to gate "
                  "against; run scripts.replay_retrieval first if the "
                  "sidecar is the missing piece")
    summary = json.loads(spath.read_text(encoding="utf-8"))
    if summary.get("partial_run"):
        _fail(f"{stem}: partial_run")
    if summary.get("generator") != generator:
        _fail(f"{stem}: summary generator {summary.get('generator')!r} != "
              f"bank's {generator!r}")
    banked = _banked_rows(jpath)

    n = 0
    sum_banked = sum_std = sum_off = 0.0
    with side.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r["query_id"])
            gold = frozenset(_atom(g) for g in r["gold"])
            ranking = [_atom(d) for d in r["doc_ranking"]]
            brow = banked.get(qid)
            if brow is None:
                _fail(f"{stem}: sidecar row {qid!r} has no banked row")
            mk = (brow.get("retrieval") or {}).get("map_at_k") or {}
            banked_map = mk.get(str(K), mk.get(K))
            if banked_map is None:
                _fail(f"{stem}: banked row {qid!r} carries no "
                      f"map_at_k[{K}] -- cannot gate the sidecar ranking")
            std = standard_ap_at_k(ranking, gold)
            if float(banked_map) != std:
                _fail(f"{stem}: GATE FAILED on {qid!r}: standard MAP@{K} "
                      f"from the sidecar ranking {std!r} != banked "
                      f"{float(banked_map)!r} -- the sidecar is not the "
                      "ranking the bank was scored on; refusing the cell")
            off = official_ap_at_k(ranking, gold)
            n += 1
            sum_banked += float(banked_map)
            sum_std += std
            sum_off += off

    expected = (int(summary["n_answerable"]) if benchmark == "multihop_rag"
                else int(summary["n_queries_scored"]))
    if n != expected:
        _fail(f"{stem}: sidecar holds {n} rows, expected {expected}")
    if sum_off > sum_std + 1e-12:
        _fail(f"{stem}: official numerator {sum_off / n} exceeds standard "
              f"AP {sum_std / n} -- impossible (ours is never lower); a "
              "bug, not a finding")
    return {
        "generator": generator, "benchmark": benchmark, "system": system,
        "n_rows": n,
        "map_at_10_standard_banked": repr(sum_banked / n),
        "map_at_10_standard_from_sidecar": repr(sum_std / n),
        "map_at_10_official_numerator": repr(sum_off / n),
        "standard_minus_official": repr((sum_std - sum_off) / n),
        "label": LABEL,
    }


def checksum_line(rows: list[dict]) -> str:
    nn = {c: sum(1 for r in rows if str(r[c]) != "") for c in COLUMNS}
    return ("MAP_OFFICIAL checksum: rows=%d non-null " % len(rows)
            + " ".join(f"{c}:{nn[c]}" for c in COLUMNS))


def write_csv(rows: list[dict], out_dir: Path) -> Path:
    """CSV only. No Markdown is written, by ruling (never tabled)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "MAP_OFFICIAL.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p10", required=True)
    ap.add_argument("--p11", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None,
                    help="one cell: {p10|p11}:BENCH:SYS")
    args = ap.parse_args()
    banks = {"p10": (QWEN, Path(args.p10)), "p11": (LLAMA, Path(args.p11))}
    targets = []
    if args.only:
        tag, bench, sysid = args.only.split(":")
        targets.append((banks[tag][0], banks[tag][1], bench, sysid))
    else:
        for tag in ("p10", "p11"):
            gen, bank = banks[tag]
            for bench in RANKED_BENCHMARKS:
                for sysid in RANKED_SYSTEMS:
                    targets.append((gen, bank, bench, sysid))
    rows = [derive_cell(bank, gen, bench, sysid)
            for gen, bank, bench, sysid in targets]
    path = write_csv(rows, Path(args.out))
    print(f"[map] wrote {path} (generated @ {_generating_commit()}; "
          "CSV only, never tabled)")
    print(checksum_line(rows))


if __name__ == "__main__":
    main()
