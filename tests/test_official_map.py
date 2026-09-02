"""official_map: the official-numerator MAP@10 derived from replay sidecars.

The oracle is a VERBATIM copy of the official `calculate_metrics`
(yixuantt/MultiHop-RAG :: retrieval_evaluate.py @ cde8e844, fetched
2026-09-02), executed here against `official_ap_at_k` on random
document-level cases -- the second audit's AF2-1 finding, pinned so the
transcription cannot drift from the source it claims to transcribe.
The standard path is asserted to be THE FROZEN SCORER (a spy on
`score_retrieval_rank_aware`), not a copy of it.
"""

from __future__ import annotations

import fnmatch
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts.export_matrix import QWEN
from scripts.official_map import (
    COLUMNS,
    LABEL,
    checksum_line,
    derive_cell,
    official_ap_at_k,
    standard_ap_at_k,
    write_csv,
)


def official_calculate_metrics(retrieved_lists, gold_lists):
    """VERBATIM from yixuantt/MultiHop-RAG :: retrieval_evaluate.py ::
    calculate_metrics, revision cde8e844af14b3012f20158abc2854fe8458212a.
    Do not edit; this is the oracle."""
    hits_at_10_count = 0
    hits_at_4_count = 0
    map_at_10_list = []
    mrr_list = []

    for retrieved, gold in zip(retrieved_lists, gold_lists):
        hits_at_10_flag = False
        hits_at_4_flag = False
        average_precision_sum = 0
        first_relevant_rank = None
        find_gold = []

        gold = [item.replace(" ", "").replace("\n", "") for item in gold]
        retrieved = [item.replace(" ", "").replace("\n", "") for item in retrieved]

        for rank, retrieved_item in enumerate(retrieved[:11], start=1):
            if any(gold_item in retrieved_item for gold_item in gold):
                if rank <= 10:
                    hits_at_10_flag = True
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    if rank <= 4:
                        hits_at_4_flag = True
                    # Compute precision at this rank for this query
                    count = 0
                    for gold_item in gold:
                        if gold_item in retrieved_item and not gold_item in find_gold:
                            count =  count + 1
                            find_gold.append(gold_item)
                    precision_at_rank = count / rank
                    average_precision_sum += precision_at_rank

        # Calculate metrics for this query
        hits_at_10_count += int(hits_at_10_flag)
        hits_at_4_count += int(hits_at_4_flag)
        map_at_10_list.append(average_precision_sum / min(len(gold), 10))
        mrr_list.append(1 / first_relevant_rank if first_relevant_rank else 0)

    # Calculate average metrics over all queries
    hits_at_10 = hits_at_10_count / len(gold_lists)
    hits_at_4 = hits_at_4_count / len(gold_lists)
    map_at_10 = sum(map_at_10_list) / len(gold_lists)
    mrr_at_10 = sum(mrr_list) / len(gold_lists)

    return {
        'Hits@10': hits_at_10,
        'Hits@4': hits_at_4,
        'MAP@10': map_at_10,
        'MRR@10': mrr_at_10,
    }


def _atoms(names):
    return [(n, "<whole>") for n in names]


class TestOfficialNumerator(unittest.TestCase):
    def test_matches_the_official_script_on_random_document_cases(self):
        # Document ids are single tokens with no spaces, so the official
        # substring test reduces to identity and the text-level official
        # function scores exactly the document-level case.
        rng = random.Random(20260903)
        for _ in range(5000):
            docs = [f"doc{i:02d}" for i in range(30)]
            gold = rng.sample(docs, rng.choice([2, 3, 4]))
            ranking = rng.sample(docs, rng.randint(1, 20))
            off = official_calculate_metrics([ranking], [gold])["MAP@10"]
            ours = official_ap_at_k(_atoms(ranking), set(_atoms(gold)))
            self.assertAlmostEqual(off, ours, places=12)

    def test_worked_examples_standard_vs_official(self):
        g = set(_atoms(["D1", "D2"]))
        r = _atoms(["D3", "D1", "D2"])
        self.assertAlmostEqual(official_ap_at_k(r, g), 5 / 12, places=12)
        self.assertAlmostEqual(standard_ap_at_k(r, g), 7 / 12, places=12)
        g2 = set(_atoms(["A", "B"]))
        r2 = _atoms(["A", "B", "C"])
        self.assertAlmostEqual(official_ap_at_k(r2, g2), 0.75, places=12)
        self.assertAlmostEqual(standard_ap_at_k(r2, g2), 1.0, places=12)
        # a single gold: the two numerators coincide
        g3 = set(_atoms(["Z"]))
        r3 = _atoms(["X", "Y", "Z"])
        self.assertAlmostEqual(official_ap_at_k(r3, g3), 1 / 3, places=12)
        self.assertAlmostEqual(standard_ap_at_k(r3, g3), 1 / 3, places=12)

    def test_official_never_exceeds_standard(self):
        rng = random.Random(7)
        for _ in range(2000):
            docs = [f"d{i}" for i in range(25)]
            gold = set(_atoms(rng.sample(docs, rng.choice([2, 3, 4]))))
            ranking = _atoms(rng.sample(docs, rng.randint(1, 15)))
            self.assertLessEqual(official_ap_at_k(ranking, gold),
                                 standard_ap_at_k(ranking, gold) + 1e-12)

    def test_standard_path_is_the_frozen_scorer(self):
        import src.eval.alignment as alignment
        real = alignment.score_retrieval_rank_aware
        with mock.patch.object(alignment, "score_retrieval_rank_aware",
                               side_effect=real) as spy:
            value = standard_ap_at_k(_atoms(["D3", "D1", "D2"]),
                                     set(_atoms(["D1", "D2"])))
        self.assertEqual(spy.call_count, 1)
        self.assertAlmostEqual(value, 7 / 12, places=12)


def _write_bank(bank: Path, *, banked_map_override=None,
                drop_jsonl=False):
    stem = "multihop_rag_M2_validation"
    summary = {"system": "M2", "benchmark": "multihop_rag",
               "generator": QWEN, "partial_run": False,
               "n_queries_scored": 3, "n_answerable": 2}
    (bank / f"{stem}.summary.json").write_text(json.dumps(summary),
                                               encoding="utf-8")
    cases = {
        "q0": (["D3", "D1", "D2"], ["D1", "D2"]),   # std 7/12, off 5/12
        "q1": (["A", "B", "C"], ["A", "B"]),         # std 1.0, off 0.75
    }
    if not drop_jsonl:
        with (bank / f"{stem}.jsonl").open("w", encoding="utf-8") as f:
            for qid, (ranking, gold) in cases.items():
                std = standard_ap_at_k(_atoms(ranking), set(_atoms(gold)))
                if banked_map_override is not None and qid == "q0":
                    std = banked_map_override
                f.write(json.dumps({
                    "query_id": qid,
                    "retrieval": {"skipped": False,
                                  "map_at_k": {"1": 0.0, "10": std}},
                }) + "\n")
            f.write(json.dumps({"query_id": "qnull",
                                "retrieval": {"skipped": True}}) + "\n")
    with (bank / f"rankings.{stem}.jsonl").open("w", encoding="utf-8") as f:
        for qid, (ranking, gold) in cases.items():
            f.write(json.dumps({
                "query_id": qid, "n_gold": len(gold),
                "gold": sorted([g, "<whole>"] for g in gold),
                "doc_ranking": [[d, "<whole>"] for d in ranking],
                "recall_at_k": {"1": 0.0, "5": 1.0, "10": 1.0},
            }) + "\n")
    return stem


class TestDeriveCell(unittest.TestCase):
    def test_end_to_end_row_and_csv_only(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank)
            row = derive_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertEqual(row["n_rows"], 2)
            self.assertAlmostEqual(float(row["map_at_10_standard_banked"]),
                                   (7 / 12 + 1.0) / 2, places=12)
            self.assertAlmostEqual(
                float(row["map_at_10_standard_from_sidecar"]),
                (7 / 12 + 1.0) / 2, places=12)
            self.assertAlmostEqual(
                float(row["map_at_10_official_numerator"]),
                (5 / 12 + 0.75) / 2, places=12)
            self.assertEqual(row["label"], LABEL)
            out = Path(td) / "out"
            path = write_csv([row], out)
            self.assertEqual(path.name, "MAP_OFFICIAL.csv")
            self.assertEqual([p.name for p in out.iterdir()],
                             ["MAP_OFFICIAL.csv"])      # no Markdown, ever
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header, ",".join(COLUMNS))
            self.assertIn("rows=1", checksum_line([row]))

    def test_gate_refuses_when_sidecar_disagrees_with_the_bank(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, banked_map_override=0.5)
            with self.assertRaises(SystemExit) as cm:
                derive_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertIn("GATE FAILED", str(cm.exception))

    def test_refuses_without_banked_rows(self):
        with TemporaryDirectory() as td:
            bank = Path(td)
            _write_bank(bank, drop_jsonl=True)
            with self.assertRaises(SystemExit) as cm:
                derive_cell(bank, QWEN, "multihop_rag", "M2")
            self.assertIn("banked rows missing", str(cm.exception))

    def test_output_name_is_glob_safe(self):
        for pattern in ("*.summary.json", "multihop_rag_M2_*.jsonl",
                        "*_validation*.jsonl", "rankings.*"):
            self.assertFalse(fnmatch.fnmatch("MAP_OFFICIAL.csv", pattern),
                             pattern)


if __name__ == "__main__":
    unittest.main()
