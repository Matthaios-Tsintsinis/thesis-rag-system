"""Questions that share a document, tested a document at a time (OURS_VERSIONS "Full splits").

NarrativeQA's questions come about 30 to a story, and a pooled shard's 100 share one corpus, so
they rise and fall together. The clustered tests resample (bootstrap) and sign-flip
(permutation) a whole story or shard at a time; the declared per-question tests are untouched,
their numbers bit for bit what they were."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import report_ours as ro


class TestWhichDocument(unittest.TestCase):
    def test_a_narrativeqa_question_names_its_story(self):
        self.assertEqual(ro.story_of("0a1b2c::q17"), "0a1b2c")
        with self.assertRaisesRegex(ValueError, "story"):
            ro.story_of("q17")

    def test_the_clustered_benchmarks(self):
        shards = {"h1": "shard001", "h2": "shard002"}
        for benchmark in ("narrativeqa", "narrativeqa_test"):
            self.assertEqual(ro.clusters_of(benchmark)("s9::q0"), "s9")
        self.assertEqual(ro.clusters_of("hotpotqa_pooled", shards)("h2"), "shard002")
        for benchmark in ("hotpotqa", "hotpotqa_full", "multihop_rag"):
            self.assertIsNone(ro.clusters_of(benchmark, shards))
        self.assertEqual(ro.LEVEL["narrativeqa_test"], "story")
        self.assertNotIn("hotpotqa_pooled_full", ro.LEVEL)

    def test_a_pooled_question_outside_the_shards_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no shard"):
            ro.clusters_of("hotpotqa_pooled", {"h1": "shard001"})("h9")

    def test_shards_come_from_the_loaders_units(self):
        class Units:
            def iter_eval_units(self, *, split, max_units=None):
                self.split = split
                yield type("U", (), {"corpus_id": "s1", "queries": [type("Q", (), {"query_id": "a"})]})
                yield type("U", (), {"corpus_id": "s2", "queries": [type("Q", (), {"query_id": "b"})]})
        self.assertEqual(ro.shards_of(Units()), {"a": "s1", "b": "s2"})


class TestPairedKeys(unittest.TestCase):
    def test_the_same_differences_as_paired_with_their_questions(self):
        a = {"q2": 0.5, "q1": 0.0, "q3": None, "q4": 1.0}
        b = {"q1": 1.0, "q2": 0.5, "q3": 1.0, "q5": 0.0}
        keys, d = ro.paired_keys(a, b)
        self.assertEqual(keys, ["q1", "q2"])
        np.testing.assert_array_equal(d, ro.paired(a, b))
        np.testing.assert_array_equal(d, [1.0, 0.0])


def clustered_data(n_clusters=10, per=30, seed=7, noise=0.0):
    """Each cluster's questions share one difference (plus optional noise of their own)."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.05, 0.1, n_clusters)
    d = np.repeat(shared, per) + rng.normal(0.0, noise, n_clusters * per)
    labels = [f"c{i}" for i in range(n_clusters) for _ in range(per)]
    return d, labels


class TestClusteredTests(unittest.TestCase):
    def test_the_standard_error_by_hand(self):
        d = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        labels = ["a", "a", "b", "b", "b", "c"]
        # Cluster sums 2, 0, 1 over sizes 2, 3, 1; mean 0.5; residual sums 1, -1.5, 0.5.
        expected = np.sqrt(3 / 2 * (1.0 + 2.25 + 0.25)) / 6
        self.assertAlmostEqual(ro.cluster_se(d, labels), expected)

    def test_one_question_per_cluster_is_the_ordinary_bootstrap(self):
        d = np.random.default_rng(3).normal(0.02, 0.3, 400)
        labels = [f"q{i}" for i in range(400)]
        self.assertEqual(ro.cluster_bootstrap_ci(d, labels, np.random.default_rng(11)),
                         ro.bootstrap_ci(d, np.random.default_rng(11)))

    def test_questions_that_move_together_widen_the_interval(self):
        d, labels = clustered_data()
        lo, hi = ro.cluster_bootstrap_ci(d, labels, np.random.default_rng(1))
        qlo, qhi = ro.bootstrap_ci(d, np.random.default_rng(1))
        self.assertGreater((hi - lo) / (qhi - qlo), 3.0)
        se = ro.cluster_se(d, labels)
        deff = (se / (np.std(d, ddof=1) / np.sqrt(len(d)))) ** 2
        self.assertGreater(deff, 20.0)

    def test_the_permutation_flips_a_cluster_whole(self):
        d = np.full(300, 0.1)
        labels = [f"c{i // 30}" for i in range(300)]
        p = ro.cluster_permutation_p(d, labels, np.random.default_rng(2))
        # Two of the 1,024 sign patterns reach the observed mean: p near 2 / 1,024.
        self.assertGreater(p, 0.0005)
        self.assertLess(p, 0.01)
        self.assertLess(ro.permutation_p(d, np.random.default_rng(2)), 0.001)
        self.assertEqual(ro.cluster_permutation_p(np.zeros(60), labels[:60], np.random.default_rng(2)), 1.0)

    def test_the_mean_is_the_questions_mean(self):
        d = np.array([1.0, 1.0, 1.0, 0.0])
        labels = ["a", "a", "a", "b"]
        sums, counts = ro.cluster_totals(d, labels)
        np.testing.assert_array_equal(sums, [3.0, 0.0])
        np.testing.assert_array_equal(counts, [3, 1])


class TestClusteredFamily(unittest.TestCase):
    def _tests(self):
        rng = np.random.default_rng(5)
        stories = {f"s{i}::q{j}": float(v) for i in range(40) for j, v in
                   enumerate(np.clip(rng.normal(0.4, 0.2, 30), 0, 1))}
        shift = {f"s{i}": rng.normal(0.03, 0.1) for i in range(40)}
        ours_stories = {k: v + shift[k.split("::")[0]] for k, v in stories.items()}
        plain = {f"h{i}": float(v) for i, v in enumerate(rng.random(500))}
        ours_plain = {k: v + 0.05 for k, v in plain.items()}
        return [("nqa answer: ours vs M2", stories, ours_stories, 0.0214, ro.story_of),
                ("hotpot answer: ours vs M2", plain, ours_plain, None, None)]

    def test_clustered_rows_retested_the_rest_as_declared_and_holm_again(self):
        tests = self._tests()
        declared = ro.run_family([t[:4] for t in tests], np.random.default_rng(ro.SEED),
                                 directional=False)
        rows = ro.clustered_family(declared, tests, np.random.default_rng(ro.SEED), directional=False)
        story, plain = rows
        self.assertEqual((story["level"], story["n_clusters"], story["n"]), ("story", 40, 1200))
        self.assertAlmostEqual(story["delta"], declared[0]["delta"])
        self.assertGreater(story["hi"] - story["lo"], declared[0]["hi"] - declared[0]["lo"])
        self.assertGreater(story["deff"], 1.0)
        self.assertAlmostEqual(story["mde"], ro.MDE_Z * story["se"])
        self.assertFalse(story["few_clusters"])
        self.assertEqual(plain["level"], "question")
        for key in ("delta", "lo", "hi", "p", "mde"):
            self.assertEqual(plain[key], declared[1][key])
        self.assertEqual([r["p_holm"] for r in rows], ro.holm([story["p"], plain["p"]]))
        self.assertIn(story["verdict"], ("SUPPORTED", "marginal", "NOT SUPPORTED"))

    def test_few_clusters_are_flagged(self):
        base = {f"s{i}::q{j}": 0.5 + 0.1 * (j % 2) for i in range(10) for j in range(20)}
        ours = {k: v + 0.05 for k, v in base.items()}
        tests = [("ten stories", base, ours, None, ro.story_of)]
        declared = ro.run_family([t[:4] for t in tests], np.random.default_rng(0), directional=True)
        row = ro.clustered_family(declared, tests, np.random.default_rng(0), directional=True)[0]
        self.assertEqual(row["n_clusters"], 10)
        self.assertTrue(row["few_clusters"])


# --- the stats command end to end ---------------------------------------------------


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _row(qid, answer, retr=None):
    retrieval = ({"skipped": True} if retr is None else
                 {"skipped": False, "f1": retr, "map_at_k": {"10": retr}})
    return {"query_id": qid, "answer": {"value": answer, "method": "token_f1"},
            "retrieval": retrieval, "metadata": {}}


QUESTIONS = {"multihop_rag": [f"m{i}" for i in range(60)],
             "narrativeqa": [f"s{i}::q{j}" for i in range(12) for j in range(5)],
             "hotpotqa": [f"h{i}" for i in range(60)],
             "hotpotqa_pooled": [f"h{i}" for i in range(60)]}
SHARDS = {f"h{i}": f"shard{i // 20:03d}" for i in range(60)}


def _banks(root: Path):
    rng = np.random.default_rng(9)
    p11, ours = root / "p11", root / "ours_v20"
    for benchmark, qids in QUESTIONS.items():
        scored = benchmark != "narrativeqa"
        for system, bank, lift in [(s, p11, 0.0) for s in ro.SYSTEMS] + [("OURS", ours, 0.08)]:
            rows = [_row(q, float(rng.random() + lift), float(rng.random() + lift) if scored else None)
                    for q in qids]
            _write(bank / f"{benchmark}_{system}_validation.jsonl", rows)
    return p11, ours


class TestStatsCommand(unittest.TestCase):
    def test_the_declared_families_as_before_then_the_clustered_ones(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as td:
            p11, ours = _banks(Path(td))
            with contextlib.redirect_stdout(io.StringIO()):
                rows = ro.stats(p11, ours, shard_of=SHARDS)
            text = (ours / "STATS_OURS.txt").read_text(encoding="utf-8")

            def scores(bank, benchmark, system, metric):
                return ro.row_scores(bank / f"{benchmark}_{system}_validation.jsonl", metric)
            bars = {("multihop_rag", "retrieval"): ro.PUBLISHED_TARGETS[("multihop_rag", "retrieval")],
                    ("narrativeqa", "answer"): ro.PUBLISHED_TARGETS[("narrativeqa", "answer")],
                    ("hotpotqa_pooled", "map10"): ro.POOLED_MAP10_BAR}
            rng = np.random.default_rng(ro.SEED)
            family_o = ro.run_family(
                [(f"{b} {m}: ours vs {c}", scores(p11, b, c, m), scores(ours, b, "OURS", m), bars[(b, m)])
                 for b, m, c in ro.FAMILY_O], rng, directional=True)
            family_x = ro.run_family(
                [(f"{b} {m}: ours vs {c}", scores(p11, b, c, m), scores(ours, b, "OURS", m), None)
                 for b, m in ro.GRID_FAMILIES for c in ro.SYSTEMS], rng, directional=False)
        # The declared families: their numbers are what the declared code alone gives.
        self.assertEqual(len(rows), 2 * (3 + 28))
        for got, want in zip(rows[:31], family_o + family_x):
            self.assertEqual({k: got[k] for k in want}, want)
        clustered_o, clustered_x = rows[31:34], rows[34:]
        self.assertEqual([r["level"] for r in clustered_o], ["question", "story", "shard"])
        by_name = {r["name"]: r for r in clustered_x}
        self.assertEqual(by_name["narrativeqa answer: ours vs M2"]["n_clusters"], 12)
        self.assertEqual(by_name["hotpotqa_pooled retrieval: ours vs M3"]["n_clusters"], 3)
        self.assertEqual(by_name["hotpotqa answer: ours vs M1"]["level"], "question")
        self.assertTrue(text.startswith("=== Family O (declared, Holm m = 3) ==="))
        self.assertIn("=== Family O, story and shard level (Holm m = 3) ===", text)
        self.assertIn("=== Family O-x, story and shard level (Holm m = 28) ===", text)
        self.assertIn("few clusters", text)


if __name__ == "__main__":
    unittest.main()
