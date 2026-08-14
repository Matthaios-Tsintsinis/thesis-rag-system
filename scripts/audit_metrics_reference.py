"""INDEPENDENT metric verification. Written from the textbook definitions,
NOT from the repo's code. Compares against src/eval/* on synthetic cases.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter

from src.chunking import Chunk
from src.eval.alignment import score_retrieval_ck2, score_retrieval_rank_aware
from src.eval.scorers.extractive import normalize_qasper_answer, token_f1
from src.retrievers.base import RetrievedChunk

FAILURES: list[str] = []
CHECKS = [0]


def check(name, expected, actual, tol=1e-12):
    CHECKS[0] += 1
    ok = (abs(expected - actual) <= tol) if isinstance(expected, float) else expected == actual
    if not ok:
        FAILURES.append(f"{name}: expected {expected!r}, got {actual!r}")
    print(f"  {'ok ' if ok else 'FAIL'} {name:<52} exp={expected!r:<22} got={actual!r}")


# ---------------------------------------------------------------- reference
def ref_set_prf(retrieved_atoms: set, gold: set):
    """Textbook set precision/recall/F1."""
    if not gold:
        return None
    inter = len(retrieved_atoms & gold)
    r = inter / len(gold)
    p = inter / len(retrieved_atoms) if retrieved_atoms else 0.0
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return r, p, f


def ref_rank_metrics(doc_ranking: list, gold: set, ks):
    """Textbook Hit@K / AP@K / RR over a DOCUMENT ranking."""
    rel = [d in gold for d in doc_ranking]
    rr = 0.0
    for i, x in enumerate(rel):
        if x:
            rr = 1.0 / (i + 1)
            break
    hit, ap = {}, {}
    for k in ks:
        top = rel[:k]
        hit[k] = 1.0 if any(top) else 0.0
        num, hits = 0.0, 0
        for i, x in enumerate(top):
            if x:
                hits += 1
                num += hits / (i + 1)
        ap[k] = num / min(k, len(gold)) if gold else 0.0
    return hit, ap, rr


def ref_token_f1(pred: str, gold: str) -> float:
    """SQuAD token F1, transcribed from the OFFICIAL evaluators.

    The reference moves to match the published implementations, never the
    reverse. Composition is `white_space_fix(remove_articles(remove_punc(
    lower(s))))`, identical in `hotpot_evaluate_v1.py` and SQuAD 2.0's
    `evaluate-v2.0.py`. The no-answer branch is SQuAD's
    `return int(gold_toks == pred_toks)`; HotpotQA has no such branch and
    returns 0 there, a divergence recorded in tests/test_normalisation.py
    and unreachable here.
    """
    def norm(s):
        s = (s or "").lower()
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------- helpers
def rc(atoms, rank=0):
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"c{rank}", doc_id="d", text="t", n_words=1,
                    position=rank, gold_provenance=tuple(atoms)),
        score=1.0 - rank * 0.01, rank=rank)


def A(name):  # atom helper
    return (name, "<whole>")


print("=" * 78)
print("TEST 1 — set-F1 (score_retrieval_ck2) vs independent reference")
print("=" * 78)
cases = {
    "A gold=[A] retr=[A,B,C]": ([A("A"), A("B"), A("C")], {A("A")}),
    "B gold=[A,B] retr=[C,A,B]": ([A("C"), A("A"), A("B")], {A("A"), A("B")}),
    "C gold=[A,B] retr=[B,C,D]": ([A("B"), A("C"), A("D")], {A("A"), A("B")}),
    "D gold=[A,B] retr=[C,D,E]": ([A("C"), A("D"), A("E")], {A("A"), A("B")}),
}
for label, (retr, gold) in cases.items():
    got = score_retrieval_ck2([rc([a], i) for i, a in enumerate(retr)], (frozenset(gold),))
    r, p, f = ref_set_prf(set(retr), gold)
    check(f"{label} recall", r, got.recall)
    check(f"{label} precision", p, got.precision)
    check(f"{label} f1", f, got.f1)

print("\n-- edge cases --")
got = score_retrieval_ck2([], (frozenset({A("A")}),))
check("empty retrieval -> recall 0", 0.0, got.recall)
check("empty retrieval -> precision 0", 0.0, got.precision)
check("empty retrieval -> f1 0", 0.0, got.f1)
check("empty retrieval NOT skipped", False, got.skipped)

got = score_retrieval_ck2([rc([A("A")])], (frozenset(),))
check("empty gold -> skipped", True, got.skipped)

got = score_retrieval_ck2([rc([A("A")]), rc([A("A")], 1)], (frozenset({A("A")}),))
check("duplicate retrieved atom dedup precision", 1.0, got.precision)
check("duplicate retrieved atom dedup recall", 1.0, got.recall)

got = score_retrieval_ck2([rc([A("A")])],
                          (frozenset({A("A"), A("B")}), frozenset({A("A")})))
check("max-over-annotators picks best f1", 1.0, got.f1)
check("max-over-annotators n_gold from best", 1, got.n_gold)

print("=" * 78)
print("TEST 2 — rank-aware (Hit@K / MAP@K / MRR) vs independent reference")
print("=" * 78)
rank_cases = {
    "A gold=[A] retr=[A,B,C]": ([A("A"), A("B"), A("C")], {A("A")}),
    "B gold=[A,B] retr=[C,A,B]": ([A("C"), A("A"), A("B")], {A("A"), A("B")}),
    "C gold=[A,B] retr=[B,C,D]": ([A("B"), A("C"), A("D")], {A("A"), A("B")}),
    "D gold=[A,B] retr=[C,D,E]": ([A("C"), A("D"), A("E")], {A("A"), A("B")}),
}
KS = (1, 5, 10)
for label, (retr, gold) in rank_cases.items():
    got = score_retrieval_rank_aware([rc([a], i) for i, a in enumerate(retr)],
                                     frozenset(gold), k_values=KS)
    hit, ap, rr = ref_rank_metrics(retr, gold, KS)
    check(f"{label} mrr", rr, got["mrr"])
    for k in KS:
        check(f"{label} hit@{k}", hit[k], got["hit_at_k"][k])
        check(f"{label} map@{k}", ap[k], got["map_at_k"][k])

print("\n-- MAP>1 regression (many chunks per gold doc) --")
chunks = [rc([A("A")], i) for i in range(5)] + [rc([A("B")], 5)]
got = score_retrieval_rank_aware(chunks, frozenset({A("A"), A("B")}), k_values=(1, 5, 10))
for k in (1, 5, 10):
    inb = 0.0 <= got["map_at_k"][k] <= 1.0
    check(f"map@{k} within [0,1] with 5 chunks of one doc", True, inb)
check("dedup: 2 docs ranked from 6 chunks", 2, got["n_docs_ranked"])
check("map@10 == 1.0 (both gold first, deduped)", 1.0, got["map_at_k"][10])

print("\n-- sanity invariants --")
got = score_retrieval_rank_aware([rc([A("Z")], 0), rc([A("A")], 1)],
                                 frozenset({A("A")}), k_values=(1, 2, 5))
check("rank1 non-gold, rank2 gold -> mrr=1/2", 0.5, got["mrr"])
check("hit@1 = 0", 0.0, got["hit_at_k"][1])
check("hit@2 = 1", 1.0, got["hit_at_k"][2])
hits = [score_retrieval_rank_aware([rc([A("Z")], 0), rc([A("A")], 1)],
        frozenset({A("A")}), k_values=(k,))["hit_at_k"][k] for k in (1, 2, 3, 4, 5)]
check("hit@K monotone non-decreasing in K", True, all(x <= y for x, y in zip(hits, hits[1:])))
got = score_retrieval_rank_aware([rc([A("Z")], 0)], frozenset({A("A")}), k_values=(10,))
check("no relevant -> mrr 0", 0.0, got["mrr"])
check("no relevant -> map 0", 0.0, got["map_at_k"][10])
got = score_retrieval_rank_aware([rc([A("A")], 0)], frozenset({A("A")}), k_values=(10,))
check("rank1 relevant -> rr = 1", 1.0, got["mrr"])
check("K > n_retrieved does not crash", 1.0, got["map_at_k"][10])
got = score_retrieval_rank_aware([rc([], 0)], frozenset({A("A")}), k_values=(5,))
check("summary node (empty provenance) ranks no doc", 0, got["n_docs_ranked"])
check("summary-only retrieval -> map 0", 0.0, got["map_at_k"][5])

print("=" * 78)
print("TEST 3 — token F1 / EM vs independent SQuAD reference")
print("=" * 78)
pairs = [
    ("the cat sat", "cat sat", None),
    ("Paris", "paris", None),
    ("The Answer, is: 42.", "answer is 42", None),
    ("", "something", None),
    ("", "", None),
    ("a an the", "the a an", None),
    ("no answer available", "Sam Bankman-Fried", None),
    ("Sam Bankman-Fried", "Sam Bankman-Fried", None),
    ("completely different", "nothing alike here", None),
]
for pred, gold, _ in pairs:
    check(f"token_f1({pred[:22]!r},{gold[:18]!r})", ref_token_f1(pred, gold),
          token_f1(pred, gold), tol=1e-12)

check("normaliser strips articles", "cat sat on mat", normalize_qasper_answer("The cat sat on a mat"))
check("normaliser drops punctuation", "hello world", normalize_qasper_answer("Hello, World!"))
check("token_f1 in [0,1] for all pairs", True,
      all(0.0 <= token_f1(p, g) <= 1.0 for p, g, _ in pairs))

print("\n-- token_f1 asymmetry / degenerate --")
check("both empty -> 1.0 (vacuous)", 1.0, token_f1("", ""))
check("punctuation-only pred vs real gold -> 0", 0.0, token_f1("...", "answer"))
check("punctuation-only BOTH -> 1.0 (SQuAD no-answer rule)", 1.0,
      token_f1("...", "!!!"))

print()
print("=" * 78)
print(f"{CHECKS[0]} checks, {len(FAILURES)} failures")
for f in FAILURES:
    print("  FAIL:", f)
print("=" * 78)
