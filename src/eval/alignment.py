"""Retrieval scoring over gold-passage atoms: set-F1 and rank-aware metrics.
Reads only chunk.gold_provenance, the (parent_id, span_id) pairs a chunk
touches, never chunk text, so the scores do not depend on the chunker.
"""

from __future__ import annotations

from typing import Sequence

from ..retrievers.base import RetrievedChunk
from .types import RetrievalScore


def _f1(intersect: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    """Recall, precision and F1 of the atom overlap; all zero on empty gold."""
    if n_gold == 0:
        # The caller skips empty annotators; this is a guard, not a score.
        return 0.0, 0.0, 0.0
    recall = intersect / n_gold
    precision = intersect / max(1, n_pred)
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return recall, precision, f1


def _retrieved_atoms(
    retrieved: Sequence[RetrievedChunk],
) -> frozenset[tuple[str, str]]:
    """Union of (parent_id, span_id) atoms across the retrieved chunks."""
    # Provenance read back from JSON arrives as 2-element lists, not
    # tuples; both shapes become tuple[str, str], anything else is dropped.
    atoms: set[tuple[str, str]] = set()
    for r in retrieved:
        for p in (r.chunk.gold_provenance or ()):
            try:
                parent, span = p
            except (TypeError, ValueError):
                continue
            atoms.add((str(parent), str(span)))
    return frozenset(atoms)


def score_retrieval_ck2(
    retrieved: Sequence[RetrievedChunk],
    gold_passage_sets: Sequence[frozenset[tuple[str, str]]],
) -> RetrievalScore:
    """Set-level recall/precision/F1 over atoms, max over annotators."""
    # harness choice: chunker-independent recall (METHODS §C.4)
    covered = _retrieved_atoms(retrieved)
    n_retrieved_atoms = len(covered)

    per_annotator: list[dict | None] = []
    best_f1 = -1.0
    best_block: dict | None = None
    any_scored = False

    # Score each annotator's gold set and keep the best F1; an annotator
    # with no gold atoms gets None instead of a block.
    for gold in gold_passage_sets:
        if not gold:
            per_annotator.append(None)
            continue
        any_scored = True
        intersect = len(covered & gold)
        recall, precision, f1 = _f1(intersect, n_pred=n_retrieved_atoms, n_gold=len(gold))
        block = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "n_gold": len(gold),
            "n_intersect": intersect,
        }
        per_annotator.append(block)
        if f1 > best_f1:
            best_f1 = f1
            best_block = block

    # No annotator has gold: the query is skipped here and the null rule
    # (METHODS §C.9) scores it on the answer side.
    if not any_scored:
        return RetrievalScore(
            skipped=True,
            n_retrieved_atoms=n_retrieved_atoms,
            per_annotator=tuple(per_annotator),
        )

    assert best_block is not None
    return RetrievalScore(
        skipped=False,
        recall=best_block["recall"],
        precision=best_block["precision"],
        f1=best_block["f1"],
        n_gold=best_block["n_gold"],
        n_covered=best_block["n_intersect"],
        n_retrieved_atoms=n_retrieved_atoms,
        per_annotator=tuple(per_annotator),
    )


def score_retrieval_rank_aware(
    retrieved: Sequence[RetrievedChunk],
    gold_atoms: frozenset[tuple[str, str]],
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Hit@K, MAP@K and MRR over the document ranking implied by the chunks."""
    # No gold atoms means a null query; the answer-side null rule scores it
    # (METHODS §C.9), so return skipped rather than zeros.
    if not gold_atoms:
        return {"skipped": True}

    # Collapse the chunk ranking to a document ranking: each atom is placed
    # at its first occurrence, so a gold document is credited once and
    # every metric below counts document positions, not chunks.
    # harness choice: document-level metrics (METHODS §C.5)
    doc_ranking: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in retrieved:
        for atom in (r.chunk.gold_provenance or ()):
            try:
                parent, span = atom
            except (TypeError, ValueError):
                continue
            key = (str(parent), str(span))
            if key not in seen:
                seen.add(key)
                doc_ranking.append(key)

    relevance: list[bool] = [doc in gold_atoms for doc in doc_ranking]

    n_gold = len(gold_atoms)
    n_relevant_retrieved = sum(relevance)

    # MRR: 1 / (1-based rank of the first gold document), 0 if none; the
    # rank is not capped at 10.
    # deviation from official (MRR@10): see METHODS §B.1
    mrr = 0.0
    for i, rel in enumerate(relevance):
        if rel:
            mrr = 1.0 / (i + 1)
            break

    # Hit@K is 1 if any gold document sits in the top K. MAP@K sums the
    # precision at each gold rank in the top K and divides by
    # min(K, n_gold); with one credit per document the sum never exceeds
    # the denominator, so MAP@K stays in [0, 1].
    # official: retrieval_evaluate.py @ cde8e844 (Hits@4, Hits@10); K = 1, 5 are ours
    # deviation from official (retrieval_evaluate.py adds newly-matched/rank): see METHODS §C.8
    # official: retrieval_evaluate.py @ cde8e844 (denominator only)
    hit_at_k: dict[int, float] = {}
    map_at_k: dict[int, float] = {}
    for k in k_values:
        top_k = relevance[:k]
        hit_at_k[k] = 1.0 if any(top_k) else 0.0
        n_relevant_so_far = 0
        precision_sum = 0.0
        for i, rel in enumerate(top_k):
            if rel:
                n_relevant_so_far += 1
                precision_sum += n_relevant_so_far / (i + 1)
        denom = max(1, min(k, n_gold))
        map_at_k[k] = precision_sum / denom

    return {
        "skipped": False,
        "hit_at_k": hit_at_k,
        "map_at_k": map_at_k,
        "mrr": mrr,
        "n_relevant_retrieved": n_relevant_retrieved,
        "n_gold": n_gold,
        "n_docs_ranked": len(doc_ranking),
    }


__all__ = ["score_retrieval_ck2", "score_retrieval_rank_aware"]
