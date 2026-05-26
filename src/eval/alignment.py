"""CK-2 retrieval-recall scoring — chunker-agnostic alignment to gold-passage atoms.

Each `RetrievedChunk` carries `chunk.gold_provenance`: a tuple of
(parent_id, span_id) pairs identifying the gold-passage atoms that
chunk touches. The eval loader populated this field via the default
BaseSystem.index_items fallback (see retrievers/base.py).

This module aggregates those atom sets across the retrieved chunks
and computes precision / recall / F1 against the gold atom set for
each annotator, returning max-over-annotators per QASPER convention.

PER-RULING behaviour:
  * FLOAT SELECTED (QASPER table-grounded evidence): the loader does
    NOT add table atoms to the gold set, so they shrink the
    denominator naturally without becoming an irreducible-0 floor on
    text-only retrieval.
  * No-match QASPER evidence (~6.9% after exact -> ws-normalised ->
    substring fallback): the loader DROPS unalignable evidence + bumps
    a manifest counter. No fuzzy-wrong atom corrupts the gold set.
  * Unanswerable / empty gold (every annotator's gold set is empty):
    retrieval recall is SKIPPED (returns RetrievalScore.skipped=True);
    the answer-side abstention scorer judges instead.

Chunker-independent: this function reads only chunk.gold_provenance
(atom set), never chunk text or boundaries. CK-1 chunking ablation
(eval grid runs under native + shared-word_window) does NOT change
this scorer's behaviour.
"""

from __future__ import annotations

from typing import Sequence

from ..retrievers.base import RetrievedChunk
from .types import RetrievalScore


def _f1(intersect: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    if n_gold == 0:
        # Caller should have skipped this annotator already.
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
    """Union of (parent_id, span_id) atoms across retrieved chunks.

    chunk.gold_provenance comes back from on-disk JSON as a list of
    2-element lists rather than tuple-of-tuples (json round-trip drops
    the tuple type). Normalise both shapes to tuple[str, str] for the
    frozenset.
    """
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
    """Max-over-annotators CK-2 retrieval-F1.

    `gold_passage_sets` length = number of annotators (1 for MultiHop).
    Each entry is a frozenset of (parent_id, span_id) atoms. Empty
    entries are skipped at the annotator level (the annotator marked
    the question unanswerable or every piece of their evidence was
    table-grounded). If EVERY annotator's set is empty, the whole
    query is skipped — answer-side abstention judges instead.
    """
    covered = _retrieved_atoms(retrieved)
    n_retrieved_atoms = len(covered)

    per_annotator: list[dict | None] = []
    best_f1 = -1.0
    best_block: dict | None = None
    any_scored = False

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


__all__ = ["score_retrieval_ck2"]
