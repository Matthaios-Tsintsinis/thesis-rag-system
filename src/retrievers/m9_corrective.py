"""M9 — CorrectiveRAG (Yan et al., 2024), corpus-internal variant.

Replacement baseline for M6 in the eval matrix (zero index-time LLM
cost). Pipeline:

  query -> inner M3 retrieve (natural top-15, dense+BM25 RRF)
        -> bge-reranker-v2-m3 scores every (query, chunk) pair,
           sigmoid -> confidence in [0, 1]
        -> action decision on max confidence:
             max >= tau_high  -> CORRECT    (trust the pool)
             max <  tau_low   -> INCORRECT  (pool failed; replace)
             otherwise        -> AMBIGUOUS  (mixed; augment)
        -> INCORRECT: one SHARED-GENERATOR query rewrite (temperature 0)
           re-runs the SAME M3 retriever; new pool REPLACES the old.
           AMBIGUOUS: rewrite + re-retrieve, union with the original
           pool (deduped by chunk_id), rerank the union.
        -> every branch feeds the natural top-15, reranker-ordered
           (professor's rule: baselines at natural strength).

Knowledge refinement (paper's decompose-then-recompose) happens in
`answer()` ONLY: each retained chunk splits into consecutive
2-sentence strips, the same reranker scores all strips in one batched
call, strips below tau_strip drop, survivors recompose in original
order. `retrieve()` returns the post-corrective PRE-refinement set —
that is what CK-2 / rank-aware retrieval scoring sees; refinement can
never narrow the scored retrieval output.

Index-time delegates entirely to the inner `HybridRRFSystem`, whose
own code computes its own cache key under cache/M3/<key> — bit-
identical to a standalone M3 run, so the existing M3 substrate cache
HITS and nothing new is built. M9 has no substrate namespace of its
own; thresholds / rewrite prompt / refinement flags are query-time
parameters logged per query in AnswerResult.extra (same principle as
M7's reranker exclusion from the substrate key).

Per-query logging (lands in the eval JSONL metadata via the runner):
m9_action, m9_max_conf / m9_min_conf / m9_mean_conf (initial evaluator
pass — the decision evidence), m9_rewrite_fired, m9_rewritten_query,
m9_overlap_jaccard (original vs re-retrieved top-15 chunk_id sets),
m9_n_strips_kept / m9_n_strips_total.
"""

# === DEVIATIONS FROM THE CRAG PAPER (Yan et al., 2024) — thesis footnote ===
# M9 reproduces Corrective RAG as a baseline. Three substitutions:
#
# 1. Evaluator. The paper fine-tunes T5-large on labeled relevance data
#    as its retrieval evaluator. We use the off-the-shelf shared harness
#    cross-encoder (BAAI/bge-reranker-v2-m3) and calibrate two scalar
#    thresholds (tau_high, tau_low) on labeled VALIDATION data. This is
#    strictly weaker supervision than the paper's — two scalars fit on
#    validation versus a whole fine-tuned model — so the calibration is
#    not an oracle advantage. The paper's published thresholds
#    (0.59 / -0.99) live on its T5 score scale and do not transfer;
#    ours are derived empirically (scripts/derive_corrective_thresholds.py).
#
# 2. Corrective action. The paper falls back to WEB SEARCH (external
#    knowledge) when retrieval is judged Incorrect/Ambiguous. Our
#    benchmarks are closed-corpus, so the corrective action is
#    corpus-internal re-retrieval: a shared-generator query rewrite
#    (keyword-style, temperature 0) re-runs the same hybrid retriever.
#    The rewrite itself is paper-faithful — the paper uses ChatGPT
#    keyword extraction to form its web queries — and does NOT conflict
#    with the no-LLM-in-decomposition rule, which constrains M7 (the
#    thesis contribution), not baselines reproducing their papers. It
#    is, however, a WEAKER corrective lever than web search: it can
#    only resurface what the corpus already holds. Per-query Jaccard
#    overlap between original and re-retrieved pools is logged to
#    measure exactly this. The rewrite introduces mild retrieval
#    nondeterminism (temperature 0); same smoke-test tolerance as M7.
#
# 3. Generator. The shared harness final generator (harness-level, held
#    constant across every system), not the paper's generators.
#
# MODEL NOTE, corrected 2026-08-22 (docs/FINAL_FIDELITY_AUDIT.md AF-8).
# This block named gpt-4o-mini in three places for the rewrite call and
# the final generator. That has been stale since 2026-08-02: gpt-4o-mini
# is removed from the project entirely and the rewrite resolves
# `self.config.generation` — the same local model every other system
# reads (Qwen2.5-7B-Instruct today, whatever `--generator` sets
# otherwise). Cosmetic only; M9 is WITHDRAWN (PREREGISTRATION
# ADDENDUM 6) and no M9 cell is produced or reported.
#
# 4. Base retriever. The paper layers its corrective loop over a
#    Contriever-class dense retriever (Self-RAG setup, Wikipedia); this
#    M9 composes over the M3 hybrid (bge-m3 + BM25 RRF) on the
#    benchmark corpus — forced by the closed-corpus setting and
#    approved at proposal time (2026-06-12, "reuse M3 retrieval, don't
#    re-implement"). Deliberate side benefit: M9's first stage IS the
#    matrix's M3 row, so M9 minus M3 isolates the corrective layer's
#    contribution exactly.
#
# === CALIBRATION FINDINGS (QASPER validation derivation, 2026-06-12;
#     artifact derivation_validation_20260612-014811.json) ===
#
# WEAK-EVALUATOR FINDING (limitations text). The off-the-shelf
# cross-encoder's confidence cannot replicate the separation the
# paper's fine-tuned evaluator implies. Concretely: an absolute
# precision target (v1 criterion, 0.8) was unreachable — measured
# precision-against-gold capped at ~0.50 at EVERY cut (gold base rate
# 0.0923 in the retrieved pools; topically-relevant-but-unannotated
# chunks count as false positives) — and at the baked tau_high the
# enrichment is only 2.09x over the base rate. The gold and non-gold
# confidence distributions are heavily entangled (both confined to
# [0.5, 0.73]; separation only in the thin upper tail), and the
# entanglement is partly STRUCTURAL to corpus-internal CRAG: the
# evaluator only ever sees M3's top-15, i.e. chunks pre-filtered to
# topical plausibility, so it judges "more vs less plausible among
# already-plausible," not "relevant vs irrelevant." This is precisely
# why the paper fine-tuned a dedicated evaluator; reproducing CRAG
# without that supervision inherits a weaker action signal.
#
# PER-BENCHMARK BRANCH BEHAVIOUR. On QASPER the INCORRECT branch is
# dead at the derived thresholds: it requires ALL 15 chunks below
# tau_low=0.5001, which never occurs — hybrid retrieval always
# surfaces something the reranker scores >= 0.5. On QASPER the
# corrective path therefore operates through the AMBIGUOUS branch
# only (~51% of queries at derivation time). This is a per-benchmark
# property, not a code path defect: the per-benchmark action mix is
# reported for each dataset (analyse.py M9 section), and MultiHop's
# distribution may differ. Strip refinement is ALIVE at the derived
# tau_strip (57.2% strip survival; the early smoke's 100% survival
# was an artifact of the provisional thresholds).

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..components import (
    ResolvedComponents,
    format_components_log,
    resolve_components,
)
from ..config import (
    SCORING_RANKING_DEPTH,
    BASE_ANSWER_SYSTEM_PROMPT,
    DEFAULT_CONFIG,
    EVIDENCE_TOKEN_BUDGET_TOKENIZER,
    HarnessConfig,
    RERANKER_MODEL,
)
from ..models import generate, rerank_scores
from .base import AnswerResult, BaseSystem, PreparedQuery, RetrievedChunk
from .m3_hybrid import HybridRRFSystem


ACTION_CORRECT = "correct"
ACTION_INCORRECT = "incorrect"
ACTION_AMBIGUOUS = "ambiguous"

# Versioned rewrite prompt (CorrectiveConfig.rewrite_prompt_version
# names which constant a run used). Paper-faithful in mechanism: the
# CRAG paper extracts keywords with ChatGPT to form its web queries;
# here the rewritten query feeds the same closed-corpus M3 retriever.
REWRITE_PROMPT_V1 = (
    "Rewrite the user's question into a short, self-contained search "
    "query for a document retrieval system. Extract the key entities, "
    "concepts, and relations as a keyword-style query. Output ONLY the "
    "rewritten search query, with no explanation and no punctuation "
    "beyond the query itself."
)

# Sentence terminators mirror the harness chunker's Greek-aware set
# (. ! ? ;) with the ano teleia excluded.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def decide_action(
    confs: Sequence[float],
    tau_high: float,
    tau_low: float,
) -> str:
    """CRAG action decision on the max evaluator confidence.

    Paper rule: one confidently-relevant document is enough to trust
    the retrieval (CORRECT); ALL documents below the discard line
    means retrieval failed (INCORRECT); anything between is AMBIGUOUS.
    An empty pool is a failed retrieval by definition.
    """
    if not confs:
        return ACTION_INCORRECT
    top = max(confs)
    if top >= tau_high:
        return ACTION_CORRECT
    if top < tau_low:
        return ACTION_INCORRECT
    return ACTION_AMBIGUOUS


def split_strips(text: str, n_sentences: int = 2) -> list[str]:
    """Consecutive n-sentence strips for decompose-then-recompose."""
    sentences = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s]
    if not sentences:
        return []
    return [
        " ".join(sentences[i : i + n_sentences])
        for i in range(0, len(sentences), n_sentences)
    ]


def refine_chunk_texts(
    query: str,
    texts: Sequence[str],
    scorer: Callable[[str, list[str]], Sequence[float]],
    *,
    tau_strip: float,
    strip_sentences: int = 2,
) -> tuple[list[str], int, int]:
    """Batched strip refinement over many chunk texts.

    `scorer(query, strips)` returns one confidence in [0, 1] per strip;
    it is called ONCE over the flattened strips of every chunk (one
    batched cross-encoder pass, not per-chunk calls). Strips below
    `tau_strip` drop; survivors recompose in original order. Guard: a
    chunk whose strips ALL fall below threshold is kept whole — the
    refinement step must never silently empty a document out of the
    context. Single-strip chunks pass through untouched (nothing to
    refine away).

    Returns (refined_texts, n_strips_kept, n_strips_total). Guarded
    (kept-whole) chunks count all their strips as kept.
    """
    per_chunk_strips = [split_strips(t, strip_sentences) for t in texts]
    flat = [s for strips in per_chunk_strips for s in strips]
    if not flat:
        return list(texts), 0, 0

    confs = list(scorer(query, flat))
    refined: list[str] = []
    n_kept = 0
    n_total = 0
    pos = 0
    for text, strips in zip(texts, per_chunk_strips):
        n = len(strips)
        chunk_confs = confs[pos : pos + n]
        pos += n
        n_total += n
        if n <= 1:
            refined.append(text)
            n_kept += n
            continue
        kept = [s for s, c in zip(strips, chunk_confs) if c >= tau_strip]
        if not kept:
            refined.append(text)
            n_kept += n
        else:
            refined.append(" ".join(kept))
            n_kept += len(kept)
    return refined, n_kept, n_total


class CorrectiveRAGSystem(BaseSystem):
    system_id = "M9"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        # Composition: index/retrieval substrate is M3, code reused
        # verbatim. The inner instance's system_id is "M3", so its
        # cache key and path are bit-identical to a standalone M3 run.
        self.m3 = HybridRRFSystem(config)
        self._resolved: ResolvedComponents | None = None

    # --- indexing -----------------------------------------------------------

    def index(self, corpus_path: Path) -> None:
        self._resolved = resolve_components(
            self.config.corrective,
            self.config,
            default_reranker=RERANKER_MODEL,
        )
        print(f"[components] {format_components_log(self.system_id, self._resolved)}")
        self.m3.index(corpus_path)
        # ALIAS the inner chunk list (same object). BaseSystem.index_items
        # stamps gold_provenance by iterating self.chunks after index()
        # returns; without this alias the stamping would silently no-op
        # and CK-2 retrieval scoring would break.
        self.chunks = self.m3.chunks
        self._indexed = True

    @property
    def resolved_components(self) -> ResolvedComponents | None:
        return self._resolved

    # --- evaluator ------------------------------------------------------------

    def _reranker_id(self) -> str:
        if self._resolved is not None and self._resolved.reranker_id:
            return self._resolved.reranker_id
        return RERANKER_MODEL

    def _score_pool(self, query: str, pool: list[RetrievedChunk]) -> np.ndarray:
        """Sigmoid evaluator confidences in [0, 1], one per pool entry."""
        if not pool:
            return np.zeros((0,), dtype=np.float32)
        logits = rerank_scores(
            query,
            [r.chunk.text for r in pool],
            model_name=self._reranker_id(),
        )
        return _sigmoid(np.asarray(logits, dtype=np.float32))

    # --- corrective retrieval -------------------------------------------------

    def _rewrite_query(self, query: str) -> str | None:
        """One rewrite call via the shared generate() router.

        Returns the rewritten query, or None on failure / empty output
        (callers fall back to the original pool — a rewrite failure
        must never kill the query).
        """
        try:
            cfg = dataclasses.replace(
                self.config.generation,
                temperature=0.0,
                max_new_tokens=64,
            )
            out = generate(
                system_prompt=REWRITE_PROMPT_V1,
                user_prompt=query,
                cfg=cfg,
            )
        except Exception as e:  # noqa: BLE001 — degrade, never crash the query
            print(f"[{self.system_id}] rewrite call failed ({e!r}); "
                  f"falling back to the original pool")
            return None
        out = (out or "").strip()
        return out or None

    def _corrective_retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        """Full corrective pipeline. Returns (chunks, metadata).

        The returned list is the post-corrective PRE-refinement set,
        reranker-ordered (score = sigmoid confidence vs the ORIGINAL
        query — the user's information need — in every branch, ranks
        reassigned 0..n-1). Honors an explicit caller k for the final
        cut; defaults to the natural top_k.
        """
        self._require_indexed()
        cfg = self.config.corrective
        k_final = k or self.config.retrieval.top_k

        pool = self.m3.retrieve(query, k=None)  # natural top-15
        confs = self._score_pool(query, pool)
        action = decide_action(confs.tolist(), cfg.tau_high, cfg.tau_low)

        meta: dict = {
            "m9_action": action,
            "m9_max_conf": float(confs.max()) if confs.size else 0.0,
            "m9_min_conf": float(confs.min()) if confs.size else 0.0,
            "m9_mean_conf": float(confs.mean()) if confs.size else 0.0,
            "m9_rewrite_fired": False,
            "m9_rewritten_query": None,
            "m9_overlap_jaccard": None,
            "m9_n_retrieval_calls": 1,
        }

        if action in (ACTION_INCORRECT, ACTION_AMBIGUOUS):
            rewritten = self._rewrite_query(query)
            if rewritten is None:
                meta["m9_rewrite_failed"] = True
            else:
                meta["m9_rewrite_fired"] = True
                meta["m9_rewritten_query"] = rewritten
                meta["m9_n_retrieval_calls"] = 2
                new_pool = self.m3.retrieve(rewritten, k=None)

                old_ids = {r.chunk.chunk_id for r in pool}
                new_ids = {r.chunk.chunk_id for r in new_pool}
                union_ids = old_ids | new_ids
                meta["m9_overlap_jaccard"] = (
                    len(old_ids & new_ids) / len(union_ids) if union_ids else None
                )

                if action == ACTION_INCORRECT:
                    # Paper behaviour: the failed pool is discarded and
                    # the corrective results replace it.
                    pool = new_pool
                else:
                    # AMBIGUOUS: union, deduped by chunk_id, original
                    # pool order first.
                    seen = set()
                    union: list[RetrievedChunk] = []
                    for r in pool + new_pool:
                        if r.chunk.chunk_id in seen:
                            continue
                        seen.add(r.chunk.chunk_id)
                        union.append(r)
                    pool = union
                confs = self._score_pool(query, pool)

        order = np.argsort(-confs) if confs.size else np.array([], dtype=int)
        final = [
            RetrievedChunk(
                chunk=pool[i].chunk,
                score=float(confs[i]),
                rank=rank,
                source_unit_type=pool[i].source_unit_type,
            )
            for rank, i in enumerate(order[:k_final].tolist())
        ]
        return final, meta

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        chunks, _ = self._corrective_retrieve(query, k=k)
        return chunks

    # --- answering ------------------------------------------------------------

    def prepare(self, query: str, k: int | None = None) -> PreparedQuery:
        """Corrective retrieve -> strip refinement -> pack -> prompt.

        Overrides prepare(), NOT answer(), so M9 stays batchable: the
        expensive part (final generation) goes through the shared
        phase-B batch, while the corrective loop and strip refinement
        stay here in phase A.

        M9's phase A is deliberately NOT LLM-free — the INCORRECT branch
        issues a query-rewrite call. Those prompts are tiny (~50 tokens)
        so running them sequentially costs little, but the design has to
        allow it rather than assume phase A never touches the model. The
        rewrite fire-rate is logged per query (m9_action / rewrite_fired)
        and is worth watching on MultiHop, where the corrective layer
        showed its only supported effect; on QASPER the branch was dead.

        `retrieved` keeps the UNREFINED chunks — that is the retrieval
        output CK-2 scores. `packed` carries the refined strips, which
        is what actually reaches the generator.
        """
        # Late imports mirror BaseSystem.answer (see its comment on the
        # retrievers/base <-> prompt_packing import ordering).
        from ..prompt_packing import count_tokens, pack_context

        self._require_indexed()
        cfg = self.config.corrective
        t0 = self._now()

        # ONE corrective pass, cut twice. The corrective DECISION reads a
        # fixed natural top-15 pool and does not depend on k_final, which
        # only sets the final cut after reranker ordering — so asking for
        # a deeper cut returns the SAME branch's ranking, uncut. Calling
        # the pipeline again for the scoring ranking would instead pay a
        # second reranker pass and a second rewrite LLM call, and could
        # land on a different branch if the rewrite were not bit-stable.
        scoring_ranking, meta = self._corrective_retrieve(
            query, k=SCORING_RANKING_DEPTH
        )
        k_reader = k or self.config.retrieval.top_k
        retrieved = scoring_ranking[:k_reader]

        to_pack: list[RetrievedChunk] = retrieved
        if cfg.refine and retrieved:
            tau_strip = cfg.tau_strip if cfg.tau_strip is not None else cfg.tau_low

            def _strip_scorer(q: str, strips: list[str]) -> list[float]:
                logits = rerank_scores(q, strips, model_name=self._reranker_id())
                return _sigmoid(np.asarray(logits, dtype=np.float32)).tolist()

            refined_texts, n_kept, n_total = refine_chunk_texts(
                query,
                [r.chunk.text for r in retrieved],
                _strip_scorer,
                tau_strip=tau_strip,
                strip_sentences=cfg.strip_sentences,
            )
            meta["m9_n_strips_kept"] = n_kept
            meta["m9_n_strips_total"] = n_total
            to_pack = [
                RetrievedChunk(
                    chunk=dataclasses.replace(r.chunk, text=text),
                    score=r.score,
                    rank=r.rank,
                    source_unit_type=r.source_unit_type,
                )
                for r, text in zip(retrieved, refined_texts)
            ]

        packed, evidence_tokens, evidence_block = pack_context(
            to_pack,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        user_prompt = f"Evidence:\n{evidence_block}\n\nQuestion: {query}"
        n_input_tokens = count_tokens(
            BASE_ANSWER_SYSTEM_PROMPT + "\n" + user_prompt,
            tokenizer_name=EVIDENCE_TOKEN_BUDGET_TOKENIZER,
        )
        return PreparedQuery(
            query=query,
            retrieved=retrieved,
            packed=packed,
            scoring_ranking=scoring_ranking,
            system_prompt=BASE_ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            evidence_tokens=evidence_tokens,
            n_input_tokens=n_input_tokens,
            n_retrieval_calls=int(meta.get("m9_n_retrieval_calls", 1)),
            prepare_s=self._now() - t0,
            extra=meta,
        )


__all__ = [
    "ACTION_CORRECT",
    "ACTION_INCORRECT",
    "ACTION_AMBIGUOUS",
    "REWRITE_PROMPT_V1",
    "CorrectiveRAGSystem",
    "decide_action",
    "refine_chunk_texts",
    "split_strips",
]
