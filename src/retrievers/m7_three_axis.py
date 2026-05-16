"""M7 — three-axis hybrid retrieval over RAPTOR (the thesis contribution).

PIPELINE_DESIGN.md §4 end-to-end. M7 reuses the shared RAPTOR substrate
(src/raptor.py: tree + LLM summaries + flat collapsed index + §4.4
expansion) UNCHANGED as Axis 1, and layers:

  * Axis 1b — multi-branch tree traversal (src/multibranch.py),
    RRF-fused with collapsed retrieval.
  * Axis 2 — Docling structural rerank/diversification
    (src/structural.py): section-diversity cap, aspect-section bias,
    neighbour expansion.
  * Axis 3 — intent decomposition (src/intent.py): aspects + paraphrase
    + HyDE views, protected global view, aspect scoring, quota-
    preserving budget.

plus quota-preserving cross-encoder rerank (§4.5), parent-summary
context packing (§4.8) and a per-aspect abstention signal (§4.9).

Cache: the substrate lives in the shared RAPTOR/<substrate_hash>/
namespace (raptor.raptor_substrate_extra) so M7 reuses M4's tree
instead of rebuilding. M7-only artifacts (the post-hoc chunk→section
attachment) live in M7/<m7_hash>/, whose key inherits every substrate
field plus the M7 prompt identity.

The eight ablation switches (evaluation_plan.pdf §4) are M7Config
fields read here; six are pure skips, A3 (view_types) and A6
(quota_preserving_rerank) gate the two explicit code branches.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..cache import (
    CacheDir,
    Manifest,
    compute_cache_key,
    corpus_content_hash,
    load_chunks,
    load_embeddings,
    load_pickle,
    save_chunks,
    save_embeddings,
    save_pickle,
)
from ..chunking import Chunk, chunk_corpus
from ..config import BASE_ANSWER_SYSTEM_PROMPT, DEFAULT_CONFIG, EMBEDDER_MODEL, HarnessConfig
from ..intent import (
    GLOBAL_VIEW_NAME,
    AspectPlan,
    allocate_budget,
    decompose,
    global_confidence,
    global_view_spec,
    score_aspects,
)
from ..models import embed_texts, generate, load_embedder, rerank_scores
from ..multibranch import merge_collapsed_multibranch, multi_branch_traverse, rrf_fuse
from ..parsing import walk_corpus
from ..raptor import (
    RAPTOR_SUBSTRATE_NAMESPACE,
    FlatCollapsedIndex,
    RaptorTree,
    build_flat_collapsed_index,
    build_raptor_tree,
    expand_node,
    load_flat_index,
    load_raptor_tree,
    raptor_substrate_extra,
    save_flat_index,
    save_raptor_tree,
)
from ..structural import (
    SectionRef,
    apply_aspect_section_bias,
    apply_section_diversity_cap,
    attach_sections,
    expand_section_neighbors,
)
from ..summarization import (
    SUMMARY_PROMPT_VERSION,
    summarization_identity,
    summarize_passages,
)
from .base import AnswerResult, BaseSystem, RetrievedChunk


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Bumped when the chunk→section attachment algorithm changes (M7-only
# cache; never touches the shared substrate).
STRUCTURAL_ATTACH_VERSION = "v1"

SUBSTRATE_FILES = (
    "chunks.jsonl",
    "embeddings.npy",
    "bm25.pkl",
    "raptor_tree.json",
    "raptor_summary_embeddings.npy",
    "flat_collapsed.index",
    "flat_collapsed_meta.json",
)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x))) if x >= 0 else float(
        np.exp(x) / (1.0 + np.exp(x))
    )


class ThreeAxisSystem(BaseSystem):
    system_id = "M7"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self.chunk_embeddings: np.ndarray | None = None
        self._bm25: Any | None = None
        self._tree: RaptorTree | None = None
        self._flat: FlatCollapsedIndex | None = None
        self._sections: dict[str, SectionRef] = {}
        self._id_to_idx: dict[str, int] = {}
        self._index_stats: dict = {}
        self._last_trace: dict = {}

    # --- index ------------------------------------------------------------

    def index(self, corpus_path: Path) -> None:
        from rank_bm25 import BM25Okapi

        m7 = self.config.m7
        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)

        substrate_extra = raptor_substrate_extra(
            build=m7.build,
            summary_model=m7.summary_model,
            summary_prompt_version=SUMMARY_PROMPT_VERSION,
            include_root=m7.include_root_in_flat_index,
            rrf_k=m7.rrf_k,
        )
        substrate_key = compute_cache_key(
            chunking_config=self.config.chunking,
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            extra=substrate_extra,
        )
        sdir = CacheDir(
            paths.cache_dir(), RAPTOR_SUBSTRATE_NAMESPACE, substrate_key
        )

        if sdir.is_complete(SUBSTRATE_FILES):
            print(f"[{self.system_id}] substrate cache hit: {sdir.path}")
            self.chunks = load_chunks(sdir.chunks_path)
            self.chunk_embeddings = load_embeddings(sdir.embeddings_path)
            self._bm25 = load_pickle(sdir.bm25_path)
            self._tree = load_raptor_tree(
                sdir.path / "raptor_tree.json",
                sdir.path / "raptor_summary_embeddings.npy",
            )
            self._flat = load_flat_index(
                sdir.path / "flat_collapsed.index",
                sdir.path / "flat_collapsed_meta.json",
            )
            parsed_docs = None
        else:
            print(f"[{self.system_id}] substrate cache miss -> build {sdir.path}")
            parsed_docs = list(
                walk_corpus(
                    corpus_path, min_chars=self.config.chunking.min_chars_per_doc
                )
            )
            embedder = (
                load_embedder()
                if self.config.chunking.strategy == "semantic"
                else None
            )
            self.chunks = chunk_corpus(
                parsed_docs, self.config.chunking, embedder=embedder
            )
            if not self.chunks:
                raise RuntimeError(f"No chunks produced from {corpus_path}")
            self.chunk_embeddings = embed_texts([c.text for c in self.chunks])
            self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

            def _summarize(passages: list[str]) -> str:
                return summarize_passages(passages, model=m7.summary_model)

            self._tree = build_raptor_tree(
                chunk_texts=[c.text for c in self.chunks],
                chunk_embeddings=self.chunk_embeddings,
                params=m7.build,
                summarize_fn=_summarize,
                embed_fn=embed_texts,
            )
            self._flat = build_flat_collapsed_index(
                self._tree,
                self.chunk_embeddings,
                expansion=m7.expansion,
                include_root=m7.include_root_in_flat_index,
            )
            save_chunks(self.chunks, sdir.chunks_path)
            save_embeddings(self.chunk_embeddings, sdir.embeddings_path)
            save_pickle(self._bm25, sdir.bm25_path)
            save_raptor_tree(
                self._tree,
                sdir.path / "raptor_tree.json",
                sdir.path / "raptor_summary_embeddings.npy",
            )
            save_flat_index(
                self._flat,
                sdir.path / "flat_collapsed.index",
                sdir.path / "flat_collapsed_meta.json",
            )
            Manifest(
                system_id=RAPTOR_SUBSTRATE_NAMESPACE,
                cache_key=substrate_key,
                chunking_config=asdict(self.config.chunking),
                embedder_model=EMBEDDER_MODEL,
                corpus_hash=chash,
                n_chunks=len(self.chunks),
                files=list(SUBSTRATE_FILES),
                extra={"substrate": substrate_extra},
            ).save(sdir.manifest_path)

        self._id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}

        # --- M7-only artifact: post-hoc chunk -> Docling section ---
        m7_extra = {
            **substrate_extra,
            **summarization_identity(model=m7.summary_model),
            "structural_attach_version": STRUCTURAL_ATTACH_VERSION,
        }
        m7_key = compute_cache_key(
            chunking_config=self.config.chunking,
            embedder_model=EMBEDDER_MODEL,
            corpus_hash=chash,
            extra=m7_extra,
        )
        mdir = CacheDir(paths.cache_dir(), self.system_id, m7_key)
        sec_path = mdir.path / "sections.json"
        if sec_path.exists():
            self._sections = _load_sections(sec_path)
        else:
            if parsed_docs is None:
                parsed_docs = list(
                    walk_corpus(
                        corpus_path,
                        min_chars=self.config.chunking.min_chars_per_doc,
                    )
                )
            self._sections = attach_sections(self.chunks, parsed_docs)
            _save_sections(self._sections, sec_path)

        self._index_stats = self._collect_index_stats()
        self._indexed = True

    def _collect_index_stats(self) -> dict:
        assert self._tree is not None and self._flat is not None
        from collections import Counter

        depth_counts = Counter(n.depth for n in self._tree.nodes.values())
        type_counts = Counter(self._flat.node_types)
        n_sections = len({r.section_key for r in self._sections.values()})
        return {
            "tree_n_nodes": len(self._tree.nodes),
            "tree_depth_counts": {int(d): int(c) for d, c in depth_counts.items()},
            "flat_n_chunks": int(type_counts.get("chunk", 0)),
            "flat_n_summaries": int(
                sum(v for k, v in type_counts.items() if k != "chunk")
            ),
            "flat_node_type_counts": {k: int(v) for k, v in type_counts.items()},
            "n_chunks": len(self.chunks),
            "n_docling_sections": int(n_sections),
        }

    @property
    def index_stats(self) -> dict:
        return dict(self._index_stats)

    @property
    def last_trace(self) -> dict:
        return dict(self._last_trace)

    # --- first stage ------------------------------------------------------

    def _collapsed_ranking(
        self, view_vec: np.ndarray, view_str: str, *, use_bm25: bool
    ) -> list[int]:
        """Axis 1a: hybrid dense+BM25 over the flat collapsed index, RRF,
        per-node-type summary expansion → ranked chunk-index list."""
        assert self._flat is not None and self._tree is not None
        assert self.chunk_embeddings is not None
        m7 = self.config.m7
        q2d = view_vec.reshape(1, -1).astype(np.float32, copy=False)
        n_flat = len(self._flat.refs)
        ks = min(m7.first_stage_top_k, n_flat)
        _, dense_idx = self._flat.faiss_index.search(q2d, ks)
        dense_positions = [i for i in dense_idx[0].tolist() if i >= 0]

        rankings = [dense_positions]
        if use_bm25:
            bm25_scores = self._bm25.get_scores(_tokenize(view_str))
            bm25_order = bm25_scores.argsort()[::-1][: m7.first_stage_top_k]
            rankings.append(
                [int(i) for i in bm25_order.tolist() if bm25_scores[i] > 0]
            )

        fused = rrf_fuse(rankings, k=m7.rrf_k)[: m7.first_stage_top_k]

        chunk_score: dict[int, float] = {}
        for flat_pos, sc in fused:
            ref = self._flat.refs[flat_pos]
            if ref["type"] == "chunk":
                ci = int(ref["chunk_idx"])
                chunk_score[ci] = max(chunk_score.get(ci, -1.0), sc)
                continue
            expanded, _ = expand_node(
                ref["node_id"], view_vec, self._tree, self.chunk_embeddings,
                expansion=m7.expansion, _path_trace=[],
            )
            for ci in expanded:
                chunk_score[ci] = max(chunk_score.get(ci, -1.0), sc)
        return [
            ci for ci, _ in sorted(
                chunk_score.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]

    def _view_candidates(
        self, view_str: str, *, use_bm25: bool, use_structural: bool,
        aspect_text: str,
    ) -> list[tuple[str, float]]:
        """Full per-view pipeline: Axis 1a + Axis 1b, RRF-merged, then
        Axis 2. Returns (chunk_id, score) descending."""
        assert self._tree is not None and self.chunk_embeddings is not None
        m7 = self.config.m7
        view_vec = embed_texts([view_str])[0]

        collapsed = self._collapsed_ranking(
            view_vec, view_str, use_bm25=use_bm25
        )
        hits = multi_branch_traverse(
            self._tree, view_vec, self.chunk_embeddings, m7.multi_branch
        )
        mb_seen: dict[int, float] = {}
        for h in hits:
            mb_seen[h.chunk_idx] = max(mb_seen.get(h.chunk_idx, -1.0), h.score)
        mb_ranked = [
            ci for ci, _ in sorted(
                mb_seen.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ]
        merged = merge_collapsed_multibranch(collapsed, mb_ranked, m7.rrf_k)

        scored = [
            (self.chunks[ci].chunk_id, sc)
            for ci, sc in merged
            if 0 <= ci < len(self.chunks)
        ]

        if use_structural:
            scored = apply_aspect_section_bias(
                scored, self._sections, aspect_text,
                m7.structural.aspect_section_bias_factor,
            )
            scored = apply_section_diversity_cap(
                scored, self._sections, m7.structural.section_diversity_cap
            )
            scored = expand_section_neighbors(
                scored, self._sections, m7.structural.neighbor_radius
            )
            scored = sorted(scored, key=lambda t: (-t[1], t[0]))
        return scored

    # --- aspect candidate pool + rerank ----------------------------------

    def _aspect_pool(
        self, plan: AspectPlan, *, use_bm25: bool, use_structural: bool
    ) -> list[tuple[str, float]]:
        """Union the aspect's view candidates, dedupe, cap per RAPTOR
        cluster (§4.5)."""
        m7 = self.config.m7
        best: dict[str, float] = {}
        for v in plan.views:
            for cid, sc in self._view_candidates(
                v.text, use_bm25=use_bm25, use_structural=use_structural,
                aspect_text=plan.name,
            ):
                best[cid] = max(best.get(cid, -1.0), sc)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        return self._cluster_cap(ranked)

    def _cluster_cap(
        self, ranked: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        assert self._tree is not None
        cap = self.config.m7.diversity.max_chunks_per_raptor_cluster
        if cap <= 0:
            return ranked
        per: dict[str, int] = {}
        out: list[tuple[str, float]] = []
        for cid, sc in ranked:
            idx = self._id_to_idx.get(cid)
            anc = self._tree.ancestors_of(idx) if idx is not None else []
            cluster = anc[0] if anc else f"__noclu__::{cid}"
            n = per.get(cluster, 0)
            if n >= cap:
                continue
            per[cluster] = n + 1
            out.append((cid, sc))
        return out

    def _rerank(
        self, query: str, cand: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Cross-encoder rerank (logits → sigmoid). Stable order on ties."""
        if not cand:
            return []
        ids = [c for c, _ in cand]
        passages = [self.chunks[self._id_to_idx[c]].text for c in ids]
        logits = rerank_scores(query, passages)
        scored = [
            (cid, _sigmoid(float(l))) for cid, l in zip(ids, logits.tolist())
        ]
        return sorted(scored, key=lambda t: (-t[1], t[0]))

    def _preliminary_confidence(self, view_str: str) -> float:
        """§4.2: top-1 cross-encoder score over the first-stage top-K of
        the (paraphrase) view. Sigmoid applied by _rerank."""
        m7 = self.config.m7
        view_vec = embed_texts([view_str])[0]
        ranked = self._collapsed_ranking(
            view_vec, view_str, use_bm25=m7.use_bm25
        )[: m7.scoring.preliminary_rerank_top_k]
        if not ranked:
            return 0.0
        cand = [(self.chunks[ci].chunk_id, 0.0) for ci in ranked]
        reranked = self._rerank(view_str, cand)
        return reranked[0][1] if reranked else 0.0

    # --- retrieve ---------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        sel, _ = self._run_pipeline(query)
        out: list[RetrievedChunk] = []
        for cid, sc in sel:
            idx = self._id_to_idx.get(cid)
            if idx is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk=self.chunks[idx], score=float(sc), rank=len(out)
                )
            )
        if k is not None:
            out = out[:k]
        return out

    def _run_pipeline(
        self, query: str
    ) -> tuple[list[tuple[str, float]], dict]:
        """Steps 1-7. Returns (selected [(chunk_id,score) packed by
        aspect then global], pipeline-state dict for packing/prompt)."""
        self._require_indexed()
        m7 = self.config.m7
        trace: dict = {"ablations": _ablation_flags(m7)}

        # Step 1 — decompose + views
        plans = decompose(query, m7)
        # Step 2 — aspect scoring (preliminary cross-encoder confidence)
        plans = score_aspects(plans, m7, self._preliminary_confidence)
        # Step 3 — budget
        plans = allocate_budget(plans, m7)

        trace["aspects"] = [
            {
                "name": p.name, "importance": round(p.importance, 4),
                "confidence": round(p.retrieval_confidence, 4),
                "score": round(p.score, 4), "budget": p.budget,
                "n_views": len(p.views),
                "view_types": [v.view_type for v in p.views],
            }
            for p in plans
        ]

        # Steps 4-5 — per aspect: pool → rerank → take budget
        per_aspect: list[tuple[AspectPlan, list[tuple[str, float]]]] = []
        a6_global_pool: list[tuple[str, float]] = []
        for p in plans:
            if p.budget <= 0:
                per_aspect.append((p, []))
                continue
            pool = self._aspect_pool(
                p, use_bm25=m7.use_bm25,
                use_structural=m7.use_docling_structural_axis,
            )
            if m7.quota_preserving_rerank:
                reranked = self._rerank(p.name, pool)
                per_aspect.append((p, reranked))
            else:
                # A6: defer to a single global rerank across all aspects.
                a6_global_pool.extend(pool)
                per_aspect.append((p, pool))

        # Step 6 — protected global view
        global_sel: list[tuple[str, float]] = []
        g_conf = 0.0
        if m7.always_include_global_query_view:
            gv = global_view_spec(query)
            g_conf = global_confidence(gv.text, self._preliminary_confidence)
            g_pool = self._view_candidates(
                gv.text, use_bm25=m7.use_bm25,
                use_structural=m7.use_docling_structural_axis,
                aspect_text=query,
            )
            g_pool = self._cluster_cap(g_pool)
            g_re = self._rerank(query, g_pool)
            global_sel = g_re[: m7.budget.global_view_quota]
        trace["global_confidence"] = round(g_conf, 4)

        # A6 branch: one global cross-encoder rerank, no per-aspect quota.
        if not m7.quota_preserving_rerank:
            uniq: dict[str, float] = {}
            for cid, sc in a6_global_pool:
                uniq[cid] = max(uniq.get(cid, -1.0), sc)
            flat = self._rerank(query, list(uniq.items()))
            take = m7.budget.final_context_chunks - len(global_sel)
            chosen = flat[: max(take, 0)]
            selected = _dedupe_keepfirst(chosen + global_sel)
            trace["mode"] = "A6_global_rerank"
            self._last_trace = trace
            return selected, {
                "plans": plans, "per_aspect": [(p, c) for p, c in per_aspect],
                "global_sel": global_sel, "g_conf": g_conf, "trace": trace,
            }

        # Step 7 — concat aspects + global, global dedupe, backfill
        selected: list[tuple[str, float]] = []
        seen: set[str] = set()
        for p, reranked in per_aspect:
            chosen = [c for c in reranked[: p.budget]]
            kept = [(c, s) for c, s in chosen if c not in seen]
            for c, _ in kept:
                seen.add(c)
            # backfill if dedupe pushed this aspect under its minimum
            if len(kept) < min(p.budget, m7.budget.min_chunks_per_aspect):
                for c, s in reranked[p.budget :]:
                    if c in seen:
                        continue
                    kept.append((c, s))
                    seen.add(c)
                    if len(kept) >= min(
                        p.budget, m7.budget.min_chunks_per_aspect
                    ):
                        break
            selected.extend(kept)
        for c, s in global_sel:
            if c not in seen:
                seen.add(c)
                selected.append((c, s))

        trace["n_selected"] = len(selected)
        if m7.trace:
            sel_ids = {c for c, _ in selected}
            trace["per_aspect_selected"] = {
                p.name: sum(
                    1 for c, _ in reranked[: max(p.budget, 0)]
                    if c in sel_ids
                )
                for p, reranked in per_aspect
            }
            # Multi-branch subtree breadth on the raw query (diagnostic
            # only; one extra traverse, no LLM/rerank).
            assert self._tree is not None and self.chunk_embeddings is not None
            qv = embed_texts([query])[0]
            qhits = multi_branch_traverse(
                self._tree, qv, self.chunk_embeddings, m7.multi_branch
            )
            trace["multibranch_distinct_paths"] = len(
                {h.path for h in qhits}
            )
        self._last_trace = trace
        return selected, {
            "plans": plans, "per_aspect": per_aspect,
            "global_sel": global_sel, "g_conf": g_conf, "trace": trace,
        }

    # --- step 8: parent-summary packing ----------------------------------

    def _orientation_for(self, cid: str) -> tuple[list[str], str | None]:
        """Nearest RAPTOR parent summary (+ ≤1 higher) and the Docling
        section title for a chunk (§4.8). Never the root summary."""
        assert self._tree is not None
        m7 = self.config.m7
        idx = self._id_to_idx.get(cid)
        summaries: list[str] = []
        if idx is not None:
            anc = self._tree.ancestors_of(idx)  # deepest → root
            cap = m7.packing.max_ancestor_summaries_per_chunk_group
            for nid in anc:
                if nid == self._tree.root_id and not m7.packing.include_root_summary:
                    continue
                txt = self._tree.nodes[nid].summary_text.strip()
                if not txt:
                    continue
                words = txt.split()
                if len(words) > m7.packing.max_parent_summary_tokens:
                    txt = " ".join(
                        words[: m7.packing.max_parent_summary_tokens]
                    )
                summaries.append(txt)
                if len(summaries) >= cap:
                    break
        ref = self._sections.get(cid)
        sec_title = ref.section_title if ref is not None else None
        return summaries, sec_title

    # --- answer -----------------------------------------------------------

    def answer(self, query: str, k: int | None = None) -> AnswerResult:
        self._require_indexed()
        t0 = self._now()
        selected, state = self._run_pipeline(query)
        m7 = self.config.m7
        plans: list[AspectPlan] = state["plans"]
        per_aspect = dict(
            (p.name, c) for p, c in state["per_aspect"]
        )
        sel_set = {c for c, _ in selected}

        thr = m7.abstention.retrieval_confidence_threshold
        abstained: list[str] = []
        blocks: list[str] = []
        seen_orient: set[str] = set()

        def _evidence(cids: list[str]) -> list[str]:
            lines: list[str] = []
            for cid in cids:
                idx = self._id_to_idx.get(cid)
                if idx is None:
                    continue
                lines.append(f"  Chunk {cid}: {self.chunks[idx].text}")
            return lines

        for p in plans:
            chosen = [
                c for c, _ in per_aspect.get(p.name, [])[: max(p.budget, 0)]
                if c in sel_set
            ]
            if not chosen:
                continue
            low = p.retrieval_confidence < thr
            if low:
                abstained.append(p.name)
            header = f"[Aspect: {p.name}]"
            if m7.pass_retrieval_confidence_to_llm:
                header += f" (retrieval_confidence: {p.retrieval_confidence:.2f})"
                if low:
                    header += " [LOW CONFIDENCE - abstain or hedge here]"
            blocks.append(header)
            if m7.include_parent_summaries:
                for cid in chosen:
                    summaries, sec = self._orientation_for(cid)
                    if sec and f"sec::{sec}" not in seen_orient:
                        seen_orient.add(f"sec::{sec}")
                        blocks.append(f"  [Orientation section: {sec}]")
                    for s in summaries:
                        key = f"sum::{hash(s)}"
                        if key in seen_orient:
                            continue
                        seen_orient.add(key)
                        blocks.append(f"  [Orientation: {s}]")
            blocks.append("  [Primary Evidence]")
            blocks.extend(_evidence(chosen))

        gsel = [c for c, _ in state["global_sel"] if c in sel_set]
        if gsel:
            blocks.append("[Global View]")
            if m7.pass_retrieval_confidence_to_llm:
                blocks.append(
                    f"  (retrieval_confidence: {state['g_conf']:.2f})"
                )
            blocks.append("  [Primary Evidence]")
            blocks.extend(_evidence(gsel))

        context = "\n".join(blocks)
        if m7.pass_retrieval_confidence_to_llm:
            system_prompt = (
                "Answer the user question using only the provided evidence. "
                "Address each aspect. Orientation summaries are context; "
                "primary evidence chunks are reliable. If a summary and a "
                "chunk conflict, trust the chunk. For aspects flagged as "
                "low-confidence, say so explicitly rather than fabricate. "
                "Synthesize the aspects into one final answer."
            )
        else:
            system_prompt = BASE_ANSWER_SYSTEM_PROMPT

        user_prompt = f"Evidence:\n{context}\n\nQuestion: {query}"
        ans = generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cfg=self.config.generation,
        )

        retrieved = [
            RetrievedChunk(
                chunk=self.chunks[self._id_to_idx[c]], score=float(s),
                rank=i,
            )
            for i, (c, s) in enumerate(selected)
            if c in self._id_to_idx
        ]
        extra: dict = {
            "abstained_aspects": abstained,
            "n_aspects": len(plans),
            "global_confidence": round(state["g_conf"], 4),
        }
        if m7.trace:
            extra["trace"] = state["trace"]
        return AnswerResult(
            query=query,
            answer=ans,
            retrieved=retrieved,
            latency_s=self._now() - t0,
            n_retrieval_calls=sum(len(p.views) for p in plans) + 1,
            extra=extra,
        )


# --- helpers --------------------------------------------------------------


def _ablation_flags(m7: Any) -> dict:
    return {
        "use_docling_structural_axis": m7.use_docling_structural_axis,
        "use_intent_decomposition": m7.use_intent_decomposition,
        "view_types": list(m7.view_types),
        "use_bm25": m7.use_bm25,
        "include_parent_summaries": m7.include_parent_summaries,
        "quota_preserving_rerank": m7.quota_preserving_rerank,
        "pass_retrieval_confidence_to_llm": m7.pass_retrieval_confidence_to_llm,
        "always_include_global_query_view": m7.always_include_global_query_view,
    }


def _dedupe_keepfirst(
    pairs: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for c, s in pairs:
        if c in seen:
            continue
        seen.add(c)
        out.append((c, s))
    return out


def _save_sections(refs: dict[str, SectionRef], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        cid: {**asdict(r), "section_path": list(r.section_path)}
        for cid, r in refs.items()
    }
    path.write_text(json.dumps(obj, ensure_ascii=False))


def _load_sections(path: Path) -> dict[str, SectionRef]:
    obj = json.loads(path.read_text())
    out: dict[str, SectionRef] = {}
    for cid, d in obj.items():
        d = dict(d)
        d["section_path"] = tuple(d.get("section_path", []))
        out[cid] = SectionRef(**d)
    return out
