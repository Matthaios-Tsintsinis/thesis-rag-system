"""M6 — HippoRAG 1 (legacy NeurIPS'24), single-step retrieval.

Faithful port of the OSU-NLP-Group/HippoRAG legacy branch, NOT a wrapper
of the original repo (the legacy code pins numpy==1.26.4 / torch==1.13.1,
which conflict ABI-hard with our harness). Same approach as M4, where a
RAPTOR-paper port lives in src/raptor.py rather than a wrapped repo.

Components per the paper / legacy main-experiment scripts:

  * Contriever embedder (facebook/contriever, 768-dim)
  * gpt-4o-mini for OpenIE + query NER (modernised from gpt-3.5-turbo-1106)
  * Synonymy edges with sim_threshold = 0.8
  * Personalised PageRank with damping = 0.5 (continue-walk probability)
  * Node specificity ON
  * doc_ensemble OFF, dpr_only OFF, single-step (no IRCoT)

Cache: M6 keeps a system-specific cache namespace cache/M6/<m6_hash>/.
The RAPTOR substrate is unrelated and not shared.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths
from ..cache import CacheDir, Manifest, compute_cache_key, corpus_content_hash, load_chunks, save_chunks
from ..chunking import Chunk, chunk_corpus
from ..components import ResolvedComponents, format_components_log, resolve_components
from ..config import DEFAULT_CONFIG, HarnessConfig, RETRIEVAL_RANKING_DEPTH
from ..hipporag_graph import (
    HippoGraph,
    REQUIRED_FILES,
    add_synonymy_edges,
    assemble_igraph,
    build_graph_structures,
    build_triple_edges,
    embed_phrases,
    load_graph,
    save_graph,
    verify_contriever_pooling,
)
from ..hipporag_openie import (
    OPENIE_PROMPT_VERSION,
    OpenIEResult,
    extract_corpus_parallel,
    extract_query_entities,
)
from ..hipporag_ppr import (
    build_reset_vector,
    link_query_entities_to_phrases,
    propagate_phrase_to_doc,
    run_pagerank,
    uniform_fallback,
)
from ..models import embed_texts, load_embedder
from ..parsing import walk_corpus
from .base import AnswerResult, BaseSystem, RetrievedChunk


class HippoRAGSystem(BaseSystem):
    """HippoRAG 1 — entity-graph PPR retrieval over OpenIE-extracted facts."""

    system_id = "M6"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self._resolved: ResolvedComponents | None = None
        self._graph: HippoGraph | None = None
        self.chunks: list[Chunk] = []
        self._index_stats: dict = {}
        self._last_trace: dict = {}

    @property
    def resolved_components(self) -> ResolvedComponents | None:
        return self._resolved

    @property
    def index_stats(self) -> dict:
        return dict(self._index_stats)

    @property
    def last_trace(self) -> dict:
        return dict(self._last_trace)

    # --- index ------------------------------------------------------------

    def index(self, corpus_path: Path) -> None:
        m6 = self.config.m6

        # Resolve components first so every downstream call uses the
        # correct embedder + openie_llm. Reranker stays None for M6.
        self._resolved = resolve_components(m6, self.config, default_reranker=None)
        print(f"[components] {format_components_log(self.system_id, self._resolved)}")
        chunker_cfg = self._resolved.chunker_config
        embedder_id = self._resolved.embedder_id
        openie_llm = self._resolved.index_llm_id
        if openie_llm is None:
            raise RuntimeError(
                "M6 requires an index-time LLM (M6Config.openie_llm). "
                "resolve_components returned None."
            )

        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)

        # M6 cache key — system-specific namespace, full identity of every
        # artifact-affecting input. Reranker NOT included (HippoRAG has no
        # rerank); generator NOT included (final generator is harness-level
        # and not part of the cached artifacts).
        m6_extra = {
            "openie_llm": openie_llm,
            "openie_prompt_version": m6.openie_prompt_version,
            "sim_threshold": m6.sim_threshold,
            "synonym_top_k_cap": m6.synonym_top_k_cap,
            "node_specificity": m6.node_specificity,
            "damping": m6.damping,
            "graph_type": "facts_and_sim",
        }
        ckey = compute_cache_key(
            chunking_config=chunker_cfg,
            embedder_model=embedder_id,
            corpus_hash=chash,
            extra=m6_extra,
        )
        cdir = CacheDir(paths.cache_dir(), self.system_id, ckey)

        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.chunks = load_chunks(cdir.chunks_path)
            self._graph = load_graph(cdir.path)
            self._index_stats = self._collect_stats_from_graph(self._graph)
            self._indexed = True
            return

        print(f"[{self.system_id}] cache miss -> building index at {cdir.path}")

        # --- Parse + chunk ---
        docs = list(walk_corpus(corpus_path, min_chars=chunker_cfg.min_chars_per_doc))
        embedder = (
            load_embedder(embedder_id) if chunker_cfg.strategy == "semantic" else None
        )
        self.chunks = chunk_corpus(docs, chunker_cfg, embedder=embedder)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        # --- HARD GATE: Contriever pooling parity ---
        # Run BEFORE phrase embedding so a pooling mismatch crashes
        # immediately, not after spending LLM calls on OpenIE.
        pooling_report = verify_contriever_pooling(embedder_id)
        print(
            f"[{self.system_id}] Contriever pooling gate: passed "
            f"(min_cosine={pooling_report['min_cosine']:.4f}, "
            f"floor={pooling_report['cosine_floor']})"
        )

        # --- OpenIE per chunk (parallel) ---
        passages = [c.text for c in self.chunks]

        def _progress(done: int, total: int) -> None:
            if done == total or done % max(1, total // 10) == 0:
                print(f"[{self.system_id}] OpenIE {done}/{total}")

        openie_results: list[OpenIEResult] = extract_corpus_parallel(
            passages,
            llm_model=openie_llm,
            max_workers=8,
            on_progress=_progress,
        )
        n_openie_failures = sum(1 for r in openie_results if not r.parse_ok)
        n_openie_tokens = sum(r.n_tokens for r in openie_results)
        print(
            f"[{self.system_id}] OpenIE done: {len(openie_results)} passages, "
            f"{n_openie_failures} parse failures, {n_openie_tokens} tokens"
        )

        # --- Graph build: phrases, facts, sparse mats ---
        (
            phrase_to_id,
            fact_to_id,
            d2f_mat,
            f2p_mat,
            phrase_to_num_doc,
        ) = build_graph_structures(openie_results)
        n_phrases = len(phrase_to_id)
        n_facts = len(fact_to_id)
        print(
            f"[{self.system_id}] graph structures: "
            f"{n_phrases} phrases, {n_facts} facts"
        )
        if n_phrases == 0:
            raise RuntimeError(
                "M6 index produced zero phrases — OpenIE returned no usable "
                "triples on the entire corpus. Check OpenIE parse failures + "
                "the corpus content."
            )

        # --- Triple-derived edges ---
        edge_dict = build_triple_edges(fact_to_id, phrase_to_id)
        n_triple_edges = len(edge_dict)

        # --- Contriever phrase embedding ---
        unique_phrases_in_id_order = [None] * n_phrases
        for p, i in phrase_to_id.items():
            unique_phrases_in_id_order[i] = p
        phrase_embeddings = embed_phrases(
            unique_phrases_in_id_order,
            embedder_id=embedder_id,
        )

        # --- Synonymy edges ---
        edge_dict, n_synonymy = add_synonymy_edges(
            edge_dict,
            phrase_embeddings,
            sim_threshold=m6.sim_threshold,
            top_k_cap=m6.synonym_top_k_cap,
        )
        print(
            f"[{self.system_id}] edges: {n_triple_edges} triple + {n_synonymy} synonymy "
            f"= {len(edge_dict)} total"
        )

        # --- Assemble igraph ---
        igraph_graph = assemble_igraph(edge_dict, n_phrases=n_phrases)

        # --- Bundle + persist ---
        edges_list: list[tuple[int, int, float]] = [
            (h, t, w) for (h, t), w in edge_dict.items()
        ]
        self._graph = HippoGraph(
            phrase_to_id=phrase_to_id,
            fact_to_id=fact_to_id,
            docs_to_facts_mat=d2f_mat,
            facts_to_phrases_mat=f2p_mat,
            edges=edges_list,
            graph=igraph_graph,
            phrase_embeddings=phrase_embeddings,
            phrase_to_num_doc=phrase_to_num_doc,
            n_passages=len(openie_results),
            n_triple_edges=n_triple_edges,
            n_synonymy_edges=n_synonymy,
            n_openie_parse_failures=n_openie_failures,
        )

        # Persist:
        cdir.path.mkdir(parents=True, exist_ok=True)
        save_chunks(self.chunks, cdir.chunks_path)
        # OpenIE results json (faithful to legacy on-disk layout).
        import json
        (cdir.path / "openie.json").write_text(
            json.dumps(
                [
                    {
                        "idx": r.idx,
                        "passage": r.passage,
                        "extracted_entities": r.extracted_entities,
                        "extracted_triples": r.extracted_triples,
                        "n_tokens": r.n_tokens,
                        "parse_ok": r.parse_ok,
                    }
                    for r in openie_results
                ],
                ensure_ascii=False,
            )
        )
        save_graph(self._graph, cdir.path)

        self._index_stats = self._collect_stats_from_graph(self._graph)
        self._index_stats["pooling_gate"] = pooling_report
        self._index_stats["n_openie_tokens"] = int(n_openie_tokens)

        Manifest(
            system_id=self.system_id,
            cache_key=ckey,
            chunking_config=asdict(chunker_cfg),
            embedder_model=embedder_id,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
            extra={
                "m6": asdict(m6),
                "index_stats": self._index_stats,
            },
        ).save(cdir.manifest_path)

        self._indexed = True

    @staticmethod
    def _collect_stats_from_graph(graph: HippoGraph) -> dict:
        return {
            "n_phrases": len(graph.phrase_to_id),
            "n_facts": len(graph.fact_to_id),
            "n_edges_total": len(graph.edges),
            "n_triple_edges": graph.n_triple_edges,
            "n_synonymy_edges": graph.n_synonymy_edges,
            "n_openie_parse_failures": graph.n_openie_parse_failures,
            "n_passages": graph.n_passages,
            "phrase_embedding_dim": int(graph.phrase_embeddings.shape[1])
            if graph.phrase_embeddings.size
            else 0,
        }

    # --- retrieve ---------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        self._require_indexed()
        assert self._resolved is not None
        assert self._graph is not None

        m6 = self.config.m6
        # CK-4: default to RETRIEVAL_RANKING_DEPTH (=50). Top-k from
        # the PPR-ranked doc_prob; deeper ranking has identical head.
        k = k or RETRIEVAL_RANKING_DEPTH
        embedder_id = self._resolved.embedder_id
        openie_llm = self._resolved.index_llm_id
        assert openie_llm is not None

        # Query NER.
        ner_strings, n_tokens = extract_query_entities(query, llm_model=openie_llm)
        # Soft cap to bound personalisation vector (M6Config.max_query_ner).
        if m6.max_query_ner > 0 and len(ner_strings) > m6.max_query_ner:
            ner_strings = ner_strings[: m6.max_query_ner]

        empty_ner = len(ner_strings) == 0
        trace: dict = {
            "n_query_ner": len(ner_strings),
            "query_ner": list(ner_strings),
            "empty_ner": empty_ner,
            "n_tokens_query_ner": int(n_tokens),
        }

        n_chunks = len(self.chunks)

        if empty_ner:
            # PROMINENT empty-NER log per paper-faithful uniform fallback.
            # Tracked in index_stats for a running counter across queries.
            self._index_stats["n_empty_ner_queries"] = int(
                self._index_stats.get("n_empty_ner_queries", 0)
            ) + 1
            print(
                f"[{self.system_id}] EMPTY-NER FALLBACK on query: {query!r}. "
                f"Returning uniform doc_prob (paper-faithful). "
                f"Total empty-NER count this session: "
                f"{self._index_stats['n_empty_ner_queries']}"
            )
            doc_prob = uniform_fallback(n_chunks)
            trace["mode"] = "uniform_fallback"
        else:
            # Dense link query NER -> phrase nodes.
            linked = link_query_entities_to_phrases(
                ner_strings,
                self._graph.phrase_embeddings,
                embedder_id=embedder_id,
            )
            trace["n_linked"] = len(linked)
            trace["linked_phrase_ids"] = [pid for pid, _ in linked]

            # Personalisation vector + PPR.
            reset_vec = build_reset_vector(
                linked,
                n_phrases=self._graph.phrase_embeddings.shape[0],
                phrase_to_num_doc=self._graph.phrase_to_num_doc,
                node_specificity=m6.node_specificity,
            )
            ppr = run_pagerank(self._graph.graph, reset_vec, damping=m6.damping)
            doc_prob = propagate_phrase_to_doc(
                ppr,
                self._graph.docs_to_facts_mat,
                self._graph.facts_to_phrases_mat,
            )
            trace["mode"] = "ppr"

        self._last_trace = trace

        # Top-k by descending doc_prob. argsort + stable, matches legacy
        # `np.argsort(doc_prob, kind='mergesort')[::-1]`.
        order = np.argsort(doc_prob, kind="mergesort")[::-1][:k]
        out: list[RetrievedChunk] = []
        for rank, idx in enumerate(order.tolist()):
            out.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(doc_prob[idx]),
                    rank=rank,
                )
            )
        return out

    # answer() inherits BaseSystem default (CK-4 shared packer).
    # M6 surfaces its last_trace (empty-NER, n_linked, mode) via the
    # property; the runner stores trace separately, no need to wrap
    # it through AnswerResult.extra.


__all__ = ["HippoRAGSystem"]
