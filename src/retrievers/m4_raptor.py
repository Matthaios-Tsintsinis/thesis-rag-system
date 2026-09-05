"""M4: paper-faithful RAPTOR collapsed-tree retrieval (Sarthi et al., ICLR 2024).
Chunks, embeds, builds the tree bottom-up, then ranks every node by cosine
and fills a 2,000-token evidence budget."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .. import paths
from ..cache import (
    CacheDir,
    Manifest,
    compute_cache_key,
    corpus_content_hash,
    load_chunks,
    load_embeddings,
    save_chunks,
    save_embeddings,
)
from ..chunking import Chunk, chunk_corpus
from ..components import (
    ResolvedComponents,
    format_components_log,
    resolve_components,
)
from ..config import DEFAULT_CONFIG, RETRIEVAL_RANKING_DEPTH, HarnessConfig
from ..models import embed_texts
from ..parsing import walk_corpus
from ..raptor_paper import (
    PAPER_TREE_BUILD_ENV,
    PaperCollapsedIndex,
    PaperNode,
    PaperTree,
    build_collapsed_index,
    build_paper_tree,
    count_tokens_plain,
    load_collapsed_index,
    load_paper_tree,
    paper_substrate_extra,
    save_collapsed_index,
    save_paper_tree,
    summarize_paper_style_batch,
    tree_stats,
)
from .base import BaseSystem, RetrievedChunk

# M4 keeps its own cache namespace; its artifacts match no other system's.
M4_SUBSTRATE_NAMESPACE = "M4_RAPTOR"

REQUIRED_FILES = (
    "chunks.jsonl",
    "embeddings.npy",
    "paper_tree.json",
    "paper_tree_embeddings.npy",
    "collapsed.index",
    "collapsed_meta.json",
)


def _layer_to_unit_type(layer: int) -> str:
    """Map a bottom-up layer index onto the harness's node-type labels."""
    # Layer 0 is the leaves; layer 1 is the most specific summary tier.
    if layer <= 0:
        return "chunk"
    if layer == 1:
        return "summary_low"
    if layer == 2:
        return "summary_mid"
    return "summary_high"


class RaptorSystem(BaseSystem):
    """RAPTOR: build the summary tree once, retrieve over its collapsed nodes."""

    system_id = "M4"
    # index() writes a substrate expensive enough that a warm hit must be
    # reported rather than served silently.
    has_cacheable_substrate = True

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self.chunk_embeddings: np.ndarray | None = None
        self._tree: PaperTree | None = None
        self._flat: PaperCollapsedIndex | None = None
        self._index_stats: dict = {}
        # Set by index(): True when the substrate comes from cache. The
        # runner's cold-tree preflight refuses a True here.
        self.tree_cache_hit: bool | None = None
        self._last_trace: dict = {}
        self._resolved: ResolvedComponents | None = None
        # Replay-only lever: a build_env string that replaces the host's
        # PAPER_TREE_BUILD_ENV in the substrate key, so a replay can find a
        # tree banked under another stack. None (the runner's value) leaves
        # the key unchanged; paper_substrate_extra uses the module constant.
        self.topology_env_override: str | None = None

    # --- cache identity ---------------------------------------------------

    def _cache_dir(self, corpus_hash: str) -> CacheDir:
        """Resolve the substrate cache directory for this corpus hash."""
        assert self._resolved is not None
        m4 = self.config.m4
        extra = paper_substrate_extra(
            params=m4.paper,
            summary_model=m4.summary_model,
            summary_prompt_version=m4.summary_prompt_version,
            summary_max_tokens=m4.summary_max_tokens,
            # harness choice: batch shape is in the cache key because it can move text at temperature 0
            summary_batch_size=m4.summary_batch_size,
            summary_max_padded_tokens=m4.summary_max_padded_tokens,
            rrf_k=m4.rrf_k,
            include_root=m4.include_root_in_flat_index,
            build_env=self.topology_env_override,
        )
        # harness choice: content-addressed substrates (METHODS §D)
        key = compute_cache_key(
            chunking_config=self._resolved.chunker_config,
            embedder_model=self._resolved.embedder_id,
            corpus_hash=corpus_hash,
            extra=extra,
        )
        return CacheDir(paths.cache_dir(), M4_SUBSTRATE_NAMESPACE, key)

    def substrate_warm_path(self, items) -> str | None:
        """Return the cache path if this unit's tree is already built, else None."""
        import tempfile

        if self._resolved is None:
            self._resolved = resolve_components(
                self.config.m4, self.config, default_reranker=None
            )
        # Write the corpus through the shared layout so the hash is the
        # one index_items computes; no embedding or clustering happens.
        with tempfile.TemporaryDirectory(prefix="M4_warmcheck_") as td:
            td_path = Path(td)
            self._write_corpus_layout(items, td_path)
            chash = corpus_content_hash(td_path)
        cdir = self._cache_dir(chash)
        return str(cdir.path) if cdir.is_complete(REQUIRED_FILES) else None

    def _guard_index_llm(self) -> None:
        """Refuse a cold build with an API summariser unless the override is set."""
        import os

        from ..models import _is_openai_model

        # The summariser is in the cache key, so a tree built with the
        # wrong one is thrown away as soon as the intended one is set.
        # Runs only on the cache-miss path; reading a built tree is free.
        model = self.config.m4.summary_model
        if not _is_openai_model(model):
            return
        if self.config.m4.allow_api_index_llm or os.environ.get(
            "M4_ALLOW_API_INDEX_LLM"
        ):
            print(
                f"[{self.system_id}] WARNING: building a tree with API "
                f"summariser {model!r} — override was set explicitly."
            )
            return
        raise RuntimeError(
            f"refusing to build a RAPTOR tree with the API index-time LLM "
            f"{model!r}.\n"
            "The summariser is part of the substrate cache key, so this "
            "build would produce a tree that is thrown away as soon as the "
            "intended local summariser is configured — and RAPTOR builds "
            "are the most expensive thing in the harness.\n"
            "Set config.m4.summary_model to the intended local model "
            "(JUDGE_MODEL is already local by default), or, if an API "
            "build really is what you want, pass "
            "M4Config(allow_api_index_llm=True) or export "
            "M4_ALLOW_API_INDEX_LLM=1."
        )

    # --- index ------------------------------------------------------------

    def index(self, corpus_path: Path) -> None:
        """Load the substrate from cache, or chunk, embed, build and save it."""
        m4 = self.config.m4
        # M4 does not rerank; the paper's pipeline has none.
        self._resolved = resolve_components(m4, self.config, default_reranker=None)
        print(f"[components] {format_components_log(self.system_id, self._resolved)}")
        chunker_cfg = self._resolved.chunker_config
        embedder_id = self._resolved.embedder_id

        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        cdir = self._cache_dir(chash)

        # Warm path: load every artifact and skip the build.
        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
            self.tree_cache_hit = True
            self.chunks = load_chunks(cdir.chunks_path)
            self.chunk_embeddings = load_embeddings(cdir.embeddings_path)
            self._tree = load_paper_tree(
                cdir.path / "paper_tree.json",
                cdir.path / "paper_tree_embeddings.npy",
            )
            self._flat = load_collapsed_index(
                cdir.path / "collapsed.index",
                cdir.path / "collapsed_meta.json",
            )
            self._index_stats = self._collect_index_stats()
            self._warn_if_degenerate()
            self._indexed = True
            return

        # Cold path: chunk the corpus and embed the leaves.
        self._guard_index_llm()
        print(f"[{self.system_id}] cache miss -> building index at {cdir.path}")
        self.tree_cache_hit = False
        docs = list(walk_corpus(corpus_path, min_chars=chunker_cfg.min_chars_per_doc))
        self.chunks = chunk_corpus(docs, chunker_cfg)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        self.chunk_embeddings = embed_texts(
            [c.text for c in self.chunks], model_name=embedder_id
        )

        summary_calls = [0]

        # Progress count only; a layer's summaries land together, so this
        # fires in bursts.
        def _on_summary(_n: PaperNode) -> None:
            """Count summaries and print a progress line every 200."""
            summary_calls[0] += 1
            if summary_calls[0] % 200 == 0:
                print(f"[{self.system_id}] {summary_calls[0]} summaries...")

        # One call per tree layer, so generate_batch length-sorts across
        # the whole layer rather than within pre-cut groups.
        # deviation from paper (gpt-3.5-turbo is retired; one local summariser per reader column): see METHODS §A.4.2
        def _summarize_batch(contexts: list[str]) -> list[str]:
            """Summarise one layer's cluster contexts with the local model."""
            return summarize_paper_style_batch(
                contexts,
                model=m4.summary_model,
                max_tokens=m4.summary_max_tokens,
                batch_size=m4.summary_batch_size,
                max_padded_tokens=m4.summary_max_padded_tokens,
            )

        def _embed(texts: list[str]) -> np.ndarray:
            """Embed summary texts with the same embedder as the leaves."""
            return embed_texts(texts, model_name=embedder_id)

        # Build the tree bottom-up, collapse it into one flat index, and
        # save every artifact; the manifest goes last.
        self._tree = build_paper_tree(
            [c.text for c in self.chunks],
            self.chunk_embeddings,
            params=m4.paper,
            summarize_batch_fn=_summarize_batch,
            embed_fn=_embed,
            on_summary=_on_summary,
            verbose=True,
        )
        self._flat = build_collapsed_index(self._tree)

        save_chunks(self.chunks, cdir.chunks_path)
        save_embeddings(self.chunk_embeddings, cdir.embeddings_path)
        save_paper_tree(
            self._tree,
            cdir.path / "paper_tree.json",
            cdir.path / "paper_tree_embeddings.npy",
        )
        save_collapsed_index(
            self._flat,
            cdir.path / "collapsed.index",
            cdir.path / "collapsed_meta.json",
        )

        self._index_stats = self._collect_index_stats()
        self._warn_if_degenerate()
        Manifest(
            system_id=M4_SUBSTRATE_NAMESPACE,
            cache_key=cdir.cache_key,
            chunking_config=asdict(chunker_cfg),
            embedder_model=embedder_id,
            corpus_hash=chash,
            n_chunks=len(self.chunks),
            files=list(REQUIRED_FILES),
            extra={
                "m4": asdict(m4),
                "index_stats": self._index_stats,
                # Summariser runtime and topology stack, recorded beside
                # the key: a tree reproduces against a pinned runtime, and
                # a mismatch must be visible.
                "summariser_runtime": self._summariser_runtime(),
                "build_env": PAPER_TREE_BUILD_ENV,
            },
        ).save(cdir.manifest_path)

        self._indexed = True

    def _warn_if_degenerate(self) -> None:
        """Print the flat-index warning on every path, cache hits included."""
        if not self._index_stats.get("degenerate_no_tree"):
            return
        print(
            f"[{self.system_id}] *** FLAT INDEX: "
            f"{self._index_stats.get('n_leaves', 0)} leaves produced no "
            "summary layer. M4 is running as flat dense retrieval on this "
            "corpus (effectively M2 with mpnet). Its rows here are NOT a "
            "RAPTOR result and must not be reported as one. ***"
        )

    def _summariser_runtime(self) -> dict:
        """Record the summariser's runtime identity, or just its name without torch."""
        from ..models import generator_identity

        try:
            return generator_identity(self.config.m4.summary_model)
        except Exception:  # torch absent on a CPU-only host
            return {"generator_model": self.config.m4.summary_model}

    @property
    def resolved_components(self) -> ResolvedComponents | None:
        """Return the components resolved by index(), or None before it."""
        return self._resolved

    def _collect_index_stats(self) -> dict:
        """Assemble the per-build stats block from the tree and the flat index."""
        assert self._tree is not None and self._flat is not None
        stats = dict(tree_stats(self._tree))
        type_counts = Counter(
            _layer_to_unit_type(int(r["layer"])) for r in self._flat.refs
        )
        stats.update({
            # Nodes summarised, one per summary; generate_calls counts the
            # model calls, which batching makes fewer.
            "n_summary_calls_at_index": int(self._tree.stats.get("n_summary_calls", 0)),
            "n_summary_nodes_at_index": int(self._tree.stats.get("n_summary_calls", 0)),
            # Phase timing from the build clock; tree_stats() does not
            # carry it, so copy it across.
            "phase_seconds": self._tree.stats.get("phase_seconds"),
            "phase_calls": self._tree.stats.get("phase_calls"),
            "phase_share": self._tree.stats.get("phase_share"),
            "phase_measured_total_s": self._tree.stats.get("phase_measured_total_s"),
            "generate_calls": self._tree.stats.get("generate_calls"),
            # Tree shape and flat-index composition under the key names
            # the results pipeline reads.
            "tree_n_nodes": stats["n_nodes"],
            "tree_depth_counts": stats["layer_sizes"],
            "flat_n_chunks": int(type_counts.get("chunk", 0)),
            "flat_n_summaries": int(
                sum(v for k, v in type_counts.items() if k != "chunk")
            ),
            "flat_node_type_counts": {k: int(v) for k, v in type_counts.items()},
            # Fidelity gates: children per parent is pass/fail, mean
            # summary length is informational.
            # ref: raptor/tree_builder.py::TreeBuilderConfig @ 7da1d48a (summarization_length=100); the paper's 131 is a measured mean (App. C)
            "gate_children_per_parent": stats["mean_children_per_parent"],
            "gate_mean_summary_tokens": stats["mean_summary_tokens"],
            # Non-zero on either counter is a finding to report.
            # no_progress_trips counts layers BIC left unsplit (k = 1);
            # recluster_guard_trips counts the depth bound firing.
            # deviation from ref (ref recursion has no base case): see METHODS §A.4.4 (ii)
            "recluster_guard_trips": int(
                self._tree.stats.get("recluster_guard_trips", 0)
            ),
            "no_progress_trips": int(
                self._tree.stats.get("no_progress_trips", 0)
            ),
            # Non-zero means the BIC search ran over fewer k than the
            # reference tries; the tree is still valid. Report it.
            # deviation from ref (ref crashes): see METHODS §A.4.4 (v)
            "bic_fit_failures": int(
                self._tree.stats.get("bic_fit_failures", 0)
            ),
            "gmm_final_fit_failures": int(
                self._tree.stats.get("gmm_final_fit_failures", 0)
            ),
            # A corpus at or below the layer stop condition yields layer 0
            # only, so M4 runs as flat dense retrieval on it. Carried per
            # build and per query (see prepare()).
            # ref: raptor/cluster_tree_builder.py @ 7da1d48a (len(layer) <= reduction_dimension + 1)
            "degenerate_no_tree": bool(
                self._tree.stats.get("degenerate_no_tree", False)
            ),
        })
        return stats

    @property
    def index_stats(self) -> dict:
        """Return a copy of the per-build stats block."""
        return dict(self._index_stats)

    @property
    def last_trace(self) -> dict:
        """Return a copy of the trace from the most recent retrieve()."""
        return dict(self._last_trace)

    # --- retrieve ---------------------------------------------------------

    def _node_as_chunk(self, node: PaperNode) -> Chunk:
        """Wrap a summary node as a Chunk with empty gold provenance."""
        # harness choice: a summary is abstractive text with no gold span
        return Chunk(
            chunk_id=node.node_id,
            doc_id="",  # a summary spans many source documents
            text=node.text,
            n_words=len(node.text.split()),
            position=node.layer,
            metadata={
                "raptor_layer": node.layer,
                "n_leaf_descendants": len(node.leaf_indices),
                "n_children": len(node.children),
            },
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Rank every node by cosine and fill the token budget, or take top-k."""
        self._require_indexed()
        assert self._resolved is not None
        assert self._flat is not None and self._tree is not None
        m4 = self.config.m4
        # Budget mode fills tokens; a caller-supplied k switches to count
        # mode.
        # RAPTOR paper §3: "2000 maximum tokens ... top-20 nodes" (paper over repo): see METHODS §A.4.3
        budget = m4.retrieval_budget_tokens if k is None else None
        k = k or m4.top_k_final

        q_vec = embed_texts([query], model_name=self._resolved.embedder_id)

        # Cosine over every node, leaves and summaries alike (inner
        # product over L2-normalised vectors). No sparse leg, no fusion.
        # RAPTOR paper §3: collapsed tree, the paper's main-results strategy
        n_flat = len(self._flat.refs)
        # Budget mode needs a deep candidate pool: RETRIEVAL_RANKING_DEPTH
        # nodes of ~110 tokens covers any plausible budget.
        # harness choice: one scoring depth for every system (METHODS §D)
        depth = max(k, RETRIEVAL_RANKING_DEPTH) if budget else k
        scores, idx = self._flat.faiss_index.search(q_vec, min(depth, n_flat))

        out: list[RetrievedChunk] = []
        type_counter: Counter[str] = Counter()
        paths_exercised: set[str] = set()
        budget_tokens = 0

        # Walk the ranking, admitting nodes until the budget or k is hit.
        for score, pos in zip(scores[0].tolist(), idx[0].tolist()):
            if pos < 0:
                continue
            ref = self._flat.refs[pos]
            node = self._tree.nodes[ref["node_id"]]

            if budget is not None:
                n_tok = count_tokens_plain(node.text)
                # Stop at the first node that would overflow; do not skip
                # it and look for a smaller one. The first node is always
                # admitted.
                # harness choice: unreachable at ~110-token nodes
                if out and budget_tokens + n_tok > budget:
                    break
                budget_tokens += n_tok

            unit_type = _layer_to_unit_type(node.layer)
            type_counter[unit_type] += 1
            paths_exercised.add(
                "leaf" if node.is_leaf else unit_type.replace("summary_", "")
            )

            # Leaves are corpus chunks; return the real Chunk so its
            # gold_provenance survives.
            if node.is_leaf:
                chunk = self.chunks[node.leaf_indices[0]]
            else:
                chunk = self._node_as_chunk(node)

            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    rank=len(out),
                    source_unit_type=unit_type,
                )
            )
            if budget is None and len(out) >= k:
                break

        # Per-query trace; non_leaf_share feeds the App. I gate.
        # RAPTOR paper App. I: non-leaf share of retrieved nodes 18.5-57%
        self._last_trace = {
            "collapsed_top_node_types": dict(type_counter),
            "paths_exercised": sorted(paths_exercised),
            "n_returned": len(out),
            "budget_tokens_used": budget_tokens if budget is not None else None,
            "budget_tokens_limit": budget,
            "non_leaf_share": (
                sum(v for kk, v in type_counter.items() if kk != "chunk")
                / max(1, sum(type_counter.values()))
            ),
        }
        return out

    def prepare(self, query: str, k: int | None = None):
        """Attach M4's per-query diagnostics to the eval row under m4_* keys."""
        prepared = super().prepare(query, k=k)
        trace = self._last_trace
        # The flat-index flag rides on every row so it reaches the results
        # table, where the index log does not.
        prepared.extra.update({
            "m4_non_leaf_share": trace.get("non_leaf_share"),
            "m4_budget_tokens_used": trace.get("budget_tokens_used"),
            "m4_tree_degenerate": bool(
                self._index_stats.get("degenerate_no_tree", False)
            ),
            "m4_bic_fit_failures": int(
                self._index_stats.get("bic_fit_failures", 0)
            ),
        })
        # Record pool composition beside retrieved share: a small pool has
        # few summary nodes and is mostly retrieved, so retrieved share
        # tracks available share and the App. I floor can be out of reach.
        n_chunks = int(self._index_stats.get("flat_n_chunks") or 0)
        n_summaries = int(self._index_stats.get("flat_n_summaries") or 0)
        pool = n_chunks + n_summaries
        prepared.extra.update({
            "m4_pool_n_nodes": pool,
            "m4_pool_non_leaf_available": (n_summaries / pool) if pool else None,
        })
        return prepared

    # answer() inherits the BaseSystem default: retrieved node text,
    # summaries included, goes into the evidence block, as in the paper.
    # harness choice: one reader across all systems (METHODS §D)
