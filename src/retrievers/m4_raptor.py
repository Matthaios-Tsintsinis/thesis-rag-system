"""M4 — RAPTOR collapsed retrieval, paper-faithful (Sarthi et al., ICLR 2024).

Rebuilt 2026-07-29 under the M4 paper-fidelity directive: M4 is to be as
close to arXiv:2401.18059 as it is possible to make it, and the only
admissible reasons to deviate are dramatic cost, literal impossibility,
or an infrastructure contract the harness cannot function without.
Comparability with the other systems, harness uniformity, implementation
effort, and "it might beat M7" are explicitly NOT reasons. A faithful
RAPTOR beating M7 is a real result.

Pipeline, end to end:
  chunk        100 tiktoken tokens, sentence-preserving, no overlap
               (raptor_paper.split_text_raptor)
  embed        SBERT multi-qa-mpnet-base-cos-v1, the checkpoint the paper names
  tree         BOTTOM-UP: UMAP global+local -> BIC-selected GMM soft
               clustering -> summarise each cluster -> re-embed -> repeat
               (raptor_paper.build_paper_tree)
  retrieve     collapse the ENTIRE tree into one layer, rank ALL nodes by
               dense cosine, take top-k
  context      retrieved summary nodes go into the prompt AS THEIR OWN TEXT

# === DEVIATIONS FROM THE RAPTOR PAPER — thesis footnote ===
# 1. Summariser. The paper uses gpt-3.5-turbo. GENUINELY FORCED: the
#    model is deprecated and cannot be relied on for the lifetime of the
#    thesis. M4Config.summary_model carries the replacement.
# 2. Reader. The paper reads with GPT-3.5 / GPT-4 / UnifiedQA-3B; this
#    harness holds ONE reader constant across every system so that
#    per-system deltas attribute to retrieval rather than to reader
#    capacity. Experimental control, not a fidelity choice.
# 3. Reader prompt. The paper's QA prompt ("You are Question Answering
#    Portal" / "Given Context: ... Give the best full answer amongst the
#    option to question ...") is replaced by the harness-wide answer
#    prompt, whose exact abstention string is load-bearing for the
#    unanswerable / abstention scorers across every benchmark.
# 4. Summarisation temperature is 0.0; the reference leaves it unset
#    (=1.0). Infrastructure contract: a cache key must determine the
#    artifact it names. Consequence recorded — the reference's own trees
#    are not reproducible run to run.
# 5. UMAP is seeded and the re-cluster recursion is depth-guarded; the
#    reference does neither. See the micro-divergence block in
#    src/raptor_paper.py for all four, with reasoning.
# 6. Chunk terminators are restored rather than discarded. See ruling 1
#    and its newline sub-ruling in src/raptor_paper.py.
# 7. NO SPARSE RETRIEVAL. The paper's collapsed retrieval is dense cosine
#    only, so this rebuild drops the BM25 index and the opt-in
#    dense+BM25 RRF first stage that the previous M4 carried. Both
#    existed because M4 used to SHARE a substrate with M7, which needs
#    BM25; M4 now owns its namespace and the paper has no sparse
#    component, so carrying one would be non-paper machinery kept for no
#    reason. rrf_k / first_stage_top_k remain on M4Config only so that
#    existing callers construct.
#
# NOT a deviation, recorded because it looks like one: retrieved SUMMARY
# nodes carry an EMPTY gold_provenance, so the CK-2 retrieval scorer
# cannot credit them. That is honest rather than convenient — a summary
# is abstractive text with no gold span, and crediting it with its
# descendants' atoms would inflate recall while collapsing precision, and
# would destroy MultiHop's document ranking outright. The consequence is
# real and must be reported: M4's retrieval-F1 is NOT directly comparable
# to systems that return only leaf chunks, because 18.5-57% of its
# retrieved units (paper App. I) are unscoreable by construction. The
# leaf-expanded diagnostic twin exists to quantify exactly that gap.

Cache: M4 owns `cache/M4_RAPTOR/<key>/`, its own namespace. It no longer
shares the RAPTOR/ namespace with the frozen M7 — the tree schema, the
chunker and the clustering algorithm all differ. The key is derived by
raptor_paper.paper_substrate_extra(), which emits the same seven base
fields the shared extras emit plus the M4-only ones, so src/raptor.py is
never opened and M7's key cannot move by construction.
"""

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
from ..config import DEFAULT_CONFIG, HarnessConfig
from ..models import embed_texts, load_embedder
from ..parsing import walk_corpus
from ..raptor_paper import (
    PaperCollapsedIndex,
    PaperNode,
    PaperTree,
    build_collapsed_index,
    build_paper_tree,
    load_collapsed_index,
    load_paper_tree,
    paper_substrate_extra,
    save_collapsed_index,
    save_paper_tree,
    summarize_paper_style,
    tree_stats,
)
from .base import BaseSystem, RetrievedChunk


# Own namespace: the artifacts are not interchangeable with the legacy
# RAPTOR/ substrate that M7 consumes.
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
    """Map a bottom-up layer index onto the harness's node-type labels.

    The analyser slices on {chunk, summary_low, summary_mid, summary_high}
    (config.NodeType), which were defined for a TOP-DOWN tree where depth
    0 is the root. Bottom-up inverts that: layer 0 is the leaves and the
    top layer is the broadest summary. Layer 1 is therefore the LOWEST
    (most specific) summary tier, and the mapping keeps existing analyser
    slices meaningful without redefining the vocabulary.
    """
    if layer <= 0:
        return "chunk"
    if layer == 1:
        return "summary_low"
    if layer == 2:
        return "summary_mid"
    return "summary_high"


class RaptorSystem(BaseSystem):
    system_id = "M4"

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self.chunk_embeddings: np.ndarray | None = None
        self._tree: PaperTree | None = None
        self._flat: PaperCollapsedIndex | None = None
        self._index_stats: dict = {}
        self._last_trace: dict = {}
        self._resolved: ResolvedComponents | None = None

    # --- cache identity ---------------------------------------------------

    def _cache_dir(self, corpus_hash: str) -> CacheDir:
        assert self._resolved is not None
        m4 = self.config.m4
        extra = paper_substrate_extra(
            params=m4.paper,
            summary_model=m4.summary_model,
            summary_prompt_version=m4.summary_prompt_version,
            summary_max_tokens=m4.summary_max_tokens,
            rrf_k=m4.rrf_k,
            include_root=m4.include_root_in_flat_index,
        )
        key = compute_cache_key(
            chunking_config=self._resolved.chunker_config,
            embedder_model=self._resolved.embedder_id,
            corpus_hash=corpus_hash,
            extra=extra,
        )
        return CacheDir(paths.cache_dir(), M4_SUBSTRATE_NAMESPACE, key)

    # --- index ------------------------------------------------------------

    def index(self, corpus_path: Path) -> None:
        m4 = self.config.m4
        # M4 does not rerank — the published pipeline has none.
        self._resolved = resolve_components(m4, self.config, default_reranker=None)
        print(f"[components] {format_components_log(self.system_id, self._resolved)}")
        chunker_cfg = self._resolved.chunker_config
        embedder_id = self._resolved.embedder_id

        corpus_path = Path(corpus_path)
        chash = corpus_content_hash(corpus_path)
        cdir = self._cache_dir(chash)

        if cdir.is_complete(REQUIRED_FILES):
            print(f"[{self.system_id}] cache hit: {cdir.path}")
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
            self._indexed = True
            return

        print(f"[{self.system_id}] cache miss -> building index at {cdir.path}")
        docs = list(walk_corpus(corpus_path, min_chars=chunker_cfg.min_chars_per_doc))
        embedder = (
            load_embedder(embedder_id) if chunker_cfg.strategy == "semantic" else None
        )
        self.chunks = chunk_corpus(docs, chunker_cfg, embedder=embedder)
        if not self.chunks:
            raise RuntimeError(f"No chunks produced from {corpus_path}")

        self.chunk_embeddings = embed_texts(
            [c.text for c in self.chunks], model_name=embedder_id
        )

        summary_calls = [0]

        def _on_summary(_n: PaperNode) -> None:
            summary_calls[0] += 1
            if summary_calls[0] % 200 == 0:
                print(f"[{self.system_id}] {summary_calls[0]} summaries...")

        def _summarize(context: str) -> str:
            return summarize_paper_style(
                context,
                model=m4.summary_model,
                max_tokens=m4.summary_max_tokens,
            )

        def _embed(texts: list[str]) -> np.ndarray:
            return embed_texts(texts, model_name=embedder_id)

        self._tree = build_paper_tree(
            [c.text for c in self.chunks],
            self.chunk_embeddings,
            params=m4.paper,
            summarize_fn=_summarize,
            embed_fn=_embed,
            on_summary=_on_summary,
            max_workers=m4.summary_max_workers,
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
                # Runtime identity: local decoding is not bit-identical
                # across GPU generations or library versions, so a tree is
                # reproducible against a PINNED runtime rather than
                # absolutely. Recorded so a mismatch is visible, not silent.
                "summariser_runtime": self._summariser_runtime(),
            },
        ).save(cdir.manifest_path)

        self._indexed = True

    def _summariser_runtime(self) -> dict:
        from ..models import generator_identity

        try:
            return generator_identity(
                self.config.m4.summary_model, load_in_4bit=False
            )
        except Exception:  # torch absent on a CPU-only host
            return {"generator_model": self.config.m4.summary_model}

    @property
    def resolved_components(self) -> ResolvedComponents | None:
        return self._resolved

    def _collect_index_stats(self) -> dict:
        assert self._tree is not None and self._flat is not None
        stats = dict(tree_stats(self._tree))
        type_counts = Counter(
            _layer_to_unit_type(int(r["layer"])) for r in self._flat.refs
        )
        stats.update({
            "n_summary_calls_at_index": int(self._tree.stats.get("n_summary_calls", 0)),
            # Legacy key names kept so existing smoke assertions and the
            # analyser keep reading M4 without a parallel vocabulary.
            "tree_n_nodes": stats["n_nodes"],
            "tree_depth_counts": stats["layer_sizes"],
            "flat_n_chunks": int(type_counts.get("chunk", 0)),
            "flat_n_summaries": int(
                sum(v for k, v in type_counts.items() if k != "chunk")
            ),
            "flat_node_type_counts": {k: int(v) for k, v in type_counts.items()},
            # FIDELITY GATES (paper App. C / App. I). children/parent and
            # non-leaf share are pass/fail; mean summary length is
            # INFORMATIONAL ONLY — the paper's 131 was measured with a
            # different summariser, and ruling 4 caps completions at 100.
            "gate_children_per_parent": stats["mean_children_per_parent"],
            "gate_mean_summary_tokens": stats["mean_summary_tokens"],
            # Non-zero on either counter is a FINDING to report, not an
            # error to silence. no_progress_trips counts layers the GMM
            # could not split at all (BIC chose k=1); recluster_guard_trips
            # counts the depth bound firing.
            "recluster_guard_trips": int(
                self._tree.stats.get("recluster_guard_trips", 0)
            ),
            "no_progress_trips": int(
                self._tree.stats.get("no_progress_trips", 0)
            ),
        })
        return stats

    @property
    def index_stats(self) -> dict:
        return dict(self._index_stats)

    @property
    def last_trace(self) -> dict:
        return dict(self._last_trace)

    # --- retrieve ---------------------------------------------------------

    def _node_as_chunk(self, node: PaperNode) -> Chunk:
        """Wrap a summary node so it can travel the RetrievedChunk path.

        gold_provenance stays EMPTY — see the module block. A summary is
        abstractive text with no gold span; crediting it with its
        descendants' atoms would inflate recall, collapse precision, and
        wreck MultiHop's document ranking.
        """
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
        self._require_indexed()
        assert self._resolved is not None
        assert self._flat is not None and self._tree is not None
        m4 = self.config.m4
        k = k or m4.top_k_final

        q_vec = embed_texts([query], model_name=self._resolved.embedder_id)

        # Collapsed retrieval: dense cosine over EVERY node of the tree,
        # leaves and summaries alike (IndexFlatIP over L2-normalised
        # vectors == cosine). No sparse component, no fusion, no
        # expansion — the paper ranks the flattened node set directly.
        n_flat = len(self._flat.refs)
        scores, idx = self._flat.faiss_index.search(q_vec, min(k, n_flat))

        out: list[RetrievedChunk] = []
        type_counter: Counter[str] = Counter()
        paths_exercised: set[str] = set()

        for score, pos in zip(scores[0].tolist(), idx[0].tolist()):
            if pos < 0:
                continue
            ref = self._flat.refs[pos]
            node = self._tree.nodes[ref["node_id"]]
            unit_type = _layer_to_unit_type(node.layer)
            type_counter[unit_type] += 1
            paths_exercised.add(
                "leaf" if node.is_leaf else unit_type.replace("summary_", "")
            )

            if node.is_leaf:
                # Leaf nodes ARE corpus chunks; hand back the real Chunk so
                # gold_provenance (stamped by index_items) survives.
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
            if len(out) >= k:
                break

        self._last_trace = {
            "collapsed_top_node_types": dict(type_counter),
            "paths_exercised": sorted(paths_exercised),
            "n_returned": len(out),
            # The App. I fidelity gate, per query. Retrieval-only: it
            # needs no generation, so it is measurable in the cheap stage.
            "non_leaf_share": (
                sum(v for kk, v in type_counter.items() if kk != "chunk")
                / max(1, sum(type_counter.values()))
            ),
        }
        return out

    # answer() inherits the BaseSystem default: retrieved node TEXT —
    # summaries verbatim included — is concatenated into the evidence
    # block, which is the paper's behaviour.
