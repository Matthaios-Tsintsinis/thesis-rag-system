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
# 4. MINIMUM CORPUS SIZE — 83 of 1,000 HotpotQA-distractor units (8.3%)
#    fall below it. MEASURED FROM BANKED CELL 6 by `analyse` counting
#    `metadata.m4_tree_degenerate` on the rows the cell actually produced
#    (2026-08-22). So:
#
#      917/1000 (91.7%) build a 2-layer hierarchy
#       83/1000  (8.3%) fall at or below the 11-leaf stop condition
#
#    THE OLD 3.6% (36/1000) FIGURE IS DEAD. Not superseded pending
#    something — dead. It was computed on 2026-08-16 from an inventory
#    that predates the single-item-rule corpus layout now in
#    `BaseSystem.index_items`, so it describes a population this code no
#    longer produces. It must not appear anywhere.
#
#    HOW THE DRIFT WAS LOCALISED, kept because the method is the useful
#    part: a re-derivation under current code reproduced MultiHop's leaf
#    population EXACTLY (16,523, matching its banked cell) while HotpotQA
#    moved to 17,443 leaves / 83 degenerate. A benchmark-specific change
#    was therefore implicated rather than a chunker change — and the
#    layout promotion is exactly that, its own docstring in
#    retrievers/base.py saying it "changes HotpotQA". The re-derivation's
#    83 then matched the banked count exactly, which is what makes the
#    diagnosis more than a coincidence.
#
#    `analyse` over a banked cell stays the authority for any cell that
#    has run; `scripts/measure_chunk_population.py` estimates one that
#    has not. Never quote the estimate for a cell that has run.
#
#    RAPTOR stops when a layer holds `<= reduction_dimension + 1` = 11
#    nodes, checked BEFORE the first clustering pass. So the cell is
#    OVERWHELMINGLY A REAL RAPTOR RESULT: 917 units build layer 1 (the
#    largest, 37 leaves, gives layer_sizes {0:15, 1:3} at 15 leaves and
#    scales from there), while 83 yield layer 0 only — no UMAP, no GMM,
#    no summaries — and are scored on flat dense retrieval with M4's OWN
#    components (mpnet, 100-token chunks, 2,000-token budget), which is
#    NOT M2, whose embedder, chunker and context budget all differ.
#
#    State the 8.3% rather than characterising the cell as "mixed": the
#    tail is small and naming it as a mix overstates it. `analyse`
#    reports the per-cell degenerate-row count at run time and the
#    results caption carries that number.
#
#    THE 83 ALSO MATTER FOR THE APP. I GATE, which FAILED on this cell at
#    16.4% micro / 15.6% macro against the paper's 18.5-57.0%. A
#    degenerate unit contributes leaves and zero summaries, so it
#    mechanically depresses a micro-average over a mixed population.
#    `analyse` therefore reports the gate BOTH over all rows and over
#    tree-building rows only, and the caption states which population
#    each figure describes. The split is a diagnostic, not an excuse: if
#    the 917-unit figure is still out of band, that is a real property of
#    RAPTOR on ~18-leaf corpora and belongs in the discussion.
#
#    The paper never tests this regime — RAPTOR's own corpora are far
#    above the threshold — so the small-corpus tail is a measured
#    property of the method, reportable as such.
# 5. Summarisation temperature is 0.0; the reference leaves it unset
#    (=1.0). Infrastructure contract: a cache key must determine the
#    artifact it names. Consequence recorded — the reference's own trees
#    are not reproducible run to run.
# 6. UMAP is seeded and the re-cluster recursion is depth-guarded; the
#    reference does neither. See the numbered micro-divergence block in
#    src/raptor_paper.py for all SEVEN, with reasoning.
# 7. Chunk terminators are restored rather than discarded (ruling 1 and
#    its newline sub-ruling), and the pieces of an over-long sentence
#    are placed in DOCUMENT ORDER rather than the reference's order
#    (ruling 1b). Both in src/raptor_paper.py.
# 8. NO SPARSE RETRIEVAL. The paper's collapsed retrieval is dense cosine
#    only, so this rebuild drops the BM25 index and the opt-in
#    dense+BM25 RRF first stage that the previous M4 carried. Both
#    existed because M4 used to SHARE a substrate with M7, which needs
#    BM25; M4 now owns its namespace and the paper has no sparse
#    component, so carrying one would be non-paper machinery kept for no
#    reason. rrf_k / first_stage_top_k remain on M4Config only so that
#    existing callers construct.
#
# 9. Retrieval budget RESTORED (professor-approved 2026-08-02), M4 ONLY.
#    The paper fills a 2,000-token context budget (~top-20 nodes)
#    rather than taking a fixed top-k. M1/M2/M3/M9 stay at natural
#    top-15 — their papers specify no budget, so moving them would be a
#    feasibility change wearing a fidelity justification. CONSEQUENT
#    ASYMMETRY, stated not hidden: M4 answers from ~2,000 evidence
#    tokens against their ~3,900, i.e. roughly half the context.
#    OBSERVED PAPER/CODE DIVERGENCE: the reference applies
#    indices[:top_k] with top_k=10 BEFORE its 3500-token cap, so it
#    retrieves ~10 nodes (~1,000 tokens) and the cap never binds. We
#    follow the PAPER TEXT, which is what the thesis cites.
#
# NOT a deviation, recorded because it looks like one: retrieved SUMMARY
# nodes carry an EMPTY gold_provenance, so the CK-2 retrieval scorer
# cannot credit them. That is honest rather than convenient — a summary
# is abstractive text with no gold span, and crediting it with its
# descendants' atoms would inflate recall while collapsing precision, and
# would destroy MultiHop's document ranking outright. The consequence is
# real and must be reported: M4's retrieval-F1 is NOT directly comparable
# to systems that return only leaf chunks, because its summary units are
# unscoreable by construction. The PAPER's band for that share is 18.5-57%
# (App. I); our MEASURED share is per-cell and is reported by `analyse`
# (NarrativeQA 25.5% micro, in band; HotpotQA-distractor 16.4% over all
# rows, below band, with the tree-building-only figure reported beside
# it). The leaf-expanded diagnostic twin exists to quantify the gap.

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
from ..config import DEFAULT_CONFIG, RETRIEVAL_RANKING_DEPTH, HarnessConfig
from ..models import embed_texts, load_embedder
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
    # M4 is the only system whose index() writes a key-addressed
    # substrate expensive enough that serving a warm one silently matters.
    has_cacheable_substrate = True

    def __init__(self, config: HarnessConfig = DEFAULT_CONFIG) -> None:
        super().__init__(config)
        self.chunks: list[Chunk] = []
        self.chunk_embeddings: np.ndarray | None = None
        self._tree: PaperTree | None = None
        self._flat: PaperCollapsedIndex | None = None
        self._index_stats: dict = {}
        # Set by index(): True when the substrate was served warm. P10's
        # preflight FAILS on a True here, because the cold-tree lever not
        # taking is exactly the failure it exists to catch.
        self.tree_cache_hit: bool | None = None
        self._last_trace: dict = {}
        self._resolved: ResolvedComponents | None = None
        # REPLAY-ONLY key injection (scripts/replay_retrieval). When set,
        # the substrate key is computed with THIS build_env string instead
        # of the host's PAPER_TREE_BUILD_ENV -- how a replay resolves a
        # tree banked under a pre-e907d68 (token-less) key after host
        # compatibility is asserted. None (the default, and the only
        # value the runner ever leaves it at) keeps the key computation
        # BYTE-IDENTICAL: paper_substrate_extra falls back to the module
        # constant. The safe-lever pattern from the cache-key rulings.
        self.topology_env_override: str | None = None

    # --- cache identity ---------------------------------------------------

    def _cache_dir(self, corpus_hash: str) -> CacheDir:
        assert self._resolved is not None
        m4 = self.config.m4
        extra = paper_substrate_extra(
            params=m4.paper,
            summary_model=m4.summary_model,
            summary_prompt_version=m4.summary_prompt_version,
            summary_max_tokens=m4.summary_max_tokens,
            # Batch shape is part of the artifact: composition can move
            # generated text at temperature 0, and summaries are cached.
            summary_batch_size=m4.summary_batch_size,
            summary_max_padded_tokens=m4.summary_max_padded_tokens,
            rrf_k=m4.rrf_k,
            include_root=m4.include_root_in_flat_index,
            build_env=self.topology_env_override,
        )
        key = compute_cache_key(
            chunking_config=self._resolved.chunker_config,
            embedder_model=self._resolved.embedder_id,
            corpus_hash=corpus_hash,
            extra=extra,
        )
        return CacheDir(paths.cache_dir(), M4_SUBSTRATE_NAMESPACE, key)

    def substrate_warm_path(self, items) -> str | None:
        """Would this unit's tree be served from cache? Read-only.

        Materialises the corpus layout through the SHARED
        `_write_corpus_layout`, so the hash this computes is the hash
        `index_items` would compute — a preflight that wrote the corpus
        even slightly differently would answer about a substrate no cell
        would ever use. No embedding, no clustering, no summarisation:
        writing text files and hashing them.

        Used by the runner's cold-tree PREFLIGHT, which scans every unit
        before indexing anything so all warm substrates are reported at
        once. The per-unit gate downstream remains as a backstop.
        """
        import tempfile

        if self._resolved is None:
            self._resolved = resolve_components(
                self.config.m4, self.config, default_reranker=None
            )
        with tempfile.TemporaryDirectory(prefix="M4_warmcheck_") as td:
            td_path = Path(td)
            self._write_corpus_layout(items, td_path)
            chash = corpus_content_hash(td_path)
        cdir = self._cache_dir(chash)
        return str(cdir.path) if cdir.is_complete(REQUIRED_FILES) else None

    def _guard_index_llm(self) -> None:
        """Refuse to start a tree build with an API index-time LLM.

        A RAPTOR build is the most expensive operation in the harness,
        and `summary_model` is baked into the substrate cache key — so
        building with the wrong summariser does not merely cost money,
        it produces an artifact that is discarded wholesale the moment
        the intended model is configured.

        This exists because it already happened: a run began building
        `M4_RAPTOR/34b630d8...` with gpt-4o-mini summaries and was
        stopped only by a missing API key. An absent credential is not a
        safety mechanism — it fails for the right reason by accident,
        and it stops failing the moment a key is exported for some
        unrelated purpose. The guard is the mechanism; the missing key
        was luck.

        Fires only on the cache-MISS path: reading a tree that was
        legitimately built earlier costs nothing and stays allowed.
        """
        import os

        from ..models import _is_openai_model

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

        self._guard_index_llm()
        print(f"[{self.system_id}] cache miss -> building index at {cdir.path}")
        self.tree_cache_hit = False
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

        # Post-hoc count only. Since summarisation is batched per LAYER,
        # these fire in bursts once a layer's call returns; live progress
        # inside a layer comes from generate_batch's own reporting.
        def _on_summary(_n: PaperNode) -> None:
            summary_calls[0] += 1
            if summary_calls[0] % 200 == 0:
                print(f"[{self.system_id}] {summary_calls[0]} summaries...")

        def _summarize_batch(contexts: list[str]) -> list[str]:
            # ONE call per tree layer. Bottom-up construction produces a
            # whole layer's clusters at once, so the layer is the natural
            # batch, and handing over the full layer lets generate_batch
            # length-sort across all of it rather than within arbitrary
            # pre-cut groups.
            return summarize_paper_style_batch(
                contexts,
                model=m4.summary_model,
                max_tokens=m4.summary_max_tokens,
                batch_size=m4.summary_batch_size,
                max_padded_tokens=m4.summary_max_padded_tokens,
            )

        def _embed(texts: list[str]) -> np.ndarray:
            return embed_texts(texts, model_name=embedder_id)

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
                # Runtime identity: local decoding is not bit-identical
                # across GPU generations or library versions, so a tree is
                # reproducible against a PINNED runtime rather than
                # absolutely. Recorded so a mismatch is visible, not silent.
                "summariser_runtime": self._summariser_runtime(),
                # Which stack built this tree, recorded beside the key it
                # is named by.
                "build_env": PAPER_TREE_BUILD_ENV,
            },
        ).save(cdir.manifest_path)

        self._indexed = True

    def _warn_if_degenerate(self) -> None:
        """Announce a flat index on EVERY path, cache hits included.

        `build_paper_tree` prints this when it builds one, but a cached
        degenerate tree is loaded rather than built, and a warning that
        only fires on the cold path is the same as no warning by the
        second run.
        """
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
        from ..models import generator_identity

        try:
            return generator_identity(self.config.m4.summary_model)
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
            # NAMED FOR WHAT IT IS: this counts NODES summarised, one per
            # summary, NOT model.generate() invocations. 24 nodes may be
            # one batched call or 24 unbatched ones and this number
            # cannot tell them apart — see generate_calls below, which
            # can.
            "n_summary_calls_at_index": int(self._tree.stats.get("n_summary_calls", 0)),
            "n_summary_nodes_at_index": int(self._tree.stats.get("n_summary_calls", 0)),
            # PHASE ATTRIBUTION. Lands in tree.stats from the build
            # clock; copied here because tree_stats() builds its own dict
            # and would otherwise drop it, which is exactly why the first
            # instrumented build reported no phase block.
            "phase_seconds": self._tree.stats.get("phase_seconds"),
            "phase_calls": self._tree.stats.get("phase_calls"),
            "phase_share": self._tree.stats.get("phase_share"),
            "phase_measured_total_s": self._tree.stats.get("phase_measured_total_s"),
            "generate_calls": self._tree.stats.get("generate_calls"),
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
            # Guard (v). Non-zero means the BIC search ran over a REDUCED
            # candidate set because some k could not be fitted — the tree
            # is still valid, but the clustering was chosen from fewer
            # options than the reference nominally considers. Report it.
            "bic_fit_failures": int(
                self._tree.stats.get("bic_fit_failures", 0)
            ),
            "gmm_final_fit_failures": int(
                self._tree.stats.get("gmm_final_fit_failures", 0)
            ),
            # Structural, not a fault: a corpus at or below the layer
            # stop condition yields layer 0 only, so M4 is flat dense
            # retrieval on it. Carried per build AND per query (see
            # prepare()) so it cannot vanish between the index log and
            # the results table.
            "degenerate_no_tree": bool(
                self._tree.stats.get("degenerate_no_tree", False)
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
        # BUDGET MODE vs COUNT MODE. The paper fills a token budget
        # rather than taking a fixed k ("Keep adding nodes to the result
        # set until you reach a predefined maximum number of tokens").
        # An explicit caller-supplied k always wins — CK-2 harness paths
        # and the smoke test pass one, and they mean it.
        budget = m4.retrieval_budget_tokens if k is None else None
        k = k or m4.top_k_final

        q_vec = embed_texts([query], model_name=self._resolved.embedder_id)

        # Collapsed retrieval: dense cosine over EVERY node of the tree,
        # leaves and summaries alike (IndexFlatIP over L2-normalised
        # vectors == cosine). No sparse component, no fusion, no
        # expansion — the paper ranks the flattened node set directly.
        n_flat = len(self._flat.refs)
        # In budget mode the stopping point is not known in advance, so
        # pull a deep-enough candidate pool. RETRIEVAL_RANKING_DEPTH
        # nodes of ~110 tokens is several times any plausible budget.
        depth = max(k, RETRIEVAL_RANKING_DEPTH) if budget else k
        scores, idx = self._flat.faiss_index.search(q_vec, min(depth, n_flat))

        out: list[RetrievedChunk] = []
        type_counter: Counter[str] = Counter()
        paths_exercised: set[str] = set()
        budget_tokens = 0

        for score, pos in zip(scores[0].tolist(), idx[0].tolist()):
            if pos < 0:
                continue
            ref = self._flat.refs[pos]
            node = self._tree.nodes[ref["node_id"]]

            if budget is not None:
                n_tok = count_tokens_plain(node.text)
                # Stop at the first node that would overflow. The
                # reference's `break` is a hard stop, not a skip — it
                # does not continue looking for a smaller node that
                # would still fit, and neither do we.
                if out and budget_tokens + n_tok > budget:
                    break
                budget_tokens += n_tok

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
            if budget is None and len(out) >= k:
                break

        self._last_trace = {
            "collapsed_top_node_types": dict(type_counter),
            "paths_exercised": sorted(paths_exercised),
            "n_returned": len(out),
            "budget_tokens_used": budget_tokens if budget is not None else None,
            "budget_tokens_limit": budget,
            # The App. I fidelity gate, per query. Retrieval-only: it
            # needs no generation, so it is measurable in the cheap stage.
            "non_leaf_share": (
                sum(v for kk, v in type_counter.items() if kk != "chunk")
                / max(1, sum(type_counter.values()))
            ),
        }
        return out

    def prepare(self, query: str, k: int | None = None):
        """Attach M4's per-query diagnostics to the eval row.

        The runner merges `AnswerResult.extra` into `ScoredQuery.metadata`
        under the `m4_*` namespace convention, which is the only way the
        App. I non-leaf-share gate reaches the JSONL — `retrieve()`'s
        trace is otherwise discarded per query.
        """
        prepared = super().prepare(query, k=k)
        trace = self._last_trace
        prepared.extra.update({
            "m4_non_leaf_share": trace.get("non_leaf_share"),
            "m4_budget_tokens_used": trace.get("budget_tokens_used"),
            # Per-QUERY, not just per-build. With thousands of tiny
            # corpora (HotpotQA standard distractor) the index logs
            # scroll past and only the results table survives, so the
            # flat-index fact has to be IN the results table.
            "m4_tree_degenerate": bool(
                self._index_stats.get("degenerate_no_tree", False)
            ),
            "m4_bic_fit_failures": int(
                self._index_stats.get("bic_fit_failures", 0)
            ),
        })
        # AVAILABILITY, for the App. I gate's ceiling diagnostic (AF-10).
        # The paper's 18.5-57.0% band describes what fraction of RETRIEVED
        # nodes are non-leaf, but that fraction cannot exceed what the
        # POOL offers: a ~18-leaf unit carries one summary layer of ~3
        # nodes, so its pool is ~14-17% non-leaf and the band's floor may
        # be unreachable at that depth. Recording the pool composition per
        # ROW lets `analyse` print retrieved share BESIDE available share,
        # so "below the band" and "below availability" stop being the
        # same sentence. On small corpora retrieval also returns most of
        # the pool, which drives retrieved share toward available share
        # mechanically — the pool size is recorded so a reader can see
        # when the statistic has lost that discriminative power.
        n_chunks = int(self._index_stats.get("flat_n_chunks") or 0)
        n_summaries = int(self._index_stats.get("flat_n_summaries") or 0)
        pool = n_chunks + n_summaries
        prepared.extra.update({
            "m4_pool_n_nodes": pool,
            "m4_pool_non_leaf_available": (n_summaries / pool) if pool else None,
        })
        return prepared

    # answer() inherits the BaseSystem default: retrieved node TEXT —
    # summaries verbatim included — is concatenated into the evidence
    # block, which is the paper's behaviour.
