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
# 8. Retrieval budget RESTORED (professor-approved 2026-08-02), M4 ONLY.
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

import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

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
from ..parsing import clean_text, walk_corpus
from ..raptor_paper import (
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
from .base import BaseSystem, RetrievedChunk, _safe_item_filename

if TYPE_CHECKING:
    from ..eval.types import CorpusItem


# Own namespace: the artifacts are not interchangeable with the legacy
# RAPTOR/ substrate that M7 consumes.
M4_SUBSTRATE_NAMESPACE = "M4_RAPTOR"

# Separator between a parent's member items in the per-parent temp file.
# Two newlines because `split_text_raptor` treats a newline RUN as a
# sentence boundary that it then drops, so members never fuse into one
# chunk mid-sentence, and `parsing.clean_text` collapses only runs of
# THREE or more — so this survives the read-back unchanged.
_PARENT_JOIN = "\n\n"


def group_items_by_parent(
    items: Sequence["CorpusItem"],
) -> dict[str, list["CorpusItem"]]:
    """Group CorpusItems by parent_id, first-appearance order preserved.

    Order matters twice over: it fixes the temp-dir write order (and so
    the collision-suffix assignment) and the order of the provenance
    pairs on a chunk. Both must be a function of the input alone.
    """
    groups: dict[str, list["CorpusItem"]] = {}
    for item in items:
        groups.setdefault(item.parent_id, []).append(item)
    return groups


def build_parent_payload(
    members: Sequence["CorpusItem"],
) -> tuple[str, list[tuple[int, int, str]]]:
    """Concatenate a multi-item parent, returning its text and member spans.

    Returns (text, [(start_char, end_char, span_id), ...]) with offsets
    into the returned text.

    THE CLEANING TRAP, measured rather than assumed. `walk_corpus` does
    not hand the chunker the file bytes — it hands it
    `parsing.clean_text(bytes)`, which collapses ` \\t` runs and 3+
    newlines and strips the ends. So chunk offsets live in CLEANED
    coordinates, and naive raw-text offsets would be silently wrong by a
    drifting amount. Worse, clean_text does NOT distribute over the
    join: clean_text("abc  " + sep + "def") is "abc \\n\\ndef" while
    joining the cleaned parts gives "abc\\n\\ndef".

    So each member is cleaned FIRST and the spans are measured on the
    joined result, which is then written to disk as-is. That is sound
    only because clean_text is IDEMPOTENT (verified: collapsing runs
    cannot create new runs), so reading the file back is a no-op and the
    chunker sees exactly the string these offsets index. The caller
    asserts that idempotence rather than trusting it.

    Empty members are skipped: they contribute no text, so no chunk can
    overlap them and no provenance is owed.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for item in members:
        text = clean_text(item.text or "")
        if not text:
            continue
        if parts:
            cursor += len(_PARENT_JOIN)
        parts.append(text)
        spans.append((cursor, cursor + len(text), item.span_id))
        cursor += len(text)
    return _PARENT_JOIN.join(parts), spans

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
            # Batch shape is part of the artifact: composition can move
            # generated text at temperature 0, and summaries are cached.
            summary_batch_size=m4.summary_batch_size,
            summary_max_padded_tokens=m4.summary_max_padded_tokens,
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

        self._guard_index_llm()
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

    # --- per-parent corpus layout (M4-local) ------------------------------

    def index_items(self, items: Sequence["CorpusItem"]) -> None:
        """Write ONE file per parent, not per item, and derive provenance
        from character offsets.

        WHY M4 NEEDS THIS. The paper chunks "the retrieval corpus" into
        contiguous 100-token pieces; the base `index_items` writes one
        file per CorpusItem, so a chunk can never span two items and
        every item boundary becomes a forced chunk boundary. On QASPER
        that means paragraph-fragmented leaves; on HotpotQA's pooled
        corpus it means paragraph-fragmented leaves too. Chunking the
        document the items came from is the paper's behaviour.

        WHY IT IS AN OVERRIDE AND NOT AN EDIT TO BaseSystem. `corpus_hash`
        is computed over the temp directory this writes, so the layout is
        a cache-key input for whichever systems use it. Changing the base
        would move M2's, M3's and M9's substrate keys and discard their
        warm bge-m3 caches — for no benefit, since none of them is
        reproducing a paper that specifies contiguous document chunking.

        THE SINGLE-ITEM RULE (rule B), and why it is not a special case
        bolted on. A parent holding exactly one item is written with the
        BASE's filename (derived from item_id) and the BASE's raw bytes,
        so its file is byte-identical to what the base produces and the
        corpus_hash does not move. Both halves are load-bearing and were
        measured, not reasoned:

          * the FILENAME matters because corpus_content_hash folds the
            relative path, not just the bytes. MultiHop happens to have
            item_id == parent_id so a parent-derived name would collide
            harmlessly, but NarrativeQA and QuALITY use "{id}::<whole>"
            against a bare parent id, so a parent-derived name would move
            their hashes despite being 1:1.
          * the RAW BYTES matter because clean_text is not the identity.
            A multi-item parent is written PRE-CLEANED (see
            build_parent_payload); doing that to a single-item parent
            would change its bytes whenever its text contains a double
            space, and move the hash that way instead.

        A single-item parent also needs no offsets: every chunk of that
        document belongs to that one item, which is exactly what the base
        stamps. Offsets are only required where a parent holds several
        items — and in that case the hash moves regardless.

        SIDE EFFECT, recorded: `walk_corpus` drops documents shorter than
        `min_chars_per_doc`, so concatenating a multi-item parent can
        rescue items that would individually have been dropped. That is a
        fidelity improvement (the paper chunks documents, not fragments)
        rather than a bug, and it cannot affect a 1:1 benchmark.
        """
        groups = group_items_by_parent(items)
        # filename -> (parent_id, spans | None, span_id_if_single)
        layout: dict[str, tuple[str, list[tuple[int, int, str]] | None, str]] = {}

        with tempfile.TemporaryDirectory(prefix=f"{self.system_id}_corpus_") as td:
            td_path = Path(td)
            for parent_id, members in groups.items():
                if len(members) == 1:
                    only = members[0]
                    seed, payload = only.item_id, only.text
                    spans, single_span = None, only.span_id
                else:
                    seed = parent_id
                    payload, spans = build_parent_payload(members)
                    single_span = ""
                    if clean_text(payload) != payload:
                        # The offsets in `spans` index `payload`, but the
                        # chunker will see clean_text(payload). They are
                        # the same string only while clean_text stays
                        # idempotent; if that ever breaks, provenance
                        # would drift silently across the document.
                        raise RuntimeError(
                            f"parsing.clean_text is no longer idempotent on "
                            f"parent {parent_id!r}: the per-parent offsets "
                            "would not match the text the chunker reads."
                        )

                filename = f"{_safe_item_filename(seed)}.txt"
                if filename in layout:
                    n = 1
                    while f"{_safe_item_filename(seed)}_{n}.txt" in layout:
                        n += 1
                    filename = f"{_safe_item_filename(seed)}_{n}.txt"
                (td_path / filename).write_text(payload, encoding="utf-8")
                layout[filename] = (parent_id, spans, single_span)

            self.index(td_path)

        # walk_corpus sets ParsedDocument.doc_id to the path relative to
        # the corpus root, which for this flat temp dir is the filename.
        n_unmapped = 0
        for chunk in self.chunks:
            entry = layout.get(chunk.doc_id)
            if entry is None:
                continue  # defensive; should not happen
            parent_id, spans, single_span = entry
            if spans is None:
                chunk.gold_provenance = ((parent_id, single_span),)
                continue
            lo = chunk.metadata.get("start_char")
            hi = chunk.metadata.get("end_char")
            if lo is None or hi is None:
                raise RuntimeError(
                    f"{self.system_id}: chunk {chunk.chunk_id!r} carries no "
                    "start_char/end_char, so per-parent provenance cannot be "
                    "derived. The raptor_100tok chunker supplies them; a "
                    "chunker override that does not cannot be used with a "
                    "multi-item parent."
                )
            # Half-open overlap. A chunk crossing an item boundary
            # legitimately carries BOTH atoms — that is the whole point
            # of contiguous chunking, and CK-2 scores it correctly.
            hits = tuple(
                (parent_id, span_id)
                for start, end, span_id in spans
                if start < hi and lo < end
            )
            if not hits:
                n_unmapped += 1
            chunk.gold_provenance = hits

        if n_unmapped:
            print(
                f"[{self.system_id}] WARNING: {n_unmapped} chunks intersected "
                "no source item and carry empty gold_provenance"
            )
        self._index_stats["n_chunks_without_gold_provenance"] = n_unmapped

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

        if m4.expand_summary_nodes:
            out = self._expand_summary_nodes(out, q_vec, m4.summary_expansion_leaves)

        self._last_trace = {
            "collapsed_top_node_types": dict(type_counter),
            "summary_expansion": bool(m4.expand_summary_nodes),
            "paths_exercised": sorted(paths_exercised),
            "n_returned": len(out),
            "budget_tokens_used": budget_tokens if budget is not None else None,
            "budget_tokens_limit": budget,
            # The App. I fidelity gate, per query. Retrieval-only: it
            # needs no generation, so it is measurable in the cheap stage.
            # Counted PRE-EXPANSION on purpose — the gate is a property
            # of the paper's retrieval, not of the diagnostic twin's
            # representation of it.
            "non_leaf_share": (
                sum(v for kk, v in type_counter.items() if kk != "chunk")
                / max(1, sum(type_counter.values()))
            ),
        }
        return out

    def _expand_summary_nodes(
        self,
        retrieved: list[RetrievedChunk],
        q_vec: np.ndarray,
        n_leaves: int,
    ) -> list[RetrievedChunk]:
        """Replace each retrieved summary with its top-N descendant leaves.

        THE DIAGNOSTIC, and what it is for. A summary node has no gold
        span, so CK-2 scores it as nothing — which means M4's retrieval
        F1 is measured on a set where a large minority of units cannot
        contribute, and is therefore NOT comparable to a leaf-only
        system's. This produces the twin that quantifies the gap: every
        returned unit becomes a leaf carrying real `gold_provenance`, so
        the same queries against the same tree yield a directly
        comparable number.

        POST-SELECTION BY DESIGN. Which nodes are retrieved — including
        the paper's 2,000-token budget fill — is decided before this runs
        and is untouched, so the twin measures the coverage of the
        PAPER'S retrieval rather than of a different retriever. Only the
        representation of the selected set changes.

        `source_unit_type` deliberately keeps the ORIGINATING summary
        tier rather than becoming "chunk": the unit was surfaced by a
        summary and the App. I gate must still see that, even though the
        text now carried is a leaf's.

        Score is inherited from the parent summary, not recomputed from
        the leaf. The selection score is the quantity that ordered the
        result set, and substituting a different similarity would break
        the descending-score invariant that rank-aware metrics assume.
        Leaf ORDER within a summary is by query cosine, so the most
        relevant descendants come first.
        """
        assert self._tree is not None
        if self.chunk_embeddings is None:
            return retrieved

        out: list[RetrievedChunk] = []
        seen: set[str] = set()

        def _emit(chunk: Chunk, score: float, unit_type: str) -> None:
            if chunk.chunk_id in seen:
                # A leaf can be reached directly AND through a summary,
                # or through two summaries. Keeping the first (best-rank)
                # copy stops duplicates from deflating precision.
                return
            seen.add(chunk.chunk_id)
            out.append(RetrievedChunk(
                chunk=chunk, score=score, rank=len(out),
                source_unit_type=unit_type,
            ))

        for item in retrieved:
            if item.source_unit_type == "chunk":
                _emit(item.chunk, item.score, "chunk")
                continue
            node = self._tree.nodes[item.chunk.chunk_id]
            idxs = [i for i in node.leaf_indices if 0 <= i < len(self.chunks)]
            if not idxs:
                continue
            sims = self.chunk_embeddings[idxs] @ q_vec[0]
            best = sorted(range(len(idxs)), key=lambda p: -float(sims[p]))
            for pos in best[:n_leaves]:
                _emit(self.chunks[idxs[pos]], item.score, item.source_unit_type)

        return out

    def prepare(self, query: str, k: int | None = None):
        """Attach M4's per-query diagnostics to the eval row.

        The runner merges `AnswerResult.extra` into `ScoredQuery.metadata`
        under the `m4_*` namespace convention, which is the only way the
        App. I non-leaf-share gate and the diagnostic-mode flag reach the
        JSONL — `retrieve()`'s trace is otherwise discarded per query.

        `m4_summary_expansion` is the important one: it makes a
        leaf-expanded diagnostic run impossible to mistake for a real M4
        cell after the fact, at the row level rather than the filename
        level.
        """
        prepared = super().prepare(query, k=k)
        trace = self._last_trace
        prepared.extra.update({
            "m4_non_leaf_share": trace.get("non_leaf_share"),
            "m4_budget_tokens_used": trace.get("budget_tokens_used"),
            "m4_summary_expansion": bool(trace.get("summary_expansion")),
        })
        return prepared

    # answer() inherits the BaseSystem default: retrieved node TEXT —
    # summaries verbatim included — is concatenated into the evidence
    # block, which is the paper's behaviour.
