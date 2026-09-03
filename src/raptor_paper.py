"""Paper-faithful RAPTOR substrate (M4 only). Sarthi et al., ICLR 2024, arXiv:2401.18059.

This module is a deliberate SIBLING of `src/raptor.py`, not a
replacement for it. `src/raptor.py` is consumed by the FROZEN M7 and
must stay byte-untouched; every paper-fidelity behaviour M4 needs lives
here instead. The duplication (serialisation helpers, a second node/tree
model) is the price of making M7's freeze safe BY CONSTRUCTION rather
than by argument.

Scope of this file, in landing order:
  * commit 1 (this one) — `split_text_raptor`, the reference chunker.
  * commit 2 — UMAP+GMM soft clustering, the bottom-up tree builder,
    the collapsed index over every layer, serialisation.
Nothing here is wired into a system yet; `M4Config` still resolves to
the shared harness chunker at this commit, so NO cache key moves.

# === FIDELITY NOTES — chunking ===
#
# PAPER (§3): "Construction of the RAPTOR tree begins with segmenting
# the retrieval corpus into short, contiguous texts of length 100" and
# "If a sentence exceeds the 100-token limit, we move the entire
# sentence to the next chunk, rather than cutting it mid-sentence."
# The paper never names a tokenizer and never states an overlap value;
# "contiguous" is the only basis for inferring zero overlap.
#
# REFERENCE CODE (raptor/utils.py, split_text) supplies what the paper
# omits, and this module follows it on every point EXCEPT one:
#   - tokenizer: tiktoken cl100k_base                      -> FOLLOWED
#   - overlap: the `overlap` parameter defaults to 0 and
#     tree_builder.py never passes it                      -> FOLLOWED
#   - sentence boundaries: . ! ? and newline               -> FOLLOWED
#   - over-long sentence: sub-split on , ; : and, if a
#     sub-phrase is still over budget, emit it oversized
#     (the 100-token bound is SOFT, not a hard truncation) -> FOLLOWED
#   - PLACEMENT of those over-long-sentence pieces         -> DIVERGED, see below
#   - delimiter handling                                   -> DIVERGED, see below
#
# DIVERGENCES FROM REFERENCE CODE, ALIGNMENT WITH PAPER TEXT
# (ruling 1 dated 2026-07-29; ruling 1b added 2026-08-22 by the final
# fidelity audit, which found the second one. Both depart from the
# reference implementation in the SAME direction: toward the paper's
# stated behaviour and away from an artifact of the reference's regex
# and control flow.):
#
#   The reference does `re.split("|".join(map(re.escape, [".", "!",
#   "?", "\\n"])), text)`. `re.split` on a pattern with NO capturing
#   group DISCARDS every separator it matches, and the rejoin is
#   `" ".join(current_chunk)` — nothing restores them. Reference chunk
#   text is therefore punctuation-free and newline-free prose.
#
#   We keep the terminators. Reasoning, recorded so the judgement is
#   auditable: (a) the paper is silent on punctuation and describes
#   only "short, contiguous texts", so nothing in the paper asks for
#   stripping; (b) the reference has no comment, no test and no
#   downstream consumer that wants stripped text, which reads as an
#   artifact of the non-capturing alternation rather than a design
#   decision; (c) this harness solves the identical problem correctly
#   elsewhere with a lookbehind (src/chunking.py `_SENTENCE_SPLIT_RE`),
#   which is what the reference would have needed; (d) reproducing it
#   would feed the generator punctuation-free text, which no reading
#   of the paper supports.
#
#   SUB-RULING on newlines (ruled 2026-07-29, alongside the above).
#   Ruling 1 restores `. ! ? , ; :` but NOT `\\n`. This is an
#   application of the ruling, not an exception to it: the ruling
#   concerns TERMINATORS destroyed by a regex artifact, and a newline is
#   not a terminator — it is layout. Three reasons it must collapse to a
#   single space rather than be preserved literally:
#     (a) the paper is silent on newlines and describes only "short,
#         contiguous texts", so nothing asks for them;
#     (b) the reference collapses them anyway at READ time — `get_text`
#         does `' '.join(node.text.splitlines())` before any node text
#         reaches an embedder or a prompt — so preserving them would
#         diverge from reference BEHAVIOUR while claiming to follow
#         reference code;
#     (c) preserving them would therefore diverge from BOTH the code and
#         the paper text, which is the one outcome no reading supports.
#   So: `. ! ? , ; :` are CONTENT and are restored and attached; `\\n`
#   runs are consumed as pure boundaries.
#
#   CONSEQUENCE, accepted, no action. Token accounting shifts. The
#   reference counts tokens on STRIPPED sentences; we count on
#   punctuated ones, so each sentence costs ~1 more token and our
#   100-token chunks hold roughly 1-3% less prose than the reference's
#   would. This is a direct and unavoidable consequence of ruling 1 and
#   is recorded rather than corrected.
#
#   RULING 1b — OVER-LONG-SENTENCE PLACEMENT (added 2026-08-22, found by
#   the final fidelity audit; see docs/FINAL_FIDELITY_AUDIT.md AF-2).
#
#   The reference SUB-SPLITS an over-long sentence the way we do, but it
#   PLACES the resulting pieces differently, and the difference is
#   structural rather than cosmetic. `split_text` appends them straight
#   to `chunks` from inside the `token_count > max_tokens` branch, while
#   `current_chunk` — holding the sentences that PRECEDE the long one —
#   keeps accumulating and is flushed later. Two consequences follow,
#   both verified against a verbatim transcription of the reference:
#     (a) the long sentence's pieces are emitted BEFORE the chunk holding
#         the text that came before them, so the chunk list is NOT in
#         document order;
#     (b) the sentences flanking the long one are packed TOGETHER, across
#         it, as though the long sentence were not between them.
#   Ours routes every piece through one packer in document order, so a
#   sub-phrase may share a chunk with an ordinary neighbouring sentence
#   and the output stays ordered.
#
#   WE FOLLOW THE PAPER. It describes "short, contiguous texts" and says
#   only that an over-long sentence moves to the next chunk; nothing in
#   it asks for reordering, and reordered chunks are contiguous in
#   neither sense. As with ruling 1, the reference behaviour reads as an
#   artifact of its control flow — an append inside a branch — rather
#   than a design decision: there is no comment, no test and no consumer
#   that wants document-order-scrambled chunks.
#
#   INCIDENCE, measured on the real corpora (2026-08-22, through the
#   pipeline's own layout and chunker) — this fires only where a single
#   sentence exceeds the 100-token budget:
#     MultiHop-RAG           45 of 70,455 sentences  (0.064%)
#     HotpotQA-distractor   137 of 46,855 sentences  (0.292%, 107/1000 units)
#     HotpotQA-pooled       137 of 46,720 sentences  (0.293%)
#     NarrativeQA             2 of 386,791 sentences (0.0005%) — measured on
#                           the run host 2026-08-23; 0/40 units degenerate.
#                           The narrative-prose regime barely produces a
#                           100-token sentence, so the divergence is close
#                           to unreachable there.
#   Uniform across all four M4 cells, and M4 is the only system using this
#   chunker, so no cross-system asymmetry arises.
#
# CACHE DISCIPLINE. The 100-token size is carried on the EXISTING
# `ChunkingConfig.chunk_words` field (read as TOKENS under
# strategy="raptor_100tok"), never on a new field. `compute_cache_key`
# folds `json.dumps(asdict(chunking_config), sort_keys=True)`, so
# adding any field to ChunkingConfig would move the substrate key of
# EVERY system — M2, M3, M9 and the frozen M7 included. The dataclass
# schema must stay byte-identical; tests/test_raptor_chunking.py pins
# it.
"""

from __future__ import annotations

import time as _time_mod

import functools
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


# Reference: `tokenizer=tiktoken.get_encoding("cl100k_base")` as the
# default argument of both `split_text` and `RAPTOR_Clustering.
# perform_clustering`. Pinned by NAME (not by model id) so it cannot
# drift with a generator swap the way `encoding_for_model` would.
REFERENCE_ENCODING = "cl100k_base"

# Bumped when the produced chunk text changes for identical input.
# Folded into M4's cache-key extras from commit 3 onward; inert here.
RAPTOR_CHUNKER_VERSION = "raptor_split_text_v1"

# Reference sentence delimiters: [".", "!", "?", "\n"]. Split into two
# classes because they are treated differently — see the DIVERGENCE
# note above. Runs are matched greedily so "..." / "?!" / "\n\n" each
# count as ONE boundary rather than producing empty segments.
_TERMINATOR_RUN = r"[.!?]+"
_NEWLINE_RUN = r"\n+"
_BOUNDARY_RE = re.compile(f"({_TERMINATOR_RUN})|({_NEWLINE_RUN})")

# Reference over-long-sentence fallback: `re.split(r"[,;:]", sentence)`.
# Same keep-the-delimiter treatment as the terminators above.
_SUBPHRASE_RE = re.compile(r"([,;:]+)")


@functools.lru_cache(maxsize=2)
def _encoding(name: str = REFERENCE_ENCODING) -> Any:
    import tiktoken

    return tiktoken.get_encoding(name)


def count_tokens_plain(text: str, *, encoding_name: str = REFERENCE_ENCODING) -> int:
    """Bare cl100k_base token count, no leading space.

    This is the convention the reference uses everywhere OUTSIDE the
    chunker: `len(tokenizer.encode(node.text))` for the 3500-token
    cluster cap and for the retrieval token budget. The chunker's
    leading-space convention (`count_tokens_reference`) is a separate
    quirk of `split_text` and must not be mixed with this one.
    """
    return len(_encoding(encoding_name).encode(text))


def count_tokens_reference(text: str, *, encoding_name: str = REFERENCE_ENCODING) -> int:
    """Token count under the reference's convention.

    The reference measures every sentence as `len(tokenizer.encode(" " +
    sentence))` — the leading space stands in for the space the rejoin
    will insert. Preserved verbatim so our packing decisions land on the
    same boundaries the reference's would, modulo the punctuation
    divergence documented in the module docstring.
    """
    return len(_encoding(encoding_name).encode(" " + text))


@dataclass(frozen=True)
class TextSpan:
    """One chunk, with its provenance span in the ORIGINAL document text.

    `text` is NOT `original[start_char:end_char]` — inter-sentence
    whitespace and newlines are normalised to single spaces during the
    rejoin. The span is a PROVENANCE range, used by M4's per-parent
    `index_items` override (commit 4) to map a chunk back to the
    CorpusItems it overlaps and derive `gold_provenance` by offset
    intersection. Treat it as "this chunk came from this region", never
    as a slice.
    """

    text: str
    start_char: int
    end_char: int
    n_tokens: int


def _iter_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into (sentence, start_char, end_char), terminators attached.

    Boundary semantics, per the module docstring:
      * a run of `. ! ?` ENDS the current sentence and is KEPT on it;
      * a run of `\\n` ends the current sentence and is DROPPED;
      * the trailing remainder after the last boundary is a sentence.
    Segments that are empty after stripping are discarded (they arise
    from consecutive boundaries such as ".\\n" or "!?").
    """
    out: list[tuple[str, int, int]] = []
    cursor = 0

    def _emit(lo: int, hi: int) -> None:
        raw = text[lo:hi]
        stripped = raw.strip()
        if not stripped:
            return
        # Re-anchor the span onto the stripped content so offsets never
        # point at leading/trailing whitespace.
        lead = len(raw) - len(raw.lstrip())
        out.append((stripped, lo + lead, lo + lead + len(stripped)))

    for m in _BOUNDARY_RE.finditer(text):
        if m.group(1) is not None:
            # Terminator run: keep it with the sentence it closes.
            _emit(cursor, m.end())
        else:
            # Newline run: boundary only, not content.
            _emit(cursor, m.start())
        cursor = m.end()

    _emit(cursor, len(text))
    return out


def _split_long_sentence(
    sentence: str,
    start_char: int,
    max_tokens: int,
    encoding_name: str,
) -> list[tuple[str, int, int]]:
    """Reference fallback for a sentence that alone exceeds max_tokens.

    Reference: `sub_sentences = re.split(r"[,;:]", sentence)`, keeping
    non-empty stripped pieces. We keep the delimiters attached for the
    same reason we keep terminators. A sub-phrase that is STILL over
    budget is returned as-is — the reference emits it oversized and so
    do we, which is why the 100-token bound is soft rather than a hard
    truncation.
    """
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    parts: list[tuple[int, int]] = []
    for m in _SUBPHRASE_RE.finditer(sentence):
        parts.append((cursor, m.end()))
        cursor = m.end()
    parts.append((cursor, len(sentence)))

    for lo, hi in parts:
        raw = sentence[lo:hi]
        stripped = raw.strip()
        if not stripped:
            continue
        lead = len(raw) - len(raw.lstrip())
        abs_lo = start_char + lo + lead
        pieces.append((stripped, abs_lo, abs_lo + len(stripped)))

    if not pieces:
        return [(sentence, start_char, start_char + len(sentence))]
    return pieces


def split_text_raptor(
    text: str,
    *,
    max_tokens: int = 100,
    encoding_name: str = REFERENCE_ENCODING,
) -> list[TextSpan]:
    """Port of the reference `utils.split_text`, 100 tokens, no overlap.

    Sentence-preserving: a sentence that would push the current chunk
    past `max_tokens` starts the next chunk instead of being cut, which
    is the paper's stated rule. A single sentence longer than
    `max_tokens` is sub-split on `, ; :`; a sub-phrase still over budget
    is emitted alone and oversized (soft bound, reference behaviour).

    Returns provenance-carrying spans, oldest-first. Empty / whitespace
    input returns [].
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not text or not text.strip():
        return []

    # Flatten to (piece, start, end, n_tokens), splitting over-long
    # sentences first so the packer below only ever sees pieces it can
    # reason about.
    pieces: list[tuple[str, int, int, int]] = []
    for sentence, lo, hi in _iter_sentences(text):
        n_tok = count_tokens_reference(sentence, encoding_name=encoding_name)
        if n_tok <= max_tokens:
            pieces.append((sentence, lo, hi, n_tok))
            continue
        for sub, sub_lo, sub_hi in _split_long_sentence(
            sentence, lo, max_tokens, encoding_name
        ):
            pieces.append((
                sub,
                sub_lo,
                sub_hi,
                count_tokens_reference(sub, encoding_name=encoding_name),
            ))

    spans: list[TextSpan] = []
    cur_texts: list[str] = []
    cur_lo = 0
    cur_hi = 0
    cur_tokens = 0

    def _flush() -> None:
        nonlocal cur_texts, cur_lo, cur_hi, cur_tokens
        if not cur_texts:
            return
        spans.append(TextSpan(
            text=" ".join(cur_texts),
            start_char=cur_lo,
            end_char=cur_hi,
            n_tokens=cur_tokens,
        ))
        cur_texts = []
        cur_lo = 0
        cur_hi = 0
        cur_tokens = 0

    for piece, lo, hi, n_tok in pieces:
        # NO OVERLAP: the flushed chunk is not carried into the next one
        # (reference `overlap=0`, never overridden by tree_builder.py).
        if cur_texts and cur_tokens + n_tok > max_tokens:
            _flush()
        if not cur_texts:
            cur_lo = lo
        cur_texts.append(piece)
        cur_hi = hi
        cur_tokens += n_tok

    _flush()
    return spans


# =========================================================================
# Bottom-up tree: UMAP (global + local) -> BIC-selected GMM soft clustering
# -> per-cluster summary -> re-embed -> repeat.
#
# PAPER (§3, and the Abstract's "from the bottom up"): "Once clustered, a
# Language Model is used to summarize the grouped texts. These summarized
# texts are then re-embedded, and the cycle of embedding, clustering, and
# summarization continues until further clustering becomes infeasible,
# resulting in a structured, multi-layered tree representation."
#
# This is NOT what src/raptor.py does. That module partitions ONE
# all-chunk root top-down with recursive MiniBatchKMeans, which produces a
# strict single-parent tree of bounded branching. The paper's algorithm
# agglomerates upward and its soft clustering lets a node belong to
# several parents, so the result is a DAG. That difference is why this
# module carries its own node model instead of extending RaptorNode
# (whose single `parent_id` cannot express it) — and why src/raptor.py,
# which the frozen M7 consumes, is never opened.
#
# Every constant below is the reference implementation's default. The
# paper states none of them; see the module docstring's UNVERIFIED
# framing — they are attributable to the code, not the paper.
#
# DOCUMENTED MICRO-DIVERGENCES from the reference, all ruled 2026-07-29:
#
#  (i) UMAP is SEEDED. The reference passes no `random_state`, so its
#      trees differ run to run. We seed, because a cache key that does
#      not determine the artifact it names is not a cache key — the same
#      infrastructure-contract reasoning that keeps summarisation at
#      temperature 0. Cost: seeding forces UMAP to n_jobs=1, i.e.
#      single-threaded, which is the dominant wall-clock term in tree
#      construction. Accepted.
#
# (ii) RE-CLUSTER RECURSION IS DEPTH-GUARDED. The reference's
#      `perform_clustering` recurses whenever a cluster's combined text
#      exceeds `max_length_in_cluster`, with no base case beyond
#      `len(nodes) == 1`; a cluster that keeps re-forming identically
#      recurses forever. A build that cannot terminate cannot produce an
#      artifact, so the guard is admissible on impossibility grounds. It
#      accepts the oversized cluster instead of recursing further and
#      increments `stats["recluster_guard_trips"]` — a non-zero count is
#      a FINDING about the clustering, to be reported, not silenced.
#
#(iii) The reference's recursive call drops `reduction_dimension` and
#      `threshold`, silently reverting them to 10 and 0.1 even when the
#      caller overrode them. We thread the params through instead. At our
#      configuration the two are identical (our defaults ARE 10 and 0.1),
#      so this is observable only under a non-default override we do not
#      use. Recorded for completeness.
#
# (iv) Two degenerate cases that CRASH or silently drop nodes in the
#      reference are guarded, both impossibility-class: an empty BIC
#      search range (`np.arange(1, 1)` -> argmin on empty), and a label
#      set where no node cleared the GMM threshold (-> zero clusters, so
#      every node at that layer vanishes from the tree). Both guards are
#      counted in `stats`.
#
#  (v) GMM FITS THAT RAISE ARE SKIPPED, not fatal. MEASURED at production
#      params, and the reason this guard exists at all: a layer of 16 or
#      of 25 nodes killed the build with "ill-defined empirical
#      covariance", while 20, 30 and 40 survived. The sweep tries k up to
#      n-1, and UMAP reducing few points into 10 components leaves tight
#      local clumps that a high-k component collapses onto. NOT monotone
#      in n and data-dependent, so passing at one corpus size proves
#      nothing about another — which is why this shipped before any tree
#      build rather than after one died. The reference crashes here and
#      loses the entire tree, so this is impossibility class.
#      `bic_fit_failures` counts skipped k in the sweep;
#      `gmm_final_fit_failures` counts downward steps in the final fit.
#      Non-zero is a FINDING to report, not noise: it means the BIC
#      search was run over a reduced candidate set.
#
# (vi) DUPLICATE-EMBEDDING MEMBERSHIP. Added 2026-08-22 by the final
#      fidelity audit (docs/FINAL_FIDELITY_AUDIT.md AF-3). The reference
#      recovers local-cluster membership by VALUE matching --
#      `np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))`
#      -- so two nodes whose texts are BYTE-IDENTICAL have identical
#      embeddings and each receives the UNION of the other's cluster
#      labels. Ours tracks membership by INDEX, so duplicates keep their
#      own labels. Observable only where a unit holds byte-identical
#      chunk texts; measured incidence 2026-08-22: MultiHop 73/16,523
#      chunks (0.44%), HotpotQA-distractor 21/17,443 (0.12%), pooled
#      21/17,396 (0.12%), NarrativeQA 0 (run host, 2026-08-23). The reference's form is also O(n^2) in the layer
#      size. Recorded as a divergence rather than adopted: label-union for
#      coincidentally identical text is an artifact of the lookup
#      strategy, not a stated behaviour.
#
#(vii) SMALL-n ENTRY GUARD in `_two_stage_labels`. Added to this list
#      2026-08-22 (AF-6); the guard itself predates it. `n <= dim + 1`
#      returns a single cluster before any UMAP call. The reference's
#      MODULE-level `perform_clustering` has no such check and would
#      raise inside UMAP for tiny n; it is unreachable at the reference's
#      own defaults only because the builder's stop condition fires
#      first. Impossibility class, same as (iv) and (v). Reachable here
#      only via recursion on a tiny oversized cluster.
#
# The reference's SEED INCONSISTENCY is reproduced exactly, not fixed:
# `random_state=224` for the BIC search, `random_state=0` for the final
# fit. Noted as an observed oddity of the reference; the paper is silent,
# the code specifies, cost is zero, so the code wins.
# =========================================================================


PAPER_TREE_SCHEMA_VERSION = "raptor_paper_bottom_up_v1"

# Paper App. I: across its benchmarks, between 18.5% and 57% of the nodes
# RAPTOR retrieves are non-leaf (summary) nodes. This is the one fidelity
# gate that is measurable only at QUERY time — the other two (children per
# parent, mean summary length) are properties of the built tree and live in
# `tree_stats`. Kept here so all three paper-derived bands sit together and
# the analyser imports rather than restates them.
PAPER_NON_LEAF_SHARE_BAND = (0.185, 0.57)

# Reference cluster_utils.py module constant: RANDOM_SEED = 224.
REFERENCE_RANDOM_SEED = 224


@dataclass(frozen=True)
class PaperTreeParams:
    """Reference-implementation defaults for the paper-faithful tree.

    Field-by-field provenance (all from parthsarthi03/raptor@master):
      reduction_dimension   ClusterTreeConfig(reduction_dimension=10)
      gmm_threshold         RAPTOR_Clustering.perform_clustering(threshold=0.1)
                            — the GMM posterior-membership cutoff. NOT the
                            same quantity as TreeBuilderConfig.threshold=0.5,
                            which is a retrieval selection threshold in a
                            different file. Easy to conflate; don't.
      max_length_in_cluster RAPTOR_Clustering.perform_clustering(3500),
                            measured in cl100k_base TOKENS summed over the
                            cluster's node texts.
      num_layers            TreeBuilderConfig(num_layers=5) — an upper
                            bound; the realised depth is usually lower.
      local_n_neighbors     local_cluster_embeddings(num_neighbors=10).
                            The GLOBAL n_neighbors is not a parameter —
                            the reference computes int((n-1)**0.5).
      metric                both UMAP calls use "cosine".
      bic_max_clusters      get_optimal_clusters(max_clusters=50).
      bic_random_state      224, the module RANDOM_SEED.
      gmm_random_state      0, GMM_cluster's own default. The mismatch
                            with 224 is the reference's, reproduced.
      umap_random_state     OURS. The reference seeds nothing.
      max_recluster_depth   OURS. The reference has no bound at all.

    Frozen and asdict-able: this lands in M4's cache-key extras as the
    "tree" field, replacing RaptorBuildParams. A different schema here is
    exactly what forks M4's substrate key away from the KMeans-era one.
    """

    reduction_dimension: int = 10
    gmm_threshold: float = 0.1
    max_length_in_cluster: int = 3500
    num_layers: int = 5
    local_n_neighbors: int = 10
    metric: str = "cosine"
    bic_max_clusters: int = 50
    bic_random_state: int = REFERENCE_RANDOM_SEED
    gmm_random_state: int = 0
    umap_random_state: int = 42
    max_recluster_depth: int = 8


@dataclass
class PaperNode:
    """One node of the paper-faithful DAG. Layer 0 nodes are leaf chunks.

    `parent_ids` is a LIST because the paper's soft clustering puts a node
    in every cluster whose GMM posterior exceeds the threshold, so a node
    can be summarised into several parents. `leaf_indices` is the
    transitive closure down to layer-0 chunk indices, which is what
    provenance and diagnostics need; it can overlap between siblings, and
    that is correct rather than a bug.
    """

    node_id: str
    layer: int
    text: str
    children: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    leaf_indices: list[int] = field(default_factory=list)
    embedding: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.layer == 0


@dataclass
class PaperTree:
    nodes: dict[str, PaperNode]
    layer_to_nodes: dict[int, list[str]]
    n_layers: int
    params: PaperTreeParams
    stats: dict = field(default_factory=dict)

    def all_node_ids(self) -> list[str]:
        """Every node, every layer, leaves included — the collapsed set."""
        out: list[str] = []
        for layer in sorted(self.layer_to_nodes):
            out.extend(self.layer_to_nodes[layer])
        return out

    def summary_nodes(self) -> list[PaperNode]:
        return [n for n in self.nodes.values() if n.layer > 0]


# --- clustering (port of cluster_utils.py) --------------------------------


class _PhaseClock:
    """Cumulative wall-time and call counts per build phase.

    PURE INSTRUMENTATION. It records; nothing branches on what it
    records, so a timed build produces a byte-identical tree to an
    untimed one. It changes nothing the cache key reads —
    `paper_substrate_extra` folds PARAMETERS, not code — so this does not
    invalidate a substrate.

    It exists because a 4,953-leaf story took 20,691 s and the harness
    could not say WHERE. Summarisation and clustering want opposite
    fixes: one is a token-cap and batch-width question on the GPU, the
    other is single-threaded CPU work that no VRAM metric can see.
    Guessing between them costs a five-hour build per guess.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        # One entry per phase currently on the stack, holding the time
        # its CHILDREN have consumed. See `enter`/`exit_`.
        self._stack: list[list] = []

    def reset(self) -> None:
        self.seconds.clear()
        self.calls.clear()
        self._stack.clear()

    def enter(self, phase: str) -> None:
        """Open a frame. Its child time accrues here, not to the phase."""
        self._stack.append([phase, 0.0])

    def exit_(self, elapsed: float) -> None:
        """Close the innermost frame, crediting it only its OWN time.

        PHASES MUST PARTITION THE BUILD, NOT OVERLAP IT. `_gmm_cluster`
        is timed as `gmm_final_fit` and calls `_get_optimal_clusters`,
        timed as `gmm_bic_sweep`, so a flat accumulator charged the sweep
        to both and the phases summed to ~19% MORE than the build they
        described. Wall clock was never wrong; the ATTRIBUTION was, and
        attribution is what retired UMAP as a suspect and surfaced GMM as
        the second cost. Decisions are being made on these numbers.

        Subtracting child time makes the invariant structural rather than
        something a future nesting can quietly break.
        """
        phase, child_s = self._stack.pop()
        own = elapsed - child_s
        if own < 0:  # clock skew only; never let a phase go negative
            own = 0.0
        self.seconds[phase] = self.seconds.get(phase, 0.0) + own
        self.calls[phase] = self.calls.get(phase, 0) + 1
        if self._stack:
            self._stack[-1][1] += elapsed

    def add(self, phase: str, dt: float) -> None:
        """Flat accrual, for phases timed without the decorator.

        Still charged to any open parent frame, so an inline `add` inside
        a decorated function does not re-create the double count.
        """
        self.seconds[phase] = self.seconds.get(phase, 0.0) + dt
        self.calls[phase] = self.calls.get(phase, 0) + 1
        if self._stack:
            self._stack[-1][1] += dt

    def as_stats(self) -> dict:
        total = sum(self.seconds.values())
        return {
            "phase_seconds": {k: round(v, 2) for k, v in sorted(
                self.seconds.items(), key=lambda kv: -kv[1])},
            "phase_calls": dict(self.calls),
            "phase_share": ({k: round(v / total, 4)
                             for k, v in self.seconds.items()}
                            if total else {}),
            "phase_measured_total_s": round(total, 2),
        }


_CLOCK = _PhaseClock()


def get_phase_clock() -> _PhaseClock:
    """The build's phase timings. Read after build_paper_tree returns."""
    return _CLOCK


def _timed(phase: str):
    """Wrap a phase function so its wall time accrues to `phase`.

    A decorator rather than inline `with` blocks: the timed calls are
    multi-line constructor expressions, and re-indenting them to insert a
    context manager would be a behavioural edit dressed as instrumentation.
    """
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import time as _time

            _CLOCK.enter(phase)
            t0 = _time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                # `finally`, so a raising phase still closes its frame —
                # a dirty stack would mis-attribute every later phase to
                # whatever was open when the exception passed through.
                _CLOCK.exit_(_time.perf_counter() - t0)

        return wrapper

    return deco


@_timed("umap")
def _umap_reduce(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int,
    metric: str,
    random_state: int,
) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric,
        random_state=random_state,
    ).fit_transform(embeddings)


def _global_cluster_embeddings(
    embeddings: np.ndarray, dim: int, params: PaperTreeParams
) -> np.ndarray:
    """Reference: n_neighbors defaults to int((len(embeddings) - 1) ** 0.5)."""
    n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return _umap_reduce(
        embeddings, dim, max(2, n_neighbors), params.metric, params.umap_random_state
    )


def _local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, params: PaperTreeParams
) -> np.ndarray:
    """Reference: fixed num_neighbors=10.

    Note the reference clamps `dim` for the GLOBAL call
    (`min(dim, len(embeddings) - 2)`) but passes the raw `dim` here. We
    clamp both, because an unclamped local call raises on a small
    cluster — impossibility-class, and unreachable at the reference's own
    defaults only because the `len <= dim + 1` escape hatch fires first.
    """
    return _umap_reduce(
        embeddings,
        min(dim, max(2, len(embeddings) - 2)),
        min(params.local_n_neighbors, max(2, len(embeddings) - 1)),
        params.metric,
        params.umap_random_state,
    )


@_timed("gmm_bic_sweep")
def _get_optimal_clusters(
    embeddings: np.ndarray, params: PaperTreeParams, stats: dict
) -> int:
    """BIC sweep. Reference: arange(1, min(max_clusters, n)), seed 224.

    `np.arange` excludes its stop, so the reference searches
    1..min(50, n)-1 and never evaluates n itself. Reproduced verbatim.
    """
    from sklearn.mixture import GaussianMixture

    max_clusters = min(params.bic_max_clusters, len(embeddings))
    candidates = np.arange(1, max_clusters)
    if len(candidates) == 0:
        # GUARD (iv): the reference would call argmin on an empty list.
        # Reachable only for n <= 1, which the escape hatches upstream
        # normally prevent.
        stats["empty_bic_range_trips"] = stats.get("empty_bic_range_trips", 0) + 1
        return 1
    # GUARD (v): a component that collapses onto a single point has an
    # ill-defined covariance and `fit` RAISES. Measured, not anticipated:
    # at production params a layer of 16 or 25 nodes died this way, while
    # 20/30/40 survived — the sweep runs k up to n-1, and UMAP's
    # reduction of few points into 10 components leaves tight local
    # clumps for the high-k fits to collapse onto. The reference has the
    # same hole and simply crashes, losing the whole build.
    #
    # Impossibility class, so a guard is admissible; SKIPPING the failing
    # k is the most conservative shape available. It leaves every k that
    # CAN be fitted in contention, so the selected k is exactly the one
    # the reference would have chosen had it not crashed — unless the
    # reference's own argmin was a k that cannot be fitted, in which case
    # there is no faithful answer to reproduce.
    bics: list[float] = []
    fitted: list[int] = []
    for n in candidates:
        gm = GaussianMixture(
            n_components=int(n), random_state=params.bic_random_state
        )
        try:
            gm.fit(embeddings)
        except ValueError:
            stats["bic_fit_failures"] = stats.get("bic_fit_failures", 0) + 1
            continue
        bics.append(float(gm.bic(embeddings)))
        fitted.append(int(n))
    if not fitted:
        # Every k failed. One cluster is the only assignment left that is
        # certainly well-defined.
        stats["bic_all_fits_failed"] = stats.get("bic_all_fits_failed", 0) + 1
        return 1
    return fitted[int(np.argmin(bics))]


@_timed("gmm_final_fit")
def _gmm_cluster(
    embeddings: np.ndarray, params: PaperTreeParams, stats: dict
) -> tuple[list[np.ndarray], int]:
    """Soft assignment: a node joins EVERY component above the threshold.

    Reference seeds the final fit with random_state=0 while the BIC
    search above uses 224. The inconsistency is the reference's and is
    reproduced deliberately (ruling 3).

    GUARD (v), second half. The seed mismatch means the final fit can
    raise at a k the BIC sweep fitted happily under seed 224 — different
    initialisation, different collapse. Walking k DOWNWARD keeps as much
    structure as can actually be fitted, where falling straight to k=1
    would silently turn a splittable layer into one parent. Terminates:
    k=1 is a single component over every point and cannot be a singleton
    collapse.
    """
    from sklearn.mixture import GaussianMixture

    n_clusters = _get_optimal_clusters(embeddings, params, stats)
    for k in range(n_clusters, 0, -1):
        gm = GaussianMixture(
            n_components=k, random_state=params.gmm_random_state
        )
        try:
            gm.fit(embeddings)
        except ValueError:
            stats["gmm_final_fit_failures"] = (
                stats.get("gmm_final_fit_failures", 0) + 1
            )
            continue
        probs = gm.predict_proba(embeddings)
        labels = [np.where(p > params.gmm_threshold)[0] for p in probs]
        return labels, k

    # Unreachable in practice (k=1 always fits); the explicit fallback
    # exists so a future sklearn cannot turn this into an UnboundLocal.
    stats["gmm_all_fits_failed"] = stats.get("gmm_all_fits_failed", 0) + 1
    return [np.array([0]) for _ in range(len(embeddings))], 1


def _two_stage_labels(
    embeddings: np.ndarray, params: PaperTreeParams, stats: dict
) -> list[np.ndarray]:
    """Global-then-local soft clustering, flat global label space.

    Reference `perform_clustering(embeddings, dim, threshold)`: reduce
    globally, GMM, then for each global cluster reduce locally and GMM
    again, offsetting local label ids by a running total so the returned
    label space is flat.
    """
    dim = params.reduction_dimension
    n = len(embeddings)
    if n <= dim + 1:
        return [np.array([0]) for _ in range(n)]

    reduced_global = _global_cluster_embeddings(
        embeddings, min(dim, n - 2), params
    )
    global_labels, n_global = _gmm_cluster(reduced_global, params, stats)

    out: list[np.ndarray] = [np.array([], dtype=int) for _ in range(n)]
    total = 0
    for gi in range(n_global):
        members = [i for i, lab in enumerate(global_labels) if gi in lab]
        if not members:
            continue
        sub = embeddings[members]
        if len(sub) <= dim + 1:
            # Reference escape hatch: too small to reduce, one cluster.
            local_labels = [np.array([0]) for _ in members]
            n_local = 1
        else:
            reduced_local = _local_cluster_embeddings(sub, dim, params)
            local_labels, n_local = _gmm_cluster(reduced_local, params, stats)
        for pos, lab in enumerate(local_labels):
            for li in np.atleast_1d(lab).tolist():
                out[members[pos]] = np.append(out[members[pos]], int(li) + total)
        total += n_local
    return out


def perform_clustering(
    nodes: list[PaperNode],
    params: PaperTreeParams,
    stats: dict | None = None,
    *,
    _depth: int = 0,
) -> list[list[PaperNode]]:
    """Reference RAPTOR_Clustering.perform_clustering, guarded.

    Returns a list of clusters; a node may appear in several of them
    (soft membership). A cluster whose combined text exceeds
    `max_length_in_cluster` tokens is recursively re-clustered, bounded
    by `max_recluster_depth` — see micro-divergence (ii).
    """
    stats = stats if stats is not None else {}
    if len(nodes) <= 1:
        return [list(nodes)]

    embeddings = np.vstack([
        np.asarray(n.embedding, dtype=np.float32).reshape(1, -1) for n in nodes
    ])
    labels = _two_stage_labels(embeddings, params, stats)

    non_empty = [lab for lab in labels if len(lab) > 0]
    if not non_empty:
        # GUARD (iv): nothing cleared the threshold. The reference would
        # produce zero clusters and drop every node at this layer.
        stats["empty_label_trips"] = stats.get("empty_label_trips", 0) + 1
        return [list(nodes)]

    clusters: list[list[PaperNode]] = []
    for label in sorted({int(x) for lab in non_empty for x in lab.tolist()}):
        members = [nodes[i] for i, lab in enumerate(labels) if label in lab]
        if not members:
            continue
        if len(members) == 1:
            clusters.append(members)
            continue
        total_tokens = sum(
            len(_encoding().encode(m.text)) for m in members
        )
        if total_tokens <= params.max_length_in_cluster:
            clusters.append(members)
            continue
        if len(members) == len(nodes):
            # NO-PROGRESS STOP. The "cluster" is the entire input, which
            # happens whenever BIC selects k=1 (low-structure data, or a
            # layer whose members are near-uniform). Re-clustering an
            # identical set yields an identical result, so the reference
            # recurses forever here — this is the concrete shape of the
            # unbounded recursion, and catching it directly costs one
            # step instead of unwinding max_recluster_depth levels of
            # wasted UMAP+GMM. Counted separately from the depth guard
            # because it means something different: not "too deep" but
            # "clustering cannot split this at all".
            stats["no_progress_trips"] = stats.get("no_progress_trips", 0) + 1
            clusters.append(members)
            continue
        if _depth >= params.max_recluster_depth:
            stats["recluster_guard_trips"] = (
                stats.get("recluster_guard_trips", 0) + 1
            )
            clusters.append(members)
            continue
        stats["recluster_calls"] = stats.get("recluster_calls", 0) + 1
        clusters.extend(
            perform_clustering(members, params, stats, _depth=_depth + 1)
        )
    return clusters


# --- summarisation input format (port of utils.get_text) ------------------


def get_text(nodes: list[PaperNode]) -> str:
    """Reference utils.get_text: collapse newlines per node, join on blank lines.

    This is the exact string the reference hands the summariser, so it is
    reproduced verbatim including the trailing "\\n\\n". Note it collapses
    internal newlines — the same treatment this module's chunker already
    applies at split time.
    """
    out = ""
    for n in nodes:
        out += f"{' '.join(n.text.splitlines())}"
        out += "\n\n"
    return out


# --- summarisation prompt (paper Appendix D) ------------------------------
#
# The paper and the reference code DISAGREE on the system prompt, and the
# paper wins because it is explicit:
#   paper App. D : "You are a Summarizing Text Portal"
#   code         : "You are a helpful assistant."
# The user prompt is identical in both, trailing colon included.
#
# This is M4-LOCAL. Its version id lives on M4Config.summary_prompt_version,
# never on summarization.SUMMARY_PROMPT_VERSION — that constant is a module
# global the FROZEN M7 also reads, and bumping it would move M7's substrate
# key.
#
# Deliberately NOT reusing summarization.SUMMARY_PROMPT_TEMPLATE, whose
# wording ("in the same language they are written in", "3-5 sentences",
# "Do not invent") is ours, not the paper's.

PAPER_SUMMARY_SYSTEM_PROMPT = "You are a Summarizing Text Portal"
PAPER_SUMMARY_USER_TEMPLATE = (
    "Write a summary of the following, including as many key details as "
    "possible: {context}:"
)


def summarize_paper_style_batch(
    contexts: list[str],
    *,
    model: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    batch_size: int = 32,
    max_padded_tokens: int | None = 16000,
    progress_every: int = 1,
) -> list[str]:
    """A whole layer's cluster summaries in one batched call. M4's path.

    WHY A LAYER IS THE BATCH. Bottom-up construction produces an entire
    layer's clusters before any of them is summarised, so the layer is a
    natural batch with no reordering, no queueing and no partial results
    to reconcile. Passing the WHOLE layer (rather than pre-slicing it)
    is deliberate: `generate_batch` length-sorts internally, and sorting
    across the full layer removes far more padding waste than sorting
    within arbitrary pre-cut groups.

    WHY NOT THREADS, which this replaces. `M4Config.summary_max_workers`
    dispatched `summarize_fn` through a ThreadPoolExecutor, which was
    right against an API and is wrong against a local model on three
    counts: threads contend on the GIL, they serialise onto one CUDA
    stream anyway, and each concurrent `model.generate` allocates its own
    KV cache. The decisive one is subtler — `models.load_generator` is
    `lru_cache`d, so every thread shares ONE tokenizer object, and
    `generate_batch` MUTATES it (forces `padding_side="left"`, restores
    in a `finally`). Concurrent callers can therefore observe a
    right-padded tokenizer mid-call, which does not raise; it produces
    fluent text continued from PAD. Silent corruption, not a crash.

    WHY SUMMARIES BATCH WELL WHEN 4k ANSWERS DO NOT. A batch runs until
    its LONGEST member stops, so one non-terminating generation makes
    every member pay the cap. At answer time that cap is
    GEN_MAX_NEW_TOKENS=512 and the tail dominates. Here `max_tokens` is
    the reference's summarization_length=100, so the worst case is 100
    decode steps for the batch — the uncapped-tail failure cannot occur.
    This is the one place batching is unambiguously correct.

    BATCH SHAPE IS PART OF THE ARTIFACT'S IDENTITY. Batch composition can
    change generated text at temperature 0 (padding plus batched-matmul
    reduction order flipping argmax on near-ties), and unlike answers,
    summaries are CACHED — they are the artifact a substrate key names.
    So `batch_size` and `max_padded_tokens` are both folded into
    `paper_substrate_extra` rather than assumed inert. Naming them beats
    pretending invariance; the cost is that changing either invalidates
    every tree built at the old value.

    `max_padded_tokens` is needed rather than optional: cluster contexts
    run from ~110 tokens up to `PaperTreeParams.max_length_in_cluster`
    (3500), so a fixed count sized for the short case OOMs on a layer of
    long ones — the same raggedness argument that produced
    `models.token_budget_batches`.

    The GenerationConfig names the SAME model as the answer path:
    `load_generator` is keyed on the model name, so a different spelling
    would load a second ~15 GB copy of the same weights instead of
    reusing the resident one.
    """
    from .config import GenerationConfig
    from .models import generate_batch

    if not contexts:
        return []
    return generate_batch(
        [PAPER_SUMMARY_SYSTEM_PROMPT] * len(contexts),
        [PAPER_SUMMARY_USER_TEMPLATE.format(context=c) for c in contexts],
        cfg=GenerationConfig(
            model=model, max_new_tokens=max_tokens, temperature=temperature
        ),
        batch_size=batch_size,
        sort_by_length=True,
        progress_every=progress_every,
        max_padded_tokens=max_padded_tokens,
    )


def summarize_paper_style(
    context: str,
    *,
    model: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """One summary call, model-agnostic. Retained for single-call use.

    M4's index path uses `summarize_paper_style_batch`; this stays as the
    reference-faithful one-context form (and the place the prompt and
    temperature reasoning below is recorded).

    Routes through `models.generate`, which dispatches on the model id
    (OpenAI prefixes to the API, anything else to a local causal LM), so
    the same code path serves an API summariser and a local one. Late
    import keeps raptor_paper free of src imports at module load — the
    property that lets config.py import PaperTreeParams from here without
    a cycle.

    temperature defaults to 0.0, NOT the reference's unset (=1.0). That
    is a deliberate deviation on infrastructure grounds: a non-zero
    temperature makes the tree a random function of its inputs, and a
    cache key that does not determine the artifact it names is not a
    cache key. Consequence, recorded: the reference implementation's own
    trees are not reproducible run to run.
    """
    from .config import GenerationConfig
    from .models import generate

    return generate(
        system_prompt=PAPER_SUMMARY_SYSTEM_PROMPT,
        user_prompt=PAPER_SUMMARY_USER_TEMPLATE.format(context=context),
        cfg=GenerationConfig(
            model=model, max_new_tokens=max_tokens, temperature=temperature
        ),
    )


# --- bottom-up construction (port of cluster_tree_builder.construct_tree) -


ClusterFn = Callable[[list[PaperNode], PaperTreeParams, dict], list[list[PaperNode]]]
SummarizeBatchFn = Callable[[list[str]], list[str]]
EmbedFn = Callable[[list[str]], np.ndarray]


def _cluster_sort_key(
    cluster: list[PaperNode], position: dict[str, int]
) -> tuple:
    """Deterministic cluster ordering, so node ids never depend on timing.

    The reference assigns node indices inside a Lock-guarded dict while
    summarising on a ThreadPoolExecutor, which makes its ids a function
    of completion order. Ours are a function of member position in the
    input layer, so no concurrency or batching decision can perturb the
    tree's SHAPE or its node IDS — the cache-identity contract again (see
    micro-divergence (i)). Summary TEXT is a separate matter: batch
    composition can move it, which is why batch shape is in the key.
    """
    positions = sorted(position[n.node_id] for n in cluster)
    return (positions[0], len(positions), tuple(positions))


def build_paper_tree(
    chunk_texts: list[str],
    chunk_embeddings: np.ndarray,
    *,
    params: PaperTreeParams,
    summarize_batch_fn: SummarizeBatchFn,
    embed_fn: EmbedFn,
    cluster_fn: ClusterFn | None = None,
    on_summary: Callable[[PaperNode], None] | None = None,
    verbose: bool = False,
) -> PaperTree:
    """Build the paper's bottom-up tree.

    Layer 0 is the leaf chunks. Each iteration soft-clusters the current
    layer, summarises every cluster over `get_text(cluster)`, embeds the
    new summaries, and repeats. Stops at `params.num_layers` iterations
    or, per the reference, as soon as a layer holds no more than
    `reduction_dimension + 1` nodes.

    `summarize_batch_fn` takes a WHOLE LAYER's concatenated context
    strings and returns summaries POSITIONALLY ALIGNED to them. Each
    context is what the reference passes as `summarize(context=...)`; the
    batching is ours, and it is the layer-granularity form because a
    layer is produced all at once (see `summarize_paper_style_batch` for
    why this replaced a ThreadPoolExecutor, and why the batch shape is
    part of the substrate cache key).

    `cluster_fn` is a test seam: it defaults to `perform_clustering` and
    exists so the deterministic bookkeeping (ids, multi-parent links,
    leaf-index closure) can be tested without paying for UMAP fits.

    Node ids and tree shape are computed before any summary call is
    dispatched, so they are invariant under every batching decision.
    Summary TEXT is not promised to be, which is exactly why batch size
    and the padded-token budget are named in the key.
    """
    if chunk_embeddings.ndim != 2:
        raise ValueError("chunk_embeddings must be 2D (n_chunks, dim)")
    if len(chunk_texts) != chunk_embeddings.shape[0]:
        raise ValueError(
            f"chunk_texts ({len(chunk_texts)}) and chunk_embeddings "
            f"({chunk_embeddings.shape[0]}) length mismatch"
        )
    if not chunk_texts:
        raise ValueError("chunk_texts must be non-empty")

    cluster = cluster_fn if cluster_fn is not None else perform_clustering
    _CLOCK.reset()
    try:
        from .models import reset_generate_calls

        reset_generate_calls()
    except Exception:
        pass
    stats: dict = {"n_summary_calls": 0}

    nodes: dict[str, PaperNode] = {}
    layer_to_nodes: dict[int, list[str]] = {0: []}
    for i, text in enumerate(chunk_texts):
        nid = f"L0_{i:06d}"
        nodes[nid] = PaperNode(
            node_id=nid,
            layer=0,
            text=text,
            leaf_indices=[i],
            embedding=chunk_embeddings[i].astype(np.float32, copy=False),
        )
        layer_to_nodes[0].append(nid)

    current = list(layer_to_nodes[0])
    realised_layers = 1

    for layer in range(params.num_layers):
        if len(current) <= params.reduction_dimension + 1:
            # Reference stop condition: `len(node_list_current_layer) <=
            # self.reduction_dimension + 1`.
            if verbose:
                print(
                    f"[raptor_paper] stop at layer {layer}: "
                    f"{len(current)} nodes <= {params.reduction_dimension + 1}"
                )
            break

        layer_nodes = [nodes[nid] for nid in current]
        position = {nid: i for i, nid in enumerate(current)}
        clusters = cluster(layer_nodes, params, stats)
        clusters = [c for c in clusters if c]
        if not clusters:
            break
        clusters.sort(key=lambda c: _cluster_sort_key(c, position))

        contexts = [get_text(c) for c in clusters]
        _t0 = _time_mod.perf_counter()
        summaries = list(summarize_batch_fn(contexts))
        _CLOCK.add("summarize", _time_mod.perf_counter() - _t0)
        if len(summaries) != len(contexts):
            # Alignment is POSITIONAL, so a length mismatch does not
            # raise on its own — it silently attaches each summary to the
            # wrong cluster and produces a plausible, wrong tree. Refuse.
            raise RuntimeError(
                f"summarize_batch_fn returned {len(summaries)} summaries for "
                f"{len(contexts)} clusters at layer {layer + 1}"
            )
        stats["n_summary_calls"] += len(summaries)
        stats["n_summary_layers"] = stats.get("n_summary_layers", 0) + 1

        next_layer = layer + 1
        new_ids: list[str] = []
        for rank, (members, summary) in enumerate(zip(clusters, summaries)):
            nid = f"L{next_layer}_{rank:06d}"
            leaves: set[int] = set()
            for m in members:
                leaves.update(m.leaf_indices)
            node = PaperNode(
                node_id=nid,
                layer=next_layer,
                text=summary or "",
                children=[m.node_id for m in members],
                leaf_indices=sorted(leaves),
            )
            for m in members:
                m.parent_ids.append(nid)
            nodes[nid] = node
            new_ids.append(nid)
            if on_summary is not None:
                on_summary(node)

        # Batch-embed the new layer's summaries in one pass. Empty
        # summaries get no embedding and are excluded from the collapsed
        # index, mirroring src/raptor.py's handling.
        texts = [nodes[nid].text for nid in new_ids]
        keep = [bool(t and t.strip()) for t in texts]
        to_embed = [t for t, k in zip(texts, keep) if k]
        if to_embed:
            _t0 = _time_mod.perf_counter()
            emb = embed_fn(to_embed)
            _CLOCK.add("embed", _time_mod.perf_counter() - _t0)
            it = iter(emb)
            for nid, k in zip(new_ids, keep):
                nodes[nid].embedding = (
                    next(it).astype(np.float32, copy=False) if k else None
                )

        layer_to_nodes[next_layer] = new_ids
        realised_layers = next_layer + 1
        # A node whose summariser returned empty text has no embedding and
        # cannot be clustered further (np.asarray(None) would raise). It
        # stays in the tree for provenance but drops out of the next
        # round, and out of the collapsed index. The reference does not
        # check this because it never inspects the summariser's output.
        current = [nid for nid in new_ids if nodes[nid].embedding is not None]
        if not current:
            break
        if verbose:
            print(
                f"[raptor_paper] layer {next_layer}: {len(new_ids)} nodes "
                f"from {len(layer_nodes)} ({len(layer_nodes) / max(1, len(new_ids)):.1f} children/parent)"
            )

    tree = PaperTree(
        nodes=nodes,
        layer_to_nodes=layer_to_nodes,
        n_layers=realised_layers,
        params=params,
        stats=stats,
    )
    tree.stats.update(tree_stats(tree))

    # DEGENERATE BUILD — no summary layer was ever produced, so this is
    # not a RAPTOR tree, it is a flat list of chunks. The stop condition
    # (`len(current) <= reduction_dimension + 1`) fires on the FIRST
    # iteration for any corpus of <= 11 leaves, and the reference exits
    # just as quietly.
    #
    # This must be loud. A silently flat M4 still retrieves, still
    # answers, and still produces a plausible row in a results table --
    # it just is not the system the row claims. It is a structural
    # property of small corpora (HotpotQA's standard distractor setting
    # gives ~10 paragraphs per question, i.e. ~8-12 leaves) rather than a
    # bug, which is exactly why nothing else would ever flag it.
    try:
        from .models import generate_calls_summary

        # WHOLESALE, not a hand-picked list. This block used to name four
        # keys of GENERATE_CALLS one at a time, which is the same
        # enumeration bug that dropped phase_seconds and generate_calls
        # from the probe row and cost two cold builds. The per-call
        # breakdown added for the 230 s/call investigation arrives here
        # because the summary is copied whole.
        tree.stats["generate_calls"] = generate_calls_summary()
    except Exception:
        tree.stats["generate_calls"] = None
    tree.stats.update(_CLOCK.as_stats())
    tree.stats["degenerate_no_tree"] = not tree.summary_nodes()
    if tree.stats["degenerate_no_tree"]:
        print(
            f"[raptor_paper] *** NO TREE BUILT: {len(chunk_texts)} leaves is "
            f"at or below the stop condition "
            f"({params.reduction_dimension + 1}), so this index has layer 0 "
            "ONLY. M4 degenerates to flat dense retrieval here and its "
            "numbers are NOT a RAPTOR result. ***"
        )
    return tree


# --- fidelity gates -------------------------------------------------------


def tree_stats(tree: PaperTree) -> dict:
    """Diagnostics, including the three paper-comparable fidelity gates.

    Paper targets (App. C Table 10 / App. I), to be checked on the FIRST
    real tree before any further benchmark is built:
      * children per parent  5.7 - 6.8
      * mean summary length  ~131 tokens  — but see ruling 4: the
        reference caps completions at summarization_length=100, so a
        faithful build should land NEAR 100 with visible truncation. A
        mean at ~100 rather than ~131 is itself the finding, namely that
        the paper's reported figure cannot have come from this config.
      * non-leaf share of RETRIEVED nodes 18.5% - 57% — query-time, so
        it is measured by the analyser, not here.
    """
    summaries = tree.summary_nodes()
    n_children = [len(n.children) for n in summaries]
    summary_tokens = [
        len(_encoding().encode(n.text)) for n in summaries if n.text.strip()
    ]
    multi_parent = [n for n in tree.nodes.values() if len(n.parent_ids) > 1]
    return {
        "n_nodes": len(tree.nodes),
        "n_leaves": len(tree.layer_to_nodes.get(0, [])),
        "n_summary_nodes": len(summaries),
        "n_layers": tree.n_layers,
        "layer_sizes": {
            int(k): len(v) for k, v in sorted(tree.layer_to_nodes.items())
        },
        "mean_children_per_parent": (
            float(np.mean(n_children)) if n_children else 0.0
        ),
        "mean_summary_tokens": (
            float(np.mean(summary_tokens)) if summary_tokens else 0.0
        ),
        "max_summary_tokens": max(summary_tokens) if summary_tokens else 0,
        "n_multi_parent_nodes": len(multi_parent),
        "multi_parent_share": (
            len(multi_parent) / len(tree.nodes) if tree.nodes else 0.0
        ),
        "parent_count_histogram": dict(
            Counter(len(n.parent_ids) for n in tree.nodes.values())
        ),
    }


# --- collapsed index (paper: the ENTIRE tree in one layer) ----------------


@dataclass
class PaperCollapsedIndex:
    """FAISS index over EVERY node of the tree, leaves and summaries alike.

    Paper §3 Querying: "First, collapse the entire RAPTOR tree into a
    single layer... calculate the cosine similarity between the query
    embedding and the embeddings of all nodes present in the collapsed
    set." There is no root exclusion — src/raptor.py's
    `include_root_in_flat_index=False` has no counterpart here, because
    the paper's top layer is whatever the clustering produced rather than
    a synthetic all-corpus root.
    """

    faiss_index: Any
    refs: list[dict]  # {"node_id": str, "layer": int, "is_leaf": bool}
    dim: int


def build_collapsed_index(tree: PaperTree) -> PaperCollapsedIndex:
    import faiss

    rows: list[np.ndarray] = []
    refs: list[dict] = []
    for nid in tree.all_node_ids():
        node = tree.nodes[nid]
        if node.embedding is None:
            continue
        rows.append(np.asarray(node.embedding, dtype=np.float32).reshape(1, -1))
        refs.append({
            "node_id": nid,
            "layer": int(node.layer),
            "is_leaf": bool(node.is_leaf),
        })
    if not rows:
        raise RuntimeError("collapsed index: no embedded nodes")

    combined = np.vstack(rows)
    dim = int(combined.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(combined)
    return PaperCollapsedIndex(faiss_index=index, refs=refs, dim=dim)


# --- serialisation (JSON topology + .npy embeddings + FAISS binary) -------
# Same contract as src/raptor.py: no pickle, because these artifacts get
# re-inspected months later during thesis writing and JSON survives
# Python upgrades that pickle does not.


def save_paper_tree(tree: PaperTree, tree_json_path: Path, emb_path: Path) -> None:
    tree_json_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = tree.all_node_ids()
    obj = {
        "schema": PAPER_TREE_SCHEMA_VERSION,
        "n_layers": tree.n_layers,
        "params": asdict(tree.params),
        "stats": tree.stats,
        "layer_to_nodes": {
            str(k): list(v) for k, v in sorted(tree.layer_to_nodes.items())
        },
        "nodes": [
            {
                "node_id": n.node_id,
                "layer": n.layer,
                "text": n.text,
                "children": list(n.children),
                "parent_ids": list(n.parent_ids),
                "leaf_indices": list(n.leaf_indices),
                "has_embedding": n.embedding is not None,
            }
            for n in (tree.nodes[nid] for nid in ordered)
        ],
    }
    tree_json_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    embs = [
        tree.nodes[nid].embedding
        for nid in ordered
        if tree.nodes[nid].embedding is not None
    ]
    mat = (
        np.vstack([e.reshape(1, -1) for e in embs]).astype(np.float32)
        if embs
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.save(emb_path, mat)


def load_paper_tree(tree_json_path: Path, emb_path: Path) -> PaperTree:
    obj = json.loads(tree_json_path.read_text())
    schema = obj.get("schema")
    if schema != PAPER_TREE_SCHEMA_VERSION:
        raise ValueError(
            f"paper tree schema mismatch: on-disk {schema!r} != "
            f"{PAPER_TREE_SCHEMA_VERSION!r}. The cache key should have "
            "prevented this; do not silently coerce."
        )
    params = PaperTreeParams(**obj["params"])
    embs = np.load(emb_path) if Path(emb_path).exists() else None
    emb_iter = iter(embs) if embs is not None and len(embs) > 0 else None

    nodes: dict[str, PaperNode] = {}
    for d in obj["nodes"]:
        node = PaperNode(
            node_id=d["node_id"],
            layer=int(d["layer"]),
            text=d.get("text", ""),
            children=list(d.get("children", ())),
            parent_ids=list(d.get("parent_ids", ())),
            leaf_indices=[int(i) for i in d.get("leaf_indices", ())],
        )
        if d.get("has_embedding") and emb_iter is not None:
            node.embedding = next(emb_iter).astype(np.float32, copy=False)
        nodes[node.node_id] = node

    return PaperTree(
        nodes=nodes,
        layer_to_nodes={
            int(k): list(v) for k, v in obj["layer_to_nodes"].items()
        },
        n_layers=int(obj["n_layers"]),
        params=params,
        stats=dict(obj.get("stats", {})),
    )


def save_collapsed_index(
    idx: PaperCollapsedIndex, faiss_path: Path, meta_path: Path
) -> None:
    import faiss

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx.faiss_index, str(faiss_path))
    meta_path.write_text(
        json.dumps({"dim": idx.dim, "refs": idx.refs}, ensure_ascii=False)
    )


def load_collapsed_index(
    faiss_path: Path, meta_path: Path
) -> PaperCollapsedIndex:
    import faiss

    meta = json.loads(meta_path.read_text())
    return PaperCollapsedIndex(
        faiss_index=faiss.read_index(str(faiss_path)),
        refs=list(meta["refs"]),
        dim=int(meta["dim"]),
    )


# --- cache identity (Lever B, strict reading) -----------------------------


def _topology_env_id() -> str:
    """Resolved versions of the libraries that DETERMINE tree topology.

    UMAP + GMM output is version-sensitive even when seeded, which is why
    P9 pins the stack at all. A tree built under an unpinned environment
    is therefore not reproducible under the pinned one, and serving it
    from cache would put M4 cells on artifacts the reproducibility
    control declares unreproducible.

    Folding this into the substrate key is the deliberate lever that
    makes those old trees unreachable: the key changes with the stack, so
    a cold build is forced rather than requested.

    ONLY the three libraries that actually move topology are named.
    Keying on the whole lockfile would rebuild every tree when an
    unrelated test-only dependency moved, which is invalidation without
    a reason. A missing package resolves to "absent" rather than raising:
    this runs at import time on hosts that never build a tree.

    THE INTERPRETER IS NAMED TOO, added 2026-08-19 after it drifted
    unguarded. Cells 1-5 built under CPython 3.12.13 and cell 6 under
    3.13.15 with every package version identical, so the lockfile hash
    was unchanged, the pin reported OK, and both trees shared a cache
    key. **Identical package versions are not identical package code**:
    cp312 and cp313 ship different compiled wheels for numpy, numba and
    llvmlite, and UMAP's JIT paths are where a last-digit float move can
    flip a GMM argmax on a near-tie.

    Only MAJOR.MINOR is keyed. A patch bump (3.12.13 -> 3.12.14) does
    not change the ABI or the wheel tag, so keying the patch would force
    cold rebuilds for a change that cannot move topology — invalidation
    without a reason, the same argument that keeps this to three
    packages. The full version including the patch is still CHECKED by
    `pin_environment` and recorded per cell; it is the KEY that is
    coarser, deliberately.
    """
    import sys
    from importlib.metadata import PackageNotFoundError, version

    parts = [f"python={sys.version_info.major}.{sys.version_info.minor}"]
    for pkg in ("umap-learn", "scikit-learn", "numpy"):
        try:
            parts.append(f"{pkg}={version(pkg)}")
        except PackageNotFoundError:
            parts.append(f"{pkg}=absent")
    return ";".join(parts)


# Resolved once at import. Recorded in every M4 manifest and run summary
# so a cell says which stack built its tree.
PAPER_TREE_BUILD_ENV = _topology_env_id()


def paper_substrate_extra(
    *,
    params: PaperTreeParams,
    summary_model: str,
    summary_prompt_version: str,
    chunker_version: str = RAPTOR_CHUNKER_VERSION,
    summary_max_tokens: int = 100,
    summary_batch_size: int = 32,
    summary_max_padded_tokens: int = 16000,
    sparse: str = "bm25okapi",
    fusion: str = "rrf",
    rrf_k: int = 60,
    include_root: bool = True,
    build_env: str | None = None,
) -> dict:
    """M4's substrate-key extras. Deliberately NOT a call into src/raptor.py.

    The approved lever was "an optional kwarg on raptor_substrate_extra,
    written into the dict only when passed", which is provably safe for
    the frozen M7 — but it requires editing a file M7 imports. This
    function is the stricter reading of the same intent: it emits the
    SAME seven base fields that `raptor.raptor_substrate_extra` emits,
    plus the M4-only keys, so `src/raptor.py` is never opened at all and
    M7's key cannot move by construction rather than by argument.

    The M4-only keys exist because of the original landmine: the shared
    extras fold tree PARAMETERS but never the clustering ALGORITHM, so
    swapping KMeans for UMAP+GMM would have changed the artifacts without
    changing the key and every warm cache would have silently served the
    old tree. `clustering.algo` and `tree_schema` are what make the swap
    visible to the key.

    `include_root` defaults True: the paper collapses the ENTIRE tree,
    and there is no synthetic all-corpus root to exclude. The field is
    kept only so the base schema stays recognisable next to M7's.

    `summary_batch_size` / `summary_max_padded_tokens` name the SHAPE of
    the batched summariser call. They are here because batch composition
    can change generated text at temperature 0, and summaries — unlike
    answers — are cached, so they ARE the artifact this key names.
    Naming the shape beats assuming invariance; the accepted cost is that
    retuning either knob (after an OOM, say) invalidates every tree built
    at the old value. Both are M4-local: they live on M4Config, which
    nothing on M7's key derivation reads.
    """
    return {
        # --- the seven base fields, same names as raptor_substrate_extra ---
        "tree": asdict(params),
        "summary_model": summary_model,
        "summary_prompt_version": summary_prompt_version,
        "include_root_in_flat_index": bool(include_root),
        "sparse": sparse,
        "fusion": fusion,
        "rrf_k": int(rrf_k),
        # --- M4-only: what the shared extras could not express ---
        "clustering": {"algo": "umap_gmm_bic"},
        "tree_schema": PAPER_TREE_SCHEMA_VERSION,
        "chunker_impl": chunker_version,
        "summary_max_tokens": int(summary_max_tokens),
        "summary_batch_size": int(summary_batch_size),
        "summary_max_padded_tokens": int(summary_max_padded_tokens),
        # THE COLD-TREE LEVER. See _topology_env_id: keys the substrate on
        # the stack that determines UMAP+GMM topology, so a tree built
        # under a different stack cannot satisfy this key.
        "build_env": PAPER_TREE_BUILD_ENV if build_env is None else build_env,
    }


__all__ = [
    "REFERENCE_ENCODING",
    "REFERENCE_RANDOM_SEED",
    "RAPTOR_CHUNKER_VERSION",
    "PAPER_TREE_SCHEMA_VERSION",
    "PAPER_NON_LEAF_SHARE_BAND",
    "TextSpan",
    "PaperNode",
    "PaperTree",
    "PaperTreeParams",
    "PaperCollapsedIndex",
    "PAPER_SUMMARY_SYSTEM_PROMPT",
    "PAPER_SUMMARY_USER_TEMPLATE",
    "count_tokens_reference",
    "split_text_raptor",
    "summarize_paper_style",
    "summarize_paper_style_batch",
    "perform_clustering",
    "get_text",
    "build_paper_tree",
    "tree_stats",
    "build_collapsed_index",
    "save_paper_tree",
    "load_paper_tree",
    "save_collapsed_index",
    "load_collapsed_index",
    "paper_substrate_extra",
]
