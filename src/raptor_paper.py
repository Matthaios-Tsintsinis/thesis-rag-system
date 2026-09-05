"""RAPTOR substrate for M4 (Sarthi et al., ICLR 2024, arXiv 2401.18059):
reference chunker, bottom-up UMAP+GMM tree, collapsed index, serialisation
and the substrate cache-key extras."""

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


# ref: raptor/utils.py::split_text @ 7da1d48a (tiktoken cl100k_base)
REFERENCE_ENCODING = "cl100k_base"

# Names the chunker's output; part of M4's cache-key extras.
RAPTOR_CHUNKER_VERSION = "raptor_split_text_v1"

# Sentence boundaries. A run of . ! ? stays on the sentence it closes; a
# run of newlines is a boundary only. Each run counts as one boundary.
# deviation from ref (ref's re.split drops . ! ? \n): see METHODS §A.4.4 ruling 1
_TERMINATOR_RUN = r"[.!?]+"
_NEWLINE_RUN = r"\n+"
_BOUNDARY_RE = re.compile(f"({_TERMINATOR_RUN})|({_NEWLINE_RUN})")

# Sub-split of a sentence that alone exceeds the chunk size, delimiters
# kept on the piece they close.
# ref: raptor/utils.py::split_text @ 7da1d48a
_SUBPHRASE_RE = re.compile(r"([,;:]+)")


@functools.lru_cache(maxsize=2)
def _encoding(name: str = REFERENCE_ENCODING) -> Any:
    """Cached tiktoken encoding by name."""
    import tiktoken

    return tiktoken.get_encoding(name)


def count_tokens_plain(text: str, *, encoding_name: str = REFERENCE_ENCODING) -> int:
    """Bare token count, as the ref counts cluster text and the budget."""
    return len(_encoding(encoding_name).encode(text))


def count_tokens_reference(text: str, *, encoding_name: str = REFERENCE_ENCODING) -> int:
    """Token count with a leading space, the ref chunker's convention."""
    return len(_encoding(encoding_name).encode(" " + text))


@dataclass(frozen=True)
class TextSpan:
    """One chunk plus the region of the original text it came from."""

    # `text` is not `original[start_char:end_char]`: whitespace between
    # sentences is collapsed on rejoin. The span maps a chunk back to the
    # source items it overlaps; never use it as a slice.
    text: str
    start_char: int
    end_char: int
    n_tokens: int


def _iter_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into (sentence, start, end) with terminators attached."""
    out: list[tuple[str, int, int]] = []
    cursor = 0

    def _emit(lo: int, hi: int) -> None:
        raw = text[lo:hi]
        stripped = raw.strip()
        if not stripped:
            return
        # Anchor the span on the stripped content so offsets skip
        # surrounding whitespace.
        lead = len(raw) - len(raw.lstrip())
        out.append((stripped, lo + lead, lo + lead + len(stripped)))

    # A terminator run stays on its sentence; a newline run is dropped;
    # the remainder after the last boundary is the final sentence.
    for m in _BOUNDARY_RE.finditer(text):
        if m.group(1) is not None:
            _emit(cursor, m.end())
        else:
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
    """Sub-split an over-long sentence on , ; : keeping the delimiters."""
    # ref: raptor/utils.py::split_text @ 7da1d48a
    # A piece that is still over budget is returned as-is, so the chunk
    # size is a soft bound.
    pieces: list[tuple[str, int, int]] = []
    cursor = 0
    parts: list[tuple[int, int]] = []
    for m in _SUBPHRASE_RE.finditer(sentence):
        parts.append((cursor, m.end()))
        cursor = m.end()
    parts.append((cursor, len(sentence)))

    # Keep non-empty pieces with spans anchored on their stripped text.
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


# RAPTOR paper §3: "short, contiguous texts of length 100"
# ref: raptor/utils.py::split_text @ 7da1d48a (overlap never passed, 0)
def split_text_raptor(
    text: str,
    *,
    max_tokens: int = 100,
    encoding_name: str = REFERENCE_ENCODING,
) -> list[TextSpan]:
    """Port of the reference split_text: 100-token chunks, no overlap."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not text or not text.strip():
        return []

    # Flatten to (piece, start, end, n_tokens). A sentence over the limit
    # is sub-split first; a piece still over it stays oversized.
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

    # Pack pieces in document order. A piece that would overflow starts
    # the next chunk, and nothing carries over between chunks.
    # RAPTOR paper §3: "we move the entire sentence to the next chunk"
    # ref: raptor/utils.py::split_text @ 7da1d48a (overlap never passed, 0)
    # deviation from ref (ref emits pieces out of order): see METHODS §A.4.4 ruling 1b
    for piece, lo, hi, n_tok in pieces:
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
# Bottom-up tree: UMAP (global then local) -> BIC-selected GMM soft
# clustering -> one summary per cluster -> re-embed -> repeat. Soft
# membership lets a node join several clusters, so the result is a DAG
# and a node carries a list of parents.
# RAPTOR paper §3: "cycle of embedding, clustering, and summarization"
# =========================================================================


# Names the on-disk tree layout; part of M4's cache-key extras.
PAPER_TREE_SCHEMA_VERSION = "raptor_paper_bottom_up_v1"

# Share of retrieved nodes that are summaries, measured at query time.
# RAPTOR paper App. I: 18.5%-57% of retrieved nodes are non-leaf
PAPER_NON_LEAF_SHARE_BAND = (0.185, 0.57)

# ref: raptor/cluster_utils.py @ 7da1d48a (RANDOM_SEED = 224)
REFERENCE_RANDOM_SEED = 224


@dataclass(frozen=True)
class PaperTreeParams:
    """Tree-build parameters; asdict() is the "tree" cache-key field."""

    # ref: raptor/cluster_tree_builder.py::ClusterTreeConfig @ 7da1d48a (reduction_dimension=10)
    reduction_dimension: int = 10
    # GMM posterior cutoff for soft membership, not a retrieval threshold.
    # ref: raptor/cluster_utils.py::perform_clustering @ 7da1d48a (threshold=0.1)
    gmm_threshold: float = 0.1
    # cl100k_base tokens summed over a cluster's node texts.
    # ref: raptor/cluster_utils.py::perform_clustering @ 7da1d48a (max_length_in_cluster=3500)
    max_length_in_cluster: int = 3500
    # Upper bound; the realised depth is usually lower.
    # ref: raptor/tree_builder.py::TreeBuilderConfig @ 7da1d48a (num_layers=5)
    num_layers: int = 5
    # ref: raptor/cluster_utils.py::local_cluster_embeddings @ 7da1d48a (num_neighbors=10)
    local_n_neighbors: int = 10
    # Both UMAP calls; the global n_neighbors is int(sqrt(n-1)), not a field.
    # ref: raptor/cluster_utils.py::global_cluster_embeddings @ 7da1d48a
    metric: str = "cosine"
    # ref: raptor/cluster_utils.py::get_optimal_clusters @ 7da1d48a (max_clusters=50)
    bic_max_clusters: int = 50
    # ref: raptor/cluster_utils.py @ 7da1d48a (RANDOM_SEED = 224)
    bic_random_state: int = REFERENCE_RANDOM_SEED
    # ref: raptor/cluster_utils.py::GMM_cluster @ 7da1d48a (random_state=0; the ref uses both seeds)
    gmm_random_state: int = 0
    # deviation from ref (ref seeds nothing; a cache key must fix its artifact): see METHODS §A.4.4 (i)
    umap_random_state: int = 42
    # deviation from ref (ref recursion has no base case): see METHODS §A.4.4 (ii)
    max_recluster_depth: int = 8


@dataclass
class PaperNode:
    """One tree node; layer 0 is a leaf chunk and parent_ids may hold many."""

    # Soft clustering puts a node in every cluster over the threshold, so
    # it can be summarised into several parents. `leaf_indices` is the
    # closure down to layer-0 chunk indices and may overlap between
    # siblings.
    node_id: str
    layer: int
    text: str
    children: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    leaf_indices: list[int] = field(default_factory=list)
    embedding: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        """True for a layer-0 chunk node."""
        return self.layer == 0


@dataclass
class PaperTree:
    """The built tree: nodes by id, ids per layer, params and build stats."""

    nodes: dict[str, PaperNode]
    layer_to_nodes: dict[int, list[str]]
    n_layers: int
    params: PaperTreeParams
    stats: dict = field(default_factory=dict)

    def all_node_ids(self) -> list[str]:
        """Every node in every layer, leaves first: the collapsed set."""
        out: list[str] = []
        for layer in sorted(self.layer_to_nodes):
            out.extend(self.layer_to_nodes[layer])
        return out

    def summary_nodes(self) -> list[PaperNode]:
        """Every node above layer 0."""
        return [n for n in self.nodes.values() if n.layer > 0]


# --- clustering (port of raptor/cluster_utils.py) -------------------------


class _PhaseClock:
    """Wall time and call counts per build phase; instrumentation only."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        # One frame per open phase: [name, seconds its children consumed].
        self._stack: list[list] = []

    def reset(self) -> None:
        """Clear every phase total, call count and open frame."""
        self.seconds.clear()
        self.calls.clear()
        self._stack.clear()

    def enter(self, phase: str) -> None:
        """Open a frame; child time accrues here, not to the phase."""
        self._stack.append([phase, 0.0])

    def exit_(self, elapsed: float) -> None:
        """Close the innermost frame and credit it only its own time."""
        # Phases partition the build: subtract child time so a nested
        # phase is charged once, then pass the whole span to the parent.
        phase, child_s = self._stack.pop()
        own = elapsed - child_s
        if own < 0:  # clock skew only; never let a phase go negative
            own = 0.0
        self.seconds[phase] = self.seconds.get(phase, 0.0) + own
        self.calls[phase] = self.calls.get(phase, 0) + 1
        if self._stack:
            self._stack[-1][1] += elapsed

    def add(self, phase: str, dt: float) -> None:
        """Accrue an inline-timed phase; the open parent frame is charged."""
        self.seconds[phase] = self.seconds.get(phase, 0.0) + dt
        self.calls[phase] = self.calls.get(phase, 0) + 1
        if self._stack:
            self._stack[-1][1] += dt

    def as_stats(self) -> dict:
        """Phase seconds, call counts and shares as a stats dict."""
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


# One clock per process; build_paper_tree resets it.
_CLOCK = _PhaseClock()


def get_phase_clock() -> _PhaseClock:
    """The build's phase timings; read after build_paper_tree returns."""
    return _CLOCK


def _timed(phase: str):
    """Decorator: the wrapped function's wall time accrues to `phase`."""
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
                # A raising phase still closes its frame.
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
    """Seeded UMAP down to `dim` components."""
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
    """Global UMAP; n_neighbors is int(sqrt(n - 1)), floored at 2."""
    # ref: raptor/cluster_utils.py::global_cluster_embeddings @ 7da1d48a
    n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return _umap_reduce(
        embeddings, dim, max(2, n_neighbors), params.metric, params.umap_random_state
    )


def _local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, params: PaperTreeParams
) -> np.ndarray:
    """Local UMAP with num_neighbors=10; dim and neighbours clamped to n."""
    # ref: raptor/cluster_utils.py::local_cluster_embeddings @ 7da1d48a (num_neighbors=10)
    # The ref clamps dim only on the global call; an unclamped local call
    # raises on a small cluster, so both are clamped here.
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
    """BIC sweep over k in arange(1, min(50, n)); the stop is excluded."""
    from sklearn.mixture import GaussianMixture

    # ref: raptor/cluster_utils.py::get_optimal_clusters @ 7da1d48a (max_clusters=50)
    max_clusters = min(params.bic_max_clusters, len(embeddings))
    candidates = np.arange(1, max_clusters)
    if len(candidates) == 0:
        # Empty range (n <= 1): one cluster, counted.
        # deviation from ref (ref drops the layer silently): see METHODS §A.4.4 (iv)
        stats["empty_bic_range_trips"] = stats.get("empty_bic_range_trips", 0) + 1
        return 1
    # A k whose fit raises (a component collapses onto one point) is
    # skipped and counted, so the argmin runs over every k that fits.
    # deviation from ref (ref crashes): see METHODS §A.4.4 (v)
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
        # Every k failed: one cluster is the only well-defined assignment.
        stats["bic_all_fits_failed"] = stats.get("bic_all_fits_failed", 0) + 1
        return 1
    return fitted[int(np.argmin(bics))]


@_timed("gmm_final_fit")
def _gmm_cluster(
    embeddings: np.ndarray, params: PaperTreeParams, stats: dict
) -> tuple[list[np.ndarray], int]:
    """Final GMM fit; a node joins every component above the threshold."""
    from sklearn.mixture import GaussianMixture

    # Walk k down from the BIC choice until a fit succeeds, so the layer
    # keeps as much structure as can be fitted. k=1 always fits.
    # ref: raptor/cluster_utils.py::GMM_cluster @ 7da1d48a (random_state=0; the ref uses both seeds)
    # deviation from ref (ref crashes): see METHODS §A.4.4 (v)
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

    # Unreachable while k=1 fits; keeps the return well-defined.
    stats["gmm_all_fits_failed"] = stats.get("gmm_all_fits_failed", 0) + 1
    return [np.array([0]) for _ in range(len(embeddings))], 1


def _two_stage_labels(
    embeddings: np.ndarray, params: PaperTreeParams, stats: dict
) -> list[np.ndarray]:
    """Global-then-local soft clustering with a flat label space."""
    dim = params.reduction_dimension
    n = len(embeddings)
    # Too few points to reduce: one cluster, no UMAP call.
    # deviation from ref (ref raises inside UMAP): see METHODS §A.4.4 (vii)
    if n <= dim + 1:
        return [np.array([0]) for _ in range(n)]

    # Reduce globally and fit a GMM, then reduce and fit again inside
    # each global cluster; local labels are offset by a running total so
    # the returned label space is flat.
    # ref: raptor/cluster_utils.py::perform_clustering @ 7da1d48a
    reduced_global = _global_cluster_embeddings(
        embeddings, min(dim, n - 2), params
    )
    global_labels, n_global = _gmm_cluster(reduced_global, params, stats)

    out: list[np.ndarray] = [np.array([], dtype=int) for _ in range(n)]
    total = 0
    for gi in range(n_global):
        # Members are tracked by index, so identical embeddings keep
        # their own labels.
        # deviation from ref (duplicate embeddings share labels in the ref): see METHODS §A.4.4 (vi)
        members = [i for i, lab in enumerate(global_labels) if gi in lab]
        if not members:
            continue
        sub = embeddings[members]
        if len(sub) <= dim + 1:
            # Too small to reduce: one local cluster.
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
    """Soft clusters; a cluster over the token cap is re-clustered."""
    # ref: raptor/cluster_utils.py::perform_clustering @ 7da1d48a
    stats = stats if stats is not None else {}
    if len(nodes) <= 1:
        return [list(nodes)]

    embeddings = np.vstack([
        np.asarray(n.embedding, dtype=np.float32).reshape(1, -1) for n in nodes
    ])
    labels = _two_stage_labels(embeddings, params, stats)

    # Nothing cleared the threshold: keep the layer as one cluster.
    # deviation from ref (ref drops the layer silently): see METHODS §A.4.4 (iv)
    non_empty = [lab for lab in labels if len(lab) > 0]
    if not non_empty:
        stats["empty_label_trips"] = stats.get("empty_label_trips", 0) + 1
        return [list(nodes)]

    # One cluster per label. A cluster within the token cap is kept; an
    # oversized one is re-clustered unless that cannot make progress.
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
            # The cluster is the whole input (BIC chose k=1), so another
            # round cannot split it; accept it and count the stop.
            # deviation from ref (ref recursion has no base case): see METHODS §A.4.4 (ii)
            stats["no_progress_trips"] = stats.get("no_progress_trips", 0) + 1
            clusters.append(members)
            continue
        if _depth >= params.max_recluster_depth:
            # Depth bound: accept the oversized cluster and count the trip.
            # deviation from ref (ref recursion has no base case): see METHODS §A.4.4 (ii)
            stats["recluster_guard_trips"] = (
                stats.get("recluster_guard_trips", 0) + 1
            )
            clusters.append(members)
            continue
        # Recurse with the same params.
        # deviation from ref (ref reverts them to defaults): see METHODS §A.4.4 (iii)
        stats["recluster_calls"] = stats.get("recluster_calls", 0) + 1
        clusters.extend(
            perform_clustering(members, params, stats, _depth=_depth + 1)
        )
    return clusters


# --- summarisation input format (port of raptor/utils.py::get_text) -------


def get_text(nodes: list[PaperNode]) -> str:
    """Summariser input; the ref's get_text, trailing blank line included."""
    # ref: raptor/utils.py::get_text @ 7da1d48a
    out = ""
    for n in nodes:
        out += f"{' '.join(n.text.splitlines())}"
        out += "\n\n"
    return out


# --- summarisation prompt -------------------------------------------------

# The ref's system prompt is "You are a helpful assistant."; the paper's
# wins. The user prompt is the same in both, trailing colon included.
# RAPTOR paper App. D Table 11 (paper over repo): see METHODS §A.4.3
PAPER_SUMMARY_SYSTEM_PROMPT = "You are a Summarizing Text Portal"
# RAPTOR paper App. D Table 11
PAPER_SUMMARY_USER_TEMPLATE = (
    "Write a summary of the following, including as many key details as "
    "possible: {context}:"
)


# ref: raptor/tree_builder.py::TreeBuilderConfig @ 7da1d48a (summarization_length=100); the paper's 131 is a measured mean (App. C)
# deviation from ref (ref leaves it unset; a cached summary must be reproducible): see METHODS §A.4.2
# harness choice: batch shape is in the cache key because it can move text at temperature 0
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
    """Summarise a whole layer's clusters in one batched call."""
    from .config import GenerationConfig
    from .models import generate_batch

    # The layer is the batch: generate_batch length-sorts across it, and
    # the padded-token cap bounds a batch of long cluster contexts. The
    # config names the same model id as the reader, so the resident copy
    # is reused.
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


# ref: raptor/tree_builder.py::TreeBuilderConfig @ 7da1d48a (summarization_length=100); the paper's 131 is a measured mean (App. C)
# deviation from ref (ref leaves it unset; a cached summary must be reproducible): see METHODS §A.4.2
def summarize_paper_style(
    context: str,
    *,
    model: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """One summary through models.generate; the single-context form."""
    # Late imports keep this module free of src imports at load time, so
    # config can import PaperTreeParams without a cycle.
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


# Seams: clustering, batched summarisation and embedding are injected.
ClusterFn = Callable[[list[PaperNode], PaperTreeParams, dict], list[list[PaperNode]]]
SummarizeBatchFn = Callable[[list[str]], list[str]]
EmbedFn = Callable[[list[str]], np.ndarray]


def _cluster_sort_key(
    cluster: list[PaperNode], position: dict[str, int]
) -> tuple:
    """Order clusters by member position, so ids do not depend on timing."""
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
    """Build the tree bottom-up: cluster, summarise, embed, next layer."""
    # Refuse mismatched or empty input.
    if chunk_embeddings.ndim != 2:
        raise ValueError("chunk_embeddings must be 2D (n_chunks, dim)")
    if len(chunk_texts) != chunk_embeddings.shape[0]:
        raise ValueError(
            f"chunk_texts ({len(chunk_texts)}) and chunk_embeddings "
            f"({chunk_embeddings.shape[0]}) length mismatch"
        )
    if not chunk_texts:
        raise ValueError("chunk_texts must be non-empty")

    # Fresh clock and generate-call counters for this build. `cluster_fn`
    # is a test seam that defaults to perform_clustering.
    cluster = cluster_fn if cluster_fn is not None else perform_clustering
    _CLOCK.reset()
    try:
        from .models import reset_generate_calls

        reset_generate_calls()
    except Exception:
        pass
    stats: dict = {"n_summary_calls": 0}

    # Layer 0: one node per chunk.
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
        # Stop when the layer is too small to cluster.
        # ref: raptor/cluster_tree_builder.py @ 7da1d48a (len(layer) <= reduction_dimension + 1)
        if len(current) <= params.reduction_dimension + 1:
            if verbose:
                print(
                    f"[raptor_paper] stop at layer {layer}: "
                    f"{len(current)} nodes <= {params.reduction_dimension + 1}"
                )
            break

        # Cluster the layer and pin the cluster order before any summary
        # call, so node ids and tree shape do not depend on batching.
        layer_nodes = [nodes[nid] for nid in current]
        position = {nid: i for i, nid in enumerate(current)}
        clusters = cluster(layer_nodes, params, stats)
        clusters = [c for c in clusters if c]
        if not clusters:
            break
        clusters.sort(key=lambda c: _cluster_sort_key(c, position))

        # One batched summary call per layer. Alignment is positional, so
        # a length mismatch is refused rather than attached to the wrong
        # cluster.
        contexts = [get_text(c) for c in clusters]
        _t0 = _time_mod.perf_counter()
        summaries = list(summarize_batch_fn(contexts))
        _CLOCK.add("summarize", _time_mod.perf_counter() - _t0)
        if len(summaries) != len(contexts):
            raise RuntimeError(
                f"summarize_batch_fn returned {len(summaries)} summaries for "
                f"{len(contexts)} clusters at layer {layer + 1}"
            )
        stats["n_summary_calls"] += len(summaries)
        stats["n_summary_layers"] = stats.get("n_summary_layers", 0) + 1

        # One parent node per cluster: each member gains a parent id and
        # the parent's leaf set is the union of its members' sets.
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

        # Embed the new summaries in one pass. An empty summary gets no
        # embedding and stays out of the collapsed index.
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

        # A node without an embedding stays in the tree for provenance
        # but cannot be clustered further.
        layer_to_nodes[next_layer] = new_ids
        realised_layers = next_layer + 1
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

    # Copy the generate-call summary whole, then the phase timings.
    try:
        from .models import generate_calls_summary

        tree.stats["generate_calls"] = generate_calls_summary()
    except Exception:
        tree.stats["generate_calls"] = None
    tree.stats.update(_CLOCK.as_stats())

    # No summary layer means a flat list of chunks, not a tree. The stop
    # condition fires on the first iteration for any corpus of at most
    # reduction_dimension + 1 leaves, and a flat M4 still retrieves and
    # still scores, so say so loudly.
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
    """Tree diagnostics, including the paper-comparable fidelity gates."""
    # Children per parent and mean summary tokens are tree properties;
    # the non-leaf share of retrieved nodes is measured at query time.
    # RAPTOR paper App. C: 131 tokens is the measured mean summary length
    # RAPTOR paper App. I: 18.5%-57% of retrieved nodes are non-leaf
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


# --- collapsed index (the whole tree in one layer) ------------------------


# RAPTOR paper §3: collapsed tree, the paper's main-results strategy
@dataclass
class PaperCollapsedIndex:
    """FAISS index over every node of the tree, leaves and summaries alike."""

    faiss_index: Any
    refs: list[dict]  # {"node_id": str, "layer": int, "is_leaf": bool}
    dim: int


def build_collapsed_index(tree: PaperTree) -> PaperCollapsedIndex:
    """Exact inner-product index over every embedded node, in layer order."""
    import faiss

    # Every embedded node, leaves first; no layer is excluded.
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
# No pickle: JSON and .npy survive Python upgrades.


def save_paper_tree(tree: PaperTree, tree_json_path: Path, emb_path: Path) -> None:
    """Write the topology as JSON and the embeddings as one .npy matrix."""
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

    # Embeddings in node order, skipping nodes that have none.
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
    """Read a tree back; refuses a schema string other than the current one."""
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

    # Rebuild nodes in file order; embeddings are consumed in the order
    # save_paper_tree writes them.
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
    """Write the FAISS index and its node refs beside it as JSON."""
    import faiss

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx.faiss_index, str(faiss_path))
    meta_path.write_text(
        json.dumps({"dim": idx.dim, "refs": idx.refs}, ensure_ascii=False)
    )


def load_collapsed_index(
    faiss_path: Path, meta_path: Path
) -> PaperCollapsedIndex:
    """Read a collapsed index and its node refs back."""
    import faiss

    meta = json.loads(meta_path.read_text())
    return PaperCollapsedIndex(
        faiss_index=faiss.read_index(str(faiss_path)),
        refs=list(meta["refs"]),
        dim=int(meta["dim"]),
    )


# --- cache identity -------------------------------------------------------


def _topology_env_id() -> str:
    """python MAJOR.MINOR plus umap-learn, scikit-learn and numpy versions."""
    import sys
    from importlib.metadata import PackageNotFoundError, version

    # UMAP and GMM output is version-sensitive even when seeded, so the
    # substrate key names the stack that determines topology. Only these
    # three packages and python MAJOR.MINOR are keyed: a patch bump does
    # not change a wheel tag, and an unrelated package cannot move a
    # tree. A missing package reads "absent" so import never raises.
    parts = [f"python={sys.version_info.major}.{sys.version_info.minor}"]
    for pkg in ("umap-learn", "scikit-learn", "numpy"):
        try:
            parts.append(f"{pkg}={version(pkg)}")
        except PackageNotFoundError:
            parts.append(f"{pkg}=absent")
    return ";".join(parts)


# Resolved once at import; recorded in every M4 manifest and run summary.
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
    """M4's substrate cache-key extras."""
    return {
        # Base fields shared with the other substrate keys. M4 collapses
        # the whole tree, so include_root is True; sparse, fusion and
        # rrf_k only keep the base schema.
        "tree": asdict(params),
        "summary_model": summary_model,
        "summary_prompt_version": summary_prompt_version,
        "include_root_in_flat_index": bool(include_root),
        "sparse": sparse,
        "fusion": fusion,
        "rrf_k": int(rrf_k),
        # M4-only fields: the clustering algorithm, the tree schema, the
        # chunker, and the summariser's cap and batch shape.
        # ref: raptor/tree_builder.py::TreeBuilderConfig @ 7da1d48a (summarization_length=100); the paper's 131 is a measured mean (App. C)
        # harness choice: batch shape is in the cache key because it can move text at temperature 0
        "clustering": {"algo": "umap_gmm_bic"},
        "tree_schema": PAPER_TREE_SCHEMA_VERSION,
        "chunker_impl": chunker_version,
        "summary_max_tokens": int(summary_max_tokens),
        "summary_batch_size": int(summary_batch_size),
        "summary_max_padded_tokens": int(summary_max_padded_tokens),
        # The topology stack; a tree built under another stack cannot
        # satisfy this key.
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
