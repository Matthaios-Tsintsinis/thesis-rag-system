"""Ours: three-axis retrieval on top of M4's RAPTOR tree.

The design is docs/three_axis_pipeline.pdf, section 4. This file follows its
steps 1-9 in order, then the answer (step 10, from development v7); the class
the runner calls is at the bottom. The tree,
its 100-token leaves and their vectors are M4's own, except that a unit holding a
document under M4's filter gets its own tree from M4's builder (development v12).
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from ours import config as settings

# (a) M4's own code loads or builds the tree, so the tree and its cache key stay
# M4's; a tree ours needs apart from M4's has its own place in the cache.
from src import paths
from src.cache import CacheDir
from src.parsing import walk_corpus
from src.retrievers.m4_raptor import RaptorSystem
# (b) The shared reader every system answers with; it also holds the embedder
# M4's leaves were made with.
from src import models
# (c) The interface the runner calls.
from src.chunking import Chunk
from src.retrievers.base import BaseSystem, PreparedQuery, RetrievedChunk
# (d) The scorers' own normaliser and refusal test, so the answer step reads an
# answer as it will be scored.
from src.eval.scorers.extractive import normalize_qasper_answer
from src.eval.scorers.unanswerable import is_abstention


# =========================================================================
# 0. The tree, as the query steps read it
# =========================================================================


@dataclass
class TreeIndex:
    """M4's tree for one unit, plus a BM25 table and the documents' sections."""

    chunks: list                      # M4's leaf chunks; leaf i is chunks[i]
    leaf_vectors: np.ndarray          # one unit-length row per leaf
    node_ids: list[str]               # searchable nodes: every leaf, then every summary but the root
    node_vectors: np.ndarray          # one row per searchable node
    vector_of: dict                   # node -> vector, for nodes that have one
    layer: dict                       # node -> 0 for a leaf, 1+ for a summary
    text: dict                        # node -> its text
    children: dict
    parents: dict
    leaves_under: dict                # node -> the leaf indices below it
    leaf_node: list[str]              # leaf index -> its node id
    root: str | None                  # a single top node; never searched or shown
    top_nodes: list[str]              # where the multi-branch walk starts
    bm25: object                      # BM25 over the leaves only
    section: list                     # leaf -> (document, section number)
    section_title: dict               # (document, section number) -> title
    detector: dict                    # document -> which section detector fired
    prev_leaf: list                   # the leaf before, in the same section
    next_leaf: list                   # the leaf after, in the same section
    cluster: list                     # leaf -> its nearest parent (its RAPTOR cluster)
    headers: list                     # leaf -> its document's name, or "" with nothing to tell apart
    source: list                      # leaf -> its document's source, or ""
    date: list                        # leaf -> its document's date in words, or ""
    source_names: list                # the unit's sources, longest first
    second_vectors: object = None     # one row per searchable node, in the second embedder


def build_tree_index(chunks, leaf_vectors, tree, docs, headers=None, fields=None) -> TreeIndex:
    """Collect what the query steps need from M4's tree and the unit's documents."""
    from rank_bm25 import BM25Okapi

    nodes = tree.nodes
    n = len(chunks)
    leaf_node = [""] * n
    for node_id in tree.layer_to_nodes.get(0, []):
        leaf_node[nodes[node_id].leaf_indices[0]] = node_id

    # The top layer: one node there is a root, whose children start the walk.
    top_layer = max(tree.layer_to_nodes)
    top = tree.layer_to_nodes[top_layer]
    root = top[0] if top_layer > 0 and len(top) == 1 else None
    if top_layer == 0:
        top_nodes = []
    elif root is not None:
        top_nodes = list(nodes[root].children)
    else:
        top_nodes = list(top)

    # Searchable nodes: leaves first (so search position i is leaf i), then
    # every summary that has a vector, the root excluded.
    summaries = [nid for layer in sorted(tree.layer_to_nodes) if layer > 0
                 for nid in tree.layer_to_nodes[layer]
                 if nid != root and nodes[nid].embedding is not None]
    node_ids = leaf_node + summaries
    vector_of = {leaf_node[i]: leaf_vectors[i] for i in range(n)}
    vector_of.update({nid: nodes[nid].embedding for nid in summaries})
    node_vectors = np.vstack([vector_of[nid] for nid in node_ids]).astype(np.float32)

    parents = {nid: list(node.parent_ids) for nid, node in nodes.items()}
    cluster = [parents[leaf_node[i]][0] if parents[leaf_node[i]] else leaf_node[i]
               for i in range(n)]

    # Sections, document by document, in chunk order.
    section = [None] * n
    section_title, detector = {}, {}
    prev_leaf, next_leaf = [None] * n, [None] * n
    by_doc: dict = {}
    for i, chunk in enumerate(chunks):
        by_doc.setdefault(chunk.doc_id, []).append(i)
    for doc, leaves in by_doc.items():
        leaves.sort(key=lambda i: chunks[i].position)
        _, title, text = docs.get(doc, (doc, doc, ""))
        numbers, titles, which = find_sections(
            [chunks[i].text for i in leaves], text, title)
        for i, k in zip(leaves, numbers):
            section[i] = (doc, k)
        for k, t in enumerate(titles):
            section_title[(doc, k)] = t
        detector[doc] = which
        for a, b in zip(leaves, leaves[1:]):
            if section[a] == section[b]:
                next_leaf[a], prev_leaf[b] = b, a

    names = [(headers or {}).get(chunk.doc_id, "") for chunk in chunks]
    known = [(fields or {}).get(chunk.doc_id, ("", "")) for chunk in chunks]
    return TreeIndex(
        chunks=list(chunks),
        leaf_vectors=np.asarray(leaf_vectors, dtype=np.float32),
        node_ids=node_ids,
        node_vectors=node_vectors,
        vector_of=vector_of,
        layer={nid: node.layer for nid, node in nodes.items()},
        text={nid: node.text for nid, node in nodes.items()},
        children={nid: list(node.children) for nid, node in nodes.items()},
        parents=parents,
        leaves_under={nid: list(node.leaf_indices) for nid, node in nodes.items()},
        leaf_node=leaf_node,
        root=root,
        top_nodes=top_nodes,
        bm25=BM25Okapi([tokenize(_headed(name, c.text)) for name, c in zip(names, chunks)]),
        section=section,
        section_title=section_title,
        detector=detector,
        prev_leaf=prev_leaf,
        next_leaf=next_leaf,
        cluster=cluster,
        headers=names,
        source=[source for source, _ in known],
        date=[date for _, date in known],
        source_names=sorted({source for source, _ in known if source}, key=lambda s: (-len(s), s)),
    )


def add_second_vectors(index) -> None:
    """The second embedder over the index's nodes, as the settings' mode wants them."""
    mode = settings.SECOND_EMBEDDER_MODE
    if mode not in ("off", "both", "only"):
        raise ValueError(f"unknown SECOND_EMBEDDER_MODE {mode!r}")
    if mode == "off":
        return
    # Leaves as the chunks read, with no document's name: as M4's vectors were made.
    n_leaves = len(index.chunks)
    texts = ([chunk.text for chunk in index.chunks]
             + [index.text[node] for node in index.node_ids[n_leaves:]])
    vectors = np.asarray(models.embed_texts(texts, model_name=settings.SECOND_EMBEDDER),
                         dtype=np.float32)
    if mode == "both":
        index.second_vectors = vectors
        return
    index.leaf_vectors = vectors[:n_leaves]
    index.node_vectors = vectors
    index.vector_of = dict(zip(index.node_ids, vectors))


def embed_query(texts):
    """A question's texts in the embedders searching: (the main vectors, the second's or None)."""
    mode = settings.SECOND_EMBEDDER_MODE
    main = settings.SECOND_EMBEDDER if mode == "only" else settings.EMBEDDER
    vectors = models.embed_texts(texts, model_name=main)
    second = models.embed_texts(texts, model_name=settings.SECOND_EMBEDDER) if mode == "both" else None
    return vectors, second


def _headed(name, text):
    """A text under its document's name, when it has one."""
    return f"[{name}] {text}" if name else text


def headed_text(index, leaf):
    """A leaf as the reader, the reranker and BM25 read it."""
    return _headed(index.headers[leaf], index.chunks[leaf].text)


def published_date(value):
    """A timestamp in the words questions use ("October 7, 2023"); "" when there is none to read."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return ""
    return f"{moment:%B} {moment.day}, {moment.year}"


def document_headers(items, chunks) -> dict:
    """Each chunked document's name: its source, date and title, as the corpus gives them.

    Empty for a one-document unit, whose chunks have nothing to tell apart."""
    first = _first_of_each_document(items)
    if not settings.DOCUMENT_HEADERS or len(first) < 2:
        return {}
    out = {}
    for chunk in chunks:
        if chunk.doc_id in out or not chunk.gold_provenance:
            continue
        parent = str(chunk.gold_provenance[0][0])
        metadata = first[parent].metadata if parent in first else {}
        parts = (metadata.get("source"), published_date(metadata.get("published_at")),
                 metadata.get("title") or parent)
        out[chunk.doc_id] = " | ".join(p for p in parts if p)
    return out


def document_fields(items, chunks) -> dict:
    """Each chunked document's source and date, for the steps a question's words can steer."""
    first = _first_of_each_document(items)
    out = {}
    for chunk in chunks:
        if chunk.doc_id in out or not chunk.gold_provenance:
            continue
        parent = str(chunk.gold_provenance[0][0])
        metadata = first[parent].metadata if parent in first else {}
        out[chunk.doc_id] = (metadata.get("source") or "",
                             published_date(metadata.get("published_at")))
    return out


def _first_of_each_document(items) -> dict:
    """Each document's first item; its metadata speaks for the document."""
    first: dict = {}
    for item in items:
        first.setdefault(str(item.parent_id), item)
    return first


def documents_of(items, chunks) -> dict:
    """Map each chunked document to (parent id, title, full text)."""
    members: dict = {}
    for item in items:
        members.setdefault(item.parent_id, []).append(item)
    out = {}
    for chunk in chunks:
        if chunk.doc_id in out or not chunk.gold_provenance:
            continue
        parent = str(chunk.gold_provenance[0][0])
        group = members.get(parent, [])
        title = (group[0].metadata.get("title") if group else "") or parent
        out[chunk.doc_id] = (parent, title, "\n\n".join(i.text for i in group))
    return out


# --- Sections (the structural axis without Docling) ----------------------


def find_sections(chunk_texts, doc_text, doc_title):
    """Split one document's chunks into sections: (section of each chunk, titles, detector)."""
    # Docling (PDF/DOCX layout) is an extension point, off by default: every
    # benchmark corpus is plain text.
    if settings.USE_DOCLING:
        raise NotImplementedError(
            "Docling sections are an extension point and are not implemented; "
            "set ours/config.py USE_DOCLING = False.")
    n = len(chunk_texts)
    # A document no longer than one block is one section.
    if n <= settings.SECTION_SIZE_CHUNKS:
        return [0] * n, [doc_title], "fixed"

    # 1. Headings: the first family that places enough of them wins.
    lines = [ln.strip() for ln in doc_text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) <= settings.MAX_HEADING_CHARS]
    for _, pattern in settings.HEADING_PATTERNS:
        placed = _place_headings([ln for ln in lines if re.match(pattern, ln)],
                                 chunk_texts)
        if len(placed) >= settings.MIN_HEADINGS:
            starts = dict(placed)
            titles = [] if 0 in starts else [doc_title]
            numbers = []
            for i in range(n):
                if i in starts:
                    titles.append(starts[i])
                numbers.append(len(titles) - 1)
            return numbers, titles, "headings"

    # 2. No headings: blocks of SECTION_SIZE_CHUNKS chunks.
    size = settings.SECTION_SIZE_CHUNKS
    numbers = [i // size for i in range(n)]
    titles = [f"{doc_title} [{k + 1}]" for k in range(numbers[-1] + 1)]
    return numbers, titles, "fixed"


def _words(text: str) -> str:
    """Lower-case letters and digits only, single-spaced: what matching compares."""
    return " ".join(re.findall(r"[0-9a-z]+", text.lower()))


def _place_headings(lines, chunk_texts):
    """(chunk index, heading) for each heading line found in the chunks, in order."""
    # Chunk text has no line breaks (M4's chunker joins sentences with
    # spaces), so headings are found as lines of the document and placed on
    # the chunk where their words appear.
    stream, starts = " ", []
    for text in chunk_texts:
        starts.append(len(stream))
        stream += _words(text) + " "
    placed, cursor = [], 0
    for line in lines:
        key = _words(line)
        if len(key) < settings.MIN_HEADING_WORD_CHARS:
            continue
        pos = stream.find(" " + key + " ", cursor)
        if pos < 0:
            continue
        cursor = pos + len(key) + 1
        chunk = bisect.bisect_right(starts, pos + 1) - 1
        # A heading right after the last one (a contents page) adds nothing.
        if not placed or chunk - placed[-1][0] >= settings.MIN_SECTION_CHUNKS:
            placed.append((chunk, line))
    return placed


# =========================================================================
# Step 1. Aspects and views
# =========================================================================


@dataclass
class Aspect:
    """One part of the question, with its two views and its share of the context."""

    name: str              # label in the prompt header
    text: str              # what retrieval, rerank and the title bias use
    importance: float
    views: list = field(default_factory=list)
    confidence: float = 0.0
    score: float = 0.0
    budget: int = 0
    focus: object = None   # the leaves its searches may return, when its words name a source


def decompose(question, ask):
    """One reader call splits the question into aspects; (aspects, parse failed)."""
    reply = ask(settings.DECOMPOSE_PROMPT, f"Question: {question}",
                settings.DECOMPOSE_MAX_TOKENS)
    parsed = parse_aspects(reply, question)
    if parsed is None:
        return [Aspect("main", question, 1.0)], True
    return choose_aspects(parsed, question), False


def parse_aspects(raw, question):
    """Read the decomposition reply; None when it cannot be used."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    obj = None
    # The reply itself, the first {...} span when prose surrounds it, or the
    # reply with the brackets it leaves open closed: Llama often ends a
    # multi-aspect reply one } short.
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1],
                      close_brackets(text)):
        try:
            obj = json.loads(candidate)
            break
        except ValueError:
            continue
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "simple":
        return [Aspect("main", question, 1.0)]
    items = obj.get("aspects")
    if obj.get("type") != "multi_aspect" or not isinstance(items, list) or not items:
        return None
    aspects = []
    for item in items:
        if not isinstance(item, dict):
            return None
        name = str(item.get("name") or "").strip()
        try:
            importance = float(item.get("importance"))
        except (TypeError, ValueError):
            return None
        if not name or not math.isfinite(importance):
            return None
        aspects.append(Aspect(name, name, importance))
    return aspects


def close_brackets(text):
    """The text from its first { with the brackets it leaves open closed; ""
    when there is nothing to close or it cannot be closed (it ends inside a
    string, or a bracket closes the wrong kind)."""
    start = text.find("{")
    if start < 0:
        return ""
    closers, in_string, escaped = [], False, False
    for ch in text[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            closers.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not closers or closers.pop() != ch:
                return ""
    if in_string or not closers:
        return ""
    return text[start:] + "".join(reversed(closers))


def choose_aspects(aspects, question):
    """Drop unimportant aspects (when there are several) and keep the three most important."""
    if len(aspects) > 1:
        aspects = [a for a in aspects
                   if a.importance >= settings.MIN_ASPECT_IMPORTANCE]
    if not aspects:
        return [Aspect("main", question, 1.0)]
    return sorted(aspects, key=lambda a: -a.importance)[: settings.MAX_ASPECTS]


def add_views(aspects, ask):
    """Two retrieval views per aspect: a paraphrase and a HyDE passage."""
    for a in aspects:
        paraphrase = ask(settings.PARAPHRASE_PROMPT, f"Query: {a.text}",
                         settings.PARAPHRASE_MAX_TOKENS).strip()
        hyde = ask(settings.HYDE_PROMPT, f"Question: {a.text}",
                   settings.HYDE_MAX_TOKENS).strip()
        a.views = [paraphrase or a.text, hyde or a.text]


# =========================================================================
# Step 2. Aspect scores
# =========================================================================


def preliminary_confidence(index, aspect, paraphrase_vector, second_vector=None):
    """Top cross-encoder score over the paraphrase view's first ten hits."""
    hits = collapsed_candidates(index, paraphrase_vector,
                                bm25_scores(index, aspect.views[0], aspect.focus), aspect.focus,
                                second_vector)
    top = [leaf for leaf, _ in hits[: settings.PRELIMINARY_RERANK_TOP_K]]
    return max((score for _, score in rerank(index, aspect.text, top)), default=0.0)


def score_aspects(aspects):
    """aspect score = 0.5 * importance + 0.5 * retrieval confidence."""
    for a in aspects:
        a.score = (settings.ASPECT_IMPORTANCE_WEIGHT * a.importance
                   + settings.RETRIEVAL_CONFIDENCE_WEIGHT * a.confidence)


# =========================================================================
# Step 3. Budgets
# =========================================================================


def allocate_budgets(aspects):
    """Share 13 slots among the aspects (the global view keeps its 2)."""
    total = settings.FINAL_CONTEXT_CHUNKS - settings.GLOBAL_VIEW_QUOTA
    # One aspect takes every slot; the 2-8 bounds apply between aspects.
    if len(aspects) == 1:
        aspects[0].budget = total
        return
    low, high, n = settings.MIN_CHUNKS_PER_ASPECT, settings.MAX_CHUNKS_PER_ASPECT, len(aspects)
    if not n * low <= total <= n * high:
        raise ValueError(f"{n} aspects cannot share {total} slots within [{low}, {high}]")
    weights = [max(a.score, 0.0) for a in aspects]
    raw = [total * w / sum(weights) if sum(weights) > 0 else total / n
           for w in weights]
    budgets = [min(high, max(low, math.floor(r))) for r in raw]
    # Slots left by rounding go to the largest remainders first ...
    order = sorted(range(n), key=lambda i: (-(raw[i] - math.floor(raw[i])),
                                            -aspects[i].score, i))
    while sum(budgets) < total:
        for i in order:
            if sum(budgets) < total and budgets[i] < high:
                budgets[i] += 1
    # ... and slots over the total come back from the weakest aspects.
    weakest = sorted(range(n), key=lambda i: (aspects[i].score, i))
    while sum(budgets) > total:
        for i in weakest:
            if sum(budgets) > total and budgets[i] > low:
                budgets[i] -= 1
    for a, b in zip(aspects, budgets):
        a.budget = b


# =========================================================================
# Step 4. Candidates for one view
# =========================================================================


def view_candidates(index, view_text, view_vector, aspect_text, focus=None, second_vector=None):
    """Collapsed and multi-branch rankings fused by RRF, then the structural step."""
    view_bm25 = bm25_scores(index, view_text, focus)
    collapsed = collapsed_candidates(index, view_vector, view_bm25, focus, second_vector)
    branches = multi_branch_candidates(index, view_vector, view_bm25, focus)
    merged = rrf([[leaf for leaf, _ in collapsed], [leaf for leaf, _ in branches]])
    return apply_structure(index, merged, aspect_text)


# --- 4a. Collapsed retrieval and summary expansion ------------------------


def collapsed_candidates(index, view_vector, view_bm25, focus=None, second_vector=None):
    """Dense search over every node and BM25 over the leaves, fused; summaries expand to leaves."""
    depth = settings.FLAT_TREE_TOP_K_PER_VIEW
    dense = _dense_ranking(index.node_vectors, view_vector, depth, focus)
    # BM25 covers leaves only, and leaf i is search position i.
    sparse = [i for i in _best(view_bm25, depth) if view_bm25[i] > 0]
    rankings = [dense, sparse]
    if second_vector is not None:
        rankings.append(_dense_ranking(index.second_vectors, second_vector, depth, focus))
    best: dict = {}
    for position, score in rrf(rankings)[:depth]:
        node = index.node_ids[position]
        leaves = ([position] if index.layer[node] == 0
                  else expand_summary(index, node, view_vector, focus=focus))
        for leaf in leaves:
            best[leaf] = max(best.get(leaf, 0.0), score)
    return _ranked(best)


def _dense_ranking(vectors, view_vector, depth, focus):
    """The nodes nearest a view, best first; with a focus, only those holding a leaf of it."""
    scores = vectors @ view_vector
    if focus is None:
        return _best(scores, depth)
    allowed = np.flatnonzero(focus.nodes)
    return allowed[_best(scores[allowed], depth)].tolist()


def expand_summary(index, node, view_vector, depth=0, focus=None):
    """Turn a summary hit into leaves, by the design's rule for its layer."""
    # Layer 1 is the lowest summary tier, 2 the middle, 3 and up the broad tiers.
    leaves = index.leaves_under[node]
    layer = index.layer[node]
    if focus is not None:
        leaves = [leaf for leaf in leaves if leaf in focus.leaves]
    direct = layer == 1 or (
        layer == 2 and len(leaves) <= settings.MAX_DESCENDANT_CHUNKS_FOR_DIRECT_EXPANSION)
    if (direct or depth >= settings.MAX_EXPANSION_RECURSION_DEPTH
            or not index.children[node]):
        return _best_leaves(index, leaves, view_vector,
                            settings.SUMMARY_EXPANSION_TOP_K_CHUNKS)
    # A broad summary routes to its best children and expands each of them.
    out = []
    best_children = sorted(_inside(index, index.children[node], focus),
                           key=lambda c: (-_cosine(index, c, view_vector), c))
    for child in best_children[: settings.MAX_CHILDREN_TO_FOLLOW_FROM_BROAD_SUMMARY]:
        for leaf in expand_summary(index, child, view_vector, depth + 1, focus):
            if leaf not in out:
                out.append(leaf)
    return out


# --- 4b. Multi-branch traversal -------------------------------------------


def multi_branch_candidates(index, view_vector, view_bm25, focus=None):
    """Walk down from the top keeping several branches alive; collect leaves at the ends."""
    if not index.top_nodes:
        return []
    frontier = _hybrid_nodes(index, _inside(index, index.top_nodes, focus), view_vector, view_bm25)
    frontier = frontier[: settings.MULTI_BRANCH_TOP_K_DEPTH_1]
    ends, depth = [], 1
    while frontier:
        below = []
        for node in frontier:
            kids = index.children[node]
            if (index.layer[node] <= 1 or depth >= settings.MULTI_BRANCH_MAX_DEPTH
                    or not kids):
                ends.append(node)
            else:
                below += _hybrid_nodes(index, _inside(index, kids, focus), view_vector, view_bm25)[
                    : settings.MULTI_BRANCH_TOP_K_PER_LEVEL]
        frontier = list(dict.fromkeys(below))
        depth += 1
    best: dict = {}
    for node in dict.fromkeys(ends):
        leaves = index.leaves_under[node]
        if focus is not None:
            leaves = [leaf for leaf in leaves if leaf in focus.leaves]
        order = hybrid_order(
            leaves,
            dense={i: float(index.leaf_vectors[i] @ view_vector) for i in leaves},
            sparse={i: float(view_bm25[i]) for i in leaves})
        for leaf in order[: settings.MULTI_BRANCH_LEAVES_PER_PATH]:
            best[leaf] = float(index.leaf_vectors[leaf] @ view_vector)
    return _ranked(best)


def _hybrid_nodes(index, nodes, view_vector, view_bm25):
    """Order sibling nodes by dense similarity and BM25 together; a node's BM25 is its best leaf's."""
    return hybrid_order(
        nodes,
        dense={n: _cosine(index, n, view_vector) for n in nodes},
        sparse={n: float(np.max(view_bm25[index.leaves_under[n]])) for n in nodes})


# --- BM25 and RRF ---------------------------------------------------------


def tokenize(text):
    """Lower-case \\w+ tokens, the same split M3's BM25 uses."""
    return re.findall(r"\w+", text.lower())


def bm25_scores(index, text, focus=None):
    """BM25 score of every leaf for a text; with a focus, nothing outside it scores."""
    scores = np.asarray(index.bm25.get_scores(tokenize(text)), dtype=float)
    # Leaves come first in search order, so the focus's first entries are the leaves.
    return scores if focus is None else scores * focus.nodes[: len(scores)]


def rrf(rankings):
    """Reciprocal rank fusion: each ranking adds 1 / (k + rank), rank from 1."""
    # RRF, Cormack 2009, k=60
    scores: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (settings.RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def hybrid_order(keys, dense, sparse):
    """Keys ordered by RRF of their dense ranking and their BM25 ranking (zero BM25 ranks nowhere)."""
    by_dense = sorted(keys, key=lambda x: (-dense[x], x))
    by_sparse = [x for x in sorted(keys, key=lambda x: (-sparse[x], x)) if sparse[x] > 0]
    return [x for x, _ in rrf([by_dense, by_sparse])]


# --- 4c. Structure: section cap, title bias, neighbours -------------------


def apply_structure(index, scored, aspect_text):
    """At most 3 leaves per section, a bias for sections titled like the aspect, then neighbours."""
    kept, per_section = [], {}
    for leaf, score in scored:
        key = index.section[leaf]
        if per_section.get(key, 0) < settings.SECTION_DIVERSITY_CAP:
            per_section[key] = per_section.get(key, 0) + 1
            kept.append((leaf, score))
    wanted = content_words(aspect_text)
    kept = [(leaf, score * settings.ASPECT_SECTION_BIAS_FACTOR
             if wanted & content_words(index.section_title[index.section[leaf]])
             else score)
            for leaf, score in kept]
    # Neighbours enter just below the leaf that brought them.
    present = {leaf for leaf, _ in kept}
    out = list(kept)
    for leaf, score in kept:
        for step in (index.prev_leaf, index.next_leaf):
            near, weight = leaf, score
            for _ in range(settings.NEIGHBOR_RADIUS):
                near = step[near]
                if near is None:
                    break
                weight *= settings.NEIGHBOR_SCORE_DISCOUNT
                if near not in present:
                    present.add(near)
                    out.append((near, weight))
    return _ranked(dict(out))


def content_words(text):
    """The words of a text that are not English stopwords."""
    return {w for w in tokenize(text) if w not in _stopwords()}


@functools.lru_cache(maxsize=1)
def _stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return ENGLISH_STOP_WORDS


# --- A question's words steering the search (development v16) -----------


@dataclass(frozen=True)
class Focus:
    """The leaves a search may return, and which searchable nodes hold one."""

    leaves: frozenset
    nodes: np.ndarray


def focus_on(index, leaves):
    """A search limited to these leaves."""
    leaves = frozenset(leaves)
    nodes = np.fromiter((not leaves.isdisjoint(index.leaves_under[n]) for n in index.node_ids),
                        dtype=bool, count=len(index.node_ids))
    return Focus(leaves, nodes)


def _inside(index, nodes, focus):
    """The nodes holding a leaf of the focus; all of them without one."""
    if focus is None:
        return nodes
    return [n for n in nodes if not focus.leaves.isdisjoint(index.leaves_under[n])]


def named_sources(index, text):
    """The unit's sources a text names as the corpus writes them; a name inside a longer one does not count."""
    found, rest = [], text
    for name in index.source_names:
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        if re.search(pattern, rest):
            found.append(name)
            rest = re.sub(pattern, " ", rest)
    return found


def named_dates(text):
    """The full dates a text names, in the form published_date gives."""
    return list(dict.fromkeys(f"{month} {int(day)}, {year}"
                              for month, day, year in re.findall(settings.FULL_DATE, text)))


def source_focus(index, text):
    """A search limited to the sources a text names; None when it names none."""
    names = named_sources(index, text) if settings.SOURCE_FOCUS else []
    if not names:
        return None
    return focus_on(index, [leaf for leaf, source in enumerate(index.source) if source in names])


def search_targets(index, question):
    """What the evidence must hold some of: each article the question dates, then each source it names."""
    if not settings.SOURCE_FOCUS or not index.source_names:
        return []
    names = named_sources(index, question)
    targets = []
    for date in named_dates(question):
        dated = {leaf for leaf, day in enumerate(index.date)
                 if day == date and (not names or index.source[leaf] in names)}
        if dated:
            targets.append(dated)
    for name in names:
        targets.append({leaf for leaf, source in enumerate(index.source) if source == name})
    return targets


def _make_room(groups, keep):
    """The group a new chunk joins: a free slot first, else the place of the weakest chunk no target needs."""
    aspects = sorted(groups[:-1], key=len, reverse=True)
    if sum(len(group) for group in groups) < settings.FINAL_CONTEXT_CHUNKS:
        return aspects[0]
    for group in aspects + [groups[-1]]:
        for at in range(len(group) - 1, -1, -1):
            if group[at][0] not in keep:
                del group[at]
                return group
    return None


def cover_targets(groups, targets, search):
    """Give each target MIN_CHUNKS_PER_TARGET chunks; returns how many had to be placed."""
    least, placed = settings.MIN_CHUNKS_PER_TARGET, 0
    held = lambda: {leaf for group in groups for leaf, _ in group}  # noqa: E731
    in_order = [leaf for group in groups for leaf, _ in group]
    # The chunks already covering a target stay.
    keep = {leaf for target in targets for leaf in [x for x in in_order if x in target][:least]}
    for target in targets:
        need = least - len(held() & set(target))
        if need <= 0:
            continue
        for leaf, score in search(target):
            if need == 0:
                break
            if leaf in held():
                continue
            group = _make_room(groups, keep)
            if group is None:
                return placed
            group.append((leaf, score))
            keep.add(leaf)
            need -= 1
            placed += 1
    return placed


# =========================================================================
# Step 5. The aspect's pool, reranked
# =========================================================================


def aspect_pool(index, view_rankings):
    """Union of the views' candidates (best score kept), at most 4 leaves per RAPTOR cluster."""
    best: dict = {}
    for ranking in view_rankings:
        for leaf, score in ranking:
            best[leaf] = max(best.get(leaf, 0.0), score)
    pool, per_cluster = [], {}
    for leaf, score in _ranked(best):
        cluster = index.cluster[leaf]
        if per_cluster.get(cluster, 0) < settings.MAX_CHUNKS_PER_RAPTOR_CLUSTER:
            per_cluster[cluster] = per_cluster.get(cluster, 0) + 1
            pool.append((leaf, score))
    return pool


def rerank(index, query, leaves):
    """Leaves ordered by cross-encoder relevance to the query."""
    if not leaves:
        return []
    scores = cross_encoder_scores(query, [headed_text(index, i) for i in leaves])
    return sorted(zip(leaves, scores), key=lambda x: (-x[1], x[0]))


@functools.lru_cache(maxsize=1)
def _cross_encoder():
    """Load the pinned reranker once: fp16 on the GPU, fp32 on a CPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        settings.RERANKER, revision=settings.RERANKER_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.RERANKER, revision=settings.RERANKER_REVISION, torch_dtype=dtype)
    model.to(device).eval()
    return tokenizer, model, device


def cross_encoder_scores(query, passages):
    """Relevance of each passage in [0, 1]: the reranker's logit through a sigmoid."""
    import torch

    tokenizer, model, device = _cross_encoder()
    scores = []
    for start in range(0, len(passages), settings.RERANK_BATCH):
        batch = passages[start: start + settings.RERANK_BATCH]
        inputs = tokenizer([query] * len(batch), batch, padding=True,
                           truncation=True, max_length=settings.RERANK_MAX_LENGTH,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float()
        scores += torch.sigmoid(logits).tolist()
    return scores


# =========================================================================
# Steps 6-7. The global view, then merge and backfill
# =========================================================================

# Step 6 runs the same view_candidates / aspect_pool / rerank on the question
# itself (see OursSystem.prepare); its quota is GLOBAL_VIEW_QUOTA.


def merge_and_backfill(rankings, quotas):
    """Fill each holder's quota from its ranking, skipping leaves an earlier holder took."""
    taken, out = set(), []
    for ranking, quota in zip(rankings, quotas):
        mine = []
        for leaf, score in ranking:
            if len(mine) >= quota:
                break
            if leaf not in taken:
                taken.add(leaf)
                mine.append((leaf, score))
        out.append(mine)
    return out


# =========================================================================
# Step 8. Orientation
# =========================================================================


def orientation(index, groups, chunk_tokens):
    """Per aspect: section titles, then parent summaries, within 15% of the context."""
    allowance = (chunk_tokens * settings.SUMMARY_CONTEXT_TOKEN_RATIO
                 / settings.CHUNK_CONTEXT_TOKEN_RATIO)
    used, seen, out = 0, set(), []
    for leaves in groups:
        wanted = []
        for leaf in leaves:
            key = index.section[leaf]
            if key not in seen:
                seen.add(key)
                wanted.append(("title", key, index.section_title[key]))
        n_summaries = 0
        for leaf in leaves:
            ancestors = _ancestors(index, leaf)
            for node in ancestors[: settings.MAX_ANCESTOR_SUMMARIES_PER_CHUNK_GROUP]:
                if node in seen or n_summaries >= settings.MAX_PARENT_SUMMARIES_PER_ASPECT:
                    continue
                seen.add(node)
                n_summaries += 1
                wanted.append(("summary", node, truncate_tokens(
                    index.text[node], settings.MAX_PARENT_SUMMARY_TOKENS)))
        lines = []
        for kind, key, text in wanted:
            cost = count_tokens(text)
            if used + cost <= allowance:
                used += cost
                lines.append((kind, key, text))
        out.append(lines)
    return out


def _ancestors(index, leaf):
    """The leaf's nearest parent, that parent's parent, and so on; never the root."""
    chain, node = [], index.leaf_node[leaf]
    while index.parents.get(node):
        node = index.parents[node][0]
        if node != index.root and index.text[node].strip():
            chain.append(node)
    return chain


# =========================================================================
# Step 9. The prompt
# =========================================================================


def build_context(aspects, aspect_chunks, aspect_orientation, global_chunks):
    """The evidence block in the design's layout: each aspect, then the global view."""
    lines, number = [], 0
    for a, chunks, orient in zip(aspects, aspect_chunks, aspect_orientation):
        header = f"[Aspect: {a.name}]"
        if settings.PASS_RETRIEVAL_CONFIDENCE_TO_LLM:
            header += f" (retrieval_confidence: {a.confidence:.2f})"
        if (settings.MARK_LOW_CONFIDENCE and a.confidence
                < settings.ABSTENTION_RETRIEVAL_CONFIDENCE_THRESHOLD):
            header += " [LOW CONFIDENCE]"
        lines.append(header)
        lines += [f"  [Orientation: {text}]" for _, _, text in orient]
        lines.append("  [Primary Evidence]")
        for text in chunks:
            number += 1
            lines.append(f"  Chunk {number}: {text}")
    if global_chunks:
        lines += ["[Global View]", "  [Primary Evidence]"]
        for text in global_chunks:
            number += 1
            lines.append(f"  Chunk {number}: {text}")
    return "\n".join(lines)


def doc_ranking(chunks):
    """Documents in order of first appearance; a summary has no document and ranks nothing."""
    out = []
    for chunk in chunks:
        for atom in chunk.gold_provenance or ():
            parent = str(atom[0])
            if parent not in out:
                out.append(parent)
    return out


# =========================================================================
# Step 10. The answer
# =========================================================================

# The reader answers the step-9 prompt in OursSystem.answer, a refusal is
# asked once more there (v8), and this cuts the answer down (v7).


ANSWER_LINE = r"(?im)^\W*(?:final\s+)?answer\s*:"


def final_answer(reply):
    """What the reader's last "Answer:" line says; the whole reply when it has none."""
    marks = list(re.finditer(ANSWER_LINE, reply))
    tail = reply[marks[-1].end():].strip().strip("*").strip() if marks else ""
    return tail.splitlines()[0].strip() if tail else reply


def unread_refusal(reasoning, final):
    """True when the reader has nothing but did not say it in the refusal's words: its
    last reply has no "Answer:" line and ends by saying it has nothing, or the final
    answer is a few words that say so. Never for an answer that already refuses."""
    final = final or ""
    if is_abstention(final):
        return False
    reply = (reasoning or "").strip()
    last = (reply.splitlines() or [""])[-1]
    says_nothing = re.compile(settings.NOTHING_SAID, re.IGNORECASE)
    ends_empty = re.search(ANSWER_LINE, reply) is None and says_nothing.search(last) is not None
    short = len(final.split()) <= settings.SHORT_NOTHING_WORDS and says_nothing.search(final) is not None
    return ends_empty or short


def is_notes(final):
    """True when the final answer spans more than one line: the reader's notes, gone on
    whole with no "Answer:" line the cut could shorten. Never for a refusal."""
    final = (final or "").strip()
    return "\n" in final and not is_abstention(final)


def asks_for_a_number(question):
    """True when the question's own words ask for a number or a date."""
    return re.search(settings.NUMBER_QUESTION, question, re.IGNORECASE) is not None


def asks_for_a_reason(question):
    """True for a why question, or a how question that does not ask for a number."""
    first = (question.split() or [""])[0].lower()
    return first == "why" or (first == "how" and not asks_for_a_number(question))


def is_a_number(cut):
    """True when a cut is nothing but a number or a date."""
    words = normalize_qasper_answer(cut).split()
    return bool(words) and all(w in settings.NUMBER_WORDS or any(c.isdigit() for c in w)
                               for w in words)


def _named_in(phrase, text):
    """Where the text names the phrase as whole words, or None; stopwords alone never count."""
    if all(word.lower() in settings.OPTION_STOPWORDS for word in phrase.split()):
        return None
    found = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE)
    return found.start() if found else None


def chosen_option(question, answer):
    """The side of an "A or B" question the answer names first, in the question's words, or None."""
    body = question.strip().rstrip("?").strip()
    if " or " not in body:
        return None
    left, right = body.rsplit(" or ", 1)
    left = re.split(r"[,:]\s+", left)[-1].split()
    right = right.split()
    # The longest end of the left side and the longest start of the right that the answer names.
    sides = ([" ".join(left[i:]) for i in range(len(left))],
             [" ".join(right[:i]) for i in range(len(right), 0, -1)])
    named = []
    for phrases in sides:
        for phrase in phrases:
            at = _named_in(phrase, answer)
            if at is not None:
                named.append((at, phrase))
                break
    return min(named)[1] if named else None


def shorten_answer(question, answer, ask):
    """The answer cut to what the question asks for; the answer itself when the
    cut is empty, refuses, is not shorter, or brings in a word the answer lacks (yes
    and no excepted). A number or date the question did not ask for gives way to the
    side of an "A or B" question the answer names first, or else to the answer."""
    reply = ask(settings.SHORT_ANSWER_PROMPT, f"Question: {question}\nAnswer: {answer}",
                settings.SHORT_ANSWER_MAX_TOKENS).strip()
    first_line = reply.splitlines()[0] if reply else ""
    short = re.sub(r"^(short answer|answer)\s*:\s*", "", first_line,
                   flags=re.IGNORECASE).strip().strip("\"'").strip()
    words = normalize_qasper_answer(short).split()
    have = normalize_qasper_answer(answer).split()
    if not words or len(words) >= len(have) or is_abstention(short):
        return answer
    if words in (["yes"], ["no"]):
        return short
    if not set(words) <= set(have):
        return answer
    if is_a_number(short) and not asks_for_a_number(question):
        return chosen_option(question, answer) or answer
    return short


# =========================================================================
# Small helpers
# =========================================================================


def _best(scores, k):
    """Indices of the k highest scores; ties keep the lower index first."""
    return np.argsort(-np.asarray(scores), kind="stable")[:k].tolist()


def _best_leaves(index, leaves, view_vector, k):
    """The k leaves (of those given) most similar to the view."""
    sims = index.leaf_vectors[leaves] @ view_vector
    return [leaves[i] for i in _best(sims, k)]


def _cosine(index, node, view_vector):
    """Similarity of a node to the view; a node without a vector ranks last."""
    vector = index.vector_of.get(node)
    return float(vector @ view_vector) if vector is not None else float("-inf")


def _ranked(scores):
    """(key, score) pairs, best first; ties by key."""
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@functools.lru_cache(maxsize=1)
def _encoding():
    import tiktoken
    return tiktoken.encoding_for_model(settings.TOKEN_COUNTER_MODEL)


def count_tokens(text):
    """Tokens as the harness counts evidence."""
    return len(_encoding().encode(text))


def truncate_tokens(text, n):
    """The text cut to its first n tokens."""
    ids = _encoding().encode(text)
    return text if len(ids) <= n else _encoding().decode(ids[:n]).strip()


def _unit_type(layer):
    """M4's label for a node's tier."""
    return {0: "chunk", 1: "summary_low", 2: "summary_mid"}.get(layer, "summary_high")


def code_fingerprint():
    """Short sha256 of this file and config.py, line endings normalised."""
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(settings.__file__)):
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


CODE_FINGERPRINT = code_fingerprint()


def check_model_revisions():
    """Refuse unless the hub still serves the pinned reader and embedders."""
    # The harness loads both by name, which means the hub's current revision.
    from huggingface_hub import HfApi

    api = HfApi()
    pins = [(settings.READER, settings.READER_REVISION), (settings.EMBEDDER, settings.EMBEDDER_REVISION)]
    if settings.SECOND_EMBEDDER_MODE != "off":
        pins.append((settings.SECOND_EMBEDDER, settings.SECOND_EMBEDDER_REVISION))
    for model_id, pinned in pins:
        served = api.model_info(model_id).sha
        if served != pinned:
            raise SystemExit(
                f"OURS REFUSED: the hub serves {model_id} at {served}, not the "
                f"pinned {pinned}. The harness would load the new revision; "
                "nothing was indexed.")


# =========================================================================
# The documents M4's filter leaves out
# =========================================================================


class WholeCorpusTree(RaptorSystem):
    """M4's builder with M4's short-document filter lifted, its trees in ours' own cache."""

    system_id = "OURS_TREE"

    def __init__(self, config) -> None:
        # The chunker is in M4's cache key, so lifting the filter moves the key too.
        chunker = replace(config.m4.chunker, min_chars_per_doc=0)
        super().__init__(replace(config, m4=replace(config.m4, chunker=chunker)))

    def _cache_dir(self, corpus_hash):
        """M4's key under ours' own namespace."""
        return CacheDir(paths.cache_dir(), settings.TREE_NAMESPACE,
                        super()._cache_dir(corpus_hash).cache_key)


def short_documents(items, m4):
    """How many of a unit's documents M4's filter leaves out, laid out and judged as M4 does."""
    with tempfile.TemporaryDirectory(prefix="OURS_short_") as td:
        m4._write_corpus_layout(items, Path(td))
        return sum(len(doc.text) < m4.config.m4.chunker.min_chars_per_doc
                   for doc in walk_corpus(Path(td)))


# =========================================================================
# The system the runner calls
# =========================================================================


class OursSystem(BaseSystem):
    """Ours as the runner sees it: index a unit through M4, then prepare each question."""

    system_id = "OURS"

    def __init__(self, config) -> None:
        super().__init__(config)
        problems = []
        if config.generation.model != settings.READER:
            problems.append(f"reader {config.generation.model!r}")
        if config.m4.summary_model != settings.READER:
            problems.append(f"tree summariser {config.m4.summary_model!r}")
        if config.m4.embedder != settings.EMBEDDER:
            problems.append(f"embedder {config.m4.embedder!r}")
        if config.generation.temperature != 0:
            problems.append(f"temperature {config.generation.temperature}")
        if problems:
            raise SystemExit(
                "OURS REFUSED: " + ", ".join(problems) + f". Ours runs with one "
                f"model, {settings.READER}, greedy, on M4's tree: pass "
                f"--generator {settings.READER} and keep M4's defaults.")
        self._m4 = RaptorSystem(config)
        self._whole = WholeCorpusTree(config)
        # The one holding the current unit's tree.
        self._tree_system = self._m4
        self._n_short_documents = 0
        self._tree_index: TreeIndex | None = None
        self._revisions_checked = False

    def index(self, corpus_path):
        """Not used: the runner indexes through index_items."""
        raise NotImplementedError(
            "OURS indexes through index_items, which hands the corpus to M4's own code.")

    def index_items(self, items) -> None:
        """Load or build this unit's tree through M4's own code, then index it for the query steps."""
        items = list(items)
        if settings.TREE_BUILD_ENV is None:
            raise SystemExit(
                "OURS REFUSED: ours/config.py TREE_BUILD_ENV is unset. Run "
                "`python -m scripts.dev_ours census` and pin the value it prints.")
        if not self._revisions_checked:
            check_model_revisions()
            self._revisions_checked = True
        self._tree_system, self._n_short_documents = self.tree_for(items)
        # (a) M4's own code: lays out the corpus, computes its cache key, and
        # loads the tree or builds it.
        self._tree_system.index_items(items)
        self.chunks = self._tree_system.chunks
        self.build_index(items)
        self._indexed = True

    def build_index(self, items) -> None:
        """What the query steps read, from the tree M4 loaded or built (kept on _tree)."""
        self._tree_index = build_tree_index(
            self.chunks, self._tree_system.chunk_embeddings, self._tree_system._tree,
            documents_of(items, self.chunks), document_headers(items, self.chunks),
            document_fields(items, self.chunks))
        add_second_vectors(self._tree_index)

    def tree_for(self, items):
        """The system holding a unit's tree, M4 unless its filter would leave a document out, and that count."""
        n_short = short_documents(items, self._m4)
        tree = self._whole if n_short else self._m4
        # The banked Llama trees were built under this topology stack, and
        # M4's cache key folds it in.
        tree.topology_env_override = settings.TREE_BUILD_ENV
        return tree, n_short

    @property
    def tree_cache_hit(self):
        """True when this unit's tree came from the cache."""
        return self._tree_system.tree_cache_hit

    @property
    def resolved_components(self):
        """M4's components plus the reranker, for the run summary."""
        resolved = self._tree_system.resolved_components
        return None if resolved is None else replace(resolved, reranker_id=settings.RERANKER)

    def retrieve(self, query, k=None):
        """The evidence chunks for a question, in prompt order."""
        retrieved = self.prepare(query).retrieved
        return retrieved[:k] if k else retrieved

    def answer(self, query, k=None):
        """Steps 1-9, the reader's reasoned answer to their prompt, then step 10."""
        prepared = self.prepare(query, k=k)
        # The baselines' own prompt on the same evidence: for the report, never
        # the answer, and not one of the system's calls.
        plain = models.generate(system_prompt=settings.ANSWER_PROMPT,
                                user_prompt=prepared.user_prompt,
                                cfg=self.config.generation)
        # The reader's call, as BaseSystem.answer makes it.
        reasoning = models.generate(system_prompt=prepared.system_prompt,
                                    user_prompt=prepared.user_prompt,
                                    cfg=self.config.generation)
        first = final_answer(reasoning)
        reply, more_calls = first, 0
        reasked = settings.REASK_REFUSALS and is_abstention(first)
        if reasked:
            reasoning = models.generate(system_prompt=settings.READER_REASK_PROMPT,
                                        user_prompt=prepared.user_prompt,
                                        cfg=self.config.generation)
            reply = final_answer(reasoning)
            more_calls += 1
        final = reply
        if settings.SHORT_ANSWER and not is_abstention(reply) and not asks_for_a_reason(query):
            final = shorten_answer(query, reply, self._ask)
            more_calls += 1
        if settings.READ_UNREAD_REFUSALS and unread_refusal(reasoning, final):
            final = settings.REFUSAL
        if settings.NOTES_ARE_REFUSALS and is_notes(final):
            final = settings.REFUSAL
        result = self.finish(prepared, final)
        result.extra["ours_n_llm_calls"] += more_calls
        result.extra["ours_first_answer"] = first
        result.extra["ours_reasked"] = reasked
        result.extra["ours_reader_answer"] = reply
        result.extra["ours_reasoning"] = reasoning
        result.extra["ours_plain_answer"] = plain
        return result

    def _ask(self, system_prompt, user_prompt, max_tokens):
        """(b) The shared reader, greedy, with this step's output cap."""
        return models.generate(
            system_prompt=system_prompt, user_prompt=user_prompt,
            cfg=replace(self.config.generation, max_new_tokens=max_tokens))

    def prepare(self, query, k=None) -> PreparedQuery:
        """Steps 1-9 for one question: everything up to the reader's answer."""
        self._require_indexed()
        index = self._tree_index
        t0 = self._now()
        calls = []

        def ask(system_prompt, user_prompt, max_tokens):
            calls.append(max_tokens)
            return self._ask(system_prompt, user_prompt, max_tokens)

        # Step 1: aspects and their views; every text embedded in one call.
        aspects, parse_failed = decompose(query, ask)
        add_views(aspects, ask)
        texts = [view for a in aspects for view in a.views] + [query]
        vectors, seconds = embed_query(texts)
        view_vectors = [vectors[2 * i: 2 * i + 2] for i in range(len(aspects))]
        question_vector = vectors[-1]
        view_seconds = [seconds[2 * i: 2 * i + 2] if seconds is not None else (None, None)
                        for i in range(len(aspects))]
        question_second = seconds[-1] if seconds is not None else None

        # Each aspect searches the sources it names, or else the question's.
        question_focus = source_focus(index, query)
        for a in aspects:
            a.focus = source_focus(index, a.text) or question_focus

        # Step 2: aspect scores.
        for a, (paraphrase_vector, _), (paraphrase_second, _) in zip(aspects, view_vectors, view_seconds):
            a.confidence = preliminary_confidence(index, a, paraphrase_vector, paraphrase_second)
        score_aspects(aspects)

        # Step 3: budgets.
        allocate_budgets(aspects)

        # Steps 4-5: each aspect's views, pooled and reranked against the aspect.
        own_vectors, own_seconds = (embed_query([a.text for a in aspects])
                                    if settings.QUESTION_VIEW else (None, None))
        rankings = []
        for n, (a, vecs, secs) in enumerate(zip(aspects, view_vectors, view_seconds)):
            views = [view_candidates(index, text, vec, a.text, a.focus, second_vector=sec)
                     for text, vec, sec in zip(a.views, vecs, secs)]
            if settings.QUESTION_VIEW:
                views.append(view_candidates(index, a.text, own_vectors[n], a.text, a.focus,
                                             second_vector=None if own_seconds is None else own_seconds[n]))
            pool = aspect_pool(index, views)
            rankings.append(rerank(index, a.text, [leaf for leaf, _ in pool]))

        # Step 6: the untouched question as the protected global view.
        pool = aspect_pool(index, [view_candidates(index, query, question_vector, query,
                                                   second_vector=question_second)])
        rankings.append(rerank(index, query, [leaf for leaf, _ in pool]))

        # Step 7: merge and backfill.
        groups = merge_and_backfill(
            rankings, [a.budget for a in aspects] + [settings.GLOBAL_VIEW_QUOTA])

        # Some of every article the question dates and every source it names.
        def search_target(leaves):
            focus = focus_on(index, leaves)
            found = aspect_pool(index, [view_candidates(index, query, question_vector, query, focus,
                                                        second_vector=question_second)])
            return rerank(index, query, [leaf for leaf, _ in found])

        n_placed = cover_targets(groups, search_targets(index, query), search_target)
        aspect_groups, global_group = groups[:-1], groups[-1]

        # Step 8: orientation for each aspect.
        evidence = [pair for group in groups for pair in group]
        chunk_tokens = sum(count_tokens(index.chunks[leaf].text) for leaf, _ in evidence)
        orient = orientation(index, [[leaf for leaf, _ in g] for g in aspect_groups],
                             chunk_tokens)

        # Step 9: the prompt.
        def texts_of(group):
            return [headed_text(index, leaf) for leaf, _ in group]
        context = build_context(aspects, [texts_of(g) for g in aspect_groups],
                                orient, texts_of(global_group))
        user_prompt = f"Evidence:\n{context}\n\nQuestion: {query}"

        # Retrieval is the evidence chunks only; orientation is packed context.
        retrieved = [RetrievedChunk(chunk=index.chunks[leaf], score=float(score), rank=n)
                     for n, (leaf, score) in enumerate(evidence)]
        summaries = [(key, text) for lines in orient for kind, key, text in lines
                     if kind == "summary"]
        packed = retrieved + [
            RetrievedChunk(
                chunk=Chunk(chunk_id=node, doc_id="", text=text,
                            n_words=len(text.split()), position=index.layer[node]),
                score=0.0, rank=len(retrieved) + n,
                source_unit_type=_unit_type(index.layer[node]))
            for n, (node, text) in enumerate(summaries)]
        # The ranking the rank metrics read: evidence by its selection score.
        by_score = sorted(evidence, key=lambda pair: (-pair[1], pair[0]))
        scoring = [RetrievedChunk(chunk=index.chunks[leaf], score=float(score), rank=n)
                   for n, (leaf, score) in enumerate(by_score)]

        detectors: dict = {}
        for doc in dict.fromkeys(index.section[leaf][0] for leaf, _ in evidence):
            detectors[index.detector[doc]] = detectors.get(index.detector[doc], 0) + 1
        threshold = settings.ABSTENTION_RETRIEVAL_CONFIDENCE_THRESHOLD
        extra = {
            "ours_aspects": [
                {"name": a.name, "importance": a.importance,
                 "confidence": a.confidence, "score": a.score, "budget": a.budget,
                 "n_selected": len(group), "low_confidence": a.confidence < threshold}
                for a, group in zip(aspects, aspect_groups)],
            "ours_global_selected": len(global_group),
            "ours_parse_failed": parse_failed,
            # The reader's answer is one more call; OursSystem.answer adds
            # step 10's.
            "ours_n_llm_calls": len(calls) + 1,
            "ours_doc_ranking": doc_ranking([r.chunk for r in scoring]),
            "ours_n_orientation_summaries": len(summaries),
            "ours_orientation_tokens": sum(count_tokens(text) for lines in orient
                                           for _, _, text in lines),
            "ours_section_detectors": detectors,
            "ours_tree_cache_hit": self._tree_system.tree_cache_hit,
            "ours_tree_degenerate": bool(self._tree_system.index_stats.get("degenerate_no_tree")),
            "ours_n_short_documents": self._n_short_documents,
            "ours_named_sources": named_sources(index, query),
            "ours_named_dates": named_dates(query) if index.source_names else [],
            "ours_n_chunks_placed": n_placed,
            "ours_code": CODE_FINGERPRINT,
            "ours_reranker_revision": settings.RERANKER_REVISION,
        }
        return PreparedQuery(
            query=query,
            retrieved=retrieved,
            packed=packed,
            scoring_ranking=scoring,
            system_prompt=settings.READER_PROMPT,
            user_prompt=user_prompt,
            evidence_tokens=count_tokens(context),
            n_input_tokens=count_tokens(settings.READER_PROMPT + "\n" + user_prompt),
            n_retrieval_calls=(3 if settings.QUESTION_VIEW else 2) * len(aspects) + 1,
            prepare_s=self._now() - t0,
            extra=extra,
        )
