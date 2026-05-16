"""Axis 2 — Docling structural hierarchy (PIPELINE_DESIGN.md §3.3, §4.4).

The shared RAPTOR substrate stores chunks whose boundaries are fixed by
the chunking strategy. Re-chunking per section would fork the substrate
cache key and break the M4/M7 share, so section assignment here is a
**post-hoc tag** over the already-produced shared chunks: every chunk is
mapped to the Docling section it falls in, without moving any chunk
boundary and without mutating the shared chunks.jsonl.

Mapping is by normalized-text alignment. `parse_pdf` gives each section
a body `text`; we locate each section body inside the document text with
a monotonic cursor, producing per-section character spans, then assign
each chunk to the section covering its start. Non-PDF inputs degrade to
one fallback section per file (PIPELINE_DESIGN §3.3) — every chunk maps
to section 0 and neighbour links fall back to chunk order, exactly the
degenerate-but-exercised path the smoke corpus hits.

Query-time Axis-2 ops (§4.4):
  * section-diversity cap (flat, =3)
  * aspect-section bias (×1.15 on section-title term overlap)
  * neighbour expansion (radius 1, within the same section)

All ablation-A1 gating lives in the orchestrator: when
`use_docling_structural_axis` is False it simply does not call these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chunking import Chunk
from .parsing import ParsedDocument


_NORM_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _normalize(text: str) -> str:
    """Lowercase + collapse all whitespace. Used only for alignment, never
    stored or embedded (metadata is a control signal, not a search signal)."""
    return _NORM_RE.sub(" ", (text or "").lower()).strip()


def _terms(text: str) -> set[str]:
    """Alphabetic word set (Greek-aware via the \\W class on a unicode re)."""
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2}


# --- Per-chunk section reference -----------------------------------------


@dataclass
class SectionRef:
    """The Docling section a chunk was assigned to, plus neighbour links.

    `section_key` is unique per (doc, section) and is what the
    diversity cap and parent-summary section-header dedup group on.
    `prev_chunk_id` / `next_chunk_id` are the immediate neighbours
    *within the same section*, in document order (PIPELINE_DESIGN §3.3).
    """
    doc_id: str
    section_title: str
    section_depth: int
    section_path: tuple[str, ...]
    order_in_document: int
    page_start: int | None
    page_end: int | None
    section_key: str
    pos_in_section: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


# --- Index-time: post-hoc chunk -> section attachment ---------------------


def _section_spans(doc_text: str, sections: list[dict]) -> list[tuple[int, int]]:
    """Char spans of each section body inside the normalized doc text.

    Monotonic cursor: each section body is searched only forward of the
    previous match, so out-of-body text (headings, page markers, clean_text
    normalization drift) cannot cause a section to capture earlier text.
    A body that cannot be located inherits the previous section's end as
    its start (assignment stays monotonic and never crashes).
    """
    norm_doc = _normalize(doc_text)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, sec in enumerate(sections):
        body = _normalize(sec.get("text", ""))
        probe = body[:80]
        start = norm_doc.find(probe, cursor) if probe else -1
        if start < 0:
            # Unlocatable body: start where the previous section ended.
            start = spans[-1][1] if spans else cursor
        end = start + len(body) if body else start
        if i + 1 == len(sections):
            end = max(end, len(norm_doc))
        spans.append((start, end))
        cursor = max(cursor, start + 1)
    # Make spans non-overlapping & monotonic: each ends where the next starts.
    fixed: list[tuple[int, int]] = []
    for i, (s, e) in enumerate(spans):
        nxt = spans[i + 1][0] if i + 1 < len(spans) else len(norm_doc)
        fixed.append((s, max(s, min(e, nxt) if nxt > s else len(norm_doc))))
    return fixed


def attach_sections(
    chunks: list[Chunk],
    parsed_docs: list[ParsedDocument],
) -> dict[str, SectionRef]:
    """Map every chunk to its Docling section (PIPELINE_DESIGN §3.3).

    Pure and deterministic: same chunks + same parsed docs → same map.
    Does not mutate `chunks` or any cached artifact. Chunks whose doc has
    no section payload (should not happen — parsing always emits ≥1
    fallback section) are assigned a synthetic file-level section.
    """
    docs_by_id = {d.doc_id: d for d in parsed_docs}

    chunks_by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)

    refs: dict[str, SectionRef] = {}

    for doc_id, doc_chunks in chunks_by_doc.items():
        doc = docs_by_id.get(doc_id)
        sections: list[dict] = (
            list(doc.metadata.get("sections", [])) if doc is not None else []
        )
        if not sections:
            sections = [{
                "section_title": doc_id,
                "section_depth": 0,
                "section_path": [doc_id],
                "page_start": None,
                "page_end": None,
                "order_in_document": 0,
                "text": doc.text if doc is not None else "",
            }]

        doc_text = doc.text if doc is not None else ""
        spans = _section_spans(doc_text, sections)
        norm_doc = _normalize(doc_text)

        # Document-order chunks; semantic + word-window both yield them in
        # ascending `position`. A monotonic search cursor maps each chunk's
        # normalized prefix to a char offset, then to its covering section.
        ordered = sorted(doc_chunks, key=lambda c: c.position)
        cursor = 0
        # collect chunk ids per section to wire within-section neighbours
        per_section: dict[int, list[str]] = {}

        for c in ordered:
            probe = _normalize(c.text)[:60]
            off = norm_doc.find(probe, cursor) if probe else -1
            if off < 0:
                off = norm_doc.find(probe) if probe else -1
            if off < 0:
                off = cursor
            cursor = max(cursor, off + 1)

            sec_idx = 0
            for i, (s, e) in enumerate(spans):
                if s <= off < e:
                    sec_idx = i
                    break
                if off >= e:
                    sec_idx = i
            sec = sections[sec_idx]
            key = f"{doc_id}::sec{int(sec.get('order_in_document', sec_idx))}"
            seq = per_section.setdefault(sec_idx, [])
            refs[c.chunk_id] = SectionRef(
                doc_id=doc_id,
                section_title=str(sec.get("section_title", doc_id)),
                section_depth=int(sec.get("section_depth", 0) or 0),
                section_path=tuple(sec.get("section_path", [doc_id]) or [doc_id]),
                order_in_document=int(sec.get("order_in_document", sec_idx) or 0),
                page_start=sec.get("page_start"),
                page_end=sec.get("page_end"),
                section_key=key,
                pos_in_section=len(seq),
            )
            seq.append(c.chunk_id)

        # within-section prev/next links (document order)
        for ids in per_section.values():
            for j, cid in enumerate(ids):
                refs[cid].prev_chunk_id = ids[j - 1] if j > 0 else None
                refs[cid].next_chunk_id = (
                    ids[j + 1] if j + 1 < len(ids) else None
                )

    return refs


# --- Query-time Axis-2 ops (PIPELINE_DESIGN.md §4.4) ----------------------


def apply_section_diversity_cap(
    scored: list[tuple[str, float]],
    refs: dict[str, SectionRef],
    cap: int,
) -> list[tuple[str, float]]:
    """Flat cap (=3): keep at most `cap` chunks per Docling section, drop
    the lowest-scoring excess. Order of the surviving list is preserved
    (callers pass it score-descending). Chunks with no section ref are
    grouped under a per-doc-unknown key, never silently dropped."""
    if cap <= 0:
        return list(scored)
    kept_per_section: dict[str, int] = {}
    out: list[tuple[str, float]] = []
    for cid, sc in sorted(scored, key=lambda t: t[1], reverse=True):
        ref = refs.get(cid)
        key = ref.section_key if ref is not None else f"__nosec__::{cid}"
        n = kept_per_section.get(key, 0)
        if n >= cap:
            continue
        kept_per_section[key] = n + 1
        out.append((cid, sc))
    return out


def apply_aspect_section_bias(
    scored: list[tuple[str, float]],
    refs: dict[str, SectionRef],
    aspect_text: str,
    factor: float,
) -> list[tuple[str, float]]:
    """Multiply a chunk's score by `factor` when its section title shares
    a content term with the aspect text (PIPELINE_DESIGN §4.4 Axis-2).
    Degenerate single-section docs (smoke .txt) almost never match —
    that is the expected no-op, not a bug."""
    a_terms = _terms(aspect_text)
    if not a_terms or factor == 1.0:
        return list(scored)
    out: list[tuple[str, float]] = []
    for cid, sc in scored:
        ref = refs.get(cid)
        if ref is not None and _terms(ref.section_title) & a_terms:
            out.append((cid, sc * factor))
        else:
            out.append((cid, sc))
    return out


def expand_section_neighbors(
    scored: list[tuple[str, float]],
    refs: dict[str, SectionRef],
    radius: int,
) -> list[tuple[str, float]]:
    """Pull in each surviving chunk's immediate same-section neighbours up
    to `radius` (PIPELINE_DESIGN §4.4 Axis-2). Added neighbours inherit a
    slightly-discounted score so they rank just below their anchor and do
    not displace independently-retrieved evidence. Deterministic."""
    if radius <= 0:
        return list(scored)
    by_id = {cid: sc for cid, sc in scored}
    out: list[tuple[str, float]] = list(scored)
    seen = set(by_id)
    for cid, sc in scored:
        ref = refs.get(cid)
        if ref is None:
            continue
        # walk prev
        cur = ref.prev_chunk_id
        for step in range(radius):
            if cur is None or cur in seen:
                break
            out.append((cur, sc * (0.9 ** (step + 1))))
            seen.add(cur)
            cur = refs[cur].prev_chunk_id if cur in refs else None
        # walk next
        cur = ref.next_chunk_id
        for step in range(radius):
            if cur is None or cur in seen:
                break
            out.append((cur, sc * (0.9 ** (step + 1))))
            seen.add(cur)
            cur = refs[cur].next_chunk_id if cur in refs else None
    return out


__all__ = [
    "SectionRef",
    "attach_sections",
    "apply_section_diversity_cap",
    "apply_aspect_section_bias",
    "expand_section_neighbors",
]
