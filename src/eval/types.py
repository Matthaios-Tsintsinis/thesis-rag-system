"""Shared dataclasses for the benchmark eval layer.

Pass-1 lands `CorpusItem` because it is required by
`BaseSystem.index_items` (the structural retrieval-side change in C5b).
The rest of the eval types — EvalQuery, GoldAnswer, RetrievalScore,
AnswerScore, EvalReport — land in C5c with the benchmark loaders and
scorers that produce them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusItem:
    """One indexable unit fed to a system at benchmark eval time.

    For QASPER each item is one paragraph from a paper (item granularity
    matches the gold-passage atom). For MultiHop-RAG each item is one
    full article body.

    Fields:
      item_id   — globally unique stable id. Convention: f"{parent_id}::{span_id}".
                  Used by the harness to name temp files when the default
                  index_items fallback writes to disk before calling
                  index(corpus_path). MUST be filesystem-safe under the
                  sanitiser in `BaseSystem._safe_item_filename`.
      parent_id — paper_id (QASPER) | article_url (MultiHop). Carried
                  into every produced Chunk's gold_provenance via the
                  default index_items fallback.
      span_id   — "sec{N}.para{M}" (QASPER) | "<whole>" (MultiHop).
                  The within-parent gold-span identifier.
      text      — the indexable text. The chunker may split this further;
                  default index_items writes one .txt file per CorpusItem
                  before calling self.index.
      metadata  — arbitrary loader-provided extras (section_name,
                  source, published_at, category, ...). Not part of the
                  index pipeline; surfaces in eval logs only.
    """

    item_id: str
    parent_id: str
    span_id: str
    text: str
    metadata: dict = field(default_factory=dict)


__all__ = ["CorpusItem"]
