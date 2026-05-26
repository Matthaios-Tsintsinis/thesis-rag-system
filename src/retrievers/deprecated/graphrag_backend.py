"""GraphRAG (Microsoft) backend glue for M5.

M5 wraps Microsoft GraphRAG (graphrag 3.0.9) as a benchmark baseline — a
non-RAPTOR hierarchical paradigm (evaluation_plan.pdf section 3). This
module is the integration layer: settings generation, bge-m3
embedding-model injection, in-process indexing, parquet artifact
loading, and local-search context retrieval. The M5 BaseSystem wrapper
(src/retrievers/m5_graphrag.py, milestone C3) consumes this module.

Verified against graphrag 3.0.9 by live introspection on Colab:

  - graphrag_llm.embedding.register_embedding(embedding_type, initializer,
    scope='transient') registers a custom embedding model. The custom
    model subclasses graphrag_llm.embedding.LLMEmbedding (an ABC) — the
    abstract surface is __init__, embedding, embedding_async,
    metrics_store, tokenizer.
  - GraphRagConfig.embedding_models / .completion_models are
    dict[str, ModelConfig] keyed by model id; embed_text.embedding_model_id
    defaults to 'default_embedding_model'.
  - graphrag.api.build_index(config, method=...) is async.
  - graphrag.config.load_config.load_config(root_dir, cli_overrides=None).
  - The default vector store is 3072-dim (OpenAI text-embedding-3-large);
    bge-m3 is 1024-dim, so build_settings overrides vector_size to
    EMBEDDING_DIM.
  - The graphrag.language_model.* namespace does NOT exist in 3.0.9 — the
    model layer is the separate top-level graphrag_llm package.

graphrag / graphrag_llm / graphrag_vectors / pandas / yaml / tiktoken
imports are deliberately lazy (inside functions, and the LLMEmbedding
subclass is defined inside a factory) so this module imports on hosts
without graphrag installed — e.g. the Windows dev worktree, where
AST/import verification runs. Code paths that touch the graphrag API
cannot be exercised off-Colab; those are marked COLAB-VERIFIED-ONLY and
are smoke-verified on Colab T4 at milestone C4, not in local CI. The
build_context result shape and a few parquet column names were not
captured by introspection — those spots are flagged inline as C4 fix
points.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import EMBEDDING_DIM, M5Config


# --- Constants ------------------------------------------------------------

GRAPHRAG_INPUT_SUBDIR = "input"
GRAPHRAG_OUTPUT_SUBDIR = "output"
GRAPHRAG_CACHE_SUBDIR = "cache"
GRAPHRAG_LANCEDB_SUBDIR = f"{GRAPHRAG_OUTPUT_SUBDIR}/lancedb"

# GraphRAG default model ids — workflows (embed_text, extract_graph, ...)
# resolve these names by default, so reusing them avoids per-workflow
# *_model_id overrides in the generated settings.
EMBEDDING_MODEL_ID = "default_embedding_model"
COMPLETION_MODEL_ID = "default_completion_model"

# Registry key for the bge-m3 embedder, placed in the embedding
# ModelConfig's `type` field so GraphRAG resolves it to our model.
BGE_M3_EMBEDDING_TYPE = "harness_bge_m3"

# Fields embedded by GraphRAG's embed_text workflow — each gets its own
# lancedb table and requires an index_schema entry sized to bge-m3's
# 1024-dim vectors (else the table defaults to OpenAI's 3072 dim and the
# bge-m3 writes are rejected).
EMBED_TEXT_FIELDS = (
    "entity_description",
    "community_full_content",
    "text_unit_text",
)


# --- Paradigm-neutral retrieval unit --------------------------------------


@dataclass
class GraphRAGContextUnit:
    """One unit of GraphRAG local-search context, paradigm-neutral.

    GraphRAG retrieves text units and community reports, not RAPTOR-style
    leaf chunks. The M5 wrapper (C3) adapts these into the harness
    RetrievedChunk type; the unit-count divergence from chunk-based
    systems is documented per evaluation_plan.pdf section 7.
    """

    unit_id: str
    text: str
    kind: str  # "text_unit" (primary evidence) | "community_report" (orientation)
    score: float
    source: str = ""  # document / community id, best-effort


@dataclass
class GraphRAGArtifacts:
    """The parquet outputs of a GraphRAG index run (pandas DataFrames)."""

    entities: Any
    communities: Any
    relationships: Any
    community_reports: Any
    text_units: Any


# --- Settings generation (pure) -------------------------------------------


def build_settings(cfg: M5Config) -> dict[str, Any]:
    """GraphRAG 3.0.9 settings as a dict (serialised to settings.yaml).

    All paths are relative, so the dict is environment-independent and
    safe to fold whole into the M5 cache key. The schema is validated by
    GraphRagConfig at load_config time (COLAB-VERIFIED-ONLY); any field
    mismatch surfaces there at C4.

    The vector store is pinned to EMBEDDING_DIM (1024) because bge-m3 is
    1024-dimensional and GraphRAG's default store is 3072-dim — a
    mismatch would misconfigure LanceDB.
    """
    return {
        "completion_models": {
            COMPLETION_MODEL_ID: {
                # `type` selects the resolution strategy; "litellm" is the
                # ModelConfig schema default and routes OpenAI calls.
                "type": "litellm",
                # model_provider is a required ModelConfig field but is
                # label-only — it forms the model_id label
                # f"{model_provider}/{model}", it does not drive resolution.
                "model_provider": "openai",
                "model": cfg.index_llm_model,
                "api_key": "${OPENAI_API_KEY}",
            },
        },
        "embedding_models": {
            EMBEDDING_MODEL_ID: {
                # `type` is the register_embedding() strategy key — the
                # embedding factory resolves the in-process bge-m3 model on
                # this exact string. It must equal BGE_M3_EMBEDDING_TYPE,
                # the value passed to register_embedding().
                "type": BGE_M3_EMBEDDING_TYPE,
                # Required by ModelConfig; label-only for a custom type.
                "model_provider": "harness",
                "model": cfg.embedder_model,
            },
        },
        # `input` and `input_storage` are SEPARATE top-level keys per the
        # canonical INIT_YAML template. The earlier "input.storage"
        # nesting parsed into an ignored extra; the loader silently used
        # defaults.
        "input": {"type": "text"},
        "input_storage": {
            "type": "file",
            "base_dir": GRAPHRAG_INPUT_SUBDIR,
        },
        # Outputs live under top-level `output_storage` — GraphRagConfig
        # has no `output` field. The earlier `output:` key was an ignored
        # extra falling back to a default that happened to match.
        "output_storage": {
            "type": "file",
            "base_dir": GRAPHRAG_OUTPUT_SUBDIR,
        },
        # `reporting` is a separate top-level section; defaults work but
        # set it explicitly to match the canonical template.
        "reporting": {"type": "file", "base_dir": "logs"},
        # `cache.storage` is nested under cache (CacheType declares only
        # json / memory / none — no "file"). The earlier `cache.base_dir`
        # at the top was an ignored extra; the on-disk cache path was
        # the default, not our value.
        "cache": {
            "type": "json",
            "storage": {
                "type": "file",
                "base_dir": GRAPHRAG_CACHE_SUBDIR,
            },
        },
        # `vector_store` IS a VectorStoreConfig in GraphRAG 3.0.9 — NOT
        # a dict-of-stores keyed by name. type / db_uri / vector_size /
        # index_schema sit directly under this key. An extra wrapper
        # (vector_store.default_vector_store: {...}) parses as an
        # ignored extraneous attribute and the loader silently uses
        # every VectorStoreConfig default, notably vector_size=3072.
        "vector_store": {
            "type": "lancedb",
            "db_uri": GRAPHRAG_LANCEDB_SUBDIR,
            # Store-level default — harmless; the indexer reads the
            # per-field index_schema entries, but a reader consulting
            # the store-level value also lands on 1024.
            "vector_size": EMBEDDING_DIM,
            # IndexSchema.vector_size defaults to 3072 (OpenAI 3-large)
            # PER ENTRY, so every embed_text field needs an explicit
            # 1024-dim entry — otherwise the write store is built at
            # 3072 and bge-m3's 1024 vectors are rejected. `fields` is
            # required by the schema (no default) but is not consulted
            # for the dimension; an empty dict validates.
            "index_schema": {
                name: {
                    "index_name": name,
                    "vector_size": EMBEDDING_DIM,
                    "fields": {},
                }
                for name in EMBED_TEXT_FIELDS
            },
        },
        # `chunking` (not `chunks`). The earlier `chunks` key was an
        # ignored extra — every prior run used the GraphRAG default
        # chunk size (1200 tokens), NOT our configured cfg.chunk_size.
        "chunking": {
            "type": "tokens",
            "size": cfg.chunk_size,
            "overlap": cfg.chunk_overlap,
        },
        "embed_text": {"embedding_model_id": EMBEDDING_MODEL_ID},
    }


def graphrag_settings_hash(cfg: M5Config) -> str:
    """Stable short hash of the generated settings (cache-key component)."""
    canon = json.dumps(build_settings(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def graphrag_identity(cfg: M5Config) -> dict[str, Any]:
    """M5 cache-key extras — folded into compute_cache_key(extra=...) by C3.

    A graphrag library bump (graphrag_version) or any settings change
    (settings hash: chunk size/overlap, models, vector dim, community
    level) invalidates the M5 cache. Same discipline as
    parsing_identity() / summarization_identity().
    """
    return {
        "graphrag_version": cfg.graphrag_version,
        "graphrag_settings_hash": graphrag_settings_hash(cfg),
        "index_llm_model": cfg.index_llm_model,
        "embedder_model": cfg.embedder_model,
        "community_level": cfg.community_level,
        "retrieval_mode": cfg.retrieval_mode,
    }


# --- Filesystem handoff ---------------------------------------------------


def write_settings(cfg: M5Config, project_dir: Path) -> Path:
    """Write the generated settings.yaml into project_dir. yaml lazy-imported."""
    import yaml

    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "settings.yaml"
    path.write_text(
        yaml.safe_dump(build_settings(cfg), sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_input_documents(docs: dict[str, str], project_dir: Path) -> Path:
    """Write {doc_id: text} into the GraphRAG input dir as UTF-8 .txt files.

    GraphRAG ingests the corpus its own way; this is the harness -> GraphRAG
    handoff. The corpus is parsed once by the shared Docling parser (the
    single PDF backend across all systems, evaluation_plan.pdf section 7)
    and handed to GraphRAG as plain text.
    """
    in_dir = Path(project_dir) / GRAPHRAG_INPUT_SUBDIR
    in_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in docs.items():
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(doc_id))
        (in_dir / f"{safe}.txt").write_text(text, encoding="utf-8")
    return in_dir


# === COLAB-VERIFIED-ONLY ==================================================
# Everything below touches the graphrag 3.0.9 API. It cannot run on a host
# without graphrag installed and is smoke-verified on Colab T4 at C4. The
# build_context result shape and a few parquet column names were not
# captured by introspection — inline notes mark the C4 fix points.
# ==========================================================================


def _make_tokenizer() -> Any:
    """Build a graphrag_llm Tokenizer for embed-input batch sizing.

    GraphRAG's _create_text_batches sizes embed batches with
    tokenizer.num_tokens(...) — a concrete method on the
    graphrag_llm.tokenizer.tokenizer.Tokenizer ABC, derived from the
    abstract encode/decode. A bare tiktoken Encoding has encode/decode
    but is not a Tokenizer subclass, so it lacks num_tokens. This wraps a
    tiktoken cl100k_base encoding in a real Tokenizer subclass, which
    inherits num_tokens / num_prompt_tokens.

    Token counts here affect only batch boundaries, never the embeddings
    themselves — bge-m3's embedding() produces every vector — so this
    approximation does not touch embedder parity. COLAB-VERIFIED-ONLY.
    """
    import tiktoken
    from graphrag_llm.tokenizer.tokenizer import Tokenizer

    class _TiktokenTokenizer(Tokenizer):
        """tiktoken-backed Tokenizer; inherits num_tokens / num_prompt_tokens."""

        def __init__(self) -> None:
            self._enc = tiktoken.get_encoding("cl100k_base")

        def encode(self, text: str) -> list[int]:
            return self._enc.encode(text)

        def decode(self, tokens: Any) -> str:
            return self._enc.decode(list(tokens))

    return _TiktokenTokenizer()


def _build_embedding_response(vectors: list[list[float]]) -> Any:
    """Construct an LLMEmbeddingResponse with one entry per input vector.

    graphrag_llm.utils.create_embedding_response is for SINGLE-vector
    responses — it takes one vector and replicates it `batch_size` times,
    which produced the silent shape bug that made the entity_description
    table reject bge-m3 writes (one batch-flat list per input rather than
    one 1024-dim vector per input).

    For our bge-m3 batch — one vector per input — we build the response
    directly against the verified pydantic schema: data is a list of
    LLMEmbedding objects, one per input. LLMEmbeddingResponse.embeddings
    is a derived property over data, so downstream code that does
    zip(ids, response.embeddings, strict=True) gets one vector per id.
    """
    from graphrag_llm.types import (
        LLMEmbedding,
        LLMEmbeddingResponse,
        LLMEmbeddingUsage,
    )

    data = [
        LLMEmbedding(object="embedding", embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]
    return LLMEmbeddingResponse(
        object="list",
        data=data,
        model="harness/BAAI/bge-m3",
        usage=LLMEmbeddingUsage(prompt_tokens=0, total_tokens=0),
    )


class _MinimalMetricsStore:
    """Duck-typed stand-in for the graphrag_llm metrics store.

    bge-m3 runs locally — there are no API call metrics to record. Provides
    the surface observed in graphrag_llm usage (id / get_metrics /
    clear_metrics). C4 fix point if a richer interface is required.
    """

    id = "harness-bge-m3"

    def get_metrics(self) -> dict:
        return {}

    def clear_metrics(self) -> None:
        return None


def _make_bge_m3_embedding_class(cfg: M5Config) -> Any:
    """Define and return the bge-m3 LLMEmbedding subclass.

    Deferred to a factory so graphrag_llm is imported only when M5 runs,
    keeping this module importable without graphrag. The subclass routes
    GraphRAG's embedding calls through the shared harness bge-m3 embedder
    (src/models.embed_texts) — the same model M2/M3/M4/M7 use, giving the
    embedder parity that evaluation_plan.pdf section 7 requires.

    The verified abstract surface is __init__, embedding, embedding_async,
    metrics_store, tokenizer. embedding / embedding_async are implemented
    against the introspected signature (self, /, **kwargs: Unpack[
    LLMEmbeddingArgs]) — the input texts arrive under the 'input' key.
    metrics_store is a minimal duck-typed stub; tokenizer is a real
    graphrag_llm Tokenizer subclass (see _make_tokenizer).
    """
    from graphrag_llm.embedding import LLMEmbedding

    from .models import embed_texts

    class BgeM3Embedding(LLMEmbedding):
        """bge-m3 embedder exposed through the graphrag_llm interface."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._model_name = cfg.embedder_model
            self._metrics = _MinimalMetricsStore()
            self._tokenizer = _make_tokenizer()

        def _encode(self, model_input: Any) -> list[list[float]]:
            texts = [model_input] if isinstance(model_input, str) else list(model_input)
            vecs = embed_texts(texts, self._model_name)
            return [v.tolist() for v in vecs]

        @staticmethod
        def _input_texts(kwargs: dict[str, Any]) -> Any:
            # LLMEmbedding.embedding takes **kwargs: Unpack[LLMEmbeddingArgs];
            # the texts to embed arrive under the 'input' key.
            for key in ("input", "texts", "text"):
                if kwargs.get(key) is not None:
                    return kwargs[key]
            raise KeyError("LLMEmbedding call missing an 'input' argument")

        def embedding(self, /, **kwargs: Any) -> Any:
            return _build_embedding_response(self._encode(self._input_texts(kwargs)))

        async def embedding_async(self, /, **kwargs: Any) -> Any:
            # bge-m3 runs locally and synchronously; no real async work.
            return self.embedding(**kwargs)

        @property
        def metrics_store(self) -> Any:
            return self._metrics

        @property
        def tokenizer(self) -> Any:
            return self._tokenizer

    return BgeM3Embedding


_REGISTERED = False


def register_bge_m3_embedding(cfg: M5Config) -> str:
    """Register the bge-m3 embedder in the graphrag_llm registry.

    Idempotent within a process. Returns the registry key to place in the
    embedding ModelConfig's `type` field. Must be called before
    build_index so GraphRAG's indexing workflows resolve the custom type.
    """
    global _REGISTERED
    if not _REGISTERED:
        from graphrag_llm.embedding import register_embedding

        embedding_cls = _make_bge_m3_embedding_class(cfg)
        register_embedding(
            BGE_M3_EMBEDDING_TYPE,
            lambda *a, **k: embedding_cls(),
            scope="transient",
        )
        _REGISTERED = True
    return BGE_M3_EMBEDDING_TYPE


def make_bge_m3_embedder(cfg: M5Config) -> Any:
    """Instantiate the bge-m3 LLMEmbedding for the query-side text_embedder.

    LocalSearchMixedContext takes a text_embedder: LLMEmbedding directly,
    so the query side needs no registry — just the instance.
    """
    return _make_bge_m3_embedding_class(cfg)()


def _preflight_check_settings(config: Any, m5: M5Config) -> None:
    """Pre-flight: confirm critical settings actually parsed where we
    expect, not into ignored extras.

    Catches the default_vector_store / input.storage / output /
    cache.base_dir / chunks-vs-chunking class of structure-mismatch
    bug — fields that look right in YAML but parse into ignored extras,
    leaving GraphRAG to silently use its defaults. Bugs of this shape
    passed several full rebuilds before surfacing via dimension mismatch
    or empty tables; this raises before any work is done.

    Checks:
      - vector_store.vector_size and every index_schema entry's
        vector_size equal EMBEDDING_DIM (the original pre-flight).
      - input_storage.base_dir and output_storage.base_dir match the
        configured subdirs.
      - cache.storage.base_dir matches.
      - chunking.size / chunking.overlap match the M5Config values
        (catches the silent fallback to GraphRAG's 1200/100 default).
    """
    errors: list[str] = []

    # vector_store + index_schema dimensions
    vs = getattr(config, "vector_store", None)
    if vs is None:
        errors.append("config.vector_store missing")
    else:
        if getattr(vs, "vector_size", None) != EMBEDDING_DIM:
            errors.append(
                f"vector_store.vector_size="
                f"{getattr(vs, 'vector_size', '?')} (expected {EMBEDDING_DIM})"
            )
        schema = getattr(vs, "index_schema", None) or {}
        if isinstance(schema, dict):
            entries = list(schema.values())
        elif hasattr(schema, "values") and callable(getattr(schema, "values", None)):
            try:
                entries = list(schema.values())
            except Exception:
                entries = []
        else:
            entries = []
        for entry in entries:
            dim = getattr(entry, "vector_size", None)
            if dim is not None and dim != EMBEDDING_DIM:
                errors.append(
                    f"index_schema[{getattr(entry, 'index_name', '?')}]."
                    f"vector_size={dim} (expected {EMBEDDING_DIM})"
                )

    # input_storage / output_storage base_dirs reach the parsed config.
    # GraphRAG resolves base_dir to an absolute path relative to the
    # project dir at load time, so compare on basename rather than the
    # literal string we wrote.
    def _basename_matches(actual: Any, expected: str) -> bool:
        if actual is None:
            return False
        return Path(str(actual)).name == Path(expected).name

    for field, expected in (
        ("input_storage", GRAPHRAG_INPUT_SUBDIR),
        ("output_storage", GRAPHRAG_OUTPUT_SUBDIR),
    ):
        sect = getattr(config, field, None)
        if sect is None:
            errors.append(f"config.{field} missing")
            continue
        actual = getattr(sect, "base_dir", None)
        if not _basename_matches(actual, expected):
            errors.append(
                f"{field}.base_dir={actual!r} "
                f"(expected basename {expected!r})"
            )

    # cache.storage.base_dir — nested, not at the cache top level
    cache = getattr(config, "cache", None)
    if cache is None:
        errors.append("config.cache missing")
    else:
        cstorage = getattr(cache, "storage", None)
        if cstorage is None:
            errors.append("cache.storage missing")
        else:
            actual = getattr(cstorage, "base_dir", None)
            if not _basename_matches(actual, GRAPHRAG_CACHE_SUBDIR):
                errors.append(
                    f"cache.storage.base_dir={actual!r} "
                    f"(expected basename {GRAPHRAG_CACHE_SUBDIR!r})"
                )

    # chunking — verifies our chunk_size/overlap reach the parsed config
    chunking = getattr(config, "chunking", None)
    if chunking is None:
        errors.append("config.chunking missing")
    else:
        actual_size = getattr(chunking, "size", None)
        if actual_size != m5.chunk_size:
            errors.append(
                f"chunking.size={actual_size} (expected {m5.chunk_size})"
            )
        actual_overlap = getattr(chunking, "overlap", None)
        if actual_overlap != m5.chunk_overlap:
            errors.append(
                f"chunking.overlap={actual_overlap} (expected {m5.chunk_overlap})"
            )

    if errors:
        raise RuntimeError(
            "GraphRAG settings did not parse as expected — fields landed "
            "in ignored extras or fell back to defaults. Errors:\n  - "
            + "\n  - ".join(errors)
        )


def build_index(cfg: M5Config, project_dir: Path) -> list:
    """Run the GraphRAG indexing pipeline in-process.

    Registers the bge-m3 embedder, force-imports GraphRAG's factory
    modules so their built-in types register, loads the generated
    settings, and runs the async build_index pipeline to completion.
    Returns the list of PipelineRunResult; raises RuntimeError if any
    workflow errored.

    settings.yaml and the input documents must already be written into
    project_dir (write_settings / write_input_documents) — the C3 wrapper
    sequences that. Indexing is in-process (not the GraphRAG CLI) because
    custom embedding-model injection is library-only.
    """
    register_bge_m3_embedding(cfg)

    # GraphRAG's built-in cache / storage / table / vector-store types
    # register at module import time via module-level register_*() calls.
    # The CLI pulls these in transitively; the in-process path must import
    # the factory modules explicitly, or the factories stay empty and
    # run_pipeline raises "type '...' is not registered".
    import graphrag_cache.cache_factory  # noqa: F401
    import graphrag_storage.storage_factory  # noqa: F401
    import graphrag_storage.tables.table_provider_factory  # noqa: F401
    import graphrag_vectors.vector_store_factory  # noqa: F401

    import graphrag.api as api
    from graphrag.config.enums import IndexingMethod
    from graphrag.config.load_config import load_config

    config = load_config(Path(project_dir))
    _preflight_check_settings(config, cfg)
    results = asyncio.run(
        api.build_index(config=config, method=IndexingMethod.Standard)
    )

    failed = [r for r in results if getattr(r, "errors", None)]
    if failed:
        detail = "; ".join(
            f"{getattr(r, 'workflow', '?')}: {r.errors}" for r in failed
        )
        raise RuntimeError(f"GraphRAG indexing failed — {detail}")
    return results


def load_artifacts(project_dir: Path) -> GraphRAGArtifacts:
    """Load the GraphRAG parquet outputs. pandas lazy-imported."""
    import pandas as pd

    out = Path(project_dir) / GRAPHRAG_OUTPUT_SUBDIR

    def _read(name: str) -> Any:
        path = out / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"GraphRAG artifact missing: {path}")
        return pd.read_parquet(path)

    return GraphRAGArtifacts(
        entities=_read("entities"),
        communities=_read("communities"),
        relationships=_read("relationships"),
        community_reports=_read("community_reports"),
        text_units=_read("text_units"),
    )


# Local-search context parameters. Tuned toward text units (primary
# evidence) with a smaller community-report share (orientation), matching
# the M5 "local search + community orientation" retrieval mode.
_LOCAL_CONTEXT_PARAMS: dict[str, Any] = {
    "text_unit_prop": 0.6,
    "community_prop": 0.2,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "max_tokens": 12_000,
}


def _get(records: Any, key: str) -> Any:
    """Best-effort lookup of a named record set from a build_context result."""
    if records is None:
        return None
    if hasattr(records, "get"):
        return records.get(key)
    return getattr(records, key, None)


def _frame_to_units(frame: Any, kind: str) -> list[GraphRAGContextUnit]:
    """Map a build_context record DataFrame into GraphRAGContextUnit records.

    Column names vary by record type and GraphRAG version, so each field
    is resolved best-effort across known aliases. Score is the row's rank
    if present, else descending by position. C4 fix point.
    """
    if frame is None or getattr(frame, "empty", True):
        return []

    text_cols = ("text", "content", "description", "summary")
    id_cols = ("id", "human_readable_id", "short_id", "title")
    source_cols = ("document_ids", "source_id", "community", "title")
    rank_cols = ("rank", "weight", "score")

    units: list[GraphRAGContextUnit] = []
    n = len(frame)
    for pos, (_, row) in enumerate(frame.iterrows()):

        def _first(cols: tuple[str, ...], default: Any = "") -> Any:
            for c in cols:
                if c in row and row[c] is not None and str(row[c]) != "nan":
                    return row[c]
            return default

        text = str(_first(text_cols, "")).strip()
        if not text:
            continue
        rank = _first(rank_cols, None)
        score = float(rank) if rank is not None else float(n - pos)
        units.append(
            GraphRAGContextUnit(
                unit_id=str(_first(id_cols, f"{kind}-{pos}")),
                text=text,
                kind=kind,
                score=score,
                source=str(_first(source_cols, "")),
            )
        )
    return units


def _context_to_units(build_context_result: Any, k: int | None) -> list[GraphRAGContextUnit]:
    """Map a LocalSearchMixedContext.build_context result into neutral units.

    build_context returns context records keyed by record type. Text units
    ('sources') become primary evidence; community reports ('reports')
    become orientation. The result object's exact shape was not
    introspected — handles a `.context_records` attribute, a `(text,
    records)` tuple, and a `.context_data` fallback. C4 fix point.
    """
    records = getattr(build_context_result, "context_records", None)
    if records is None and isinstance(build_context_result, tuple):
        records = build_context_result[1] if len(build_context_result) > 1 else None
    if records is None:
        records = getattr(build_context_result, "context_data", build_context_result)

    units = _frame_to_units(_get(records, "sources"), "text_unit")
    units += _frame_to_units(_get(records, "reports"), "community_report")
    units.sort(key=lambda u: u.score, reverse=True)
    return units[:k] if k else units


def local_search_context(
    cfg: M5Config,
    project_dir: Path,
    query: str,
    k: int | None = None,
) -> list[GraphRAGContextUnit]:
    """Local-search retrieval — ranked text units plus community reports.

    Builds a LocalSearchMixedContext over the indexed artifacts and calls
    build_context for the query, then maps the selected units into
    paradigm-neutral GraphRAGContextUnit records. This does NOT generate an
    answer — the harness generator (Qwen2.5-3B-Instruct) does that in the
    C3 wrapper, holding the generator constant per evaluation_plan.pdf
    section 7. M5 differs from the other systems only in retrieval and
    embedding.
    """
    artifacts = load_artifacts(project_dir)

    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.indexer_adapters import (
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.query.structured_search.local_search.mixed_context import (
        LocalSearchMixedContext,
    )
    # LanceDBVectorStore is not a top-level graphrag_vectors export — it
    # lives in the graphrag_vectors.lancedb submodule.
    from graphrag_vectors.lancedb import LanceDBVectorStore

    level = cfg.community_level
    entities = read_indexer_entities(artifacts.entities, artifacts.communities, level)
    relationships = read_indexer_relationships(artifacts.relationships)
    reports = read_indexer_reports(artifacts.community_reports, artifacts.communities, level)
    text_units = read_indexer_text_units(artifacts.text_units)

    # index_name must equal the entity-embedding table GraphRAG wrote
    # ("entity_description"). A wrong name makes connect() silently skip
    # opening the table — no error, but retrieval returns nothing.
    # vector_size must be bge-m3's 1024; the base default is 3072 (the
    # OpenAI dimension) and would mismatch the 1024-dim table. db_uri is
    # a constructor argument; connect() takes none. project_dir is the
    # local staging copy — lancedb cannot be read off a Drive mount.
    store = LanceDBVectorStore(
        db_uri=str(Path(project_dir) / GRAPHRAG_LANCEDB_SUBDIR),
        index_name="entity_description",
        vector_size=EMBEDDING_DIM,
    )
    store.connect()

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=None,
        entity_text_embeddings=store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=make_bge_m3_embedder(cfg),
        tokenizer=_make_tokenizer(),
    )

    result = context_builder.build_context(query=query, **_LOCAL_CONTEXT_PARAMS)
    return _context_to_units(result, k or cfg.top_k_final)
