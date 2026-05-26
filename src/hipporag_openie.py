"""HippoRAG 1 OpenIE — entity + triple extraction prompts and runner.

Faithful port of the OSU-NLP-Group/HippoRAG legacy-branch OpenIE pipeline.
Prompts are reproduced VERBATIM from the legacy source files:

  * NER + post-NER OpenIE passage prompts:
        scratch_hipporag/src/openie_extraction_instructions.py
  * Query NER prompt:
        scratch_hipporag/src/named_entity_extraction_parallel.py

The legacy code wraps these via LangChain ChatPromptTemplate; we translate
the same message content to the OpenAI Python client's message list shape
(role + content), keeping every system / user / assistant message string
unchanged. OpenAI client paths (model name + JSON response_format + the
exact temperature / max_tokens / stop params per call site) match the
legacy `init_langchain_model('openai', ...).invoke(...)` requests
byte-for-byte at the wire level.

Why a port and not a wrapper: the legacy repo pins numpy==1.26.4 /
torch==1.13.1, which conflict ABI-hard with our harness (numpy>=2.1,
torch>=2.2). Same precedent as M4 (RAPTOR port in src/raptor.py rather
than a wrapped repo).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from .summarization import _resolve_openai_key


# --- Prompt constants (verbatim from legacy openie_extraction_instructions.py) ---

# The one-shot passage that anchors both NER and OpenIE few-shot examples.
ONE_SHOT_PASSAGE = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

ONE_SHOT_PASSAGE_ENTITIES = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

ONE_SHOT_PASSAGE_TRIPLES = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

NER_INSTRUCTION = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.
"""

NER_INPUT_ONE_SHOT = f"Paragraph:\n```\n{ONE_SHOT_PASSAGE}\n```\n"

OPENIE_POST_NER_INSTRUCTION = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

OPENIE_POST_NER_FRAME = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""

# Query NER prompt (verbatim from legacy named_entity_extraction_parallel.py).
QUERY_NER_SYSTEM = "You're a very effective entity extraction system."

QUERY_NER_ONE_SHOT_INPUT = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""

QUERY_NER_ONE_SHOT_OUTPUT = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

QUERY_NER_USER_TEMPLATE = """
Question: {question}

"""


# --- Prompt versioning -----------------------------------------------------

OPENIE_PROMPT_VERSION = "v1"


# --- Phrase normalisation (verbatim from legacy processing.processing_phrases) ---


def processing_phrases(phrase: str) -> str:
    """Lower-case + replace non-alphanumeric (except space) with space + strip.

    Verbatim from `scratch_hipporag/src/processing.py:39`. The legacy
    pipeline applies this to every entity and triple element before
    indexing into the phrase dictionary; preserving it byte-for-byte is
    required for entity-id stability vs. their published artifacts.

    Side note: destroys non-ASCII characters (Greek included). It is a
    paper-faithfulness cost, not a bug. Documented as an M6 limitation.
    """
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


# --- OpenAI client + low-level JSON-mode call ------------------------------


_CLIENT: Any | None = None


def _get_client() -> Any:
    """Lazy OpenAI client. Cached for the process lifetime."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. `pip install openai` is "
            "required for HippoRAG OpenIE."
        ) from e
    _CLIENT = OpenAI(api_key=_resolve_openai_key())
    return _CLIENT


def _invoke_json_chat(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
    _label: str = "openie",
) -> tuple[str, int]:
    """Invoke chat.completions with JSON mode. Returns (content_str, n_tokens).

    Mirrors the legacy LangChain calls:
        client.invoke(messages, temperature=0, max_tokens=..., stop=...,
                      response_format={"type":"json_object"})

    Retries on transient errors (rate limit, timeout, transient API
    errors). Non-transient errors propagate immediately so misconfig
    surfaces during smoke instead of hiding in retry loops.
    """
    client = _get_client()
    try:
        from openai import APIError, APITimeoutError, RateLimitError
        transient_excs: tuple[type[BaseException], ...] = (
            RateLimitError,
            APITimeoutError,
            APIError,
        )
    except ImportError:
        transient_excs = ()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop

    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = (resp.choices[0].message.content or "").strip()
            n_tokens = int(resp.usage.total_tokens) if resp.usage else 0
            return content, n_tokens
        except transient_excs as e:
            last_exc = e
            if attempt == max_retries - 1:
                break
            time.sleep(retry_backoff_s * (2**attempt))

    raise RuntimeError(
        f"{_label}: exhausted {max_retries} retries against {model}"
    ) from last_exc


# --- Extraction API --------------------------------------------------------


@dataclass
class OpenIEResult:
    """One extracted passage's NER + triple output.

    Mirrors the legacy on-disk shape from `output/openie_*_results_*.json`
    so anyone reading both can cross-reference fields directly.

    `extracted_entities` and `extracted_triples` hold the RAW LLM output
    strings, not normalised. processing_phrases() is applied later at
    graph-build time, matching the legacy two-stage pattern (OpenIE saves
    raw, create_graph.py normalises).

    `parse_ok` is False if either the NER or the OpenIE post-NER JSON
    failed to parse. Failures are recorded but do not crash the
    pipeline; the entry has empty entities/triples and contributes
    nothing to the graph.
    """
    idx: int
    passage: str
    extracted_entities: list[str]
    extracted_triples: list[list[str]]
    n_tokens: int
    parse_ok: bool


def _build_passage_ner_messages(passage: str) -> list[dict]:
    """Mirror legacy ner_prompts.format_prompt(user_input=passage).to_messages()."""
    return [
        {"role": "system", "content": NER_INSTRUCTION},
        {"role": "user", "content": NER_INPUT_ONE_SHOT},
        {"role": "assistant", "content": ONE_SHOT_PASSAGE_ENTITIES},
        {"role": "user", "content": f"Paragraph:```\n{passage}\n```"},
    ]


def _build_openie_messages(passage: str, entities: list[str]) -> list[dict]:
    """Mirror legacy openie_post_ner_prompts.format_prompt(...).to_messages()."""
    named_entity_json = json.dumps({"named_entities": entities})
    one_shot_input = OPENIE_POST_NER_FRAME.format(
        passage=ONE_SHOT_PASSAGE,
        named_entity_json=ONE_SHOT_PASSAGE_ENTITIES,
    )
    user_input = OPENIE_POST_NER_FRAME.format(
        passage=passage,
        named_entity_json=named_entity_json,
    )
    return [
        {"role": "system", "content": OPENIE_POST_NER_INSTRUCTION},
        {"role": "user", "content": one_shot_input},
        {"role": "assistant", "content": ONE_SHOT_PASSAGE_TRIPLES},
        {"role": "user", "content": user_input},
    ]


def _build_query_ner_messages(query: str) -> list[dict]:
    """Mirror legacy query_ner_prompts.format_prompt(...).to_messages()."""
    return [
        {"role": "system", "content": QUERY_NER_SYSTEM},
        {"role": "user", "content": QUERY_NER_ONE_SHOT_INPUT},
        {"role": "assistant", "content": QUERY_NER_ONE_SHOT_OUTPUT},
        {"role": "user", "content": QUERY_NER_USER_TEMPLATE.format(question=query)},
    ]


def extract_passage_entities_and_triples(
    passage: str,
    *,
    idx: int,
    llm_model: str,
) -> OpenIEResult:
    """NER + post-NER OpenIE for a single passage. Two LLM calls.

    Call 1 (NER): passage -> {"named_entities": [...]}. Temp=0, JSON mode.
    Call 2 (OpenIE): passage + named-entity JSON -> {"triples": [[h,r,t],...]}.
    Temp=0, max_tokens=4096 (legacy openie_post_ner_extract line 72), JSON mode.

    Token counts summed across both calls. Malformed JSON in either
    call: parse_ok=False, empty fields for that level — index pipeline
    continues, the failure is counted in the manifest.
    """
    total_tokens = 0
    parse_ok = True

    # --- Call 1: NER ---
    entities: list[str] = []
    try:
        ner_content, ner_tokens = _invoke_json_chat(
            _build_passage_ner_messages(passage),
            model=llm_model,
            temperature=0.0,
            _label=f"passage_ner[{idx}]",
        )
        total_tokens += ner_tokens
        ner_obj = json.loads(ner_content)
        raw_entities = ner_obj.get("named_entities", [])
        if isinstance(raw_entities, list):
            entities = [str(e) for e in raw_entities]
        else:
            parse_ok = False
    except (json.JSONDecodeError, RuntimeError, KeyError, TypeError):
        parse_ok = False

    # --- Call 2: OpenIE post-NER (skip if NER produced no entities — the
    # legacy code still calls but with empty list; we replicate that exactly
    # so the prompt/cache behaviour is identical) ---
    triples: list[list[str]] = []
    try:
        openie_content, openie_tokens = _invoke_json_chat(
            _build_openie_messages(passage, entities),
            model=llm_model,
            temperature=0.0,
            max_tokens=4096,
            _label=f"passage_openie[{idx}]",
        )
        total_tokens += openie_tokens
        openie_obj = json.loads(openie_content)
        raw_triples = openie_obj.get("triples", [])
        if isinstance(raw_triples, list):
            # Keep only well-formed length-3 triples; the legacy graph
            # builder filters incorrectly-formatted ones at create_graph
            # time anyway, but doing it here keeps the OpenIEResult shape
            # tight and the failure visible.
            for t in raw_triples:
                if isinstance(t, list) and len(t) == 3 and all(isinstance(x, (str, int, float)) for x in t):
                    triples.append([str(x) for x in t])
                else:
                    parse_ok = False
        else:
            parse_ok = False
    except (json.JSONDecodeError, RuntimeError, KeyError, TypeError):
        parse_ok = False

    return OpenIEResult(
        idx=idx,
        passage=passage,
        extracted_entities=entities,
        extracted_triples=triples,
        n_tokens=total_tokens,
        parse_ok=parse_ok,
    )


def extract_query_entities(
    query: str,
    *,
    llm_model: str,
) -> tuple[list[str], int]:
    """Query-side NER. One LLM call, returns (entity_strings, n_tokens).

    Empty-NER outcome (model returns no entities or JSON parse fails) is
    NOT an error — returns ([], n_tokens). The M6 retrieve path falls
    back to a uniform PPR reset per the paper and logs the empty-NER
    event prominently. See `hipporag_ppr.uniform_fallback` +
    `m6_hipporag.HippoRAGSystem.retrieve`.

    Legacy call params (named_entity_extraction_parallel.py:48):
    temperature=0, max_tokens=300, stop=['\\n\\n'], JSON mode.
    """
    try:
        content, n_tokens = _invoke_json_chat(
            _build_query_ner_messages(query),
            model=llm_model,
            temperature=0.0,
            max_tokens=300,
            stop=["\n\n"],
            _label="query_ner",
        )
        obj = json.loads(content)
        raw = obj.get("named_entities", [])
        if isinstance(raw, list):
            return [str(e) for e in raw], n_tokens
        return [], n_tokens
    except (json.JSONDecodeError, RuntimeError, KeyError, TypeError):
        return [], 0


def extract_corpus_parallel(
    passages: list[str],
    *,
    llm_model: str,
    max_workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[OpenIEResult]:
    """Run OpenIE over an entire passage list with thread parallelism.

    Threads (not processes) — OpenIE is IO-bound on OpenAI HTTP calls,
    not CPU-bound. Returns results in input order (idx-sorted). The
    on_progress callback fires after each completed passage with
    (n_done, n_total) so callers can attach a tqdm bar or smoke log.
    """
    results: list[OpenIEResult | None] = [None] * len(passages)
    n_done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                extract_passage_entities_and_triples,
                p,
                idx=i,
                llm_model=llm_model,
            ): i
            for i, p in enumerate(passages)
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
            n_done += 1
            if on_progress is not None:
                on_progress(n_done, len(passages))

    # All slots filled (futures complete before exiting `with`).
    return [r for r in results if r is not None]


__all__ = [
    # Prompt constants (verbatim from legacy)
    "ONE_SHOT_PASSAGE",
    "ONE_SHOT_PASSAGE_ENTITIES",
    "ONE_SHOT_PASSAGE_TRIPLES",
    "NER_INSTRUCTION",
    "NER_INPUT_ONE_SHOT",
    "OPENIE_POST_NER_INSTRUCTION",
    "OPENIE_POST_NER_FRAME",
    "QUERY_NER_SYSTEM",
    "QUERY_NER_ONE_SHOT_INPUT",
    "QUERY_NER_ONE_SHOT_OUTPUT",
    "QUERY_NER_USER_TEMPLATE",
    "OPENIE_PROMPT_VERSION",
    # Helpers
    "processing_phrases",
    "OpenIEResult",
    # Public API
    "extract_passage_entities_and_triples",
    "extract_query_entities",
    "extract_corpus_parallel",
]
