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
unchanged. OpenAI client paths (model name + JSON response_format) match
the legacy `init_langchain_model('openai', ...).invoke(..., response_
format={"type":"json_object"})` path byte-for-byte at the request layer.

Why a port and not a wrapper: the legacy repo pins numpy==1.26.4 /
torch==1.13.1, which conflict ABI-hard with our harness (numpy>=2.1,
torch>=2.2). See docs/PROJECT_BRIEF and the M6 design proposal.

THIS FILE IS A C4a SKELETON — function bodies raise NotImplementedError.
C4b lands the working extraction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

# Bumped when ANY of the constants above changes. Folded into the M6 cache
# key so a prompt revision invalidates the OpenIE artifacts cleanly.
OPENIE_PROMPT_VERSION = "v1"


# --- Phrase normalisation (verbatim from legacy processing.processing_phrases) ---

def processing_phrases(phrase: str) -> str:
    """Lower-case + replace non-alphanumeric (except space) with space + strip.

    Verbatim from `scratch_hipporag/src/processing.py:39`. The legacy
    pipeline applies this to every entity and triple element before
    indexing into the phrase dictionary; preserving it byte-for-byte is
    required for entity-id stability vs. their published artifacts (if
    we ever want to diff).

    Side note: this destroys non-ASCII characters (Greek included). It is
    a paper-faithfulness cost, not a bug. Documented as a M6 limitation
    in the methods section alongside the English-centric Contriever
    embedder.
    """
    import re
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


# --- Extraction API (C4b implementation) ----------------------------------


@dataclass
class OpenIEResult:
    """One extracted passage's NER + triple output.

    Mirrors the legacy on-disk shape `output/openie_{corpus}_results_ner_*.json`
    so anyone reading both can cross-reference fields directly.
    """
    idx: int
    passage: str
    extracted_entities: list[str]
    extracted_triples: list[list[str]]
    n_tokens: int  # for cost accounting + the manifest


def extract_passage_entities_and_triples(
    passage: str,
    *,
    idx: int,
    llm_model: str,
    client: Any | None = None,
) -> OpenIEResult:
    """NER + post-NER OpenIE for a single passage. Two LLM calls.

    Call 1 (NER): passage -> {"named_entities": [...]}.
    Call 2 (OpenIE): passage + named-entity JSON -> {"triples": [[h,r,t], ...]}.

    Both calls use OpenAI JSON mode (response_format={"type":"json_object"}).
    Token counts are summed across the two calls for the returned
    n_tokens. Malformed JSON is caught and logged; a degenerate
    OpenIEResult with empty entities/triples is returned so the index
    pipeline can continue.

    NOT IMPLEMENTED in C4a — function signature only.
    """
    raise NotImplementedError("C4b: implement OpenIE extraction.")


def extract_query_entities(
    query: str,
    *,
    llm_model: str,
    client: Any | None = None,
) -> tuple[list[str], int]:
    """Query-side NER. One LLM call, returns (entity_strings, n_tokens).

    Empty-NER outcome (LLM returns no entities) is NOT an error — the
    M6 retrieve path falls back to a uniform PPR reset per the paper,
    and the empty-NER event is logged prominently for analysis. See
    M6Config / m6_hipporag.HippoRAGSystem.retrieve.

    NOT IMPLEMENTED in C4a — function signature only.
    """
    raise NotImplementedError("C4b: implement query-side NER.")


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
    # Public API (C4b)
    "extract_passage_entities_and_triples",
    "extract_query_entities",
]
