"""Settings for ours: the design's values (docs/three_axis_pipeline.pdf §5),
the prompts, and the pinned models. Development may change any of them on
held-out data; the scored cells run with the values frozen here."""

# --- Models ---------------------------------------------------------------

# One model plays every LLM role: reader, aspects, paraphrases, HyDE, and
# (through M4's own code) the tree summaries.
READER = "meta-llama/Llama-3.1-8B-Instruct"
READER_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

# M4's leaf embedder. Questions and views are embedded with it too, so they
# land in the same space as the tree.
EMBEDDER = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
EMBEDDER_REVISION = "d51b22a1dfa8184e9258074e56e2875e50612dca"

RERANKER = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RERANK_BATCH = 32
RERANK_MAX_LENGTH = 512

# The topology stack the banked Llama trees were built under, as the p11
# summaries record it (tree_build_env). M4's cache key folds this string, so
# it must match for the existing trees to be found. Read from the four p11
# M4 summaries by `python -m scripts.dev_ours census`, which found all 1,051
# scored units warm under it.
TREE_BUILD_ENV: str | None = "python=3.12;umap-learn=0.5.12;scikit-learn=1.6.1;numpy=2.5.2"

# Token counts use the harness's counter (o200k_base), so evidence_tokens
# compares across cells.
TOKEN_COUNTER_MODEL = "gpt-4o-mini"

# --- Step 1: aspects and views --------------------------------------------

MAX_ASPECTS = 3
MIN_ASPECT_IMPORTANCE = 0.25
DECOMPOSE_MAX_TOKENS = 220
PARAPHRASE_MAX_TOKENS = 80
HYDE_MAX_TOKENS = 160

# --- Step 2: aspect scores ------------------------------------------------

ASPECT_IMPORTANCE_WEIGHT = 0.5
RETRIEVAL_CONFIDENCE_WEIGHT = 0.5
PRELIMINARY_RERANK_TOP_K = 10

# --- Step 3: budgets ------------------------------------------------------

FINAL_CONTEXT_CHUNKS = 15
GLOBAL_VIEW_QUOTA = 2
MIN_CHUNKS_PER_ASPECT = 2
# Applies only with two or more aspects; one aspect takes all 13 slots.
MAX_CHUNKS_PER_ASPECT = 8

# --- Step 4: candidates for one view --------------------------------------

FLAT_TREE_TOP_K_PER_VIEW = 50
MAX_DESCENDANT_CHUNKS_FOR_DIRECT_EXPANSION = 50
MAX_CHILDREN_TO_FOLLOW_FROM_BROAD_SUMMARY = 2
SUMMARY_EXPANSION_TOP_K_CHUNKS = 3
MAX_EXPANSION_RECURSION_DEPTH = 2

MULTI_BRANCH_TOP_K_DEPTH_1 = 3
MULTI_BRANCH_TOP_K_PER_LEVEL = 2
MULTI_BRANCH_MAX_DEPTH = 4
MULTI_BRANCH_LEAVES_PER_PATH = 5

# RRF, Cormack 2009, k=60
RRF_K = 60

SECTION_DIVERSITY_CAP = 3
NEIGHBOR_RADIUS = 1
ASPECT_SECTION_BIAS_FACTOR = 1.15
NEIGHBOR_SCORE_DISCOUNT = 0.9

# Section detection. A document of at most SECTION_SIZE_CHUNKS chunks is one
# section; a longer one is split at headings, or else into blocks of that
# many chunks. Heading families are tried in order and the first with at
# least MIN_HEADINGS placed headings wins; a heading closer than
# MIN_SECTION_CHUNKS to the one before is dropped. Tuned only on the
# held-out NarrativeQA stories. Development v3 tests 60 against the starting
# value of 30 (decision rule, set before the run: keep 60 only if NarrativeQA
# F1 rises with a paired CI clear of zero).
SECTION_SIZE_CHUNKS = 60
HEADING_PATTERNS = (
    ("markdown", r"^#{1,6}\s+\S"),
    ("numbered", r"^\d+(\.\d+)*\.?\s+[A-Z][^.!?]*$"),
    ("chapter", r"^(CHAPTER|Chapter)\s+\S"),
    ("scene", r"^(INT|EXT|INT\./EXT|I/E)[.\s]"),
    ("caps", r"^(?=.*[A-Z]{3})[^a-z]+$"),
)
MAX_HEADING_CHARS = 60
MIN_HEADING_WORD_CHARS = 4
MIN_HEADINGS = 2
MIN_SECTION_CHUNKS = 3
# Docling (PDF/DOCX layout) is an extension point, off: every benchmark
# corpus is plain text, so the evaluation never exercises it.
USE_DOCLING = False

# --- Step 5: aspect pool --------------------------------------------------

MAX_CHUNKS_PER_RAPTOR_CLUSTER = 4

# --- Step 8: orientation --------------------------------------------------

MAX_PARENT_SUMMARIES_PER_ASPECT = 3
MAX_ANCESTOR_SUMMARIES_PER_CHUNK_GROUP = 2
MAX_PARENT_SUMMARY_TOKENS = 80
SUMMARY_CONTEXT_TOKEN_RATIO = 0.15
CHUNK_CONTEXT_TOKEN_RATIO = 0.85

# --- Step 9: prompt -------------------------------------------------------

ABSTENTION_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.40
# The design shows each aspect's confidence and marks those below the
# threshold. Development v0 found the marker drove refusals (60 % of flagged
# NarrativeQA questions refused) and misfired (correct answers at confidence
# 0.03-0.22), so v1 showed neither; v2 shows the score alone.
PASS_RETRIEVAL_CONFIDENCE_TO_LLM = True
MARK_LOW_CONFIDENCE = False

# The exact refusal the scorers recognise (src/config.ABSTENTION_RESPONSE).
REFUSAL = "No answer available."

DECOMPOSE_PROMPT = (
    "You analyze a question for a retrieval system. Classify it as "
    "\"simple\" (one information need) or \"multi_aspect\" (several distinct "
    "sub-questions that need separate evidence). For \"simple\", return "
    "exactly one aspect named \"main\" with importance 1.0. For "
    "\"multi_aspect\", return up to 3 aspects; each is a short, "
    "self-contained sub-question covering one part of the answer, with an "
    "importance in [0, 1] for how central it is (importances should sum to "
    "roughly 1). Output ONLY a JSON object, no prose and no code fences: "
    "{\"type\": \"simple\" | \"multi_aspect\", \"aspects\": "
    "[{\"name\": string, \"importance\": number}]}"
)

PARAPHRASE_PROMPT = (
    "Rephrase the search query so the wording changes but the information "
    "need stays exactly the same. Output only the rephrased query on one "
    "line, with no preamble and no quotation marks."
)

HYDE_PROMPT = (
    "Write a short passage of 2-3 sentences that would directly answer the "
    "question, in the style of an extract from a relevant source document. "
    "Be specific and factual in tone. Do not hedge and do not say that it is "
    "hypothetical. Output only the passage."
)

# The answer prompt is the baselines' own (src/config.BASE_ANSWER_SYSTEM_PROMPT),
# so ours asks for the same answer form as M1-M4, plus one line on reading
# the orientation. v0's design prompt ("address each aspect", "synthesize")
# drew paragraph answers: 33.5 words on average against 2.6-word golds.
ORIENTATION_LINE = (
    "Orientation lines are context; primary evidence chunks are reliable; if "
    "they conflict, trust the chunk."
)
ANSWER_PROMPT = (
    "Answer the user's question using only the provided evidence. "
    f"If the evidence does not contain the answer, respond with exactly: "
    f"'{REFUSAL}'. "
    "Do not fabricate; do not guess. Be concise and factual. "
    + ORIENTATION_LINE
)
