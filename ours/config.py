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

# M4's filter (the harness's min_chars_per_doc) leaves out documents under 200
# characters. A unit holding one gets a tree of ours' own, from M4's builder with
# the filter lifted, kept under this namespace so M4's bank stays M4's. Every
# other unit reads M4's tree. (Development v12.)
TREE_NAMESPACE = "OURS_WHOLE_CORPUS"

# Token counts use the harness's counter (o200k_base), so evidence_tokens
# compares across cells.
TOKEN_COUNTER_MODEL = "gpt-4o-mini"

# Every chunk goes to the reader, the reranker and BM25 under its document's name:
# source, date and title, as the corpus gives them (development v15). Development (v14)
# found 36 / 32 % of held-out HotpotQA chunks not naming their document, rows with such a
# gold chunk at F1 0.688 / 0.620 against 0.754 / 0.752, and titles alone lifting BM25's
# pooled R@5 from 0.675 to 0.745; 91 % of MultiHop's questions name a source, and only 35 %
# of its articles hold theirs in their text. A one-document unit has nothing to tell apart,
# so NarrativeQA stays as it was. M4's tree and vectors do not see the name.
DOCUMENT_HEADERS = True

# A question naming a source or a date steers retrieval (development v16). 2,331 of
# MultiHop's 2,556 questions name a source as the corpus writes it, which leaves about 94
# of its 609 articles, and 810 of the 823 full dates they name belong to an article of a
# named source. Each aspect searches the sources it names, or else the question's; the
# global view stays open; and the evidence holds at least MIN_CHUNKS_PER_TARGET chunks of
# every named source and dated article. No call: names are matched as written. On this
# dataset a source and date filter took Hits@10 from 0.749 to 0.891 with bge-large-en-v1.5
# (Multi-Meta-RAG, arXiv 2406.13213). HotpotQA and NarrativeQA have no sources, so they stay
# as they were; MultiHop has no held-out questions, so this rests on label-free checks.
SOURCE_FOCUS = True
MIN_CHUNKS_PER_TARGET = 2
FULL_DATE = (
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})\b"
)

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
# held-out NarrativeQA stories.
SECTION_SIZE_CHUNKS = 30
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

# --- Step 10: the answer --------------------------------------------------

# A refusal is asked once more, the evidence unchanged, under REASK_PROMPT
# (development v8); that prompt may still refuse. Development (round 2, v4)
# found every gold sentence in the context for 45 / 42 of the 97 / 103
# held-out HotpotQA refusals, and questions refused under one version but
# answered under another with the same chunks scored F1 0.33 / 0.50 / 0.24
# when answered (distractor / pooled / NarrativeQA).
REASK_REFUSALS = True
REASK_PROMPT = (
    "Answer the user's question using only the provided evidence. The "
    "evidence was retrieved for this question and usually contains the "
    "answer, sometimes spread over several chunks that must be combined. "
    "Give the answer the evidence best supports, in a few words. Only if "
    "nothing in the evidence bears on the question, respond with exactly: "
    f"'{REFUSAL}'. Do not use facts that are not in the evidence. "
    + ORIENTATION_LINE
)

# The reader writes out its reasoning first (development v14); only its last
# "Answer:" line goes on. Development (v8) found every gold sentence in the
# context and no gold in the reader's answer on 67 / 65 held-out HotpotQA rows,
# 32 / 31 % of all F1 lost, and 81 % of HotpotQA's questions are bridge questions:
# it found the actor and stopped ("Daniel Henney", for what 2007 film he is known
# for). ANSWER_PROMPT stays as it was: it is asked once on the same evidence,
# saved for the report as the plain answer, never the answer. The example is
# invented.
REASON_LINE = (
    "Before answering, work step by step: write the facts from the evidence that "
    "bear on the question, and how they connect. A question may need two or more "
    "facts joined, so follow each step until you reach what the question actually "
    "asks for. Then end with a last line that begins 'Answer:' and gives only the "
    "answer, or the refusal above. For example, for the question 'What river runs "
    "through the birthplace of the author of Harbour Lights?': Harbour Lights was "
    "written by Anna Reed. Anna Reed was born in Tacoma. The Ember runs through "
    "Tacoma.\nAnswer: The Ember"
)
READER_PROMPT = ANSWER_PROMPT + " " + REASON_LINE
READER_REASK_PROMPT = REASK_PROMPT + " " + REASON_LINE

# One more call cuts the reader's answer to its shortest answering phrase
# (development v7). Development (round 2, v4) found the gold inside a longer
# answer for 77 / 82 of the answered held-out HotpotQA questions: answers of
# about 20 words, at F1 0.31 / 0.29. The cut keeps only the answer's own
# words, or yes / no; a cut that is not shorter, adds a word or refuses leaves
# the answer as the reader gave it, and a refusal is never cut.
#
# Development v13: asked for the "shortest phrase", v12's cut lowered F1 on 82 /
# 73 / 59 held-out rows, and the gold stood whole in the reader's answer on 80 /
# 70 / 9 rows where the final answer was not it: too short ("Monitor" for USS
# "Monitor"), other words ("Maseru" for the club based there), or a number
# nobody asked for ("1992" for Lonestar). So it now asks for what the question
# asks for, kept whole; the examples are invented. A why or how question is not
# cut at all: it wants a clause.
SHORT_ANSWER = True
SHORT_ANSWER_MAX_TOKENS = 32
SHORT_ANSWER_PROMPT = (
    "You shorten answers. Given a question and an answer to it, return only "
    "what the question asks for, taken from the answer. Keep it whole: the "
    "full name, title, place or phrase, with every word that belongs to it. "
    "If the question offers a choice between two things, return the one the "
    "answer picks, not a date or number that supports it. Return a number or "
    "date only if the question asks for one. If the question is a yes/no "
    "question, return only yes or no. Use only words that appear in the "
    "answer. Output only the short answer.\n\n"
    "Question: Which river is longer, the Ember or the Lode?\n"
    "Answer: The Ember is longer, at 640 km against 410 km.\n"
    "Short answer: The Ember\n\n"
    "Question: In what city is the Hallam Glass Museum?\n"
    "Answer: The Hallam Glass Museum stands in Tacoma, Washington, near the port.\n"
    "Short answer: Tacoma, Washington\n\n"
    "Question: Who directed the film Harbour Lights?\n"
    "Answer: Harbour Lights was directed by Anna Reed and released in 1994.\n"
    "Short answer: Anna Reed\n\n"
    "Question: In what year did the Lode Bridge open?\n"
    "Answer: The Lode Bridge opened in 1931 after six years of work.\n"
    "Short answer: 1931"
)
# What marks a question as asking for a number; what, beside digits, makes a cut
# nothing but a number or a date; and the words that alone never name one side
# of an "A or B" question.
NUMBER_QUESTION = (
    r"\bhow\s+(?:many|much|old|long|tall|far|big|large|high|wide|deep|fast|heavy|often)\b|"
    r"what year|which year|what date|\bwhen\b|\bnumber\b|\bage\b|\bdate\b|\byear\b|"
    r"\bpopulation\b|\bcapacity\b|\bheight\b|\blength\b"
)
NUMBER_WORDS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
})
OPTION_STOPWORDS = frozenset({"a", "an", "the", "of", "in", "on", "at", "to", "by", "for", "and"})
