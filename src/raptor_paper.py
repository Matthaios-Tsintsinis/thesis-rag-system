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
#   - delimiter handling                                   -> DIVERGED, see below
#
# DIVERGENCE FROM REFERENCE CODE, ALIGNMENT WITH PAPER TEXT
# (ruled 2026-07-29; the single place this module knowingly departs
# from the reference implementation):
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

import functools
import re
from dataclasses import dataclass
from typing import Any


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


__all__ = [
    "REFERENCE_ENCODING",
    "RAPTOR_CHUNKER_VERSION",
    "TextSpan",
    "count_tokens_reference",
    "split_text_raptor",
]
