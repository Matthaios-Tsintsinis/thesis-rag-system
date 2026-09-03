"""Substrate-key oracles from the Drive manifests.

THE PATTERN THIS FILE CLOSES (three-refusal session, 2026-08-31): the
review verified the scoring path in depth and gave key computation a
code-read — while every manifest on Drive was an executable oracle for
it. One test against `51a2e3f9…` would have prevented all three
refusals. These tests pin the replay's key assembly to the RECORDED
inputs and RECORDED keys of real banked substrates; a failure names the
differing input before the key comparison runs.

Recorded ground truth (operator-pulled manifests, 2026-08-31):
  * M2/MultiHop substrate `51a2e3f973962c3ca495369e773e80de`
    (created 2026-05-30, warm-hit by every P10 and P11 M2/MultiHop run)
  * M4/MultiHop P10 tree `cc0681443cb9e5d9ea8641e1b0dec05b`
    (created 2026-08-18, token-less env era, 16,523 chunks)
Both carry corpus_hash
`72aa87d69093d4b17aab2a3d51409151aab82bc28c8023d9c5ae8b8ca2306ef1` —
which this host CANNOT re-derive (its corpus bytes are not the run
host's; the b274e596 prediction failure), so the hash is FED, exactly
as the replay's key assembly is fed a recorded hash.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.cache import CacheDir, compute_cache_key
from src.components import resolve_components
from src.config import DEFAULT_CONFIG

RECORDED_CORPUS_HASH = (
    "72aa87d69093d4b17aab2a3d51409151aab82bc28c8023d9c5ae8b8ca2306ef1")
M2_KEY = "51a2e3f973962c3ca495369e773e80de"
M4_KEY = "cc0681443cb9e5d9ea8641e1b0dec05b"
# predicted only -- confirm against the M3 warm-hit directory on Drive
# before pinning; the dry-run prints it per cell.
M3_PREDICTED_KEY = "0c3c65e24def020cf35d10fb111897b1"

RECORDED_M2_CHUNKING = {
    "absolute_threshold": 0.5, "breakpoint_percentile": 90.0,
    "buffer_size": 1, "chunk_words": 200, "max_if_min_words": 500,
    "max_words": 400, "min_chars_per_doc": 200, "min_words": 80,
    "overlap_words": 50, "strategy": "word_window"}
TOKENLESS_ENV = "umap-learn=0.5.12;scikit-learn=1.6.1;numpy=2.5.2"
QWEN = "Qwen/Qwen2.5-7B-Instruct"


class TestM2Oracle(unittest.TestCase):
    def test_inputs_match_the_manifest_field_by_field(self):
        # localises a future failure BEFORE the key comparison
        res = resolve_components(None, DEFAULT_CONFIG)
        self.assertEqual(asdict(res.chunker_config), RECORDED_M2_CHUNKING)
        self.assertEqual(res.embedder_id, "BAAI/bge-m3")

    def test_key_reproduces_the_banked_directory(self):
        res = resolve_components(None, DEFAULT_CONFIG)
        key = compute_cache_key(chunking_config=res.chunker_config,
                                embedder_model=res.embedder_id,
                                corpus_hash=RECORDED_CORPUS_HASH)
        self.assertEqual(key, M2_KEY)


class TestM4Oracle(unittest.TestCase):
    def test_lever_reproduces_the_token_less_banked_tree(self):
        from src.retrievers.m4_raptor import (
            RaptorSystem,
            resolve_components as m4_resolve,
        )
        cfg = replace(
            DEFAULT_CONFIG,
            generation=replace(DEFAULT_CONFIG.generation, model=QWEN),
            m4=replace(DEFAULT_CONFIG.m4, summary_model=QWEN))
        sy = RaptorSystem(config=cfg)
        sy.topology_env_override = TOKENLESS_ENV
        sy._resolved = m4_resolve(cfg.m4, cfg, default_reranker=None)
        cdir = sy._cache_dir(RECORDED_CORPUS_HASH)
        self.assertEqual(Path(str(cdir.manifest_path)).parent.name, M4_KEY)

    def test_without_the_lever_the_key_differs(self):
        # the same inputs under the HOST env resolve elsewhere -- which
        # is exactly why the recorded env must be injected for pre-token
        # cells, and why local corpus-derived predictions were doomed
        from src.retrievers.m4_raptor import (
            RaptorSystem,
            resolve_components as m4_resolve,
        )
        cfg = replace(
            DEFAULT_CONFIG,
            generation=replace(DEFAULT_CONFIG.generation, model=QWEN),
            m4=replace(DEFAULT_CONFIG.m4, summary_model=QWEN))
        sy = RaptorSystem(config=cfg)
        sy._resolved = m4_resolve(cfg.m4, cfg, default_reranker=None)
        cdir = sy._cache_dir(RECORDED_CORPUS_HASH)
        self.assertNotEqual(Path(str(cdir.manifest_path)).parent.name,
                            M4_KEY)


class TestM3Assembly(unittest.TestCase):
    def test_extra_enters_the_key(self):
        res = resolve_components(None, DEFAULT_CONFIG)
        m3 = compute_cache_key(
            chunking_config=res.chunker_config,
            embedder_model=res.embedder_id,
            corpus_hash=RECORDED_CORPUS_HASH,
            extra={"sparse": "bm25okapi", "fusion": "rrf",
                   "rrf_k": DEFAULT_CONFIG.retrieval.rrf_k})
        self.assertNotEqual(m3, M2_KEY)
        # documented prediction; pin to the Drive directory once the
        # operator confirms the M3 warm-hit dirname
        self.assertEqual(m3, M3_PREDICTED_KEY)


class TestResolveSubstrate(unittest.TestCase):
    """The replay's warm resolution against a real temp cache tree."""

    def test_complete_substrate_resolves_and_incomplete_refuses(self):
        from src.retrievers.m2_flat_dense import (
            REQUIRED_FILES as M2_REQ,
            FlatDenseSystem,
        )
        from src.eval.types import CorpusItem
        from src.cache import corpus_content_hash
        import scripts.replay_retrieval as rr
        import tempfile

        sy = FlatDenseSystem(config=DEFAULT_CONFIG)
        items = [CorpusItem(item_id="p0::s0", parent_id="p0",
                            span_id="s0",
                            text="alpha beta gamma " * 30, metadata={})]
        with TemporaryDirectory() as cache_root:
            with mock.patch("src.paths.cache_dir",
                            return_value=Path(cache_root)):
                # derive the expected key exactly as the replay will
                with tempfile.TemporaryDirectory() as td:
                    sy._write_corpus_layout(items, Path(td))
                    chash = corpus_content_hash(Path(td))
                cdir, req, _ = rr.assemble_cdir(sy, "M2", chash)
                expected = Path(str(cdir.manifest_path)).parent
                warm, chash2, exp2 = rr.resolve_substrate(sy, "M2", items)
                self.assertIsNone(warm)          # nothing on disk yet
                self.assertEqual(chash2, chash)
                self.assertEqual(exp2, expected)
                expected.mkdir(parents=True)
                for f in M2_REQ:
                    (expected / f).write_text("x", encoding="utf-8")
                (expected / "manifest.json").write_text(
                    json.dumps({"corpus_hash": chash}), encoding="utf-8")
                warm, _, _ = rr.resolve_substrate(sy, "M2", items)
                self.assertEqual(warm, expected)
                (expected / M2_REQ[0]).unlink()  # break completeness
                warm, _, _ = rr.resolve_substrate(sy, "M2", items)
                self.assertIsNone(warm)


if __name__ == "__main__":
    unittest.main()
