"""Pin substrate cache-key assembly to the recorded inputs and keys of
banked substrates, so a drift names the differing input before the key
comparison runs."""

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

# Recorded ground truth from the banked manifests. The corpus hash is fed,
# not re-derived: this host does not hold the run host's corpus bytes, and
# the replay's key assembly is fed a recorded hash the same way.
RECORDED_CORPUS_HASH = (
    "72aa87d69093d4b17aab2a3d51409151aab82bc28c8023d9c5ae8b8ca2306ef1")
M2_KEY = "51a2e3f973962c3ca495369e773e80de"
M4_KEY = "cc0681443cb9e5d9ea8641e1b0dec05b"
# predicted only; not yet confirmed against a banked M3 directory
M3_PREDICTED_KEY = "0c3c65e24def020cf35d10fb111897b1"

# harness choice: shared default for M2/M3 (METHODS §A.2)
RECORDED_M2_CHUNKING = {
    "absolute_threshold": 0.5, "breakpoint_percentile": 90.0,
    "buffer_size": 1, "chunk_words": 200, "max_if_min_words": 500,
    "max_words": 400, "min_chars_per_doc": 200, "min_words": 80,
    "overlap_words": 50, "strategy": "word_window"}
# the topology env recorded in the M4 manifest, injected verbatim on replay
TOKENLESS_ENV = "umap-learn=0.5.12;scikit-learn=1.6.1;numpy=2.5.2"
QWEN = "Qwen/Qwen2.5-7B-Instruct"


class TestM2Oracle(unittest.TestCase):
    """The M2 key reproduces the banked M2/MultiHop substrate."""

    def test_inputs_match_the_manifest_field_by_field(self):
        """The resolved chunking config and embedder match the manifest."""
        res = resolve_components(None, DEFAULT_CONFIG)
        self.assertEqual(asdict(res.chunker_config), RECORDED_M2_CHUNKING)
        self.assertEqual(res.embedder_id, "BAAI/bge-m3")

    def test_key_reproduces_the_banked_directory(self):
        """The assembled M2 key equals the banked directory name."""
        res = resolve_components(None, DEFAULT_CONFIG)
        key = compute_cache_key(chunking_config=res.chunker_config,
                                embedder_model=res.embedder_id,
                                corpus_hash=RECORDED_CORPUS_HASH)
        self.assertEqual(key, M2_KEY)


class TestM4Oracle(unittest.TestCase):
    """The M4 key reproduces the banked tree only under the recorded env."""

    def test_lever_reproduces_the_token_less_banked_tree(self):
        """Injecting the recorded topology env yields the banked M4 key."""
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
        """The same inputs under the host env resolve to another key."""
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
    """The M3 extras move the key off M2's and land on the predicted key."""

    def test_extra_enters_the_key(self):
        """The sparse/fusion extras change the key to the M3 prediction."""
        res = resolve_components(None, DEFAULT_CONFIG)
        m3 = compute_cache_key(
            chunking_config=res.chunker_config,
            embedder_model=res.embedder_id,
            corpus_hash=RECORDED_CORPUS_HASH,
            extra={"sparse": "bm25okapi", "fusion": "rrf",
                   "rrf_k": DEFAULT_CONFIG.retrieval.rrf_k})
        self.assertNotEqual(m3, M2_KEY)
        self.assertEqual(m3, M3_PREDICTED_KEY)


class TestResolveSubstrate(unittest.TestCase):
    """The replay's warm resolution against a real temp cache tree."""

    def test_complete_substrate_resolves_and_incomplete_refuses(self):
        """A complete substrate resolves warm; a missing file refuses it."""
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
                # derive the expected key the way the replay does
                with tempfile.TemporaryDirectory() as td:
                    sy._write_corpus_layout(items, Path(td))
                    chash = corpus_content_hash(Path(td))
                cdir, req, _ = rr.assemble_cdir(sy, "M2", chash)
                expected = Path(str(cdir.manifest_path)).parent
                # nothing on disk yet: cold, but the key and path agree
                warm, chash2, exp2 = rr.resolve_substrate(sy, "M2", items)
                self.assertIsNone(warm)
                self.assertEqual(chash2, chash)
                self.assertEqual(exp2, expected)
                # a complete substrate resolves warm
                expected.mkdir(parents=True)
                for f in M2_REQ:
                    (expected / f).write_text("x", encoding="utf-8")
                (expected / "manifest.json").write_text(
                    json.dumps({"corpus_hash": chash}), encoding="utf-8")
                warm, _, _ = rr.resolve_substrate(sy, "M2", items)
                self.assertEqual(warm, expected)
                # removing one required file makes it refuse again
                (expected / M2_REQ[0]).unlink()
                warm, _, _ = rr.resolve_substrate(sy, "M2", items)
                self.assertIsNone(warm)


if __name__ == "__main__":
    unittest.main()
