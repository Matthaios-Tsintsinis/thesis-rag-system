"""The allocator config must be set before CUDA initialises, not after.

INSTANCE ELEVEN. `configure_cuda_allocator()` sets
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, and its own docstring
says the variable is read once at allocator setup so changing it later
does nothing. It was called from exactly two places, both on the
GENERATION path: `_load_generator_impl` and `generate_batch`.

In a real cell the embedder loads first — `load_embedder` constructs a
SentenceTransformer, which moves weights to CUDA and initialises the
allocator — and the generator loads afterwards. So the call ran AFTER the
thing it configures, and the setting was inert for every cell that would
ever run.

Meanwhile every probe in this investigation sets it at MODULE TOP, before
any torch import. The entire measurement history was therefore taken with
expandable_segments ON while the matrix would have run without it. The
observed OOM reported 1.16 GiB reserved-but-unallocated — precisely the
fragmentation the setting exists to prevent.

The fix sets it at import time of `src.models`, which every CUDA-touching
path in this harness goes through (embedder, reranker and generator all
load from there), and records the effective value in run provenance so a
regression is visible in the artifact rather than only in a traceback.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_fresh_process(code: str, env: dict | None = None) -> str:
    """Run `code` in a clean interpreter. Import-time behaviour cannot be
    tested in-process: another test may already have imported the module,
    and the assertion would pass on leakage rather than on the import."""
    full = {**os.environ, **(env or {})}
    full.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if env:
        full.update(env)
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, cwd=REPO, env=full, timeout=180,
    )
    if out.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{out.stderr[-2000:]}")
    return out.stdout.strip()


class TestSetAtImportTime(unittest.TestCase):
    def test_importing_models_sets_it(self):
        """The property that was missing: set by the IMPORT, so it lands
        before anything constructs a CUDA tensor."""
        got = _in_fresh_process("""
            import os
            import src.models  # noqa: F401
            print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        """)
        self.assertEqual(got, "expandable_segments:True")

    def test_importing_the_runner_sets_it(self):
        """The entry point a cell actually uses."""
        got = _in_fresh_process("""
            import os
            import src.eval.runner  # noqa: F401
            print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        """)
        self.assertEqual(got, "expandable_segments:True")

    def test_it_is_set_before_the_embedder_could_initialise_cuda(self):
        """THE ORDERING THAT BROKE. `load_embedder` builds a
        SentenceTransformer, which initialises the allocator. The config
        must already be in place when that module is merely importable."""
        got = _in_fresh_process("""
            import os
            import src.models as m
            before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            assert callable(m.load_embedder)
            print(before)
        """)
        self.assertEqual(got, "expandable_segments:True")

    def test_an_operator_override_is_respected(self):
        """setdefault, not assignment: a deliberate value must survive."""
        got = _in_fresh_process(
            """
            import os
            import src.models  # noqa: F401
            print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
            """,
            env={"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
        )
        self.assertEqual(got, "max_split_size_mb:128")


class TestProvenanceRecordsIt(unittest.TestCase):
    """A value that is set and never asserted is not a check — the
    corollary this project has earned five times over."""

    def test_environment_provenance_carries_the_effective_value(self):
        from scripts.pin_environment import environment_provenance

        prov = environment_provenance(None)
        self.assertIn("cuda_alloc_conf", prov)
        self.assertEqual(prov["cuda_alloc_conf"],
                         os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


if __name__ == "__main__":
    unittest.main()
