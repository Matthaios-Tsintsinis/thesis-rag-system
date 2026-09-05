"""Pins that importing src.models sets PYTORCH_CUDA_ALLOC_CONF before any
model can initialise CUDA, and that run provenance records the value."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_fresh_process(code: str, env: dict | None = None) -> str:
    """Run code in a fresh interpreter so import-time effects are visible."""
    # Clear the allocator variable, then apply the test's own env.
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
    """Pins that the allocator variable is set at import time."""

    def test_importing_models_sets_it(self):
        """Importing src.models sets the allocator variable."""
        got = _in_fresh_process("""
            import os
            import src.models  # noqa: F401
            print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        """)
        self.assertEqual(got, "expandable_segments:True")

    def test_importing_the_runner_sets_it(self):
        """Importing the runner, the cell entry point, sets it too."""
        got = _in_fresh_process("""
            import os
            import src.eval.runner  # noqa: F401
            print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        """)
        self.assertEqual(got, "expandable_segments:True")

    def test_it_is_set_before_the_embedder_could_initialise_cuda(self):
        """The variable is in place before load_embedder can touch CUDA."""
        got = _in_fresh_process("""
            import os
            import src.models as m
            before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            assert callable(m.load_embedder)
            print(before)
        """)
        self.assertEqual(got, "expandable_segments:True")

    def test_an_operator_override_is_respected(self):
        """A value already in the environment survives the import."""
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
    """Pins that run provenance records the allocator setting."""

    def test_environment_provenance_carries_the_effective_value(self):
        """Provenance carries cuda_alloc_conf equal to the live variable."""
        from scripts.pin_environment import environment_provenance

        prov = environment_provenance(None)
        self.assertIn("cuda_alloc_conf", prov)
        self.assertEqual(prov["cuda_alloc_conf"],
                         os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


if __name__ == "__main__":
    unittest.main()
