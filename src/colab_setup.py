"""Colab-only setup helpers.

Notebooks call mount_drive_and_setup() once at the top. The rest of
`src/` is environment-agnostic and never imports this module.

Setting HF_HOME / TRANSFORMERS_CACHE *before* transformers is imported
is mandatory — once transformers reads HF_HOME, the env var is captured
and later changes have no effect. Notebook order:

    from src.colab_setup import mount_drive_and_setup
    mount_drive_and_setup()         # <-- before anything from transformers
    from src.retrievers.m2_flat_dense import FlatDenseSystem

M4's paper-faithful clustering adds a second, EARLIER ordering
constraint. Installing umap-learn upgrades numpy, and doing that in a
kernel that already imported the stock numpy leaves a spliced module
tree that breaks faiss / torch / sentence-transformers. Full notebook
order:

    Block 0a:  pip install -r requirements.txt
    Block 0b:  RESTART RUNTIME  (Runtime -> Restart session)
    Block 0c:  verify_umap_stack()          # raises if the stack is broken
    Block 0d:  mount_drive_and_setup()      # before any transformers import
    Block 1+:  eval cells

See verify_umap_stack() below — it is a hard gate, not advice.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import paths


def _set_hf_env(hf_cache: Path) -> None:
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(hf_cache / "sentence-transformers")


def verify_umap_stack(verbose: bool = True) -> dict[str, str]:
    """Gate the M4 paper-faithful clustering stack. Call after Block 0's restart.

    M4's paper-faithful tree needs umap-learn (UMAP global+local
    reduction) and numba. Installing them on Colab UPGRADES numpy, and
    doing that inside a kernel that already imported the stock numpy
    leaves a spliced module tree. The observed failure is

        ImportError: cannot import name '_center' from 'numpy._core.umath'

    and it takes down faiss / torch / sentence-transformers, not just
    UMAP — so a run can die hours in, on a system that has nothing to do
    with M4. A runtime restart is the only fix.

    Required notebook order, no exceptions:

        Block 0a:  pip install -r requirements.txt
        Block 0b:  RESTART RUNTIME  (Runtime -> Restart session)
        Block 0c:  from src.colab_setup import verify_umap_stack
                   verify_umap_stack()
        Block 1+:  anything else

    Raises RuntimeError with the actionable fix rather than returning a
    falsy value, so a notebook cannot stumble past a broken stack into a
    paid run. Returns the resolved version map on success — worth
    printing into the run log, since UMAP output is version-sensitive
    even under a pinned random_state and M4's tree topology is therefore
    reproducible only against a pinned stack.
    """
    import importlib.metadata as md

    def _ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except Exception:
            return "MISSING"

    def _tup(v: str) -> tuple[int, ...]:
        out: list[int] = []
        for part in v.split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            if not digits:
                break
            out.append(int(digits))
        return tuple(out)

    versions = {
        p: _ver(p)
        for p in (
            "numpy", "numba", "llvmlite", "umap-learn", "pynndescent",
            "scikit-learn", "faiss-cpu", "sentence-transformers", "tiktoken",
        )
    }

    numpy_v, numba_v = versions["numpy"], versions["numba"]
    if numpy_v == "MISSING" or _tup(numpy_v) < (2, 1):
        raise RuntimeError(
            f"numpy {numpy_v} violates the project floor >=2.1 "
            "(requirements.txt). Something downgraded it — almost always an "
            "old numba pinning numpy<2.1. Upgrade numba (>=0.66), do NOT "
            "lower numpy: numpy<2.1 breaks the faiss / torch / "
            "sentence-transformers wheels."
        )
    if numba_v == "MISSING" or _tup(numba_v) < (0, 66):
        raise RuntimeError(
            f"numba {numba_v} is too old for this project. Versions below "
            "0.66 pin numpy<2.1, which collides with the numpy>=2.1 floor. "
            "pip install -U 'numba>=0.66', then RESTART THE RUNTIME."
        )

    # The spliced-module-tree check. Import the packages a post-install
    # kernel actually breaks on, and name the fix in the message.
    for mod in ("numpy", "numba", "umap", "faiss", "torch",
                "sentence_transformers", "tiktoken", "sklearn"):
        try:
            __import__(mod)
        except Exception as e:
            raise RuntimeError(
                f"import {mod} failed after the UMAP-stack install: "
                f"{type(e).__name__}: {e}\n"
                "This is the spliced-numpy symptom. RESTART THE RUNTIME "
                "(Runtime -> Restart session) and run this check again "
                "BEFORE any eval cell. Do not proceed — a paid run started "
                "in this kernel will die partway through."
            ) from e

    if verbose:
        print("[colab_setup] UMAP stack OK — pinned versions for the run log:")
        for k, v in versions.items():
            print(f"  {k:22s} {v}")
        print("  NOTE: seeding UMAP forces n_jobs=1 (single-threaded); the "
              "'n_jobs value 1 overridden' warning is expected and correct.")

    return versions


def mount_drive_and_setup(verbose: bool = True) -> dict[str, Path]:
    """Mount Drive on Colab, create the four working dirs, route HF cache.

    Safe to call multiple times. Returns the resolved paths so the caller
    can print or log them.
    """
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        if verbose:
            print("[colab_setup] Not on Colab — skipping drive.mount; "
                  "using local fallback paths.")
    else:
        drive.mount("/content/drive")

    resolved = paths.ensure_all()
    _set_hf_env(resolved["HF_CACHE_DIR"])

    if verbose:
        print(paths.describe())
        print(f"  HF_HOME       = {os.environ['HF_HOME']}")

    return resolved
