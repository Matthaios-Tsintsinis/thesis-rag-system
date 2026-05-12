"""Colab-only setup helpers.

Notebooks call mount_drive_and_setup() once at the top. The rest of
`src/` is environment-agnostic and never imports this module.

Setting HF_HOME / TRANSFORMERS_CACHE *before* transformers is imported
is mandatory — once transformers reads HF_HOME, the env var is captured
and later changes have no effect. Notebook order:

    from src.colab_setup import mount_drive_and_setup
    mount_drive_and_setup()         # <-- before anything from transformers
    from src.retrievers.m2_flat_dense import FlatDenseSystem
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
