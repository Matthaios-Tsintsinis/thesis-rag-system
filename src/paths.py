"""Filesystem layout resolution.

Resolves four roles to concrete paths with environment-variable overrides
and a multi-tier fallback chain so the same code runs on:

  * Colab with Drive mounted (preferred for persistence)
  * Colab without Drive   (results lost on session end, but smoke runs)
  * Local dev (Windows / Linux / Mac)

Resolution order for each role (first match wins):

  1. Explicit env var (THESIS_INPUT_DIR, THESIS_CACHE_DIR,
     THESIS_OUTPUT_DIR, THESIS_HF_CACHE_DIR)
  2. Drive subdir at /content/drive/MyDrive/thesis_rag/<role>
     (only when /content/drive/MyDrive exists)
  3. Colab local subdir at /content/thesis_rag/<role>
     (only when /content exists; HF cache uses /content/hf_cache)
  4. Repo-local fallback at <repo>/local_runs/<role>

HF model cache is kept local by default even when Drive is mounted —
the Drive sync layer is known to corrupt large .safetensors files
mid-download. Set THESIS_HF_CACHE_DIR explicitly to opt into Drive.

This module has no side effects at import time. Callers invoke
ensure_all() (typically once at notebook startup) to create dirs.
"""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_FALLBACK_ROOT = _REPO_ROOT / "local_runs"
_DRIVE_ROOT = Path("/content/drive/MyDrive/thesis_rag")
_COLAB_LOCAL_ROOT = Path("/content/thesis_rag")
_DRIVE_MARKER = Path("/content/drive/MyDrive")
_COLAB_MARKER = Path("/content")


def _drive_mounted() -> bool:
    return _DRIVE_MARKER.exists()


def _on_colab() -> bool:
    return _COLAB_MARKER.exists()


def _base_root() -> Path:
    if _drive_mounted():
        return _DRIVE_ROOT
    if _on_colab():
        return _COLAB_LOCAL_ROOT
    return _LOCAL_FALLBACK_ROOT


def _resolve(env_var: str, subdir: str) -> Path:
    val = os.environ.get(env_var)
    if val:
        return Path(val).expanduser()
    return _base_root() / subdir


def input_dir() -> Path:
    return _resolve("THESIS_INPUT_DIR", "inputs")


def cache_dir() -> Path:
    return _resolve("THESIS_CACHE_DIR", "cache")


def output_dir() -> Path:
    return _resolve("THESIS_OUTPUT_DIR", "outputs")


def hf_cache_dir() -> Path:
    """HF model cache. Local by default; opt into Drive with env var."""
    val = os.environ.get("THESIS_HF_CACHE_DIR")
    if val:
        return Path(val).expanduser()
    if _on_colab():
        return Path("/content/hf_cache")
    return _LOCAL_FALLBACK_ROOT / "hf_cache"


def all_paths() -> dict[str, Path]:
    return {
        "INPUT_DIR": input_dir(),
        "CACHE_DIR": cache_dir(),
        "OUTPUT_DIR": output_dir(),
        "HF_CACHE_DIR": hf_cache_dir(),
    }


def ensure_all() -> dict[str, Path]:
    paths = all_paths()
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def describe() -> str:
    """Human-readable summary for notebook startup banner."""
    flags = []
    if _drive_mounted():
        flags.append("drive=mounted")
    elif _on_colab():
        flags.append("drive=NOT mounted, using /content")
    else:
        flags.append("local dev")
    lines = ["thesis_rag paths (" + ", ".join(flags) + "):"]
    for k, v in all_paths().items():
        lines.append(f"  {k:14s} = {v}")
    return "\n".join(lines)
