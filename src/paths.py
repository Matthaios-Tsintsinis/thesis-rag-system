"""Resolve the input, cache, output, HF-cache and staging directories.
Env vars win, then Drive under /content/drive, then /content, then
<repo>/local_runs. Nothing runs at import; ensure_all() creates the dirs.
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
    """True when the Colab Drive mount is present."""
    return _DRIVE_MARKER.exists()


def _on_colab() -> bool:
    """True when /content exists, i.e. the code runs on Colab."""
    return _COLAB_MARKER.exists()


def _base_root() -> Path:
    """Pick the root for this host: Drive, Colab local disk, or the repo."""
    if _drive_mounted():
        return _DRIVE_ROOT
    if _on_colab():
        return _COLAB_LOCAL_ROOT
    return _LOCAL_FALLBACK_ROOT


def _resolve(env_var: str, subdir: str) -> Path:
    """Return the env-var override for a role, else <root>/<subdir>."""
    val = os.environ.get(env_var)
    if val:
        return Path(val).expanduser()
    return _base_root() / subdir


def input_dir() -> Path:
    """Directory holding benchmark inputs."""
    return _resolve("THESIS_INPUT_DIR", "inputs")


def cache_dir() -> Path:
    """Directory holding substrate caches."""
    return _resolve("THESIS_CACHE_DIR", "cache")


def output_dir() -> Path:
    """Directory holding run outputs."""
    return _resolve("THESIS_OUTPUT_DIR", "outputs")


def hf_cache_dir() -> Path:
    """HF model cache; local disk unless THESIS_HF_CACHE_DIR says otherwise."""
    val = os.environ.get("THESIS_HF_CACHE_DIR")
    if val:
        return Path(val).expanduser()
    # Drive's sync layer corrupts large .safetensors mid-download, so the
    # model cache stays off Drive even when Drive is mounted.
    if _on_colab():
        return Path("/content/hf_cache")
    return _LOCAL_FALLBACK_ROOT / "hf_cache"


def staging_dir() -> Path:
    """Scratch root on a real local filesystem, never on Drive."""
    val = os.environ.get("THESIS_STAGING_DIR")
    if val:
        return Path(val).expanduser()
    # Stores that commit by atomic rename get EPERM on the Drive FUSE
    # mount, so they build here and copy the result into the cache.
    if _on_colab():
        return Path("/content/thesis_staging")
    return _LOCAL_FALLBACK_ROOT / "staging"


def cache_dir_needs_staging() -> bool:
    """True when the cache dir sits on the Drive FUSE mount."""
    try:
        cd = cache_dir().resolve()
    except OSError:
        return False
    return cd == _DRIVE_MARKER or _DRIVE_MARKER in cd.parents


def all_paths() -> dict[str, Path]:
    """Map each role name to its resolved path."""
    return {
        "INPUT_DIR": input_dir(),
        "CACHE_DIR": cache_dir(),
        "OUTPUT_DIR": output_dir(),
        "HF_CACHE_DIR": hf_cache_dir(),
        "STAGING_DIR": staging_dir(),
    }


def ensure_all() -> dict[str, Path]:
    """Create every role directory and return the map."""
    paths = all_paths()
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def describe() -> str:
    """One line per role, for a startup banner."""
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
