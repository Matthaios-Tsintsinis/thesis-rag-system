"""Write and check the lockfile that pins the M4 topology stack.
    python -m scripts.pin_environment write --out requirements.lock
    python -m scripts.pin_environment check --lockfile requirements.lock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# Packages pinned exactly: they decide M4 tree topology or the numbers
# that follow from it. The M4 substrate cache key folds the first three.
TOPOLOGY_CRITICAL = (
    "umap-learn",
    "scikit-learn",
    "numpy",
    "numba",
    "llvmlite",
    "pynndescent",
    "torch",
    "transformers",
    "sentence-transformers",
    "faiss-cpu",
    "faiss-gpu",
    "rank-bm25",
    "tiktoken",
    "datasets",
    "huggingface-hub",
)


def _installed() -> dict[str, str]:
    """Map every installed distribution's lower-cased name to its version."""
    from importlib.metadata import distributions

    out: dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip()
        if name:
            out[name.lower()] = dist.version
    return out


def lockfile_hash(text: str) -> str:
    """Hash the requirement lines only, so comments and order do not count."""
    lines = sorted(
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def gpu_model() -> str:
    """Name the GPU; a tree reproduces on the same lockfile and GPU class."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
        return "cpu"
    except Exception:
        return "unknown"


def write_lockfile(out: Path) -> int:
    """Snapshot this environment's pinned packages into a lockfile."""
    installed = _installed()
    lines = [
        "# Environment lock for the thesis RAG matrix.",
        "# Written by scripts/pin_environment.py ON THE RUN HOST.",
        "# M4 tree topology is reproducible against THIS stack, on the",
        "# same GPU class, and is not claimed to reproduce against another.",
        f"# gpu={gpu_model()}",
        f"# python={sys.version.split()[0]}",
        "",
    ]
    # One requirement line per pinned package; absent ones go in a trailer.
    missing: list[str] = []
    for pkg in TOPOLOGY_CRITICAL:
        version = installed.get(pkg)
        if version is None:
            missing.append(pkg)
            continue
        lines.append(f"{pkg}=={version}")
    if missing:
        lines.append("")
        lines.append("# absent in the environment that wrote this file:")
        lines.extend(f"#   {pkg}" for pkg in missing)

    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(f"[pin] lockfile_hash={lockfile_hash(text)}")
    print(f"[pin] gpu={gpu_model()}")
    if missing:
        print(f"[pin] NOTE: {len(missing)} package(s) absent here: "
              f"{', '.join(missing)}")
    return 0


def locked_python(text: str) -> str | None:
    """Read the interpreter version from the lockfile's python= comment."""
    # The version lives in a comment, not a requirement line, so it does
    # not enter lockfile_hash.
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") and "python=" in line:
            return line.split("python=", 1)[1].strip() or None
    return None


def _pip_check_conflicts() -> list[str]:
    """Return pip check's conflict lines for this interpreter, [] if clean."""
    import subprocess

    # A pip check that cannot run degrades to a warning, never a crash.
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:  # pip missing, timeout, sandboxing
        print(f"[pin] WARN: could not run pip check ({type(e).__name__}: "
              f"{e}); dependency conflicts were NOT screened")
        return []
    if out.returncode == 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _canon(name: str) -> str:
    """Fold case and -/_ so two spellings of one package name compare equal."""
    return name.strip().lower().replace("_", "-")


def _line_package_tokens(line: str) -> set[str]:
    """Collect whole tokens in a pip-check line that could name a package."""
    import re

    # Tokens start with a letter, so bare versions drop out; English words
    # ride along and collide with no locked name.
    return {_canon(t) for t in re.findall(r"[A-Za-z][A-Za-z0-9._-]*", line)}


def classify_conflicts(
    conflicts: list[str], locked_names: list[str]
) -> tuple[list[str], list[str]]:
    """Split pip-check lines into (naming a locked package, warn-only)."""
    # Match on whole canonical tokens, not substrings: "torchvision" must
    # not match locked "torch", and "rank_bm25" must match "rank-bm25".
    # A conflict that names no locked package is noise from the host's
    # preinstalled packages and only warns.
    failing: list[str] = []
    warn: list[str] = []
    names = {_canon(n) for n in locked_names}
    for line in conflicts:
        if _line_package_tokens(line) & names:
            failing.append(line)
        else:
            warn.append(line)
    return failing, warn


def check_lockfile(lockfile: Path) -> int:
    """Compare this environment to a lockfile; 0 on match, 1 on mismatch."""
    text = lockfile.read_text(encoding="utf-8")
    installed = _installed()
    mismatches: list[str] = []
    checked = 0
    locked_names: list[str] = []
    # Every requirement line must be installed at exactly the locked version.
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        pkg, _, want = line.partition("==")
        locked_names.append(pkg.strip())
        have = installed.get(pkg.strip().lower())
        checked += 1
        if have is None:
            mismatches.append(f"{pkg}: locked {want}, ABSENT")
        elif have != want:
            mismatches.append(f"{pkg}: locked {want}, installed {have}")

    # The interpreter is a pinned input too: cp312 and cp313 install
    # different compiled wheels, and UMAP's JIT paths can move a float's
    # last digits and flip a GMM argmax. It counts toward `checked`.
    want_py = locked_python(text)
    have_py = sys.version.split()[0]
    if want_py is None:
        print("[pin] WARN: this lockfile predates the python check and "
              "records no interpreter; regenerate it to gain the guard")
    else:
        checked += 1
        if have_py != want_py:
            mismatches.append(
                f"python: locked {want_py}, running {have_py} "
                "(different compiled wheels; M4 topology is not claimed "
                "to reproduce across interpreters)")

    # Screen pip check after the version loop so the locked-name list is
    # complete; a conflict naming a locked package fails the pin.
    failing, warn_only = classify_conflicts(_pip_check_conflicts(),
                                            locked_names)
    for line in failing:
        mismatches.append(f"pip check: {line}")
    for line in warn_only:
        print(f"[pin] WARN (pip check, unlocked packages): {line}")

    # Report, then fail on any mismatch.
    print(f"[pin] lockfile_hash={lockfile_hash(text)}")
    print(f"[pin] python={have_py} (locked {want_py or 'UNRECORDED'})")
    print(f"[pin] gpu={gpu_model()}")
    print(f"[pin] checked {checked} pinned package(s)")
    if mismatches:
        print("[pin] MISMATCH:")
        for m in mismatches:
            print(f"  - {m}")
        print("[pin] FAILED — this environment is not the locked one. M4 "
              "tree topology is not claimed to reproduce here.")
        return 1
    print("[pin] OK — environment matches the lockfile.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch the write, check and json subcommands."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="snapshot THIS environment")
    w.add_argument("--out", type=Path, default=Path("requirements.lock"))

    c = sub.add_parser("check", help="compare this environment to a lockfile")
    c.add_argument("--lockfile", type=Path, default=Path("requirements.lock"))

    j = sub.add_parser("json", help="emit the provenance block the runner records")
    j.add_argument("--lockfile", type=Path, default=Path("requirements.lock"))

    args = ap.parse_args(argv)
    if args.cmd == "write":
        return write_lockfile(args.out)
    if args.cmd == "check":
        return check_lockfile(args.lockfile)
    print(json.dumps(environment_provenance(args.lockfile), indent=2))
    return 0


def environment_provenance(lockfile: Path | None = None) -> dict:
    """Build the environment block every cell summary carries."""
    # Hash the lockfile when present; warn loudly when it is not, because
    # a null hash leaves the cell with no record of its environment.
    lock_hash = None
    if lockfile is not None and Path(lockfile).exists():
        lock_hash = lockfile_hash(Path(lockfile).read_text(encoding="utf-8"))
    else:
        print(
            f"[pin] WARNING: lockfile_hash=null — no lockfile at "
            f"{lockfile!r}. This run's provenance records NO environment "
            "pin. M4 tree topology is not claimed to reproduce from it. "
            "Generate one with: python -m scripts.pin_environment write"
        )
    installed = _installed()
    return {
        "lockfile_hash": lock_hash,
        "gpu": gpu_model(),
        # Recorded so the allocator setting the run started under is visible.
        "cuda_alloc_conf": __import__("os").environ.get(
            "PYTORCH_CUDA_ALLOC_CONF"),
        "python": sys.version.split()[0],
        "versions": {
            pkg: installed.get(pkg) for pkg in TOPOLOGY_CRITICAL
            if installed.get(pkg) is not None
        },
    }


if __name__ == "__main__":
    sys.exit(main())
