"""P9: write and verify the environment lockfile that M4's topology needs.

WHY A LOCKFILE AT ALL. UMAP + GMM output is version-sensitive even when
seeded, so an M4 tree is reproducible against a PINNED stack rather than
absolutely. Without a lockfile "reproducible" is a claim nobody can
check; with one it is a command.

WHY THIS IS WRITTEN ON THE RUN HOST, NOT COMMITTED BY HAND. The stack
that matters is the one the matrix runs on — Colab — and a lockfile
transcribed from a developer laptop would pin the wrong versions with
full confidence. `write` snapshots the environment it is executed in;
`check` compares a later environment against that snapshot.

THE ACCEPTANCE TEST IS OPERATOR-EXECUTED, not agent-verified: build a
fresh environment from the lockfile, on the SAME GPU CLASS, and confirm
one M4 unit's tree node count reproduces exactly. It cannot be run from a
machine without the GPU, and it is recorded as an operator line for the
same reason the tree-cache preflight is.

    python -m scripts.pin_environment write --out requirements.lock
    python -m scripts.pin_environment check --lockfile requirements.lock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# Pinned exactly, because they determine M4 tree topology or the numbers
# that flow from it. A drift here invalidates trees; a drift elsewhere
# does not, which is why the cold-tree cache lever keys on the first
# three of these rather than on the whole file.
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
    from importlib.metadata import distributions

    out: dict[str, str] = {}
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip()
        if name:
            out[name.lower()] = dist.version
    return out


def lockfile_hash(text: str) -> str:
    """Hash of the REQUIREMENT LINES only, so a comment or a reordering
    does not read as an environment change."""
    lines = sorted(
        ln.strip() for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def gpu_model() -> str:
    """The GPU string, recorded because the reproducibility target is
    same-lockfile-SAME-GPU-CLASS. A tree that reproduces on an L4 is not
    thereby claimed to reproduce on an A100."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
        return "cpu"
    except Exception:
        return "unknown"


def write_lockfile(out: Path) -> int:
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
    """The interpreter version `write_lockfile` recorded, or None.

    IT WAS ALWAYS IN THE FILE. `write_lockfile` has always emitted
    `# python=3.12.13`, and NOTHING EVER READ IT: `lockfile_hash` strips
    comments by design (so a comment cannot read as an environment
    change) and `check_lockfile` skipped every line starting with `#`.
    A value that was written correctly and consumed by nothing — the
    fourteenth instance of this project's recurring defect, and the one
    that let cells 1-5 run on CPython 3.12.13 and cell 6 on 3.13.15 with
    the pin reporting OK both times.

    Parsed from the comment RATHER THAN promoted to a requirement line,
    deliberately: a `python==3.12.13` line would enter `lockfile_hash`
    and move it off `17878bc8740173be`, firing the environment gate on
    every remaining cell and orphaning the hash recorded in six banked
    summaries. The data is already on disk; only the reader was missing.
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") and "python=" in line:
            return line.split("python=", 1)[1].strip() or None
    return None


def check_lockfile(lockfile: Path) -> int:
    text = lockfile.read_text(encoding="utf-8")
    installed = _installed()
    mismatches: list[str] = []
    checked = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        pkg, _, want = line.partition("==")
        have = installed.get(pkg.strip().lower())
        checked += 1
        if have is None:
            mismatches.append(f"{pkg}: locked {want}, ABSENT")
        elif have != want:
            mismatches.append(f"{pkg}: locked {want}, installed {have}")

    # THE INTERPRETER IS A PINNED INPUT. Identical package VERSIONS do
    # not mean identical package CODE: cp312 and cp313 install different
    # compiled wheels for numpy, numba, llvmlite and torch, and UMAP's
    # JIT-compiled paths are exactly where a float can move in its last
    # digits and flip a GMM argmax on a near-tie. Counted with the
    # packages so `checked N pinned` is honest about what was checked.
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
    """The block every run summary carries.

    Recorded per CELL rather than per session: a matrix assembled from
    several sessions must be able to say which environment produced which
    row, and a session-level note cannot.
    """
    lock_hash = None
    if lockfile is not None and Path(lockfile).exists():
        lock_hash = lockfile_hash(Path(lockfile).read_text(encoding="utf-8"))
    else:
        # LOUD, because a null hash is not a missing nicety: it means this
        # cell's summary records NOTHING about the environment that
        # produced it, and M4's substrate key folds three of those
        # versions. Every probe run in the 2026-08-16 cost investigation
        # carried lockfile_hash: null and nobody noticed, because it was
        # written in silence.
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
        # RECORDED so it cannot go inert unnoticed again. This setting was
        # correct, was applied on the generation path only, and was
        # therefore read after the embedder had already initialised the
        # allocator — instance eleven. A value that is set and never
        # asserted is not a check.
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
