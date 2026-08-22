"""Does the interpreter version move RAPTOR tree TOPOLOGY?

WHY THIS IS CHEAP AND STILL DECISIVE FOR THE HOTPOTQA CELL. Tree shape at
layer 1 is fixed by clustering layer-0 leaf embeddings — UMAP, then a
BIC-selected GMM. Summary TEXT does not enter until after the clusters
exist, so it cannot change how many layer-1 nodes there are. On
HotpotQA-distractor 964 of 1,000 units stop at two layers, which means
**that cell's entire topology is decided by the layer-0 clustering
alone**. Feeding fixed synthetic vectors through the real
`perform_clustering` therefore tests the exact code path that decides the
shape, with no GPU, no model download and no summariser.

WHAT IT DOES NOT COVER, stated plainly: the leaf EMBEDDINGS themselves
come from mpnet under torch, whose wheels also differ across cp312 and
cp313. This probe holds those fixed by construction. So a MATCH here
narrows the risk to the embedder; it does not clear it. The end-to-end
check — rebuild one already-built HotpotQA unit under the other
interpreter and diff `n_nodes`, `layer_sizes`, `n_summary_nodes` — is
still the authority. Run this first because it costs seconds.

    python -m scripts.probe_python_topology            # human-readable
    python -m scripts.probe_python_topology --json     # diffable

Run under BOTH interpreters ON THE SAME HOST and diff the JSON. Same host
matters: comparing across machines reintroduces the CPU differences this
is trying to hold constant.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

from src.config import DEFAULT_CONFIG
from src.raptor_paper import PaperNode, PaperTreeParams, perform_clustering

# Deterministic stand-in for one unit's leaf embeddings. Shaped like the
# real thing rather than arbitrary: HotpotQA units carry 15-21 leaves
# (measured median 18) at the 768 dimensions mpnet emits. Several sizes
# are probed because the reduction_dimension+1 = 12 stop boundary and the
# BIC sweep's k range both depend on n, and a version difference could
# surface at one size and not another.
LEAF_COUNTS = (12, 15, 18, 21, 25, 37)
EMBED_DIM = 768
SEED = 20260805  # the project's sampling seed, reused for traceability


def synthetic_leaves(n: int, dim: int = EMBED_DIM) -> np.ndarray:
    """L2-normalised vectors from a fixed seed, in three loose clusters.

    Structure matters: uniform noise gives BIC an easy k=1 and would
    test nothing. Three overlapping blobs put the GMM near real
    decisions, which is where a last-digit float difference can flip an
    argmax.
    """
    rng = np.random.default_rng(SEED)
    centres = rng.normal(size=(3, dim)).astype(np.float32)
    rows = []
    for i in range(n):
        base = centres[i % 3]
        rows.append(base + 0.45 * rng.normal(size=dim).astype(np.float32))
    mat = np.vstack(rows).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def probe_one(n: int, params: PaperTreeParams) -> dict:
    emb = synthetic_leaves(n)
    nodes = [
        PaperNode(node_id=f"L0_{i:06d}", layer=0, text=f"leaf {i}",
                  leaf_indices=[i], embedding=emb[i])
        for i in range(n)
    ]
    stats: dict = {}
    clusters = perform_clustering(nodes, params, stats)
    sizes = sorted(len(c) for c in clusters)
    # Membership as sorted leaf-index tuples, itself sorted: identical
    # topology must give an identical structure, and comparing it catches
    # a reshuffle that preserves the SIZES.
    membership = sorted(
        tuple(sorted(int(m.node_id.split("_")[1]) for m in c))
        for c in clusters
    )
    return {
        "n_leaves": n,
        "n_clusters": len(clusters),
        "cluster_sizes": sizes,
        "membership": [list(m) for m in membership],
        "guard_trips": {k: v for k, v in sorted(stats.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from importlib.metadata import PackageNotFoundError, version

    def v(pkg: str) -> str:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "absent"

    params = DEFAULT_CONFIG.m4.paper
    out = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "packages": {p: v(p) for p in
                     ("umap-learn", "scikit-learn", "numpy", "numba",
                      "llvmlite", "pynndescent")},
        "params": {
            "reduction_dimension": params.reduction_dimension,
            "gmm_threshold": params.gmm_threshold,
            "umap_random_state": params.umap_random_state,
            "bic_random_state": params.bic_random_state,
            "gmm_random_state": params.gmm_random_state,
        },
        "results": [probe_one(n, params) for n in LEAF_COUNTS],
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    print(f"python  {out['python']}  ({out['implementation']})")
    print("stack   " + "  ".join(f"{k}={x}" for k, x in out["packages"].items()))
    print()
    print(f"{'leaves':>7}{'clusters':>10}  cluster sizes")
    print("-" * 46)
    for r in out["results"]:
        print(f"{r['n_leaves']:>7}{r['n_clusters']:>10}  {r['cluster_sizes']}")
    trips = {k: sum(r["guard_trips"].get(k, 0) for r in out["results"])
             for r in out["results"] for k in r["guard_trips"]}
    if trips:
        print(f"\nguard trips across all sizes: {trips}")
    print("\nRun under the OTHER interpreter on THIS host and diff --json.")
    print("Identical n_clusters AND membership at every size = the")
    print("clustering path is interpreter-stable; the residual risk is")
    print("then the EMBEDDER, which this probe holds fixed.")


if __name__ == "__main__":
    main()
