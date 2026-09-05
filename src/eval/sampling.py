"""Seeded subsampling shared by every benchmark that draws a sample."""

from __future__ import annotations

import random


# harness choice: preregistered seed (METHODS §B)
SUBSAMPLE_SEED = 20260805


def subsample_indices(n_total: int, k: int, seed: int = SUBSAMPLE_SEED) -> list[int]:
    """Draw k seeded random indices out of n_total and return them sorted."""
    # The sample is a cap, not a demand: k >= n_total returns every index.
    if k >= n_total:
        return list(range(n_total))
    # random.Random is a specified, stable generator; sorting keeps dataset
    # order, so HotpotQA shard boundaries depend on the sample alone.
    return sorted(random.Random(seed).sample(range(n_total), k))


__all__ = ["SUBSAMPLE_SEED", "subsample_indices"]
