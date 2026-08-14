"""Seeded subsampling, shared by every benchmark that draws a sample.

ONE DATED CONSTANT. Every pre-registered sample in this project traces to
seed 20260805, so the seed lives here rather than being re-declared per
benchmark. HotpotQA drew its 1,000-question sample under it; NarrativeQA
now draws its 40 stories under the same one.

WHY NOT `Dataset.shuffle`: the datasets library is free to change its
shuffle implementation between versions, and this project has already
been bitten by a version-sensitive algorithm whose seed did not fully pin
it (UMAP). Python's Mersenne Twister is specified and stable.

WHY INDICES ARE SORTED AFTER SAMPLING: the subsample then keeps dataset
order, so downstream structure that depends on order — HotpotQA's shard
boundaries, for instance — is a function of the SAMPLE rather than of the
order `sample()` happened to emit.
"""

from __future__ import annotations

import random


# The one dated constant. See docs/PREREGISTRATION.md.
SUBSAMPLE_SEED = 20260805


def subsample_indices(n_total: int, k: int, seed: int = SUBSAMPLE_SEED) -> list[int]:
    """Seeded random indices, sorted. Pure, so it is testable and quotable.

    Returns every index when `k >= n_total`, so a caller asking for more
    than exists gets the full set rather than an error — the sample is a
    cap, not a demand.
    """
    if k >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), k))


__all__ = ["SUBSAMPLE_SEED", "subsample_indices"]
