"""Archived retrievers — preserved code, not part of the active roster.

Modules in this package were built and verified at some point but are no
longer registered in `smoke_test/run_smoke.py` or the active evaluation
grid. They are kept on disk for thesis discussion, potential
resurrection, and historical reference.

NOTHING in this package is re-exported here intentionally — importing
`src.retrievers.deprecated` does not activate any archived system. See
`README.md` in this directory for why each module was archived.

Imports inside the archived modules may not resolve cleanly (e.g.
`src.graphrag_backend` is now `src.retrievers.deprecated.graphrag_backend`,
and `M5Config` / `M8Config` live in `_archived_config.py` here rather
than in `src.config`). They are preserved as-was; fix imports only on
resurrection, not eagerly.
"""
