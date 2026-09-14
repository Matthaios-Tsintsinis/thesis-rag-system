"""Pins the GMM fit guard: a fit that raises is skipped and counted.

The failure is stubbed, so the tests pin the guard's contract and not
sklearn's numerics.
"""
# deviation from ref (ref crashes): see METHODS §A.4.4 (v)

from __future__ import annotations

import unittest

import numpy as np

from src.raptor_paper import (
    PaperTreeParams,
    _get_optimal_clusters,
    _gmm_cluster,
)


class _StubGMM:
    """GaussianMixture stand-in that raises for a chosen set of k."""

    # The sweep fits at seed 224 and the final fit at seed 0, so the two
    # failure sets are told apart by random_state.
    # ref: raptor/cluster_utils.py @ 7da1d48a (RANDOM_SEED = 224)
    # ref: raptor/cluster_utils.py::GMM_cluster @ 7da1d48a (random_state=0; the ref uses both seeds)
    fail_for: set[int] = set()          # any fit
    fail_final_for: set[int] = set()    # final fit only (seed 0)
    bic_by_k: dict[int, float] = {}

    def __init__(self, n_components: int, random_state: int = 0, **kw) -> None:
        self.n_components = int(n_components)
        self.random_state = random_state

    def fit(self, X):  # noqa: D102
        final = self.random_state == 0
        if self.n_components in self.fail_for or (
            final and self.n_components in self.fail_final_for
        ):
            raise ValueError(
                "Fitting the mixture model failed because some components "
                "have ill-defined empirical covariance"
            )
        return self

    def bic(self, X):  # noqa: D102
        return self.bic_by_k.get(self.n_components, 100.0 - self.n_components)

    def predict_proba(self, X):  # noqa: D102
        n = len(X)
        out = np.zeros((n, self.n_components), dtype=float)
        for i in range(n):
            out[i, i % self.n_components] = 1.0
        return out


class _StubMixture:
    """Module stand-in for sklearn.mixture."""

    GaussianMixture = _StubGMM


def _install_stub(fail_for, bic_by_k=None, fail_final_for=()):
    """Install the stub mixture module and return what it replaced."""
    import sys

    _StubGMM.fail_for = set(fail_for)
    _StubGMM.fail_final_for = set(fail_final_for)
    _StubGMM.bic_by_k = dict(bic_by_k or {})
    saved = sys.modules.get("sklearn.mixture")
    sys.modules["sklearn.mixture"] = _StubMixture
    return saved


def _restore(saved):
    """Put the saved sklearn.mixture module back."""
    import sys

    if saved is None:
        sys.modules.pop("sklearn.mixture", None)
    else:
        sys.modules["sklearn.mixture"] = saved


class _StubCase(unittest.TestCase):
    """Shared params, a 12-point layer and stub cleanup."""

    def setUp(self):
        self.params = PaperTreeParams()
        self.X = np.zeros((12, 10), dtype=np.float32)
        self._saved = None

    def tearDown(self):
        _restore(self._saved)


class TestBicSweepSkipsFailingK(_StubCase):
    """The BIC sweep skips a k that cannot be fitted."""

    def test_a_failing_k_is_skipped_and_counted(self):
        """The best BIC k is skipped when its fit raises, and counted."""
        # k=3 wins on BIC but cannot be fitted.
        self._saved = _install_stub(
            fail_for={3}, bic_by_k={2: 50.0, 3: 1.0, 4: 60.0}
        )
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertNotEqual(k, 3)
        self.assertEqual(k, 2, "should pick the best k that CAN be fitted")
        self.assertEqual(stats["bic_fit_failures"], 1)

    def test_the_build_survives_when_most_k_fail(self):
        """The sweep falls back to k=1 when every k above 1 fails."""
        self._saved = _install_stub(fail_for=set(range(2, 12)))
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertEqual(k, 1)
        self.assertEqual(stats["bic_fit_failures"], 10)

    def test_all_failing_falls_back_to_one_cluster(self):
        """The sweep returns k=1 and sets bic_all_fits_failed when all fail."""
        self._saved = _install_stub(fail_for=set(range(1, 12)))
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertEqual(k, 1)
        self.assertEqual(stats["bic_all_fits_failed"], 1)

    def test_no_failures_leaves_no_trip_counters(self):
        """A healthy layer writes no failure counters."""
        self._saved = _install_stub(fail_for=set(), bic_by_k={5: -1.0})
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertEqual(k, 5)
        self.assertNotIn("bic_fit_failures", stats)
        self.assertNotIn("bic_all_fits_failed", stats)


class TestFinalFitWalksDown(_StubCase):
    """The final fit walks k down one step at a time when it raises."""

    def test_final_fit_failure_steps_down_instead_of_collapsing_to_one(self):
        """A final fit that fails at the sweep's k steps down, not to 1."""
        # The sweep fits everything and picks 6; the final fit fails at 6
        # and 5.
        self._saved = _install_stub(
            fail_for=set(), bic_by_k={6: -100.0}, fail_final_for={6, 5}
        )
        stats: dict = {}
        self.assertEqual(_get_optimal_clusters(self.X, self.params, {}), 6)
        labels, k = _gmm_cluster(self.X, self.params, stats)
        self.assertEqual(k, 4, "should step 6 -> 5 -> 4, not 6 -> 1")
        self.assertEqual(stats["gmm_final_fit_failures"], 2)
        self.assertEqual(len(labels), len(self.X))

    def test_labels_are_still_well_formed_after_stepping_down(self):
        """Labels after a step-down are non-empty and below the final k."""
        self._saved = _install_stub(
            fail_for=set(), bic_by_k={4: -100.0}, fail_final_for={4}
        )
        stats: dict = {}
        labels, k = _gmm_cluster(self.X, self.params, stats)
        self.assertEqual(k, 3)
        for lab in labels:
            self.assertTrue(len(lab) >= 1)
            for v in lab.tolist():
                self.assertLess(v, k)


class TestGuardIsReportedNotSilenced(unittest.TestCase):
    """The guard's counters reach the reported index stats."""

    def test_counters_reach_the_index_stats(self):
        """Tree fit-failure counters are copied into the index stats."""
        from src.retrievers.m4_raptor import RaptorSystem
        from src.raptor_paper import PaperCollapsedIndex, PaperNode, PaperTree

        sysm = RaptorSystem()
        node = PaperNode(node_id="L0_000000", layer=0, text="x", leaf_indices=[0])
        sysm._tree = PaperTree(
            nodes={node.node_id: node}, layer_to_nodes={0: [node.node_id]},
            n_layers=1, params=PaperTreeParams(),
            stats={"bic_fit_failures": 7, "gmm_final_fit_failures": 2},
        )
        sysm._flat = PaperCollapsedIndex(
            faiss_index=None,
            refs=[{"node_id": "L0_000000", "layer": 0, "is_leaf": True}],
            dim=4,
        )
        stats = sysm._collect_index_stats()
        self.assertEqual(stats["bic_fit_failures"], 7)
        self.assertEqual(stats["gmm_final_fit_failures"], 2)


if __name__ == "__main__":
    unittest.main()
