"""Guard (v): a GMM fit that raises must not kill the tree build.

MEASURED, not anticipated. At production PaperTreeParams a layer of 16
nodes and a layer of 25 nodes both died with sklearn's "ill-defined
empirical covariance", while 20, 30 and 40 survived. The BIC sweep tries
k up to n-1, and UMAP's reduction of few points into 10 components leaves
tight local clumps for a high-k component to collapse onto.

Two properties make this worth a guard rather than a retry-and-hope:

  * it is NOT monotone in n, so a build passing at one corpus size proves
    nothing about another;
  * a raise loses the ENTIRE tree, which is the most expensive artifact
    in the harness.

Most trees hit a 12-30 node layer near the top, so this is not a
HotpotQA-only concern — it is every benchmark.

The failure is stubbed rather than reproduced from real UMAP output: the
guard's contract is "skip what cannot be fitted, keep what can, count
it", and that is what these pin. A test that depended on sklearn's exact
numerics at a particular UMAP version would be testing the library.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.raptor_paper import (
    PaperTreeParams,
    _get_optimal_clusters,
    _gmm_cluster,
)


class _StubGMM:
    """GaussianMixture stand-in that raises for a chosen set of k.

    Distinguishes the BIC sweep from the final fit by SEED, exactly as
    the code under test does: the sweep runs at bic_random_state=224 and
    the final fit at gmm_random_state=0. That seed mismatch is the
    reference's own (ruling 3), and it is precisely why a k the sweep
    fitted can still fail in the final fit — so the two failure sets have
    to be settable independently or the walk-down cannot be tested.
    """

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
    GaussianMixture = _StubGMM


def _install_stub(fail_for, bic_by_k=None, fail_final_for=()):
    import sys

    _StubGMM.fail_for = set(fail_for)
    _StubGMM.fail_final_for = set(fail_final_for)
    _StubGMM.bic_by_k = dict(bic_by_k or {})
    saved = sys.modules.get("sklearn.mixture")
    sys.modules["sklearn.mixture"] = _StubMixture
    return saved


def _restore(saved):
    import sys

    if saved is None:
        sys.modules.pop("sklearn.mixture", None)
    else:
        sys.modules["sklearn.mixture"] = saved


class _StubCase(unittest.TestCase):
    def setUp(self):
        self.params = PaperTreeParams()
        self.X = np.zeros((12, 10), dtype=np.float32)
        self._saved = None

    def tearDown(self):
        _restore(self._saved)


class TestBicSweepSkipsFailingK(_StubCase):
    def test_a_failing_k_is_skipped_and_counted(self):
        # k=3 would have won on BIC, but cannot be fitted.
        self._saved = _install_stub(
            fail_for={3}, bic_by_k={2: 50.0, 3: 1.0, 4: 60.0}
        )
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertNotEqual(k, 3)
        self.assertEqual(k, 2, "should pick the best k that CAN be fitted")
        self.assertEqual(stats["bic_fit_failures"], 1)

    def test_the_build_survives_when_most_k_fail(self):
        self._saved = _install_stub(fail_for=set(range(2, 12)))
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertEqual(k, 1)
        self.assertEqual(stats["bic_fit_failures"], 10)

    def test_all_failing_falls_back_to_one_cluster(self):
        self._saved = _install_stub(fail_for=set(range(1, 12)))
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertEqual(k, 1)
        self.assertEqual(stats["bic_all_fits_failed"], 1)

    def test_no_failures_leaves_no_trip_counters(self):
        """The guard must be inert on a healthy layer, or a non-zero
        counter stops meaning anything."""
        self._saved = _install_stub(fail_for=set(), bic_by_k={5: -1.0})
        stats: dict = {}
        k = _get_optimal_clusters(self.X, self.params, stats)
        self.assertEqual(k, 5)
        self.assertNotIn("bic_fit_failures", stats)
        self.assertNotIn("bic_all_fits_failed", stats)


class TestFinalFitWalksDown(_StubCase):
    def test_final_fit_failure_steps_down_instead_of_collapsing_to_one(self):
        """The seed mismatch (224 for the sweep, 0 for the final fit) means
        the final fit can fail at a k the sweep accepted. Falling straight
        to k=1 would turn a splittable layer into a single parent."""
        # Sweep (seed 224) fits everything and picks 6; the final fit
        # (seed 0) fails at 6 and 5.
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
    def test_counters_reach_the_index_stats(self):
        """A reduced BIC candidate set is a FINDING about the clustering.
        If it never leaves the tree's stats dict it cannot be reported."""
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
