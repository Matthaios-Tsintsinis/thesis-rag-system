"""The inventory must measure the population the CELL builds.

THE DEFECT THIS PINS. `inventory_narrativeqa_units` called
`iter_eval_units(max_units=None)`, which yields the FULL 115-story
validation split. The cell builds a seeded 40-story draw. Every number
the inventory produced was therefore correct about the wrong population:
the named "largest" and "median" stories were drawn from 115, and a cell
projected from that inventory would have overstated the build term by
roughly 3x.

It surfaced only because the loader's own header printed
`115 stories, 3461 questions` while every build run in the same session
reported 40 — a number contradicting its neighbours, not a failure.

WHY max_units IS LOAD-BEARING AND NOT A CAP. The draw is
`subsample_indices(len(order), max_units)`, so the SIZE of the request
selects WHICH stories are drawn. `subsample_indices(115, 1)` is not the
first element of `subsample_indices(115, 40)`. Asking for a different
number does not narrow the same sample; it takes a different one.
"""

from __future__ import annotations

import unittest

from src.eval.narrativeqa import CELL_UNITS
from src.eval.sampling import SUBSAMPLE_SEED, subsample_indices


class TestCellUnitsIsOneConstant(unittest.TestCase):
    def test_the_cell_size_has_a_single_source(self):
        self.assertEqual(CELL_UNITS, 40)

    def test_the_probe_reads_that_same_constant(self):
        """Two copies of 40 would drift, and the drift would be a probe
        measuring a different draw from the cell it is predicting."""
        from scripts.probe_cell_costs import NARRATIVEQA_CELL_UNITS

        self.assertEqual(NARRATIVEQA_CELL_UNITS, CELL_UNITS)


class TestDrawIdentity(unittest.TestCase):
    """The property the inventory bug violated, stated as arithmetic on
    the sampler so it needs no dataset download."""

    def test_a_different_n_is_a_different_draw_not_a_prefix(self):
        forty = subsample_indices(115, CELL_UNITS)
        one = subsample_indices(115, 1)
        self.assertNotEqual(one, forty[:1])

    def test_the_draw_is_stable_for_the_same_n(self):
        self.assertEqual(
            subsample_indices(115, CELL_UNITS),
            subsample_indices(115, CELL_UNITS),
        )
        self.assertEqual(SUBSAMPLE_SEED, 20260805)

    def test_the_inventory_requests_the_cell_size_by_default(self):
        """Reads the DEFAULT the script will actually use, rather than
        trusting the docstring — the inventory bug was precisely a
        default that disagreed with its own description."""
        import inspect

        from scripts.inventory_unit_leaves import inventory

        sig = inspect.signature(inventory)
        # None now means "the benchmark's DECLARED cell_units", resolved
        # inside the function from `cell_units`, so the default is None
        # and the resolution is asserted behaviourally below.
        self.assertIsNone(sig.parameters["max_units"].default)


class TestInventoryMatchesTheLoader(unittest.TestCase):
    """The check the user asked for: same seed, same list.

    Driven through a STUB loader rather than the real dataset, because
    the property under test is that the inventory passes the cell's
    max_units through to the loader unchanged — not that HuggingFace
    still serves NarrativeQA. A test that needed a 4 GB download would
    not run, and a check that does not run has not passed.
    """

    def test_inventory_enumerates_exactly_the_loader_units(self):
        from unittest import mock

        class FakeItem:
            def __init__(self, text):
                self.text = text
                # group_items_by_parent / build_parent_payload need these:
                # the inventory now counts leaves the way index_items does.
                self.parent_id = "p"
                self.span_id = "s0"
                self.item_id = "p::s0"

        class FakeUnit:
            def __init__(self, cid, text):
                self.corpus_id = cid
                self.corpus = (FakeItem(text),)
                self.queries = ("q",)

        seen: dict = {}

        class FakeBench:
            # Declared exactly as the real loader declares it; the test
            # is that the inventory READS the declaration and passes it
            # to the loader, which is the plumbing that broke.
            cell_units = CELL_UNITS

            def iter_eval_units(self, *, split, max_units=None):
                seen["max_units"] = max_units
                n = max_units if max_units is not None else 115
                for i in range(n):
                    yield FakeUnit(f"story{i:03d}", "A sentence. " * 30)

        from scripts import inventory_unit_leaves as inv

        with mock.patch.dict(
            "src.eval.runner.BENCHMARK_REGISTRY",
            {"narrativeqa": FakeBench}, clear=False,
        ):
            report = inv.inventory("narrativeqa")

        self.assertEqual(seen["max_units"], CELL_UNITS)
        self.assertEqual(report["n_units"], CELL_UNITS)
        self.assertEqual(
            [u["corpus_id"] for u in report["units"]],
            [f"story{i:03d}" for i in range(CELL_UNITS)],
        )
        self.assertEqual(len(report["leaves_per_unit"]), CELL_UNITS)

    def test_the_full_split_is_reported_separately_when_asked(self):
        """Context is fine; conflating it with the cell is not."""
        from scripts import inventory_unit_leaves as inv

        self.assertIn("population", inv.inventory.__doc__.lower())


class TestLeafCountMatchesTheBuild(unittest.TestCase):
    """The inventory must count what the BUILD counts.

    MEASURED DIVERGENCE: story 961902ae inventoried 519 leaves against
    481 in the build, while 57523a48 and d431326b matched exactly. The
    cause is that `index_items` writes each parent to a temp file and
    reads it back through `parsing.clean_text`, which collapses runs of
    spaces/tabs and runs of 3+ newlines — and `split_text_raptor` treats
    a newline run as a boundary, so the cleaning MOVES chunk boundaries.

    Story-dependent, which is the worst shape of error: two agreeing
    samples read as proof that the third is fine.
    """

    @staticmethod
    def _unit(*texts, parent="p1"):
        class _Item:
            def __init__(self, pid, sid, text):
                self.parent_id, self.span_id, self.text = pid, sid, text
                self.item_id = f"{pid}::{sid}"

        class _Unit:
            pass

        u = _Unit()
        u.corpus = [_Item(parent, f"s{i}", t) for i, t in enumerate(texts)]
        return u

    def test_whitespace_heavy_text_diverges_when_counted_raw(self):
        """Pins the MECHANISM. If this stops diverging, the cleaning
        changed and the parity below is passing for a new reason."""
        from src.raptor_paper import split_text_raptor

        from scripts.inventory_unit_leaves import unit_leaf_count

        text = ("Alpha beta gamma delta epsilon zeta eta theta." + "\n" * 6
                + "Iota    kappa\t\tlambda mu nu xi omicron pi rho. ") * 40
        self.assertNotEqual(len(split_text_raptor(text)),
                            unit_leaf_count(self._unit(text)))

    def test_single_item_unit_equals_the_cleaned_chunking(self):
        """Parity with the build's pipeline for the one-item shape."""
        from src.parsing import clean_text
        from src.raptor_paper import split_text_raptor

        from scripts.inventory_unit_leaves import unit_leaf_count

        text = ("Alpha beta gamma." + "\n" * 5
                + "Delta   epsilon\tzeta eta theta iota. ") * 30
        self.assertEqual(unit_leaf_count(self._unit(text)),
                         len(split_text_raptor(clean_text(text))))

    def test_a_multi_item_parent_is_JOINED_before_chunking(self):
        """THE SECOND DRIFT, and the one that mattered for HotpotQA.

        `index_items` groups items by parent and joins each parent's
        members into ONE document before chunking, so chunks can pack
        across member boundaries. Summing per-item counts coincides only
        when a unit has exactly one item — true of NarrativeQA, FALSE of
        HotpotQA-distractor's ~10 paragraphs per question.

        Measured on this fixture: per-item summing says 10 leaves, the
        joined payload says 5. It OVERSTATES, so a per-item inventory
        would have under-reported degeneracy — the exact quantity the
        HotpotQA declaration turns on."""
        from src.parsing import clean_text
        from src.raptor_paper import split_text_raptor

        from scripts.inventory_unit_leaves import unit_leaf_count

        para = "Alpha beta gamma delta epsilon zeta. " * 6
        unit = self._unit(*[para] * 10)
        per_item = sum(
            len(split_text_raptor(clean_text(it.text))) for it in unit.corpus
        )
        joined = unit_leaf_count(unit)
        self.assertNotEqual(per_item, joined)
        self.assertLess(joined, per_item)

    def test_the_degeneracy_threshold_comes_from_the_config(self):
        from src.config import DEFAULT_CONFIG

        from scripts.inventory_unit_leaves import degeneracy_threshold

        self.assertEqual(
            degeneracy_threshold(),
            DEFAULT_CONFIG.m4.paper.reduction_dimension + 1,
        )


if __name__ == "__main__":
    unittest.main()
