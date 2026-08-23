"""The per-row cell differ (scripts/diff_cell_rows) and the packed_ids field.

The comparison's value is entirely in its implications being SOUND: a
"set differs" verdict must follow from a signal that cannot be equal under
identical sets, and an "order-only candidate" must require every set
signal to match. A differ whose buckets overlap or leak would put numbers
on F-X4 that mean nothing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diff_cell_rows import compare_rows


def _row(*, ev=100, npk=15, f1=0.5, ans="Paris", ids=None):
    r = {
        "evidence_tokens": ev,
        "n_packed": npk,
        "retrieval": {"f1": f1, "recall": f1, "precision": f1},
        "predicted_answer": ans,
        "metadata": {},
    }
    if ids is not None:
        r["metadata"]["packed_ids"] = list(ids)
    return r


class TestBounds(unittest.TestCase):
    def test_identical_rows_raise_no_flags(self):
        r = compare_rows(_row(), _row())
        self.assertFalse(r["set_signals_differ"])
        self.assertFalse(r["answer_differs"])
        self.assertFalse(r["order_only_candidate"])
        self.assertIsNone(r["exact"])

    def test_token_total_difference_proves_a_set_difference(self):
        r = compare_rows(_row(ev=100), _row(ev=101))
        self.assertTrue(r["set_signals_differ"])

    def test_prf_difference_proves_a_set_difference(self):
        r = compare_rows(_row(f1=0.5), _row(f1=0.51))
        self.assertTrue(r["set_signals_differ"])

    def test_order_only_candidate_requires_all_set_signals_equal(self):
        """The F-X4 mechanism bucket: answer moved, set signals did not."""
        r = compare_rows(_row(ans="Paris"), _row(ans="Lyon"))
        self.assertTrue(r["order_only_candidate"])
        # and it must NOT fire when a set signal also moved
        r2 = compare_rows(_row(ans="Paris", ev=100), _row(ans="Lyon", ev=90))
        self.assertFalse(r2["order_only_candidate"])


class TestExactWithIds(unittest.TestCase):
    def test_same_set_same_order_is_clean(self):
        r = compare_rows(_row(ids=["a", "b"]), _row(ids=["a", "b"]))
        self.assertEqual(r["exact"],
                         {"set_differs": False, "order_differs": False})

    def test_same_set_different_order_is_order_not_set(self):
        r = compare_rows(_row(ids=["a", "b"]), _row(ids=["b", "a"]))
        self.assertEqual(r["exact"],
                         {"set_differs": False, "order_differs": True})

    def test_different_set_is_set_not_order(self):
        r = compare_rows(_row(ids=["a", "b"]), _row(ids=["a", "c"]))
        self.assertEqual(r["exact"],
                         {"set_differs": True, "order_differs": False})

    def test_one_sided_ids_fall_back_to_bounds(self):
        r = compare_rows(_row(ids=["a"]), _row())
        self.assertIsNone(r["exact"])


if __name__ == "__main__":
    unittest.main()
