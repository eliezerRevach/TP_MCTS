"""Tests for the strict-bound rational DBM (``strict_dbm``).

Every case here is taken from ``artifacts/Temporal_STN_PDB_Prototype.docx``:
section 3's five endpoint placements and section 8's two numerical checks. The
spec states the expected answers, so these are spec-conformance tests rather
than tests of whatever the implementation happens to do.
"""

import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.strict_dbm import (  # noqa: E402
    INF,
    StrictDBM,
    bound_add,
    bound_is_negative,
    bound_tighter,
)


def _start_action(z, label, start, end, duration, after, deadline):
    """Spec section 2: introduce S_a, E_a with E_a - S_a = d(a), C <= S_a, E_a <= D."""
    z.add_var(start)
    z.add_var(end)
    ok = z.add_eq(end, start, duration)
    ok = z.add_le(after, start) and ok          # C <= S_a
    ok = z.add_le(end, z.origin, deadline) and ok   # E_a <= D
    ok = z.add_le(z.origin, start, 0) and ok        # 0 <= S_a
    return ok


class TestBoundArithmetic(unittest.TestCase):
    def test_strict_is_tighter_at_equal_value(self):
        self.assertTrue(bound_tighter((Fraction(3), True), (Fraction(3), False)))
        self.assertFalse(bound_tighter((Fraction(3), False), (Fraction(3), True)))

    def test_strictness_is_contagious_along_a_path(self):
        self.assertEqual(
            bound_add((Fraction(2), False), (Fraction(3), True)),
            (Fraction(5), True),
        )
        self.assertEqual(
            bound_add((Fraction(2), False), (Fraction(3), False)),
            (Fraction(5), False),
        )

    def test_infinity_absorbs(self):
        self.assertEqual(bound_add((Fraction(2), True), INF), INF)
        self.assertFalse(bound_tighter(INF, (Fraction(10), False)))

    def test_zero_strict_self_bound_is_a_negative_cycle(self):
        # t - t < 0 is unsatisfiable even though its value is not negative.
        self.assertTrue(bound_is_negative((Fraction(0), True)))
        self.assertFalse(bound_is_negative((Fraction(0), False)))
        self.assertTrue(bound_is_negative((Fraction(-1, 2), False)))


class TestStrictCycles(unittest.TestCase):
    def test_a_lt_b_lt_a_is_rejected_without_an_epsilon(self):
        z = StrictDBM()
        self.assertTrue(z.add_lt("a", "b"))
        self.assertFalse(z.add_lt("b", "a"))
        self.assertFalse(z.consistent)

    def test_a_le_b_le_a_is_accepted_and_forces_equality(self):
        z = StrictDBM()
        self.assertTrue(z.add_le("a", "b"))
        self.assertTrue(z.add_le("b", "a"))
        self.assertTrue(z.consistent)
        self.assertEqual(z.bound("a", "b"), (Fraction(0), False))
        self.assertEqual(z.bound("b", "a"), (Fraction(0), False))

    def test_strict_and_equality_are_different_constraints(self):
        eq = StrictDBM()
        eq.add_eq("S2", "E1", 0)
        self.assertTrue(eq.consistent)
        lt = StrictDBM()
        lt.add_eq("S2", "E1", 0)
        self.assertFalse(lt.add_lt("S2", "E1"))


class TestSpecSection3Placements(unittest.TestCase):
    """Section 8: with S1 = 0, d(a1) = 5, d(a2) = 2, D = 8 all three strict
    placements of section 3 must be admitted."""

    def _base(self):
        z = StrictDBM()
        z.add_eq("C", z.origin, 0)
        _start_action(z, "a1", "S1", "E1", 5, "C", 8)
        z.add_eq("S1", z.origin, 0)             # S1 = 0, as the check states
        _start_action(z, "a2", "S2", "E2", 2, "C", 8)
        self.assertTrue(z.consistent)
        return z

    def test_after(self):
        z = self._base()
        self.assertTrue(z.add_lt("E1", "S2"))
        self.assertTrue(z.add_lt("S2", "E2"))
        self.assertTrue(z.consistent)
        lo, lo_strict, hi, hi_strict = z.window("S2")
        self.assertEqual((lo, lo_strict), (Fraction(5), True))   # S2 > 5
        self.assertEqual((hi, hi_strict), (Fraction(6), False))  # S2 <= 6

    def test_crosses_e1(self):
        z = self._base()
        self.assertTrue(z.add_lt("S2", "E1"))
        self.assertTrue(z.add_lt("E1", "E2"))
        self.assertTrue(z.consistent)
        lo, lo_strict, hi, hi_strict = z.window("S2")
        self.assertEqual((lo, lo_strict), (Fraction(3), True))   # S2 > 3
        self.assertEqual((hi, hi_strict), (Fraction(5), True))   # S2 < 5

    def test_contained_before_e1(self):
        z = self._base()
        self.assertTrue(z.add_lt("S2", "E2"))
        self.assertTrue(z.add_lt("E2", "E1"))
        self.assertTrue(z.consistent)
        lo, lo_strict, hi, hi_strict = z.window("S2")
        self.assertEqual((lo, lo_strict), (Fraction(0), False))  # S2 >= 0
        self.assertEqual((hi, hi_strict), (Fraction(3), True))   # S2 < 3

    def test_the_two_equality_placements(self):
        # Section 3: "Also include S2 = E1 < E2 and S2 < E2 = E1."
        z = self._base()
        self.assertTrue(z.add_eq("S2", "E1", 0))
        self.assertTrue(z.add_lt("E1", "E2"))
        self.assertTrue(z.consistent)
        self.assertEqual(z.window("S2")[0], Fraction(5))

        z = self._base()
        self.assertTrue(z.add_eq("E2", "E1", 0))
        self.assertTrue(z.add_lt("S2", "E2"))
        self.assertTrue(z.consistent)
        self.assertEqual(z.window("S2")[0], Fraction(3))


class TestSpecSection8Windows(unittest.TestCase):
    """Section 8: D = 3, d(a1) = 2, d(a2) = 3 => 0 <= S1 <= 1 and 0 <= S2 <= 0."""

    def _both_started(self):
        z = StrictDBM()
        z.add_eq("C", z.origin, 0)
        _start_action(z, "a1", "S1", "E1", 2, "C", 3)
        _start_action(z, "a2", "S2", "E2", 3, "C", 3)
        return z

    def test_start_windows_are_forced_by_the_world(self):
        z = self._both_started()
        self.assertTrue(z.consistent)
        self.assertEqual(z.window("S1"), (Fraction(0), False, Fraction(1), False))
        self.assertEqual(z.window("S2"), (Fraction(0), False, Fraction(0), False))

    def test_neither_action_can_complete_twice_by_the_deadline(self):
        # A second instance of a1 must start at or after E1 >= 2, so its own end
        # lands at 4 > D = 3.
        z = self._both_started()
        self.assertTrue(z.add_eq("S1", z.origin, 0))
        ok = _start_action(z, "a1_retry", "S1b", "E1b", 2, "E1", 3)
        self.assertFalse(ok)
        self.assertFalse(z.consistent)

    def test_sequential_placements_are_infeasible(self):
        for earlier, later in (("E1", "S2"), ("E2", "S1")):
            z = self._both_started()
            self.assertFalse(
                z.add_lt(earlier, later),
                msg=f"{earlier} < {later} should be infeasible",
            )

    def test_e1_lt_e2_and_e1_eq_e2_are_both_feasible(self):
        z = self._both_started()
        self.assertTrue(z.add_lt("E1", "E2"))
        self.assertTrue(z.consistent)
        self.assertEqual(z.window("S1")[2], Fraction(1))
        self.assertTrue(z.window("S1")[3])              # S1 < 1 now

        z = self._both_started()
        self.assertTrue(z.add_eq("E1", "E2", 0))
        self.assertTrue(z.consistent)
        self.assertEqual(z.window("S1"), (Fraction(1), False, Fraction(1), False))


class TestClosureAndProjection(unittest.TestCase):
    def test_incremental_closure_matches_full_floyd_warshall(self):
        import random

        rng = random.Random(7)
        names = ["t%d" % i for i in range(6)]
        for _ in range(200):
            inc = StrictDBM()
            raw = StrictDBM()
            edges = []
            for _ in range(10):
                x, y = rng.sample(names, 2)
                edges.append((x, y, Fraction(rng.randint(-4, 6)), rng.random() < 0.5))
            for x, y, v, s in edges:
                inc.add_bound(x, y, v, s)
            # Same edges, but written straight into the matrix and closed once.
            for x, y, v, s in edges:
                i, j = raw.add_var(x), raw.add_var(y)
                if bound_tighter((v, s), raw._m[i][j]):
                    raw._m[i][j] = (v, s)
            raw.close()
            self.assertEqual(inc.consistent, raw.consistent)
            if inc.consistent:
                for a in names:
                    for b in names:
                        if a in inc and b in inc and a in raw and b in raw:
                            self.assertEqual(inc.bound(a, b), raw.bound(a, b))

    def test_projection_keeps_the_implied_frontier_bound(self):
        # Spec section 9's example: a1 then a2 took 2 and 4 in sequence, so the
        # frontier must retain C - O >= 6 after the internal events are dropped.
        z = StrictDBM()
        z.add_eq("S1", z.origin, 0)
        z.add_eq("E1", "S1", 2)
        z.add_le("E1", "S2")
        z.add_eq("E2", "S2", 4)
        z.add_eq("C", "E2", 0)
        self.assertTrue(z.consistent)
        f = z.project(["C"])
        self.assertEqual(f.variables, (z.origin, "C"))
        lo, lo_strict, _hi, _hi_strict = f.window("C")
        self.assertEqual((lo, lo_strict), (Fraction(6), False))

    def test_projection_does_not_set_the_next_start_to_a_number(self):
        # "Starting a5 adds S5 >= C; this does not set S5 = 6."
        z = StrictDBM()
        z.add_eq("C", z.origin, 6)
        z.add_le("C", "S5")
        z.add_le("S5", z.origin, 20)
        lo, lo_strict, hi, hi_strict = z.window("S5")
        self.assertEqual((lo, lo_strict), (Fraction(6), False))
        self.assertEqual((hi, hi_strict), (Fraction(20), False))

    def test_canonical_form_is_order_stable_and_separates_strictness(self):
        a = StrictDBM()
        a.add_lt("x", "y", 3)
        b = StrictDBM()
        b.add_le("x", "y", 3)
        self.assertNotEqual(a.canonical(["O", "x", "y"]), b.canonical(["O", "x", "y"]))
        c = StrictDBM()
        c.add_lt("x", "y", 3)
        self.assertEqual(a.canonical(["O", "x", "y"]), c.canonical(["O", "x", "y"]))

    def test_copy_is_independent(self):
        z = StrictDBM()
        z.add_le("a", z.origin, 5)
        w = z.copy()
        w.add_le("a", z.origin, 2)
        self.assertEqual(z.window("a")[2], Fraction(5))
        self.assertEqual(w.window("a")[2], Fraction(2))

    def test_rational_durations_survive(self):
        z = StrictDBM()
        z.add_eq("C", z.origin, 0)
        _start_action(z, "a", "S", "E", Fraction(5, 3), "C", Fraction(10, 3))
        self.assertTrue(z.consistent)
        self.assertEqual(z.window("S")[2], Fraction(5, 3))


if __name__ == "__main__":
    unittest.main()
