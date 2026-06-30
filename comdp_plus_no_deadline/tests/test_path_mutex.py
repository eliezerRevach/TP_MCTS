"""
Tests for the temporal path-mutex tightening (path_mutex) and the
``baseline_admissible_paths`` strategy.

Covers the scenarios from the design discussion:
- The car: two drive paths overlapping in time are mutex (self-mutex on the same
  action), so the OR layer collapses them via max; non-overlapping drives sum.
- "At least one mutex breaks the parallel": a conjunctive AND path whose chosen
  achievers conflict is infeasible (dropped).
- A surviving mutex needs only ONE conflicting segment pair anywhere across two
  paths.
- Integration: overlapping re-fires of a resource action collapse via max.
"""

import unittest
from dataclasses import dataclass

from comdp_plus_no_deadline.engines.path_mutex import (
    Path,
    PathMutexInstrumentation,
    Row,
    Segment,
    best_feasible_and,
    common_footprint,
    guaranteed_mutex,
    insert_or_absorb,
    or_aggregate_paths,
    path_internal_feasible,
    paths_mutex,
    propagate_path_mutex,
    row_from_path,
    segments_overlap,
)
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


def _drive_self_mutex(a, b):
    # Drive occupies the (single) car: mutex with itself and with any drive.
    return a == "drive" and b == "drive"


class TestPathMutexPrimitives(unittest.TestCase):
    def test_overlap_half_open(self):
        self.assertTrue(segments_overlap(Segment("drive", 0, 15), Segment("drive", 5, 20)))
        # Touching endpoints do not overlap.
        self.assertFalse(segments_overlap(Segment("a", 0, 5), Segment("b", 5, 10)))
        # Zero-width window overlaps nothing.
        self.assertFalse(segments_overlap(Segment("a", 5, 5), Segment("a", 0, 10)))

    def test_car_overlapping_drives_are_mutex(self):
        p1 = Path(frozenset({Segment("drive", 0, 15)}), 0.9)
        p2 = Path(frozenset({Segment("drive", 5, 20)}), 0.8)
        self.assertTrue(paths_mutex(p1, p2, _drive_self_mutex))

    def test_car_disjoint_drives_not_mutex(self):
        p1 = Path(frozenset({Segment("drive", 0, 5)}), 0.9)
        p2 = Path(frozenset({Segment("drive", 5, 20)}), 0.8)  # touch at 5, no overlap
        self.assertFalse(paths_mutex(p1, p2, _drive_self_mutex))

    def test_or_collapses_mutex_drives_to_max(self):
        p1 = Path(frozenset({Segment("drive", 0, 15)}), 0.9)
        p2 = Path(frozenset({Segment("drive", 5, 20)}), 0.8)
        res = or_aggregate_paths([p1, p2], _drive_self_mutex, k=4)
        self.assertAlmostEqual(res.value, 0.9)  # max(0.9, 0.8), two schedules
        self.assertEqual(res.n_rows, 2)
        self.assertTrue(res.tightened)

    def test_or_sums_disjoint_drives(self):
        p1 = Path(frozenset({Segment("drive", 0, 5)}), 0.4)
        p2 = Path(frozenset({Segment("drive", 5, 20)}), 0.5)
        res = or_aggregate_paths([p1, p2], _drive_self_mutex, k=4)
        self.assertAlmostEqual(res.value, 0.9)  # one schedule {p1,p2} sum
        self.assertEqual(res.n_rows, 1)
        self.assertFalse(res.tightened)

    def test_new_path_mutex_with_one_row_folds_free_rows(self):
        # Table {p1,p2,p3} pairwise mutex; pnew mutex with p1 only, free with p2,p3.
        # Per insert_or_absorb Case 3: the free rows (p2,p3) and pnew fold into one
        # summed row -> table {p1, p2+p3+pnew}; value = max(0.5, 0.9) = 0.9.
        p1 = Path(frozenset({Segment("a", 0, 10)}), 0.5)
        p2 = Path(frozenset({Segment("b", 0, 10)}), 0.4)
        p3 = Path(frozenset({Segment("c", 0, 10)}), 0.3)
        pnew = Path(frozenset({Segment("d", 2, 8)}), 0.2)  # action d -> mutex with a (p1) only
        pairs = {frozenset(("a", "b")), frozenset(("a", "c")),
                 frozenset(("b", "c")), frozenset(("a", "d"))}
        mutex = lambda x, y: frozenset((x, y)) in pairs
        res = or_aggregate_paths([p1, p2, p3, pnew], mutex, k=4)
        self.assertAlmostEqual(res.value, max(0.5, 0.2 + 0.4 + 0.3))  # 0.9
        self.assertEqual(res.n_rows, 2)
        self.assertTrue(res.tightened)


def _name_mutex(*pairs):
    """Symmetric action-mutex from unordered name pairs (handles a==b too)."""
    pset = {frozenset(p) if len(set(p)) == 2 else (p[0],) for p in pairs}
    def fn(a, b):
        if a == b:
            return (a,) in pset
        return frozenset((a, b)) in pset
    return fn


class TestInsertOrAbsorb(unittest.TestCase):
    """The three cases of insert_or_absorb."""

    def _row(self, action, prob, start=0, end=10):
        return Row(prob, frozenset({Segment(action, start, end)}))

    def test_guaranteed_mutex_and_common_footprint(self):
        sa = Segment("a", 0, 10)
        sb = Segment("b", 0, 10)
        mutex = _name_mutex(("a", "b"))
        self.assertTrue(guaranteed_mutex(Row(0.5, frozenset({sa})),
                                         Row(0.4, frozenset({sb})), mutex))
        # Empty footprint -> never certified mutex (uncertain = free).
        self.assertFalse(guaranteed_mutex(Row(0.5, frozenset()),
                                          Row(0.4, frozenset({sb})), mutex))
        self.assertEqual(common_footprint(Row(0.5, frozenset({sa, sb})),
                                          Row(0.4, frozenset({sb}))), frozenset({sb}))

    def test_case1_mutex_with_all_and_room_adds_row(self):
        mutex = _name_mutex(("a", "b"))
        A = self._row("a", 0.5)
        T = [A]
        new = self._row("b", 0.3)
        insert_or_absorb(T, new, 3, mutex)
        self.assertEqual(len(T), 2)
        self.assertIn(new, T)  # r_new never dropped

    def test_case3_free_row_sums_and_erases_footprint(self):
        # A free w.r.t E, B mutex with E -> E folds with A into a summed, footprint-
        # erased row; B stays.  T = {B, A+E}.
        mutex = _name_mutex(("a", "b"), ("b", "e"))  # a-e NOT mutex
        A = self._row("a", 0.5)
        B = self._row("b", 0.4)
        T = [A, B]
        E = self._row("e", 0.3)
        insert_or_absorb(T, E, 3, mutex)
        self.assertEqual(len(T), 2)
        summed = [r for r in T if not r.footprint]
        self.assertEqual(len(summed), 1)
        self.assertAlmostEqual(summed[0].prob, 0.5 + 0.3)  # A + E, counted once
        self.assertTrue(any(r.footprint and abs(r.prob - 0.4) < 1e-9 for r in T))  # B kept

    def test_case2_full_table_absorbs_via_max(self):
        # T full (K=3), all pairwise mutex; D mutex with all -> absorbed via max into
        # the best row (largest common footprint, then larger max prob). r_new not added.
        mutex = _name_mutex(("a", "b"), ("a", "c"), ("b", "c"),
                            ("a", "d"), ("b", "d"), ("c", "d"))
        A = self._row("a", 0.5)
        B = self._row("b", 0.4)
        C = self._row("c", 0.3)
        T = [A, B, C]
        D = Row(0.45, frozenset({Segment("d", 0, 10)}))
        insert_or_absorb(T, D, 3, mutex)
        self.assertEqual(len(T), 3)  # no new row
        # common footprint is empty for all (distinct segments) -> tie broken by max
        # prob: A=max(0.5,0.45)=0.5 wins. A absorbs D: prob max -> 0.5, footprint -> {}.
        self.assertAlmostEqual(A.prob, 0.5)
        self.assertEqual(A.footprint, frozenset())

    def test_hit_counter(self):
        # counter = [case1_mutex_add, case2_merge, case3_sum]
        mutex = _name_mutex(("a", "b"), ("a", "c"), ("b", "c"))
        c = [0, 0, 0]
        T = []
        insert_or_absorb(T, self._row("a", 0.5), 2, mutex, c)  # first row: trivial add, no hit
        insert_or_absorb(T, self._row("b", 0.4), 2, mutex, c)  # mutex with a, room -> Case 1 add (HIT)
        insert_or_absorb(T, self._row("c", 0.3), 2, mutex, c)  # mutex with all, full -> Case 2 merge (HIT)
        self.assertEqual(c[0], 1)  # one mutex add
        self.assertEqual(c[1], 1)  # one mutex merge
        self.assertEqual(c[2], 0)  # no free sums
        # A free path triggers Case 3 (not a hit).
        c2 = [0, 0, 0]
        T2 = [self._row("a", 0.5)]
        insert_or_absorb(T2, self._row("z", 0.3), 2, _name_mutex(("a", "b")), c2)  # z free of a
        self.assertEqual(c2, [0, 0, 1])

    def test_never_drops_r_new(self):
        mutex = _name_mutex(("a", "a"))  # only a self-mutex
        T = []
        for i in range(6):
            insert_or_absorb(T, Row(0.5 - i * 0.05, frozenset({Segment("x", i, i + 1)})), 2, mutex)
        self.assertLessEqual(len(T), 2)  # K bound respected, nothing crashed/dropped silently

    def test_shared_segment_is_not_a_conflict(self):
        # Two retry paths SHARE the same self-mutex step drive[0,5] (one physical
        # occurrence), and their distinct steps don't overlap. They must be FREE
        # (sum/accumulate), not mutex — a shared occurrence is not contention.
        shared = Segment("drive", 0, 5)
        p = Path(frozenset({shared, Segment("sample", 6, 8)}), 0.5)
        q = Path(frozenset({shared, Segment("sample", 10, 12)}), 0.5)
        self.assertFalse(paths_mutex(p, q, _drive_self_mutex))
        res = or_aggregate_paths([p, q], _drive_self_mutex, k=4)
        self.assertAlmostEqual(res.value, 1.0)  # one compatible schedule {p,q}: 0.5+0.5
        self.assertEqual(res.n_rows, 1)
        self.assertFalse(res.tightened)

    def test_distinct_overlapping_self_mutex_segments_conflict(self):
        # Two DISTINCT windows of the same self-mutex action overlap -> mutex.
        p = Path(frozenset({Segment("drive", 0, 10)}), 0.5)
        q = Path(frozenset({Segment("drive", 5, 15)}), 0.5)
        self.assertTrue(paths_mutex(p, q, _drive_self_mutex))

    def test_one_conflicting_pair_breaks_parallel(self):
        # P has many free segments + one that conflicts with Q -> mutex.
        p = Path(
            frozenset({Segment("x", 0, 3), Segment("y", 3, 6), Segment("drive", 6, 12)}),
            0.5,
        )
        q = Path(frozenset({Segment("z", 0, 2), Segment("drive", 8, 14)}), 0.5)
        self.assertTrue(paths_mutex(p, q, _drive_self_mutex))

    def test_and_at_least_one_mutex_blocks(self):
        # f3 needs f1,f2 achieved by a1@[0,10], a2@[0,10]; a1 mutex a2 -> infeasible.
        mutex = lambda a, b: frozenset((a, b)) == frozenset(("a1", "a2"))
        a1_path = Path(frozenset({Segment("a1", 0, 10)}), 0.9)
        a2_path = Path(frozenset({Segment("a2", 0, 10)}), 0.9)
        a3_seg = Segment("a3", 10, 12)
        support = best_feasible_and([[a1_path], [a2_path]], a3_seg, 1.0, mutex)
        self.assertIsNone(support)

    def test_and_feasible_when_no_conflict(self):
        mutex = lambda a, b: False
        a1_path = Path(frozenset({Segment("a1", 0, 10)}), 0.9)
        a2_path = Path(frozenset({Segment("a2", 0, 10)}), 0.8)
        a3_seg = Segment("a3", 10, 12)
        support = best_feasible_and([[a1_path], [a2_path]], a3_seg, 1.0, mutex)
        self.assertIsNotNone(support)
        self.assertTrue(path_internal_feasible(support, mutex))
        self.assertEqual(len(support.segments), 3)  # a1, a2, a3
        self.assertAlmostEqual(support.prob, 0.9 * 0.8)


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset
    add_effects: frozenset
    duration_steps: int
    del_effects: frozenset = frozenset()
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


class TestPathMutexStrategy(unittest.TestCase):
    def test_strategy_is_normalized(self):
        self.assertEqual(
            TemporalProbabilisticRPGHeuristic._normalize_strategy("baseline_admissible_paths"),
            "baseline_admissible_paths",
        )

    def test_self_mutex_detection(self):
        # place consumes free(m) (deletes a precondition it needs) -> self-mutex.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="place",
                    pos_preconditions=frozenset({"free"}),
                    add_effects=frozenset({"on"}),
                    duration_steps=3,
                    del_effects=frozenset({"free"}),
                ),
            ],
            facts={"free", "on"},
            initial_facts={"free"},
            goal_facts={"on"},
        )
        h._ensure_unbiased_structural_extracted()
        self.assertTrue(h._path_actions_mutex("place", "place"))
        # A non-resource action is not self-mutex.
        h2 = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="paint",
                    pos_preconditions=frozenset({"ready"}),
                    add_effects=frozenset({"painted"}),
                    duration_steps=2,
                ),
            ],
            facts={"ready", "painted"},
            initial_facts={"ready"},
            goal_facts={"painted"},
        )
        h2._ensure_unbiased_structural_extracted()
        self.assertFalse(h2._path_actions_mutex("paint", "paint"))

    def test_strategy_runs_and_scores_in_unit_interval(self):
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="place",
                    pos_preconditions=frozenset({"free"}),
                    add_effects=frozenset({"on"}),
                    duration_steps=3,
                    del_effects=frozenset({"free"}),
                ),
            ],
            facts={"free", "on"},
            initial_facts={"free"},
            goal_facts={"on"},
        )
        score = h.heuristic_score(
            {"free"}, {"on"}, fixed_depth=8, strategy="baseline_admissible_paths"
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIn("HITS", h.log_pathmutex_summary())


if __name__ == "__main__":
    unittest.main()
