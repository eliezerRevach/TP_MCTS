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
    Segment,
    best_feasible_and,
    or_aggregate_paths,
    path_internal_feasible,
    paths_mutex,
    propagate_path_mutex,
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
        self.assertAlmostEqual(res.value, 0.9)  # max, not 0.9+0.8
        self.assertEqual(res.n_components, 1)
        self.assertTrue(res.clique_survived)

    def test_or_sums_disjoint_drives(self):
        p1 = Path(frozenset({Segment("drive", 0, 5)}), 0.4)
        p2 = Path(frozenset({Segment("drive", 5, 20)}), 0.5)
        res = or_aggregate_paths([p1, p2], _drive_self_mutex, k=4)
        self.assertAlmostEqual(res.value, 0.9)  # sum (no overlap)
        self.assertEqual(res.n_components, 2)
        self.assertFalse(res.clique_survived)

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
        self.assertIn("survival_fraction", h.log_pathmutex_summary())


if __name__ == "__main__":
    unittest.main()
