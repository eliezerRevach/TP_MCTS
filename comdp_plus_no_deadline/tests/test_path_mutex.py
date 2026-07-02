"""
Tests for the temporal path-mutex tightening (path_mutex) and the
``baseline_admissible_paths`` strategy.

``baseline_admissible_paths`` is a MINIMAL generalization of
``baseline_admissible``: identical persistence / Frechet-min AND / cumulative-
retry cross-layer recursion, with only the per-layer OR-hazard ``H_t(f)``
upgraded from a flat union sum to a K-bounded mutex table (``insert_or_absorb``
via ``table_or_hazard``). Covers:
- ``Row``/``insert_or_absorb``: the three cases (mutex-add, mutex-merge, free-sum)
  and the never-drop-r_new / footprint-intersect-on-merge / erase-on-sum rules.
- ``table_or_hazard``: reduces to the union bound with no mutex, tightens via max
  when same-layer achievers are certified mutex, and never exceeds the union.
- The strategy: never exceeds ``baseline_admissible`` (the core invariant — every
  table operation is either a sum or a max-of-something-<=-a-sum, so the whole
  heuristic can only be <= the all-sum baseline), including across the specific
  "achiever re-fires every layer because its precondition persists" scenario that
  previously caused an over-count bug in an earlier (now-replaced) design.
"""

import math
import unittest
from dataclasses import dataclass

from comdp_plus_no_deadline.engines.path_mutex import (
    PathMutexInstrumentation,
    Row,
    Segment,
    combine_precondition_footprints,
    common_footprint,
    guaranteed_mutex,
    insert_or_absorb,
    prune_expired,
    segments_overlap,
    table_or_hazard,
)
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


def _name_mutex(*pairs):
    """Symmetric action-mutex from unordered name pairs (handles a==b too)."""
    pset = {frozenset(p) if len(set(p)) == 2 else (p[0],) for p in pairs}

    def fn(a, b):
        if a == b:
            return (a,) in pset
        return frozenset((a, b)) in pset

    return fn


class TestSegments(unittest.TestCase):
    def test_overlap_half_open(self):
        self.assertTrue(segments_overlap(Segment("drive", 0, 15), Segment("drive", 5, 20)))
        # Touching endpoints do not overlap.
        self.assertFalse(segments_overlap(Segment("a", 0, 5), Segment("b", 5, 10)))
        # Zero-width window overlaps nothing.
        self.assertFalse(segments_overlap(Segment("a", 5, 5), Segment("a", 0, 10)))


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
        # Two rows SHARE the same self-mutex segment (one physical occurrence) ->
        # not certified mutex (same occurrence, not contention) -> free/sum.
        shared = Segment("drive", 0, 5)
        mutex = lambda a, b: a == "drive" and b == "drive"
        r1 = Row(0.5, frozenset({shared}))
        r2 = Row(0.5, frozenset({shared}))
        self.assertFalse(guaranteed_mutex(r1, r2, mutex))

    def test_distinct_overlapping_self_mutex_segments_conflict(self):
        # Two DISTINCT windows of the same self-mutex action overlap -> mutex.
        mutex = lambda a, b: a == "drive" and b == "drive"
        r1 = Row(0.5, frozenset({Segment("drive", 0, 10)}))
        r2 = Row(0.5, frozenset({Segment("drive", 5, 15)}))
        self.assertTrue(guaranteed_mutex(r1, r2, mutex))


def _sup(name, prob, start, end):
    """Build a (name, prob, footprint) triple from a single-segment window —
    table_or_hazard takes pre-built footprints (possibly multi-segment, via
    combine_precondition_footprints), so tests construct the common
    single-segment case via this helper."""
    return (name, prob, frozenset({Segment(name, start, end)}))


class TestTableOrHazard(unittest.TestCase):
    def test_empty_is_zero(self):
        res = table_or_hazard([], lambda a, b: False, k=3)
        self.assertEqual(res.value, 0.0)
        self.assertEqual(res.union_value, 0.0)

    def test_reduces_to_union_when_no_mutex(self):
        supports = [_sup("a", 0.2, 0, 5), _sup("b", 0.3, 0, 5), _sup("c", 0.1, 0, 5)]
        res = table_or_hazard(supports, lambda a, b: False, k=3)
        self.assertAlmostEqual(res.value, 0.6)
        self.assertAlmostEqual(res.union_value, 0.6)
        self.assertFalse(res.tightened)

    def test_same_layer_mutex_collapses_to_max(self):
        # Two achievers landing at the SAME arrival layer, windows overlap -> mutex.
        supports = [_sup("paint_x1", 0.5, 0, 5), _sup("paint_x2", 0.4, 2, 5)]  # share machine m
        mutex = lambda a, b: {a, b} == {"paint_x1", "paint_x2"} or a == b == "paint_x1"
        res = table_or_hazard(supports, mutex, k=3)
        self.assertAlmostEqual(res.value, 0.5)  # max, not 0.9
        self.assertAlmostEqual(res.union_value, 0.9)
        self.assertTrue(res.tightened)
        self.assertEqual(res.mutex_adds, 1)

    def test_never_exceeds_union(self):
        cases = [
            ([_sup("a", 0.5, 0, 5), _sup("b", 0.5, 0, 5)], lambda a, b: a != b),
            ([_sup("a", 0.3, 0, 5), _sup("b", 0.3, 1, 6), _sup("c", 0.3, 2, 7)], lambda a, b: True),
            ([_sup("a", 0.9, 0, 5), _sup("b", 0.9, 5, 10)], lambda a, b: False),
            ([_sup("a", 0.4, 0, 5), _sup("b", 0.4, 2, 8), _sup("c", 0.4, 6, 10), _sup("d", 0.4, 0, 3)],
             lambda a, b: a < b),
        ]
        for supports, mutex in cases:
            res = table_or_hazard(supports, mutex, k=2)
            self.assertLessEqual(res.value, res.union_value + 1e-12, msg=str(supports))

    def test_self_mutex_same_arrival_layer_overlapping_windows(self):
        # Same achiever NAME firing from two different start times but landing at
        # the SAME arrival layer (different durations), overlapping windows.
        supports = [_sup("drive", 0.6, 0, 10), _sup("drive", 0.5, 6, 10)]
        mutex = lambda a, b: a == "drive" and b == "drive"
        res = table_or_hazard(supports, mutex, k=3)
        self.assertAlmostEqual(res.value, 0.6)  # max(0.6, 0.5), windows overlap on [6,10)
        self.assertTrue(res.tightened)


class TestPruneExpired(unittest.TestCase):
    """prune_expired is pure garbage collection for the AND-layer's
    active_footprints registry (combine_precondition_footprints) — kept after
    cross-layer shadowing was reverted (see path_mutex module docstring)."""

    def test_prune_expired_drops_only_past_segments(self):
        segs = [Segment("a", 0, 5), Segment("b", 3, 10), Segment("c", 8, 20)]
        kept = prune_expired(segs, current_layer=5)
        self.assertEqual(set(kept), {Segment("b", 3, 10), Segment("c", 8, 20)})


class TestCombinePreconditionFootprints(unittest.TestCase):
    """The AND-layer's "remember all paths in pai" step: union one feasible
    representative footprint per precondition, never claiming two mutually-
    exclusive occurrences "both definitely happened"."""

    def test_unions_one_segment_per_precondition_when_feasible(self):
        mutex = lambda a, b: False  # nothing mutex -> any combination feasible
        registry = {"f1": [Segment("a", 0, 5)], "f2": [Segment("b", 0, 5)]}
        result = combine_precondition_footprints(["f1", "f2"], registry, mutex)
        self.assertEqual(result, frozenset({Segment("a", 0, 5), Segment("b", 0, 5)}))

    def test_rejects_internally_conflicting_combination(self):
        # The only candidates for f1 and f2 are mutex-and-overlapping with each
        # other -> must NOT union them (would over-claim "both happened").
        mutex = lambda a, b: {a, b} == {"a", "b"}
        registry = {"f1": [Segment("a", 0, 5)], "f2": [Segment("b", 2, 7)]}
        result = combine_precondition_footprints(["f1", "f2"], registry, mutex)
        self.assertEqual(result, frozenset())  # safe fallback: nothing inherited

    def test_searches_alternate_candidates_on_conflict(self):
        # f1's most-recent candidate conflicts with f2's only candidate, but f1
        # has an OLDER, non-conflicting alternative too -> should find it.
        mutex = lambda a, b: {a, b} == {"a1", "b"}
        registry = {
            "f1": [Segment("a0", 0, 5), Segment("a1", 10, 15)],  # a1 is "most recent" (last)
            "f2": [Segment("b", 12, 16)],
        }
        result = combine_precondition_footprints(
            ["f1", "f2"], registry, mutex, per_precondition_cap=2
        )
        self.assertIn(Segment("b", 12, 16), result)
        self.assertIn(Segment("a0", 0, 5), result)  # fell back to the free alternative
        self.assertNotIn(Segment("a1", 10, 15), result)

    def test_no_tracked_footprint_contributes_nothing(self):
        mutex = lambda a, b: False
        registry = {"f1": [Segment("a", 0, 5)]}  # f2 untracked
        result = combine_precondition_footprints(["f1", "f2"], registry, mutex)
        self.assertEqual(result, frozenset({Segment("a", 0, 5)}))

    def test_no_preconditions_is_empty(self):
        self.assertEqual(combine_precondition_footprints([], {}, lambda a, b: False), frozenset())


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
        # place consumes free(m) (deletes a precondition it needs), AND free(m) is
        # a SHARED resource (another action also requires it) -> self-mutex.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="place",
                    pos_preconditions=frozenset({"free"}),
                    add_effects=frozenset({"on"}),
                    duration_steps=3,
                    del_effects=frozenset({"free"}),
                ),
                SyntheticAction(
                    name="place_other",
                    pos_preconditions=frozenset({"free"}),
                    add_effects=frozenset({"on_other"}),
                    duration_steps=3,
                ),
            ],
            facts={"free", "on", "on_other"},
            initial_facts={"free"},
            goal_facts={"on"},
        )
        h._ensure_unbiased_structural_extracted()
        self.assertTrue(h._path_actions_mutex("place", "place"))
        # A fact required by NO action other than itself (e.g. a private
        # inExecution(start-X)-style bookkeeping marker) is NOT a real resource:
        # repeated re-firing is a legitimate relaxed-graph retry, not contention.
        h_private = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="solo",
                    pos_preconditions=frozenset({"marker"}),
                    add_effects=frozenset({"done"}),
                    duration_steps=3,
                    del_effects=frozenset({"marker"}),
                ),
            ],
            facts={"marker", "done"},
            initial_facts={"marker"},
            goal_facts={"done"},
        )
        h_private._ensure_unbiased_structural_extracted()
        self.assertFalse(h_private._path_actions_mutex("solo", "solo"))
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

    def test_never_exceeds_baseline_admissible_resource_contention(self):
        # Two pieces competing for the same exclusive machine (resource self-mutex,
        # both can complete at the same discretized layer).
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="use_m_x1",
                    pos_preconditions=frozenset({"free_m"}),
                    add_effects=frozenset({"done_x1"}),
                    duration_steps=5,
                    del_effects=frozenset({"free_m"}),
                ),
                SyntheticAction(
                    name="use_m_x2",
                    pos_preconditions=frozenset({"free_m"}),
                    add_effects=frozenset({"done_x2"}),
                    duration_steps=5,
                    del_effects=frozenset({"free_m"}),
                ),
            ],
            facts={"free_m", "done_x1", "done_x2"},
            initial_facts={"free_m"},
            goal_facts={"done_x1", "done_x2"},
        )
        for depth in (5, 10, 20):
            adm = h.heuristic_score(
                {"free_m"}, {"done_x1", "done_x2"}, fixed_depth=depth,
                strategy="baseline_admissible",
            )
            paths = h.heuristic_score(
                {"free_m"}, {"done_x1", "done_x2"}, fixed_depth=depth,
                strategy="baseline_admissible_paths",
            )
            self.assertLessEqual(paths, adm + 1e-9, msg=f"depth={depth}")

    def test_never_exceeds_baseline_admissible_persistent_precondition_refire(self):
        # Regression: an action whose precondition PERSISTS in the relaxed graph
        # re-fires at every layer once enabled, contributing fresh B_e(t) at each
        # subsequent arrival layer. A flat cross-layer sum of these would blow past
        # baseline; this must stay <= baseline at every depth (the cumulative-retry
        # recursion, shared with baseline, is what discounts the redundant re-fires).
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="enable",
                    pos_preconditions=frozenset(),
                    add_effects=frozenset({"ready"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="finish",
                    pos_preconditions=frozenset({"ready"}),
                    add_effects=frozenset({"goal"}),
                    duration_steps=1,
                ),
            ],
            facts={"ready", "goal"},
            initial_facts=set(),
            goal_facts={"goal"},
        )
        for depth in (3, 6, 10, 20):
            adm = h.heuristic_score(
                set(), {"goal"}, fixed_depth=depth, strategy="baseline_admissible"
            )
            paths = h.heuristic_score(
                set(), {"goal"}, fixed_depth=depth, strategy="baseline_admissible_paths"
            )
            self.assertLessEqual(paths, adm + 1e-9, msg=f"depth={depth}")
            self.assertLessEqual(paths, 1.0 + 1e-9)

    def test_car_self_mutex_cross_layer_refire_is_a_known_gap(self):
        # The literal car example: 'drive' is self-mutex (occupies a SHARED
        # resource 'avail') and its precondition persists, so it legitimately
        # fires again before its previous occupation elapses (window [0,d) then
        # [1,d+1) etc., all pairwise overlapping for d>1). Cross-layer shadowing
        # was tried and REVERTED (see path_mutex module docstring) because it
        # could permanently block a stronger later mutex-alternative behind a
        # weak earlier one. So this specific case is NOT caught (0 hits) — same
        # -layer-only comparison stays sound; admissibility must still hold.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="drive",
                    pos_preconditions=frozenset({"avail"}),
                    add_effects=frozenset({"goal"}),
                    duration_steps=10,
                    del_effects=frozenset({"avail"}),
                ),
                SyntheticAction(
                    name="other_use",
                    pos_preconditions=frozenset({"avail"}),
                    add_effects=frozenset({"other_goal"}),
                    duration_steps=1,
                ),
            ],
            facts={"avail", "goal", "other_goal"},
            initial_facts={"avail"},
            goal_facts={"goal"},
        )
        adm = h.heuristic_score(
            {"avail"}, {"goal"}, fixed_depth=20, strategy="baseline_admissible"
        )
        paths = h.heuristic_score(
            {"avail"}, {"goal"}, fixed_depth=20, strategy="baseline_admissible_paths"
        )
        self.assertLessEqual(paths, adm + 1e-9)
        self.assertEqual(h._pathmutex_instr.hits, 0)  # known gap, not a regression

    def test_weak_early_occupant_does_not_block_strong_later_alternative(self):
        # Regression for the reverted cross-layer shadowing bug: a weak
        # achiever (X, B_e=0.01) firing early must NOT permanently suppress a
        # much stronger mutex-alternative (Y, B_e=0.9) arriving at a later,
        # overlapping layer. Without shadowing, table_or_hazard/g(p,h) handle
        # this correctly on their own (Y arrives alone in its own layer bucket).
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalRelaxedActionModel,
        )

        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("X", frozenset({"r"}), frozenset({"goal"}), 10,
                                 del_effects=frozenset({"r"})),
                SyntheticAction("Y", frozenset({"r", "unlock"}), frozenset({"goal"}), 10,
                                 del_effects=frozenset({"r"})),
                SyntheticAction("enable_Y", frozenset(), frozenset({"unlock"}), 5),
            ],
            facts={"r", "goal", "unlock"},
            initial_facts={"r"},
            goal_facts={"goal"},
        )
        h._action_models = [
            TemporalRelaxedActionModel("X", frozenset({"r"}), {"goal": 0.01}, 10),
            TemporalRelaxedActionModel("Y", frozenset({"r", "unlock"}), {"goal": 0.9}, 10),
            TemporalRelaxedActionModel("enable_Y", frozenset(), {"unlock": 1.0}, 5),
        ]
        h._actions_by_effect_fact = h._build_actions_by_effect_fact()

        paths_at_20 = h.heuristic_score(
            {"r"}, {"goal"}, fixed_depth=20, strategy="baseline_admissible_paths"
        )
        self.assertGreater(paths_at_20, 0.5)  # Y's strength must show through

    def test_degenerates_to_baseline_when_nothing_mutex(self):
        # No mutex anywhere (independent achievers of independent goals) -> the
        # table never exceeds one row -> paths == baseline_admissible exactly.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="mk_x",
                    pos_preconditions=frozenset({"a"}),
                    add_effects=frozenset({"x"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="mk_y",
                    pos_preconditions=frozenset({"b"}),
                    add_effects=frozenset({"y"}),
                    duration_steps=1,
                ),
            ],
            facts={"a", "b", "x", "y"},
            initial_facts={"a", "b"},
            goal_facts={"x", "y"},
        )
        adm = h.heuristic_score(
            {"a", "b"}, {"x", "y"}, fixed_depth=6, strategy="baseline_admissible"
        )
        paths = h.heuristic_score(
            {"a", "b"}, {"x", "y"}, fixed_depth=6, strategy="baseline_admissible_paths"
        )
        self.assertAlmostEqual(paths, adm)


if __name__ == "__main__":
    unittest.main()
