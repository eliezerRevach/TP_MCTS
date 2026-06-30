"""
Tests for the mutex-aware K-bounded OR-layer tightening (kmutex_or_bound) and the
``baseline_admissible_kmutex`` strategy.

Validates the stated sanity checks:
- optimized <= baseline (union) everywhere — never above (admissibility direction).
- all-mutex clique -> reduces to max; all-free -> reduces to baseline sum.
- worked case a,b mutex; c free -> matches the baseline union (a+b+c), NOT an
  under-count of max(a,b)+c.
- max-vs-sum is decided from the footprint only (adding a free element clears it).
- K bound is respected and stays admissible.
- The strategy reduces to baseline_admissible when nothing is mutex, and never
  scores above it.
"""

import unittest
from dataclasses import dataclass

from comdp_plus_no_deadline.engines.kmutex_or_bound import (
    KMutexInstrumentation,
    kmutex_or_hazard,
)
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


def _mutex_from_pairs(pairs):
    """Build a symmetric mutex_fn from an iterable of unordered name pairs."""
    pset = {frozenset(p) for p in pairs}

    def mutex_fn(a, b):
        return frozenset((a, b)) in pset

    return mutex_fn


class TestKMutexOrPure(unittest.TestCase):
    def test_single_support_is_union(self):
        res = kmutex_or_hazard([("a", 0.4)], _mutex_from_pairs([]), k=3)
        self.assertAlmostEqual(res.value, 0.4)
        self.assertAlmostEqual(res.union_value, 0.4)
        self.assertFalse(res.clique_survived)

    def test_all_free_reduces_to_union_sum(self):
        supports = [("a", 0.2), ("b", 0.3), ("c", 0.1)]
        res = kmutex_or_hazard(supports, _mutex_from_pairs([]), k=3)
        self.assertAlmostEqual(res.value, 0.6)
        self.assertAlmostEqual(res.union_value, 0.6)
        self.assertEqual(res.n_mutex_rows, 0)
        self.assertFalse(res.clique_survived)

    def test_all_mutex_reduces_to_max(self):
        supports = [("a", 0.2), ("b", 0.3)]
        res = kmutex_or_hazard(supports, _mutex_from_pairs([("a", "b")]), k=3)
        # max(a, b) = 0.3, strictly below union 0.5.
        self.assertAlmostEqual(res.value, 0.3)
        self.assertAlmostEqual(res.union_value, 0.5)
        self.assertTrue(res.clique_survived)
        self.assertEqual(res.max_clique_size, 2)

    def test_three_mutex_clique_reduces_to_max(self):
        supports = [("a", 0.2), ("b", 0.3), ("c", 0.25)]
        mutex = _mutex_from_pairs([("a", "b"), ("a", "c"), ("b", "c")])
        res = kmutex_or_hazard(supports, mutex, k=4)
        self.assertAlmostEqual(res.value, 0.3)  # max over the clique
        self.assertTrue(res.clique_survived)
        self.assertEqual(res.max_clique_size, 3)

    def test_worked_case_a_b_mutex_c_free_matches_union(self):
        # a,b mutex; c free w.r.t both. c folds into one mutex row and clears it,
        # leaving a lone mutex row -> result == union (a+b+c), NOT max(a,b)+c.
        supports = [("a", 0.2), ("b", 0.3), ("c", 0.1)]
        res = kmutex_or_hazard(supports, _mutex_from_pairs([("a", "b")]), k=3)
        self.assertAlmostEqual(res.value, 0.6)
        self.assertAlmostEqual(res.union_value, 0.6)
        # Not the under-count max(a,b)+c = 0.4.
        self.assertNotAlmostEqual(res.value, 0.4)

    def test_three_clique_plus_free_buys_something(self):
        # a,b,d 3-clique + free c: c clears the smallest row (a), leaving b,d as a
        # surviving 2-clique -> value = (a+c) + max(b,d) < union.
        supports = [("a", 0.2), ("b", 0.3), ("d", 0.4), ("c", 0.1)]
        mutex = _mutex_from_pairs([("a", "b"), ("a", "d"), ("b", "d")])
        res = kmutex_or_hazard(supports, mutex, k=4)
        self.assertAlmostEqual(res.value, (0.2 + 0.1) + 0.4)  # 0.7
        self.assertAlmostEqual(res.union_value, 1.0)
        self.assertLess(res.value, res.union_value)
        self.assertTrue(res.clique_survived)

    def test_never_exceeds_union(self):
        # Random-ish mixes: optimized must never be above the union bound.
        cases = [
            ([("a", 0.5), ("b", 0.5)], [("a", "b")]),
            ([("a", 0.3), ("b", 0.3), ("c", 0.3)], [("a", "b")]),
            ([("a", 0.4), ("b", 0.4), ("c", 0.4), ("d", 0.4)], [("a", "b"), ("c", "d")]),
            ([("a", 0.9), ("b", 0.9)], []),
        ]
        for supports, pairs in cases:
            res = kmutex_or_hazard(supports, _mutex_from_pairs(pairs), k=2)
            self.assertLessEqual(res.value, res.union_value + 1e-12, msg=str(supports))

    def test_k_bound_is_respected(self):
        # Five mutually-mutex supports with K=2 keep at most 2 rows and stay <= union.
        names = list("abcde")
        supports = [(n, 0.15) for n in names]
        pairs = [(names[i], names[j]) for i in range(5) for j in range(i + 1, 5)]
        res = kmutex_or_hazard(supports, _mutex_from_pairs(pairs), k=2)
        self.assertLessEqual(res.n_rows, 2)
        self.assertLessEqual(res.value, res.union_value + 1e-12)

    def test_clamps_to_one(self):
        supports = [("a", 0.7), ("b", 0.8)]
        res = kmutex_or_hazard(supports, _mutex_from_pairs([]), k=3)
        self.assertAlmostEqual(res.value, 1.0)
        self.assertAlmostEqual(res.union_value, 1.0)

    def test_instrumentation_accumulates_survival(self):
        instr = KMutexInstrumentation()
        instr.record(kmutex_or_hazard([("a", 0.2), ("b", 0.3)],
                                      _mutex_from_pairs([("a", "b")]), k=3))
        instr.record(kmutex_or_hazard([("a", 0.2), ("b", 0.3)],
                                      _mutex_from_pairs([]), k=3))
        instr.record(kmutex_or_hazard([("a", 0.2)], _mutex_from_pairs([]), k=3))
        self.assertEqual(instr.or_nodes_total, 2)  # the single-support node is skipped
        self.assertEqual(instr.or_nodes_clique_survived, 1)
        self.assertAlmostEqual(instr.clique_survival_fraction, 0.5)


@dataclass(frozen=True)
class SyntheticAction:
    """Minimal action-like object compatible with the heuristic's model builder."""

    name: str
    pos_preconditions: frozenset
    add_effects: frozenset
    duration_steps: int
    del_effects: frozenset = frozenset()
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


class TestKMutexStrategy(unittest.TestCase):
    def test_strategy_is_normalized(self):
        normalized = TemporalProbabilisticRPGHeuristic._normalize_strategy(
            "baseline_admissible_kmutex"
        )
        self.assertEqual(normalized, "baseline_admissible_kmutex")

    def _exclusive_achiever_heuristic(self):
        # Two achievers of G that delete each other's precondition: structurally
        # mutex (Graphplan delete-interference) and both achieve the same fact.
        return TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="try1",
                    pos_preconditions=frozenset({"R1"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R2"}),
                ),
                SyntheticAction(
                    name="try2",
                    pos_preconditions=frozenset({"R2"}),
                    add_effects=frozenset({"G"}),
                    duration_steps=1,
                    del_effects=frozenset({"R1"}),
                ),
            ],
            facts={"R1", "R2", "G"},
            initial_facts={"R1", "R2"},
            goal_facts={"G"},
        )

    def test_kmutex_score_at_most_admissible(self):
        h = self._exclusive_achiever_heuristic()
        admissible = h.heuristic_score(
            {"R1", "R2"}, {"G"}, fixed_depth=6, strategy="baseline_admissible"
        )
        kmutex = h.heuristic_score(
            {"R1", "R2"}, {"G"}, fixed_depth=6, strategy="baseline_admissible_kmutex"
        )
        self.assertLessEqual(kmutex, admissible + 1e-9)

    def test_kmutex_detects_surviving_clique(self):
        h = self._exclusive_achiever_heuristic()
        h.heuristic_score(
            {"R1", "R2"}, {"G"}, fixed_depth=6, strategy="baseline_admissible_kmutex"
        )
        # G has two mutex achievers, so at least one OR-node clique survived.
        self.assertGreaterEqual(h._kmutex_instr.or_nodes_clique_survived, 1)
        self.assertIn("survival_fraction", h.log_kmutex_summary())

    def test_kmutex_reduces_to_admissible_without_mutex(self):
        # Independent achievers of two different goals: no mutex anywhere, so the
        # kmutex strategy must equal baseline_admissible exactly.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="mk_x",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"X"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="mk_y",
                    pos_preconditions=frozenset({"B"}),
                    add_effects=frozenset({"Y"}),
                    duration_steps=1,
                ),
            ],
            facts={"A", "B", "X", "Y"},
            initial_facts={"A", "B"},
            goal_facts={"X", "Y"},
        )
        admissible = h.heuristic_score(
            {"A", "B"}, {"X", "Y"}, fixed_depth=6, strategy="baseline_admissible"
        )
        kmutex = h.heuristic_score(
            {"A", "B"}, {"X", "Y"}, fixed_depth=6, strategy="baseline_admissible_kmutex"
        )
        self.assertAlmostEqual(kmutex, admissible)
        self.assertEqual(h._kmutex_instr.or_nodes_clique_survived, 0)


if __name__ == "__main__":
    unittest.main()
