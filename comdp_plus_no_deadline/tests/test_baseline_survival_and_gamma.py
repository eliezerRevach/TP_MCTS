"""
Tests for the baseline_survival_and_gamma temporal heuristic strategy and the
AND-layer gamma correction machinery (comdp_plus_no_deadline/engines/and_gamma.py).

Covers the spec's required cases:
  A. No dependency  -> R_gamma equals the original product (== baseline_survival).
  B. Positive pair  -> factor = gamma_positive.
  C. Negative pair  -> factor = gamma_negative.
  D. Mutex pair     -> factor = gamma_mutex.
  E. Sparse rollout -> learned gamma stays mostly at the static default.
  F. Cache reuse    -> a repeated key hits the gamma memo.
  G. Trace reuse    -> one trace updates many candidate pairs, not only one.
  H. Absence data   -> pair n increments even when both facts are false.
"""

import unittest
from dataclasses import dataclass

from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)
from comdp_plus_no_deadline.engines.and_gamma import (
    MUTEX,
    NEGATIVE,
    POSITIVE,
    SINGLETON,
)


@dataclass(frozen=True)
class SyntheticAction:
    """Minimal action-like object compatible with the heuristic's model builder."""

    name: str
    pos_preconditions: frozenset
    add_effects: frozenset
    duration_steps: int = 1
    del_effects: frozenset = frozenset()
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


def _non_singleton_component(heuristic, action_name):
    comps = heuristic._and_gamma_components_by_name[action_name]
    non_singletons = [c for c in comps if c.comp_type != SINGLETON]
    assert non_singletons, f"expected a non-singleton component on {action_name}"
    return non_singletons[0]


class TestStrategyRecognition(unittest.TestCase):
    def test_strategy_is_normalized(self):
        normalized = TemporalProbabilisticRPGHeuristic._normalize_strategy(
            "baseline_survival_and_gamma"
        )
        self.assertEqual(normalized, "baseline_survival_and_gamma")

    def test_strategy_is_callable_through_heuristic_score(self):
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("a_to_b", frozenset({"A"}), frozenset({"B"})),
            ],
            facts={"A", "B"},
            initial_facts={"A"},
            goal_facts={"B"},
        )
        score = h.heuristic_score(
            {"A"}, {"B"}, fixed_depth=3, strategy="baseline_survival_and_gamma"
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestCaseA_NoDependency(unittest.TestCase):
    def _independent_two_precondition_domain(self):
        # X and Y are achieved by separate actions with disjoint preconditions:
        # no shared achiever, no delete relation -> no static dependency.
        return TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("mkX", frozenset({"S"}), frozenset({"X"})),
                SyntheticAction("mkY", frozenset({"T"}), frozenset({"Y"})),
                SyntheticAction("use", frozenset({"X", "Y"}), frozenset({"G"})),
            ],
            facts={"S", "T", "X", "Y", "G"},
            initial_facts={"S", "T"},
            goal_facts={"G"},
        )

    def test_components_are_all_singletons(self):
        h = self._independent_two_precondition_domain()
        h._ensure_and_gamma_built()
        comps = h._and_gamma_components_by_name["use"]
        self.assertTrue(all(c.comp_type == SINGLETON for c in comps))
        # Two independent preconditions -> factor is exactly 1.
        self.assertAlmostEqual(h._and_gamma_factor("use"), 1.0, places=9)

    def test_score_equals_baseline_survival(self):
        h = self._independent_two_precondition_domain()
        depth = 8
        base = h.heuristic_score(
            {"S", "T"}, {"G"}, fixed_depth=depth, strategy="baseline_survival"
        )
        gamma = h.heuristic_score(
            {"S", "T"}, {"G"}, fixed_depth=depth, strategy="baseline_survival_and_gamma"
        )
        self.assertAlmostEqual(base, gamma, places=9)


class TestCaseBCD_ComponentFactors(unittest.TestCase):
    def test_positive_pair_factor(self):
        # mk adds BOTH f1 and f2 -> shared achiever -> positive edge.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("mk", frozenset({"S"}), frozenset({"f1", "f2"})),
                SyntheticAction("use", frozenset({"f1", "f2"}), frozenset({"G"})),
            ],
            facts={"S", "f1", "f2", "G"},
            initial_facts={"S"},
            goal_facts={"G"},
        )
        h._ensure_and_gamma_built()
        comp = _non_singleton_component(h, "use")
        self.assertEqual(comp.comp_type, POSITIVE)
        self.assertIn(POSITIVE, comp.edge_kinds)
        self.assertAlmostEqual(
            h._and_gamma_factor("use"), h._and_gamma_config.gamma_positive, places=9
        )

    def test_negative_pair_factor(self):
        # del1 deletes f1 and requires f2 -> negative edge, but not toggling.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("use", frozenset({"f1", "f2"}), frozenset({"G"})),
                SyntheticAction(
                    "del1", frozenset({"f2"}), frozenset(), del_effects=frozenset({"f1"})
                ),
            ],
            facts={"f1", "f2", "G"},
            initial_facts={"f1", "f2"},
            goal_facts={"G"},
        )
        h._ensure_and_gamma_built()
        comp = _non_singleton_component(h, "use")
        self.assertEqual(comp.comp_type, NEGATIVE)
        self.assertAlmostEqual(
            h._and_gamma_factor("use"), h._and_gamma_config.gamma_negative, places=9
        )

    def test_mutex_pair_factor(self):
        # p: del f1, add f2 ; q: del f2, add f1 -> toggling contradiction -> mutex.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    "p", frozenset(), frozenset({"f2"}), del_effects=frozenset({"f1"})
                ),
                SyntheticAction(
                    "q", frozenset(), frozenset({"f1"}), del_effects=frozenset({"f2"})
                ),
                SyntheticAction("use", frozenset({"f1", "f2"}), frozenset({"G"})),
            ],
            facts={"f1", "f2", "G"},
            initial_facts={"f1", "f2"},
            goal_facts={"G"},
        )
        h._ensure_and_gamma_built()
        comp = _non_singleton_component(h, "use")
        self.assertEqual(comp.comp_type, MUTEX)
        self.assertIn(MUTEX, comp.edge_kinds)
        self.assertAlmostEqual(
            h._and_gamma_factor("use"), h._and_gamma_config.gamma_mutex, places=9
        )


class TestCalibrationStatistics(unittest.TestCase):
    def _positive_pair_heuristic(self):
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("mk", frozenset({"S"}), frozenset({"f1", "f2"})),
                SyntheticAction("use", frozenset({"f1", "f2"}), frozenset({"G"})),
            ],
            facts={"S", "f1", "f2", "G"},
            initial_facts={"S"},
            goal_facts={"G"},
        )
        h._ensure_and_gamma_built()
        return h

    def test_case_E_sparse_rollout_stays_near_default(self):
        h = self._positive_pair_heuristic()
        cal = h._and_gamma_calibrator
        comp = _non_singleton_component(h, "use")
        default = h._and_gamma_config.gamma_positive

        # 10 anti-correlated samples (< min_n=30): f1 alone, then f2 alone.
        trace = [frozenset({"f1"})] * 5 + [frozenset({"f2"})] * 5
        cal.ingest_trace(trace)

        learned_raw = 0.0  # P(f1&f2)=0 with P(f1)P(f2)=0.25 -> raw learned ~ 0
        gamma = cal.gamma_for_component(comp)
        # Sparse data must not dominate: result stays closer to the default than to
        # the raw learned value.
        self.assertLess(abs(gamma - default), abs(gamma - learned_raw))

    def test_case_F_cache_reuse(self):
        h = self._positive_pair_heuristic()
        cal = h._and_gamma_calibrator
        comp = _non_singleton_component(h, "use")

        hits_before = cal.cache_hits
        first = cal.gamma_for_component(comp)
        second = cal.gamma_for_component(comp)
        self.assertEqual(first, second)
        # The second lookup is served from the gamma memo.
        self.assertGreater(cal.cache_hits, hits_before)

    def test_case_G_trace_reuse_updates_many_pairs(self):
        # use has three preconditions -> three co-occurrence candidate pairs.
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("mk", frozenset({"S"}), frozenset({"f1", "f2", "f3"})),
                SyntheticAction("use", frozenset({"f1", "f2", "f3"}), frozenset({"G"})),
            ],
            facts={"S", "f1", "f2", "f3", "G"},
            initial_facts={"S"},
            goal_facts={"G"},
        )
        h._ensure_and_gamma_built()
        cal = h._and_gamma_calibrator
        self.assertGreaterEqual(len(cal.candidate_pairs), 3)

        # A SINGLE trace updates every tracked candidate pair, not just one.
        cal.ingest_trace([frozenset({"f1", "f2", "f3"})])
        updated = [p for p, s in cal.pair_stats.items() if s.n > 0]
        self.assertGreaterEqual(len(updated), 2)

    def test_case_H_absence_data_counts(self):
        h = self._positive_pair_heuristic()
        cal = h._and_gamma_calibrator
        pair = frozenset({"f1", "f2"})

        # A state where BOTH facts are absent must still increment n.
        cal.ingest_trace([frozenset()])
        stat = cal.pair_stats[pair]
        self.assertGreaterEqual(stat.n, 1)
        self.assertEqual(stat.both, 0)
        self.assertEqual(stat.singles.get("f1", 0), 0)


class TestDiagnostics(unittest.TestCase):
    def test_diagnostics_report_components_and_types(self):
        h = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction("mk", frozenset({"S"}), frozenset({"f1", "f2"})),
                SyntheticAction("use", frozenset({"f1", "f2"}), frozenset({"G"})),
            ],
            facts={"S", "f1", "f2", "G"},
            initial_facts={"S"},
            goal_facts={"G"},
        )
        # Drive a real evaluation so and_calls / cache counters move.
        h.heuristic_score(
            {"S"}, {"G"}, fixed_depth=5, strategy="baseline_survival_and_gamma"
        )
        diag = h._and_gamma_calibrator.diagnostics()
        self.assertGreater(diag["components_found"], 0)
        self.assertIn(POSITIVE, diag["component_type_counts"])
        self.assertGreater(diag["and_calls"], 0)


if __name__ == "__main__":
    unittest.main()
