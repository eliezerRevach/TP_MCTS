"""Tests for the admissible upper-bound PTRPG heuristic (baseline_admissible).

Covers the pure combination operators (Frechet-min AND, union-bound OR, cumulative
retry update), the standalone reference propagation, and the wired-in
``baseline_admissible`` strategy. The defining property is that every operator is a
provable UPPER bound, so the strategy must never under-estimate relative to the
(non-admissible) ``baseline`` product / noisy-OR combination.
"""

import unittest
from dataclasses import dataclass
from math import prod
from typing import Mapping

from comdp_plus_no_deadline.engines.admissible_temporal_rpg import (
    admissible_and_support,
    union_bound_or_hazard,
    cumulative_retry_update,
    propagate_admissible_temporal_rpg,
)
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset
    add_effects: frozenset
    duration_steps: int
    probabilistic_effects: tuple = ()

    def duration_int(self) -> int:
        return self.duration_steps


@dataclass(frozen=True)
class SyntheticProbabilisticEffect:
    outcomes: Mapping[float, Mapping[object, object]]

    def probability_function(self, state, env):
        del state, env
        return self.outcomes


def _prob_achiever(name, pre, fact, q, duration=1):
    """Action whose only effect adds ``fact`` with probability ``q``."""
    return SyntheticAction(
        name=name,
        pos_preconditions=frozenset(pre),
        add_effects=frozenset(),
        duration_steps=duration,
        probabilistic_effects=(SyntheticProbabilisticEffect({q: {fact: True}}),),
    )


class TestAdmissibleOperators(unittest.TestCase):
    def test_and_support_is_frechet_min(self):
        self.assertEqual(admissible_and_support([], {}), 1.0)  # vacuous conjunction
        self.assertEqual(
            admissible_and_support(["a", "b"], {"a": 0.4, "b": 0.9}), 0.4
        )

    def test_and_support_dominates_product(self):
        probs = {"a": 0.4, "b": 0.5, "c": 0.6}
        keys = list(probs)
        min_bound = admissible_and_support(keys, probs)
        product = prod(probs[k] for k in keys)
        self.assertGreater(min_bound, product)  # min >= prod, strict here

    def test_union_bound_caps_at_one(self):
        self.assertAlmostEqual(union_bound_or_hazard([0.3, 0.3]), 0.6)
        self.assertEqual(union_bound_or_hazard([0.7, 0.7]), 1.0)

    def test_union_bound_dominates_noisy_or(self):
        bs = [0.5, 0.4]
        union = union_bound_or_hazard(bs)
        noisy_or = 1.0 - prod(1.0 - b for b in bs)
        self.assertGreater(union, noisy_or)  # union >= noisy-OR, strict here

    def test_cumulative_update_monotone(self):
        self.assertAlmostEqual(cumulative_retry_update(0.5, 0.5), 0.75)
        self.assertGreaterEqual(
            cumulative_retry_update(0.5, 0.6), cumulative_retry_update(0.5, 0.4)
        )
        self.assertEqual(cumulative_retry_update(1.0, 0.5), 1.0)  # saturated


class TestAdmissibleStrategy(unittest.TestCase):
    def test_strategy_is_registered(self):
        h = TemporalProbabilisticRPGHeuristic(actions=[], facts={"x"})
        self.assertEqual(
            h._normalize_strategy("baseline_admissible"), "baseline_admissible"
        )

    def test_union_bound_beats_baseline_noisy_or(self):
        # Two independent achievers of G, each succeeding with probability 0.5,
        # both firing at layer 1 (duration 1) so both land at layer 1.
        actions = [
            _prob_achiever("g_via_x", [], "G", 0.5),
            _prob_achiever("g_via_y", [], "G", 0.5),
        ]
        h = TemporalProbabilisticRPGHeuristic(
            actions=actions, facts={"G"}, initial_facts=set(), goal_facts={"G"}
        )
        adm = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=1,
                                    strategy="baseline_admissible")
        base = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=1,
                                     strategy="baseline")
        self.assertAlmostEqual(adm.probabilities_by_layer[1]["G"], 1.0)   # min(1, .5+.5)
        self.assertAlmostEqual(base.probabilities_by_layer[1]["G"], 0.75)  # 1-(.5)(.5)

    def test_min_bound_beats_baseline_product(self):
        # G has one achiever needing A and B; A and B are each achieved with
        # prob 0.5 at layer 1, so the AND support for G at layer 1 is min(.5,.5)
        # (admissible) vs .5*.5 (baseline product). G lands at layer 2.
        actions = [
            _prob_achiever("mk_a", [], "A", 0.5),
            _prob_achiever("mk_b", [], "B", 0.5),
            SyntheticAction("mk_g", frozenset({"A", "B"}), frozenset({"G"}), 1),
        ]
        h = TemporalProbabilisticRPGHeuristic(
            actions=actions, facts={"A", "B", "G"},
            initial_facts=set(), goal_facts={"G"},
        )
        adm = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=2,
                                    strategy="baseline_admissible")
        base = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=2,
                                     strategy="baseline")
        self.assertAlmostEqual(adm.probabilities_by_layer[2]["G"], 0.5)    # min(.5,.5)
        self.assertAlmostEqual(base.probabilities_by_layer[2]["G"], 0.25)  # .5*.5
        self.assertGreater(
            adm.probabilities_by_layer[2]["G"], base.probabilities_by_layer[2]["G"]
        )

    def test_admissible_never_below_baseline(self):
        actions = [
            _prob_achiever("mk_a", [], "A", 0.6),
            _prob_achiever("mk_b", [], "B", 0.7),
            _prob_achiever("g_via_x", ["A", "B"], "G", 0.8),
            _prob_achiever("g_via_y", ["A"], "G", 0.4),
        ]
        h = TemporalProbabilisticRPGHeuristic(
            actions=actions, facts={"A", "B", "G"},
            initial_facts=set(), goal_facts={"G"},
        )
        for depth in range(1, 8):
            adm = h.heuristic_score({}, ["G"], aggregation="product",
                                    fixed_depth=depth, strategy="baseline_admissible")
            base = h.heuristic_score({}, ["G"], aggregation="product",
                                     fixed_depth=depth, strategy="baseline")
            self.assertGreaterEqual(float(adm) + 1e-9, float(base),
                                    msg=f"admissible < baseline at depth {depth}")

    def test_reference_matches_strategy(self):
        actions = [
            _prob_achiever("mk_a", [], "A", 0.5),
            _prob_achiever("mk_b", [], "B", 0.5),
            SyntheticAction("mk_g", frozenset({"A", "B"}), frozenset({"G"}), 1),
        ]
        h = TemporalProbabilisticRPGHeuristic(
            actions=actions, facts={"A", "B", "G"},
            initial_facts=set(), goal_facts={"G"},
        )
        depth = 4
        strat = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=depth,
                                      strategy="baseline_admissible")
        ref = propagate_admissible_temporal_rpg(
            h._action_models, h._facts, set(), depth
        )
        for layer in range(depth + 1):
            for fact in ("A", "B", "G"):
                self.assertAlmostEqual(
                    strat.probabilities_by_layer[layer][fact],
                    ref[layer][fact],
                    msg=f"mismatch at layer {layer} fact {fact}",
                )


class TestAdmissibleWiring(unittest.TestCase):
    def test_experiment_common_alias(self):
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import experiment_common as ec  # repo scripts/experiment_common.py
        for key in ("baseline_admissible", "admissible"):
            self.assertIn(key, ec.HEURISTIC_ALIASES)
            self.assertEqual(
                ec.HEURISTIC_ALIASES[key]["temporal_heuristic_strategy"],
                "baseline_admissible",
            )

    def test_parser_alias(self):
        from unified_planning.parser import _parse_temporal_heuristic_strategy as parse
        self.assertEqual(parse("25"), "baseline_admissible")
        self.assertEqual(parse("baseline_admissible"), "baseline_admissible")


if __name__ == "__main__":
    unittest.main()
