"""Tests for the marginal-consistent LP OR-layer bound (baseline_admissible_lp).

Covers the pure LP operator (doc Section 9.3) and the wired-in
``baseline_admissible_lp`` strategy. The defining properties are:

  - ADMISSIBLE: the LP maximises over all local joints consistent with the stored
    marginal upper bounds, so it never under-estimates relative to ``baseline``.
  - TIGHTER: it is never looser than the union bound used by ``baseline_admissible``
    and is strictly tighter when achievers share preconditions.
  - SAFE FALLBACK: when the local fact set is too large it degrades exactly to
    ``baseline_admissible`` (the union bound).
"""

import unittest
from dataclasses import dataclass
from typing import Mapping

from comdp_plus_no_deadline.engines.admissible_lp import (
    MarginalConsistentORBound,
    marginal_consistent_or_hazard,
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


def _exact_independent_or_of_and(achievers, marginals):
    """Exact P(OR-of-AND) when the local facts are independent with the given
    marginals and each achiever's action noise q is independent. Ground-truth
    feasible point of the LP's marginal-consistent set, so the LP must dominate it.
    """
    from itertools import product

    facts = sorted({f for pre, _q in achievers for f in pre})
    total = 0.0
    for bits in product((0, 1), repeat=len(facts)):
        world = {f: b for f, b in zip(facts, bits)}
        world_prob = 1.0
        for f in facts:
            p = marginals[f]
            world_prob *= p if world[f] else (1.0 - p)
        survive = 1.0
        for pre, q in achievers:
            if all(world[f] for f in pre):
                survive *= 1.0 - q
        total += world_prob * (1.0 - survive)
    return total


def _prob_achiever(name, pre, fact, q, duration=1):
    """Action whose only effect adds ``fact`` with probability ``q``."""
    return SyntheticAction(
        name=name,
        pos_preconditions=frozenset(pre),
        add_effects=frozenset(),
        duration_steps=duration,
        probabilistic_effects=(SyntheticProbabilisticEffect({q: {fact: True}}),),
    )


class TestLPOperator(unittest.TestCase):
    def test_shared_precondition_avoids_double_counting(self):
        # Doc example: e1 = x^y -> f, e2 = x^z -> f, P(x)=P(y)=P(z)=0.5, q=1.
        # (x^y) v (x^z) = x ^ (y v z), so the OR cannot exceed P(x)=0.5, while the
        # union bound would report 1.0.
        bound = MarginalConsistentORBound()
        value = bound.or_hazard(
            [(("x", "y"), 1.0), (("x", "z"), 1.0)],
            {"x": 0.5, "y": 0.5, "z": 0.5},
        )
        self.assertAlmostEqual(value, 0.5, places=6)

    def test_disjoint_achievers_match_union_bound(self):
        # No shared facts -> the LP can stack the mass and recovers the union bound
        # (verified via the ungated solver). The gated path skips it as a no-op.
        bound = MarginalConsistentORBound()
        ach, marg = [(("a",), 1.0), (("b",), 1.0)], {"a": 0.3, "b": 0.3}
        self.assertAlmostEqual(bound.or_hazard_ungated(ach, marg), 0.6, places=6)

    def test_gate_skips_disjoint_achievers(self):
        # The fast-exit gate returns None (caller uses the union-bound fallback)
        # whenever no precondition is shared between distinct achievers.
        bound = MarginalConsistentORBound()
        self.assertIsNone(
            bound.or_hazard([(("a",), 1.0), (("b",), 1.0)], {"a": 0.3, "b": 0.3})
        )
        self.assertIsNone(bound.or_hazard([((), 0.4)], {}))  # single achiever

    def test_unconditioned_achiever(self):
        value = MarginalConsistentORBound().or_hazard_ungated([((), 0.4)], {})
        self.assertAlmostEqual(value, 0.4, places=6)

    def test_never_looser_than_union_bound(self):
        # Random-ish structured case: LP <= union bound always.
        bound = MarginalConsistentORBound()
        achievers = [(("x", "y"), 0.9), (("x", "z"), 0.8), (("w",), 0.5)]
        marg = {"x": 0.6, "y": 0.7, "z": 0.4, "w": 0.5}
        lp = bound.or_hazard(achievers, marg)
        union = min(
            1.0,
            sum(q * min(marg[f] for f in pre) for pre, q in achievers),
        )
        self.assertLessEqual(lp, union + 1e-9)

    def test_independent_value_mode(self):
        # Two disjoint achievers, q=0.5 each, certain preconditions: union -> 1.0,
        # but independent action noise gives 1-(1-.5)(1-.5)=0.75.
        bound = MarginalConsistentORBound(value_mode="independent")
        value = bound.or_hazard_ungated(
            [(("a",), 0.5), (("b",), 0.5)], {"a": 1.0, "b": 1.0}
        )
        self.assertAlmostEqual(value, 0.75, places=6)

    def test_fallback_when_too_many_local_facts(self):
        bound = MarginalConsistentORBound(max_local_facts=2)
        value = bound.or_hazard(
            [(("a", "b", "c"), 1.0)], {"a": 0.5, "b": 0.5, "c": 0.5}
        )
        self.assertIsNone(value)  # signals "use the union-bound fallback"

    def test_signature_cache_reuses_structure(self):
        bound = MarginalConsistentORBound()
        ach = [(("x", "y"), 1.0), (("x", "z"), 1.0)]
        bound.or_hazard(ach, {"x": 0.5, "y": 0.5, "z": 0.5})
        bound.or_hazard(ach, {"x": 0.2, "y": 0.9, "z": 0.1})  # same formula, new RHS
        self.assertEqual(len(bound._prepared), 1)


class TestLPStrategy(unittest.TestCase):
    def _shared_precondition_heuristic(self):
        # x, y, z each reachable with prob 0.5 at layer 1; two G achievers
        # (x^y and x^z) land at layer 2 -> shared precondition x.
        actions = [
            _prob_achiever("mk_x", [], "x", 0.5),
            _prob_achiever("mk_y", [], "y", 0.5),
            _prob_achiever("mk_z", [], "z", 0.5),
            _prob_achiever("g_via_xy", ["x", "y"], "G", 1.0),
            _prob_achiever("g_via_xz", ["x", "z"], "G", 1.0),
        ]
        return TemporalProbabilisticRPGHeuristic(
            actions=actions,
            facts={"x", "y", "z", "G"},
            initial_facts=set(),
            goal_facts={"G"},
        )

    def test_strategy_is_registered(self):
        h = TemporalProbabilisticRPGHeuristic(actions=[], facts={"x"})
        self.assertEqual(
            h._normalize_strategy("baseline_admissible_lp"), "baseline_admissible_lp"
        )

    def test_lp_tighter_than_union_on_shared_precondition(self):
        h = self._shared_precondition_heuristic()
        lp = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=2,
                                   strategy="baseline_admissible_lp")
        adm = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=2,
                                    strategy="baseline_admissible")
        self.assertAlmostEqual(lp.probabilities_by_layer[2]["G"], 0.5, places=6)
        self.assertAlmostEqual(adm.probabilities_by_layer[2]["G"], 1.0, places=6)
        self.assertLess(
            lp.probabilities_by_layer[2]["G"],
            adm.probabilities_by_layer[2]["G"],
        )

    def test_lp_is_admissible_upper_bound_under_independence(self):
        # Admissibility: the LP (union value mode) must be >= the EXACT OR-of-AND
        # probability of any joint consistent with the marginals, including the
        # independent one. (Note: the non-admissible ``baseline`` noisy-OR can
        # over-shoot ABOVE the LP here, because it double-counts the shared
        # precondition x -- which is precisely the looseness the LP removes -- so
        # baseline is NOT a valid lower bound to compare against.)
        bound = MarginalConsistentORBound()
        achievers = [(("x", "y"), 0.9), (("x", "z"), 0.8), (("w",), 0.6)]
        marg = {"x": 0.6, "y": 0.7, "z": 0.4, "w": 0.5}
        lp = bound.or_hazard(achievers, marg)
        exact = _exact_independent_or_of_and(achievers, marg)
        self.assertGreaterEqual(lp + 1e-9, exact)

    def test_lp_at_most_admissible_every_depth(self):
        h = self._shared_precondition_heuristic()
        for depth in range(1, 8):
            lp = h.heuristic_score({}, ["G"], aggregation="product",
                                   fixed_depth=depth, strategy="baseline_admissible_lp")
            adm = h.heuristic_score({}, ["G"], aggregation="product",
                                    fixed_depth=depth, strategy="baseline_admissible")
            self.assertLessEqual(
                float(lp), float(adm) + 1e-9,
                msg=f"LP > admissible at depth {depth}",
            )

    def test_fallback_matches_admissible(self):
        # Force the fallback (cap below the local fact count): the LP strategy must
        # then reproduce baseline_admissible exactly (the union bound).
        h = self._shared_precondition_heuristic()
        h._admissible_lp_max_local_facts = 1  # |U|=3 at the G layer -> fallback
        lp = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=2,
                                   strategy="baseline_admissible_lp")
        adm = h.heuristic_propagate({}, goal_facts=["G"], fixed_depth=2,
                                    strategy="baseline_admissible")
        for layer in range(3):
            for fact in ("x", "y", "z", "G"):
                self.assertAlmostEqual(
                    lp.probabilities_by_layer[layer][fact],
                    adm.probabilities_by_layer[layer][fact],
                    places=6,
                    msg=f"fallback mismatch at layer {layer} fact {fact}",
                )


class TestLPWiring(unittest.TestCase):
    def test_experiment_common_alias(self):
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import experiment_common as ec
        for key in ("baseline_admissible_lp", "admissible_lp"):
            self.assertIn(key, ec.HEURISTIC_ALIASES)
            self.assertEqual(
                ec.HEURISTIC_ALIASES[key]["temporal_heuristic_strategy"],
                "baseline_admissible_lp",
            )

    def test_parser_alias(self):
        from unified_planning.parser import _parse_temporal_heuristic_strategy as parse
        self.assertEqual(parse("26"), "baseline_admissible_lp")
        self.assertEqual(parse("baseline_admissible_lp"), "baseline_admissible_lp")


if __name__ == "__main__":
    unittest.main()
