import unittest
from dataclasses import dataclass, field
from typing import Mapping

from comdp_plus_no_deadline.engines.probabilistic_rpg import (
    ProbabilisticOptimisticRPGHeuristic,
    compute_precondition_support,
)


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset[str]
    add_effects: frozenset[str] = frozenset()
    probabilistic_add_effects: Mapping[str, float] = field(default_factory=dict)


class TestProbabilisticOptimisticRPG(unittest.TestCase):
    def test_compute_precondition_support_conjunction(self):
        support = compute_precondition_support(
            frozenset({"A", "B"}),
            {"A": 0.8, "B": 0.5},
        )
        self.assertAlmostEqual(support, 0.4)

    def test_compute_precondition_support_disjoint_dnf(self):
        support = compute_precondition_support(
            (frozenset({"A"}), frozenset({"B"})),
            {"A": 0.8, "B": 0.5},
        )
        self.assertAlmostEqual(support, 0.9)

    def test_compute_precondition_support_overlap_strict_raises(self):
        with self.assertRaises(NotImplementedError):
            compute_precondition_support(
                (
                    frozenset({"A", "B", "C"}),
                    frozenset({"C", "D", "E"}),
                ),
                {"A": 0.8, "B": 0.5, "C": 0.6, "D": 0.4, "E": 0.7},
                strict=True,
            )

    def test_single_retryable_action(self):
        heuristic = ProbabilisticOptimisticRPGHeuristic(
            [
                SyntheticAction(
                    "a_to_b",
                    frozenset({"A"}),
                    probabilistic_add_effects={"B": 0.3},
                )
            ],
            facts={"A", "B"},
        )

        result = heuristic.heuristic_propagate(
            {"A"},
            goal_facts={"B"},
            max_layers=2,
            goal_threshold=0.999,
            debug=True,
        )

        self.assertAlmostEqual(result.traces[1].fact_probabilities["B"], 0.3)
        self.assertAlmostEqual(result.traces[2].fact_probabilities["B"], 0.51)
        self.assertEqual(
            result.traces[1].action_support_details["a_to_b"]["mode"],
            "exact_conjunctive",
        )

    def test_parallel_achievers_union_their_hazards(self):
        heuristic = ProbabilisticOptimisticRPGHeuristic(
            [
                SyntheticAction(
                    "a_to_x",
                    frozenset({"A"}),
                    probabilistic_add_effects={"X": 0.2},
                ),
                SyntheticAction(
                    "b_to_x",
                    frozenset({"B"}),
                    probabilistic_add_effects={"X": 0.4},
                ),
            ],
            facts={"A", "B", "X"},
        )

        result = heuristic.heuristic_propagate(
            {"A", "B"},
            goal_facts={"X"},
            max_layers=1,
            goal_threshold=0.999,
            debug=True,
        )

        self.assertAlmostEqual(result.traces[1].fact_probabilities["X"], 0.52)

    def test_chain_matches_worked_example(self):
        heuristic = ProbabilisticOptimisticRPGHeuristic(
            [
                SyntheticAction("a_to_ab", frozenset({"A"}), probabilistic_add_effects={"AB": 0.3}),
                SyntheticAction("b_to_ab", frozenset({"B"}), probabilistic_add_effects={"AB": 0.6}),
                SyntheticAction("c_to_c1", frozenset({"C"}), probabilistic_add_effects={"C1": 0.9}),
                SyntheticAction("ab_to_d", frozenset({"AB"}), probabilistic_add_effects={"D": 0.1}),
                SyntheticAction("c1_to_d", frozenset({"C1"}), probabilistic_add_effects={"D": 0.3}),
            ],
            facts={"A", "B", "C", "AB", "C1", "D"},
        )

        result = heuristic.heuristic_propagate(
            {"A", "B", "C"},
            goal_facts={"D"},
            max_layers=2,
            goal_threshold=0.999,
            debug=True,
        )

        self.assertAlmostEqual(result.traces[1].fact_probabilities["AB"], 0.72)
        self.assertAlmostEqual(result.traces[1].fact_probabilities["C1"], 0.9)
        self.assertAlmostEqual(result.traces[2].fact_probabilities["AB"], 0.9216)
        self.assertAlmostEqual(result.traces[2].fact_probabilities["C1"], 0.99)
        self.assertAlmostEqual(result.traces[2].fact_probabilities["D"], 0.32256)

    def test_cycles_trigger_fixed_point_mode(self):
        heuristic = ProbabilisticOptimisticRPGHeuristic(
            [
                SyntheticAction("seed_to_a", frozenset({"SEED"}), probabilistic_add_effects={"A": 0.4}),
                SyntheticAction("a_to_b", frozenset({"A"}), probabilistic_add_effects={"B": 0.5}),
                SyntheticAction("b_to_a", frozenset({"B"}), probabilistic_add_effects={"A": 0.5}),
            ],
            facts={"SEED", "A", "B"},
        )

        result = heuristic.heuristic_propagate(
            {"SEED"},
            goal_facts={"A", "B"},
            max_layers=20,
            epsilon=1e-9,
            goal_threshold=0.999999,
            debug=True,
        )

        self.assertTrue(result.cyclic_dependency)
        self.assertTrue(result.warnings)
        previous_a = 0.0
        previous_b = 0.0
        for trace in result.traces[1:]:
            self.assertGreaterEqual(trace.fact_probabilities["A"], previous_a)
            self.assertGreaterEqual(trace.fact_probabilities["B"], previous_b)
            previous_a = trace.fact_probabilities["A"]
            previous_b = trace.fact_probabilities["B"]
        self.assertGreater(result.probabilities["A"], 0.0)
        self.assertGreater(result.probabilities["B"], 0.0)

    def test_product_and_min_scores(self):
        heuristic = ProbabilisticOptimisticRPGHeuristic(
            [
                SyntheticAction("a_to_g1", frozenset({"A"}), probabilistic_add_effects={"G1": 0.6}),
                SyntheticAction("a_to_g2", frozenset({"A"}), probabilistic_add_effects={"G2": 0.8}),
            ],
            facts={"A", "G1", "G2"},
        )

        product_score = heuristic.heuristic_score(
            {"A"},
            {"G1", "G2"},
            aggregation="product",
            max_layers=1,
            goal_threshold=0.999,
        )
        min_score = heuristic.heuristic_score(
            {"A"},
            {"G1", "G2"},
            aggregation="min",
            max_layers=1,
            goal_threshold=0.999,
        )

        self.assertAlmostEqual(product_score, 0.48)
        self.assertAlmostEqual(min_score, 0.6)


if __name__ == "__main__":
    unittest.main()
