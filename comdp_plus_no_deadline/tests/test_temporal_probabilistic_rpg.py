import unittest
from dataclasses import dataclass

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


@dataclass(frozen=True)
class SyntheticAction:
    name: str
    pos_preconditions: frozenset[str]
    add_effects: frozenset[str]
    duration_steps: int

    def duration_int(self) -> int:
        return self.duration_steps


class TestTemporalProbabilisticRPG(unittest.TestCase):
    def test_duration_delays_effect_arrival(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b_d2",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=2,
                )
            ],
            facts={"A", "B"},
        )

        score_depth1 = heuristic.heuristic_score({"A"}, {"B"}, fixed_depth=1)
        score_depth2 = heuristic.heuristic_score({"A"}, {"B"}, fixed_depth=2)

        self.assertAlmostEqual(score_depth1, 0.0)
        self.assertAlmostEqual(score_depth2, 1.0)

    def test_precondition_gated_activation(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="seed_to_a",
                    pos_preconditions=frozenset({"SEED"}),
                    add_effects=frozenset({"A"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                ),
            ],
            facts={"SEED", "A", "B"},
        )

        score_depth1 = heuristic.heuristic_score({"SEED"}, {"B"}, fixed_depth=1)
        score_depth2 = heuristic.heuristic_score({"SEED"}, {"B"}, fixed_depth=2)

        self.assertAlmostEqual(score_depth1, 0.0)
        self.assertAlmostEqual(score_depth2, 1.0)

    def test_dp_cache_reuse(self):
        heuristic = TemporalProbabilisticRPGHeuristic(
            actions=[
                SyntheticAction(
                    name="a_to_b",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"B"}),
                    duration_steps=1,
                ),
                SyntheticAction(
                    name="a_to_c",
                    pos_preconditions=frozenset({"A"}),
                    add_effects=frozenset({"C"}),
                    duration_steps=1,
                ),
            ],
            facts={"A", "B", "C"},
        )

        first = heuristic.heuristic_propagate({"A"}, fixed_depth=2)
        second = heuristic.heuristic_propagate({"A"}, fixed_depth=2)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)

    def test_split_durative_end_effect_keeps_original_duration(self):
        problem = up.model.Problem("split_duration_preserved")
        done = up.model.Fluent("done", BoolType())
        ready = up.model.Fluent("ready", BoolType())
        problem.add_fluent(done, default_initial_value=False)
        problem.add_fluent(ready, default_initial_value=True)

        finish = up.model.action.DurativeAction("finish")
        finish.set_fixed_duration(4)
        finish.add_precondition(OverallPreconditionTiming(), ready, True)
        finish.add_effect(done, True)
        problem.add_action(finish)
        problem.add_goal(done)

        ground_problem = up.engines.compilers.Grounder()._compile(problem).problem
        converted_problem = up.engines.Convert_problem(ground_problem)._converted_problem
        heuristic = TemporalProbabilisticRPGHeuristic.from_problem(converted_problem)

        initial_predicates = {
            fact
            for fact, value in converted_problem.initial_values.items()
            if value.bool_constant_value()
        }
        initial_state = up.engines.State(initial_predicates)
        score_depth3 = heuristic.heuristic_score(initial_state, converted_problem.goals, fixed_depth=3)
        score_depth4 = heuristic.heuristic_score(initial_state, converted_problem.goals, fixed_depth=4)

        self.assertAlmostEqual(score_depth3, 0.0)
        self.assertAlmostEqual(score_depth4, 1.0)


if __name__ == "__main__":
    unittest.main()
