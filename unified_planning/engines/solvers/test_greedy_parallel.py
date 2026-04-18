import unittest

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from unified_planning.engines.mdp import MDP
from unified_planning.engines.solvers.greedy_parallel import plan as greedy_parallel_plan


def _build_split_problem(duration: int):
    problem = up.model.Problem("greedy_parallel_sanity")
    done = up.model.Fluent("done", BoolType())
    ready = up.model.Fluent("ready", BoolType())
    problem.add_fluent(done, default_initial_value=False)
    problem.add_fluent(ready, default_initial_value=True)

    finish = up.model.action.DurativeAction("finish")
    finish.set_fixed_duration(duration)
    finish.add_precondition(OverallPreconditionTiming(), ready, True)
    finish.add_effect(done, True)
    problem.add_action(finish)
    problem.add_goal(done)

    ground_problem = up.engines.compilers.Grounder()._compile(problem).problem
    return up.engines.Convert_problem(ground_problem)._converted_problem


class TestGreedyParallel(unittest.TestCase):
    def test_greedy_parallel_reaches_goal_when_deadline_allows(self):
        converted_problem = _build_split_problem(duration=2)
        converted_problem.set_deadline(
            up.model.timing.Timing(
                delay=3,
                timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START),
            )
        )
        mdp = MDP(converted_problem, discount_factor=0.95, reward_mode="terminal")

        success, makespan = greedy_parallel_plan(
            mdp,
            steps=20,
            search_time=1,
            search_depth=5,
            exploration_constant=10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=10,
            temporal_heuristic_strategy="atom_half_split",
        )
        self.assertEqual(success, 1)
        self.assertLessEqual(makespan, 3)

    def test_greedy_parallel_respects_deadline(self):
        converted_problem = _build_split_problem(duration=2)
        converted_problem.set_deadline(
            up.model.timing.Timing(
                delay=1,
                timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START),
            )
        )
        mdp = MDP(converted_problem, discount_factor=0.95, reward_mode="terminal")

        success, makespan = greedy_parallel_plan(
            mdp,
            steps=20,
            search_time=1,
            search_depth=5,
            exploration_constant=10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=10,
            temporal_heuristic_strategy="baseline",
        )
        self.assertEqual(success, 0)
        self.assertEqual(makespan, -float("inf"))

    def test_greedy_parallel_correlation_pessimistic_runs(self):
        converted_problem = _build_split_problem(duration=2)
        converted_problem.set_deadline(
            up.model.timing.Timing(
                delay=3,
                timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START),
            )
        )
        mdp = MDP(converted_problem, discount_factor=0.95, reward_mode="terminal")

        success, makespan = greedy_parallel_plan(
            mdp,
            steps=20,
            search_time=1,
            search_depth=5,
            exploration_constant=10,
            heuristic_name="baseline_passmistic",
            temporal_heuristic_depth=10,
            temporal_heuristic_strategy="baseline",
        )
        self.assertEqual(success, 1)
        self.assertLessEqual(makespan, 3)

    def test_greedy_parallel_correlation_optimistic_runs(self):
        converted_problem = _build_split_problem(duration=2)
        converted_problem.set_deadline(
            up.model.timing.Timing(
                delay=3,
                timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START),
            )
        )
        mdp = MDP(converted_problem, discount_factor=0.95, reward_mode="terminal")

        success, makespan = greedy_parallel_plan(
            mdp,
            steps=20,
            search_time=1,
            search_depth=5,
            exploration_constant=10,
            heuristic_name="baseline_optimstic",
            temporal_heuristic_depth=10,
            temporal_heuristic_strategy="baseline",
        )
        self.assertEqual(success, 1)
        self.assertLessEqual(makespan, 3)


if __name__ == "__main__":
    unittest.main()
