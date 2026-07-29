import unittest

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from comdp_plus_no_deadline.engines import MDP
from comdp_plus_no_deadline.engines import greedy_solver as greedy_solver_module
from comdp_plus_no_deadline.engines.greedy_solver import (
    _effective_temporal_depth,
    _score_combination_action,
    regular_greedy_plan,
)


class TestGreedySolver(unittest.TestCase):
    def test_effective_temporal_depth_caps_by_deadline_remaining_time(self):
        self.assertEqual(_effective_temporal_depth(35, 30.4, 35), 4)
        self.assertEqual(_effective_temporal_depth(35, 30.0, 35), 5)
        self.assertEqual(_effective_temporal_depth(10, 12.0, 35), 10)
        self.assertEqual(_effective_temporal_depth(10, 40.0, 35), 0)
        self.assertEqual(_effective_temporal_depth(10, 5.0, None), 10)

    def test_regular_greedy_reaches_simple_goal(self):
        problem = up.model.Problem("simple_no_deadline")
        done = up.model.Fluent("done", BoolType())
        ready = up.model.Fluent("ready", BoolType())
        problem.add_fluent(done, default_initial_value=False)
        problem.add_fluent(ready, default_initial_value=True)

        finish = up.model.action.DurativeAction("finish")
        finish.set_fixed_duration(1)
        finish.add_precondition(OverallPreconditionTiming(), ready, True)
        finish.add_effect(done, True)
        problem.add_action(finish)
        problem.add_goal(done)

        grounder = up.engines.compilers.Grounder()
        grounding_result = grounder._compile(problem)
        ground_problem = grounding_result.problem
        converted = up.engines.Convert_problem(ground_problem)._converted_problem

        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")
        result = regular_greedy_plan(mdp, max_steps=10, heuristic_weight=0.1)
        self.assertEqual(result.success, 1)
        self.assertLessEqual(result.plan_length, 10)

    def test_regular_greedy_supports_probabilistic_rpg_heuristic(self):
        problem = up.model.Problem("simple_no_deadline_probabilistic_rpg")
        done = up.model.Fluent("done", BoolType())
        ready = up.model.Fluent("ready", BoolType())
        problem.add_fluent(done, default_initial_value=False)
        problem.add_fluent(ready, default_initial_value=True)

        finish = up.model.action.DurativeAction("finish")
        finish.set_fixed_duration(1)
        finish.add_precondition(OverallPreconditionTiming(), ready, True)
        finish.add_effect(done, True)
        problem.add_action(finish)
        problem.add_goal(done)

        grounder = up.engines.compilers.Grounder()
        grounding_result = grounder._compile(problem)
        ground_problem = grounding_result.problem
        converted = up.engines.Convert_problem(ground_problem)._converted_problem

        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")
        result = regular_greedy_plan(
            mdp,
            max_steps=10,
            heuristic_weight=0.5,
            heuristic_name="probabilistic_rpg",
            heuristic_aggregation="product",
        )
        self.assertEqual(result.success, 1)
        self.assertLessEqual(result.plan_length, 10)

    def test_regular_greedy_supports_temporal_probabilistic_rpg_heuristic(self):
        problem = up.model.Problem("simple_no_deadline_temporal_probabilistic_rpg")
        done = up.model.Fluent("done", BoolType())
        ready = up.model.Fluent("ready", BoolType())
        problem.add_fluent(done, default_initial_value=False)
        problem.add_fluent(ready, default_initial_value=True)

        finish = up.model.action.DurativeAction("finish")
        finish.set_fixed_duration(1)
        finish.add_precondition(OverallPreconditionTiming(), ready, True)
        finish.add_effect(done, True)
        problem.add_action(finish)
        problem.add_goal(done)

        grounder = up.engines.compilers.Grounder()
        grounding_result = grounder._compile(problem)
        ground_problem = grounding_result.problem
        converted = up.engines.Convert_problem(ground_problem)._converted_problem

        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")
        result = regular_greedy_plan(
            mdp,
            max_steps=10,
            heuristic_weight=0.5,
            heuristic_name="temporal_probabilistic_rpg",
            heuristic_aggregation="product",
            temporal_heuristic_depth=5,
        )
        self.assertEqual(result.success, 1)
        self.assertLessEqual(result.plan_length, 10)

    def test_regular_greedy_deadline_prunes_late_plan(self):
        problem = up.model.Problem("simple_deadline_prune")
        done = up.model.Fluent("done", BoolType())
        ready = up.model.Fluent("ready", BoolType())
        problem.add_fluent(done, default_initial_value=False)
        problem.add_fluent(ready, default_initial_value=True)

        finish = up.model.action.DurativeAction("finish")
        finish.set_fixed_duration(2)
        finish.add_precondition(OverallPreconditionTiming(), ready, True)
        finish.add_effect(done, True)
        problem.add_action(finish)
        problem.add_goal(done)

        grounder = up.engines.compilers.Grounder()
        grounding_result = grounder._compile(problem)
        ground_problem = grounding_result.problem
        converted = up.engines.Convert_problem(ground_problem)._converted_problem

        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")
        result = regular_greedy_plan(
            mdp,
            max_steps=5,
            heuristic_weight=0.5,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=5,
            deadline=1,
        )
        self.assertEqual(result.success, 0)

    def test_combination_temporal_heuristic_uses_next_state_time(self):
        class FakeAction:
            name = "advance"

        class FakeState:
            def __init__(self, current_time):
                self.current_time = current_time

        class FakeMDP:
            def step(self, state, action):
                return False, FakeState(7.0), 0.0

        captured_times = []
        original = greedy_solver_module._state_heuristic_score

        def capture_time(*args, **kwargs):
            captured_times.append(args[8])
            return 0.0

        greedy_solver_module._state_heuristic_score = capture_time
        try:
            _score_combination_action(
                FakeMDP(),
                FakeState(2.0),
                FakeAction(),
                heuristic_weight=0.0,
                noop_penalty=0.0,
                heuristic_name="temporal_probabilistic_rpg",
                heuristic_aggregation="product",
                heuristic_layers=5,
                heuristic_epsilon=1e-6,
                goal_threshold=0.99,
                temporal_heuristic_depth=5,
                deadline=20.0,
            )
        finally:
            greedy_solver_module._state_heuristic_score = original

        self.assertEqual(captured_times, [7.0])


if __name__ == "__main__":
    unittest.main()

