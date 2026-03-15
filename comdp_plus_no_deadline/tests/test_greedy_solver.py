import unittest

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from comdp_plus_no_deadline.engines import MDP
from comdp_plus_no_deadline.engines.greedy_solver import regular_greedy_plan


class TestGreedySolver(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

