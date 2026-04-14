"""
Regression tests for baseline_cached cache propagation in MCTS (plan / combination_plan).

Checks:
  - baseline_cached runs through C_MCTS (plan) and MCTS (combination_plan) without crashing
    and reaches the goal when the deadline allows.
  - Classical trpg and baseline strategies are unaffected by the new cache wiring.
"""

import unittest

import unified_planning as up
from unified_planning.shortcuts import BoolType, OverallPreconditionTiming

from unified_planning.engines.mdp import MDP
from unified_planning.engines.solvers.mcts import plan as mcts_plan, combination_plan


def _build_split_problem(duration: int, deadline: int):
    problem = up.model.Problem("mcts_cache_sanity")
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
    converted = up.engines.Convert_problem(ground_problem)._converted_problem
    converted.set_deadline(
        up.model.timing.Timing(
            delay=deadline,
            timepoint=up.model.timing.Timepoint(up.model.timing.TimepointKind.START),
        )
    )
    return converted


class TestMCTSBaselineCached(unittest.TestCase):
    """baseline_cached in plan() (C_MCTS) must succeed when deadline allows."""

    def test_plan_baseline_cached_reaches_goal(self):
        converted = _build_split_problem(duration=2, deadline=5)
        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")

        success, makespan = mcts_plan(
            mdp,
            steps=30,
            search_time=1,
            search_depth=5,
            exploration_constant=0.5,
            selection_type="avg",
            k=10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=10,
            temporal_heuristic_strategy="baseline_cached",
        )
        self.assertEqual(success, 1)
        self.assertLessEqual(makespan, 5)

    def test_plan_trpg_unaffected(self):
        """Classical trpg path must still work after the cache wiring changes."""
        converted = _build_split_problem(duration=2, deadline=5)
        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")

        success, makespan = mcts_plan(
            mdp,
            steps=30,
            search_time=1,
            search_depth=5,
            exploration_constant=10.0,
            selection_type="avg",
            k=10,
            heuristic_name="trpg",
        )
        self.assertEqual(success, 1)
        self.assertLessEqual(makespan, 5)

    def test_plan_baseline_unaffected(self):
        """Non-cached baseline strategy must still work (no cache threading)."""
        converted = _build_split_problem(duration=2, deadline=5)
        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")

        success, makespan = mcts_plan(
            mdp,
            steps=30,
            search_time=1,
            search_depth=5,
            exploration_constant=0.5,
            selection_type="avg",
            k=10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=10,
            temporal_heuristic_strategy="baseline",
        )
        self.assertEqual(success, 1)
        self.assertLessEqual(makespan, 5)

    def test_plan_baseline_cached_respects_deadline(self):
        """baseline_cached must not falsely succeed when deadline is too tight."""
        converted = _build_split_problem(duration=2, deadline=1)
        mdp = MDP(converted, discount_factor=0.95, reward_mode="terminal")

        success, _ = mcts_plan(
            mdp,
            steps=30,
            search_time=1,
            search_depth=5,
            exploration_constant=0.5,
            selection_type="avg",
            k=10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=10,
            temporal_heuristic_strategy="baseline_cached",
        )
        self.assertEqual(success, 0)


if __name__ == "__main__":
    unittest.main()
