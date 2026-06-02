"""
Tests for PTRPG-guided terminal rollout (MCTS leaf evaluation).
"""

import logging
import sys
import unittest
from unittest import mock

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import unified_planning as up
    from unified_planning.shortcuts import BoolType, OverallPreconditionTiming
    from unified_planning.engines.mdp import MDP
    from unified_planning.engines.solvers.greedy_parallel import pick_best_action
    from unified_planning.engines.solvers.mcts import C_MCTS, plan as mcts_plan
    from unified_planning.engines.solvers.ptrpg_guided_rollout import (
        RolloutConfig,
        ptrpg_guided_terminal_rollout,
        remaining_deadline,
        resolve_rollout_policy,
    )
    from unified_planning.engines.utils import create_init_stn
finally:
    sys.argv = _ORIGINAL_ARGV


def _build_split_problem(duration: int, deadline: int):
    problem = up.model.Problem("ptrpg_rollout_test")
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


class TestPtrpgGuidedTerminalRollout(unittest.TestCase):
    def setUp(self):
        self.converted = _build_split_problem(duration=2, deadline=8)
        self.mdp = MDP(self.converted, discount_factor=0.95, reward_mode="terminal")
        self.stn = create_init_stn(self.mdp)
        self.state = self.mdp.initial_state()
        self.config = RolloutConfig(
            policy_strategy=resolve_rollout_policy("baseline_survival_resolution"),
            max_steps=20,
        )

    def test_rollout_returns_binary(self):
        value = ptrpg_guided_terminal_rollout(
            mdp=self.mdp,
            state=self.state,
            stn=self.stn,
            previous_action_node=None,
            config=self.config,
            temporal_heuristic_depth=8,
        )
        self.assertIn(value, (0.0, 1.0))

    def test_loop_guard_returns_zero(self):
        config = RolloutConfig(
            policy_strategy=self.config.policy_strategy,
            max_steps=50,
            loop_repeat_limit=3,
        )
        action = list(self.mdp.problem.actions)[0]
        with mock.patch(
            "unified_planning.engines.solvers.ptrpg_guided_rollout.pick_best_action"
        ) as mock_pick:
            mock_pick.return_value = (
                1.0,
                action,
                self.stn.clone(),
                None,
                False,
                {},
            )
            with mock.patch.object(self.mdp, "step") as mock_step:
                mock_step.return_value = (False, self.state, 0.0)
                value = ptrpg_guided_terminal_rollout(
                    mdp=self.mdp,
                    state=self.state,
                    stn=self.stn,
                    previous_action_node=None,
                    config=config,
                    temporal_heuristic_depth=8,
                )
        self.assertEqual(value, 0.0)

    def test_first_action_matches_greedy_stable(self):
        stable = pick_best_action(
            mdp=self.mdp,
            state=self.state,
            stn=self.stn,
            previous_action_node=None,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=8,
            temporal_heuristic_strategy="baseline_survival_resolution",
            tie_break="stable",
        )
        legacy = pick_best_action(
            mdp=self.mdp,
            state=self.state,
            stn=self.stn,
            previous_action_node=None,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=8,
            temporal_heuristic_strategy="baseline_survival_resolution",
            tie_break="legacy",
        )
        self.assertIsNotNone(stable)
        self.assertIsNotNone(legacy)
        self.assertEqual(stable[0], legacy[0])
        self.assertEqual(stable[1].name, legacy[1].name)

    def test_mcts_leaf_does_not_call_heuristic(self):
        with mock.patch.object(C_MCTS, "heuristic", autospec=True) as mock_h:
            with mock.patch(
                "unified_planning.engines.solvers.ptrpg_guided_rollout.ptrpg_guided_terminal_rollout",
                return_value=1.0,
            ) as mock_rollout:
                mcts_plan(
                    self.mdp,
                    steps=1,
                    search_time=1,
                    search_depth=3,
                    exploration_constant=0.5,
                    selection_type="avg",
                    k=10,
                    heuristic_name="temporal_probabilistic_rpg",
                    temporal_heuristic_depth=8,
                    temporal_heuristic_strategy="baseline_survival_resolution",
                    value_mode="ptrpg_guided_terminal_rollout",
                )
        mock_h.assert_not_called()
        self.assertGreater(mock_rollout.call_count, 0)

    def test_debug_trace_emits_once(self):
        config = RolloutConfig(
            policy_strategy=self.config.policy_strategy,
            max_steps=10,
            debug_first_rollout=True,
        )
        with self.assertLogs(
            "unified_planning.engines.solvers.ptrpg_guided_rollout",
            level=logging.INFO,
        ) as captured:
            ptrpg_guided_terminal_rollout(
                mdp=self.mdp,
                state=self.state,
                stn=self.stn,
                previous_action_node=None,
                config=config,
                temporal_heuristic_depth=8,
                debug_emit=True,
            )
        joined = "\n".join(captured.output)
        self.assertIn("ptrpg_guided_rollout step=", joined)
        self.assertIn("final_value=", joined)

    def test_remaining_deadline_non_negative(self):
        self.assertGreaterEqual(remaining_deadline(self.mdp, self.stn), 0)


if __name__ == "__main__":
    unittest.main()
