"""
Tests for fixed-tail PTRPG rollout (MCTS leaf evaluation).
"""

import sys
import time
import unittest
from unittest import mock

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import unified_planning as up
    from unified_planning.shortcuts import BoolType, OverallPreconditionTiming
    from unified_planning.engines.mdp import MDP
    from unified_planning.engines.solvers.mcts import C_MCTS, plan as mcts_plan
    from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
        PREFIX_POLICY_PTRPG_GREEDY,
        FixedTailConfig,
        FixedTailSafetyError,
        MAX_SECONDS_PER_FIXED_TAIL_EVAL,
        fixed_tail_ptrpg_value,
        remaining_deadline,
        reset_fixed_tail_profiler,
    )
    from unified_planning.engines.solvers.ptrpg_guided_rollout import (
        resolve_rollout_policy,
    )
    from unified_planning.engines.utils import create_init_stn
finally:
    sys.argv = _ORIGINAL_ARGV


def _build_split_problem(duration: int, deadline: int):
    problem = up.model.Problem("fixed_tail_rollout_test")
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


class TestFixedTailPtrpgRollout(unittest.TestCase):
    def setUp(self):
        reset_fixed_tail_profiler()
        self.converted = _build_split_problem(duration=2, deadline=25)
        self.mdp = MDP(self.converted, discount_factor=1.0, reward_mode="terminal")
        self.stn = create_init_stn(self.mdp)
        self.state = self.mdp.initial_state()
        self.config = FixedTailConfig(
            fixed_tail_h=10,
            policy_strategy=resolve_rollout_policy("atomic_exact_resolution"),
            tail_strategy="atom_backtrack_exact_resolution",
        )
        self.greedy_config = FixedTailConfig(
            fixed_tail_h=10,
            policy_strategy=resolve_rollout_policy("atomic_exact_resolution"),
            tail_strategy="atom_backtrack_exact_resolution",
            prefix_policy=PREFIX_POLICY_PTRPG_GREEDY,
        )

    def test_value_in_unit_interval(self):
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout._ptrpg_at_horizon",
            return_value=0.42,
        ):
            value = fixed_tail_ptrpg_value(
                mdp=self.mdp,
                state=self.state,
                stn=self.stn,
                previous_action_node=None,
                config=self.config,
                temporal_heuristic_depth=25,
            )
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_short_horizon_uses_r_not_fixed_h(self):
        short_r = 7
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.remaining_deadline",
            return_value=short_r,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout._ptrpg_at_horizon",
            return_value=0.5,
        ) as mock_tail:
            fixed_tail_ptrpg_value(
                mdp=self.mdp,
                state=self.state,
                stn=self.stn,
                previous_action_node=None,
                config=self.config,
                temporal_heuristic_depth=25,
            )
            self.assertEqual(mock_tail.call_count, 1)
            self.assertEqual(mock_tail.call_args[0][3], short_r)

    def test_long_horizon_tail_uses_fixed_h(self):
        R = remaining_deadline(self.mdp, self.stn)
        self.assertGreater(R, self.greedy_config.fixed_tail_h)
        legal = self.mdp.legal_actions(self.state)
        candidate_stn = mock.Mock()
        candidate_stn.get_current_end_time.return_value = float(R - self.greedy_config.fixed_tail_h)

        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.pick_best_action",
            return_value=(1.0, legal[0], candidate_stn, None, False, {}),
        ), mock.patch.object(
            self.mdp,
            "step",
            return_value=(False, self.state, 0.0),
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout._ptrpg_at_horizon",
            return_value=0.33,
        ) as mock_tail:
            value = fixed_tail_ptrpg_value(
                mdp=self.mdp,
                state=self.state,
                stn=self.stn,
                previous_action_node=None,
                config=self.greedy_config,
                temporal_heuristic_depth=25,
            )
        self.assertEqual(value, 0.33)
        mock_tail.assert_called_once()
        self.assertEqual(mock_tail.call_args[0][3], self.greedy_config.fixed_tail_h)

    def test_boundary_wait_no_overshoot_step(self):
        R = 25
        H = 10
        delta = R - H
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.remaining_deadline",
            return_value=R,
        ):
            legal = self.mdp.legal_actions(self.state)
            self.assertTrue(legal)

            def fake_pick(*_args, **_kwargs):
                candidate_stn = mock.Mock()
                candidate_stn.get_current_end_time.return_value = 30.0
                return (1.0, legal[0], candidate_stn, None, False, {})

            current_stn = mock.Mock()
            current_stn.get_current_end_time.return_value = 0.0
            current_stn.is_consistent.return_value = True

            with mock.patch(
                "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.pick_best_action",
                side_effect=fake_pick,
            ), mock.patch.object(self.mdp, "step") as mock_step, mock.patch(
                "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout._ptrpg_at_horizon",
                return_value=0.25,
            ) as mock_tail:
                fixed_tail_ptrpg_value(
                    mdp=self.mdp,
                    state=self.state,
                    stn=current_stn,
                    previous_action_node=None,
                    config=self.greedy_config,
                    temporal_heuristic_depth=25,
                )
        mock_step.assert_not_called()
        self.assertEqual(mock_tail.call_args[0][2], float(delta))
        self.assertEqual(mock_tail.call_args[0][3], H)

    def test_single_eval_deadline_25_h_22(self):
        converted = _build_split_problem(duration=2, deadline=25)
        mdp = MDP(converted, discount_factor=1.0, reward_mode="terminal")
        stn = create_init_stn(mdp)
        state = mdp.initial_state()
        config = FixedTailConfig(
            fixed_tail_h=22,
            policy_strategy=resolve_rollout_policy("atomic_exact_resolution"),
            tail_strategy="atom_backtrack_exact_resolution",
        )
        R = remaining_deadline(mdp, stn)
        self.assertEqual(R, 25)
        self.assertEqual(R - config.fixed_tail_h, 3)

        t0 = time.time()
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout._ptrpg_at_horizon",
            return_value=0.5,
        ) as mock_tail:
            value = fixed_tail_ptrpg_value(
                mdp=mdp,
                state=state,
                stn=stn,
                previous_action_node=None,
                config=config,
                temporal_heuristic_depth=25,
            )
        elapsed = time.time() - t0
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertEqual(mock_tail.call_count, 1)
        self.assertEqual(mock_tail.call_args[0][3], 22)
        self.assertLess(elapsed, 2.0)

    def test_safety_timeout(self):
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.MAX_SECONDS_PER_FIXED_TAIL_EVAL",
            0.001,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout._ptrpg_at_horizon",
            side_effect=lambda *_a, **_k: time.sleep(0.05) or 0.5,
        ):
            with self.assertRaises(FixedTailSafetyError):
                fixed_tail_ptrpg_value(
                    mdp=self.mdp,
                    state=self.state,
                    stn=self.stn,
                    previous_action_node=None,
                    config=self.config,
                    temporal_heuristic_depth=25,
                )

    def test_mcts_leaf_does_not_call_heuristic(self):
        with mock.patch.object(C_MCTS, "heuristic", autospec=True) as mock_h:
            with mock.patch(
                "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.fixed_tail_ptrpg_value",
                return_value=0.5,
            ) as mock_eval:
                mcts_plan(
                    self.mdp,
                    steps=1,
                    search_time=1,
                    search_depth=3,
                    exploration_constant=0.5,
                    selection_type="avg",
                    k=10,
                    heuristic_name="temporal_probabilistic_rpg",
                    temporal_heuristic_strategy="atom_backtrack_exact_resolution",
                    value_mode="fixed_tail_ptrpg_rollout",
                    final_selection="q",
                )
        mock_h.assert_not_called()
        self.assertGreater(mock_eval.call_count, 0)


if __name__ == "__main__":
    unittest.main()
