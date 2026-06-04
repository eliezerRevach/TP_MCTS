"""
Tests for ephemeral fixed-tail random rollout leaf evaluation (Option A).
"""

import math
import sys
import unittest
from unittest import mock

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import unified_planning as up
    from unified_planning.shortcuts import BoolType, OverallPreconditionTiming
    from unified_planning.engines.mdp import MDP
    from unified_planning.engines.solvers.mcts import C_MCTS
    from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
        FixedTailConfig,
        build_fixed_tail_search_context,
        node_remaining,
    )
    from unified_planning.engines.solvers.fixed_tail_random_rollout_eval import (
        FixedTailRandomRolloutConfig,
        FixedTailRandomRolloutEvaluator,
        pick_rollout_action,
        random_rollout_config_from_args,
        rollout_legal_fitting_actions,
    )
    from unified_planning.engines.utils import create_init_stn
    from unified_planning.plans.stn.stn_plan import STNPlanNode
    from unified_planning.model.timing import TimepointKind
finally:
    sys.argv = _ORIGINAL_ARGV


def _build_split_problem(duration: int, deadline: int):
    problem = up.model.Problem("fixed_tail_random_rollout_test")
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


class TestFixedTailRandomRolloutEvaluator(unittest.TestCase):
    def setUp(self):
        self.converted = _build_split_problem(duration=2, deadline=25)
        self.mdp = MDP(self.converted, discount_factor=1.0, reward_mode="terminal")
        self.stn = create_init_stn(self.mdp)
        self.state = self.mdp.initial_state()
        self.config = FixedTailConfig(
            prefix_frac=0.10,
            tail_strategy="atom_backtrack_exact_resolution",
        )
        self.ctx = build_fixed_tail_search_context(
            self.mdp, self.state, self.stn, self.config
        )
        self.prev = STNPlanNode(TimepointKind.GLOBAL_START)

    def _evaluator(self, num_samples=1, policy="random_legal_fitting"):
        return FixedTailRandomRolloutEvaluator(
            mdp=self.mdp,
            ctx=self.ctx,
            config=FixedTailRandomRolloutConfig(
                num_samples=num_samples,
                rollout_policy=policy,
            ),
            strategy=self.config.tail_strategy,
        )

    def test_tail_horizon_matches_prefix_frac(self):
        self.assertEqual(self.ctx.root_remaining, node_remaining(self.mdp, self.state, self.stn))
        self.assertEqual(
            self.ctx.prefix_budget,
            max(0, int(math.floor(0.10 * self.ctx.root_remaining))),
        )
        self.assertEqual(self.ctx.tail_horizon, self.ctx.root_remaining - self.ctx.prefix_budget)

    def test_k1_single_ptrpg_call(self):
        zero_prefix_ctx = build_fixed_tail_search_context(
            self.mdp,
            self.state,
            self.stn,
            FixedTailConfig(prefix_frac=0.0, tail_strategy=self.config.tail_strategy),
        )
        evaluator = FixedTailRandomRolloutEvaluator(
            mdp=self.mdp,
            ctx=zero_prefix_ctx,
            config=FixedTailRandomRolloutConfig(num_samples=1),
            strategy=self.config.tail_strategy,
        )
        tail_h = zero_prefix_ctx.tail_horizon
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_random_rollout_eval.ptrpg_at_horizon",
            return_value=0.6,
        ) as mock_ptrpg:
            value = evaluator.evaluate_leaf(self.state, self.stn, self.prev)
        self.assertEqual(mock_ptrpg.call_count, 1)
        self.assertEqual(mock_ptrpg.call_args[0][3], tail_h)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(value, 0.6)

    def test_deep_leaf_in_tail_zone_skips_rollout_uses_actual_remaining(self):
        evaluator = self._evaluator(num_samples=1)
        deep_rem = self.ctx.tail_horizon - 2
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_random_rollout_eval.node_remaining",
            return_value=deep_rem,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_random_rollout_eval.ptrpg_at_horizon",
            return_value=0.55,
        ) as mock_ptrpg:
            value = evaluator.evaluate_leaf(self.state, self.stn, self.prev)
        self.assertEqual(mock_ptrpg.call_count, 1)
        self.assertEqual(mock_ptrpg.call_args[0][3], deep_rem)
        self.assertAlmostEqual(value, 0.55)

    def test_k3_averages_sample_values(self):
        evaluator = self._evaluator(num_samples=3)
        sample_returns = [0.2, 0.5, 0.8]

        def fake_sample(*_args, **_kwargs):
            return sample_returns.pop(0), mock.Mock()

        with mock.patch.object(evaluator, "_run_one_sample", side_effect=fake_sample):
            value = evaluator.evaluate_leaf(self.state, self.stn, self.prev)
        self.assertAlmostEqual(value, (0.2 + 0.5 + 0.8) / 3.0)

    def test_goal_mid_rollout_skips_ptrpg(self):
        evaluator = self._evaluator(num_samples=1)
        early = mock.Mock(terminated_early=True, sample_value=1.0)
        with mock.patch.object(
            evaluator, "_run_one_sample", return_value=(1.0, early)
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_random_rollout_eval.ptrpg_at_horizon",
        ) as mock_ptrpg:
            value = evaluator.evaluate_leaf(self.state, self.stn, self.prev)
        mock_ptrpg.assert_not_called()
        self.assertAlmostEqual(value, 1.0)

    def test_first_legal_fitting_is_deterministic(self):
        class _Action:
            def __init__(self, name):
                self.name = name

        legal = [_Action("z"), _Action("a"), _Action("m")]
        rng = mock.Mock()
        chosen = pick_rollout_action(legal, "first_legal_fitting", rng)
        self.assertEqual(chosen.name, "a")
        rng.choice.assert_not_called()

    def test_rollout_config_from_args(self):
        up.args = type(
            "A",
            (),
            {
                "fixed_tail_rollout_samples": 5,
                "fixed_tail_rollout_policy": "first_legal_fitting",
            },
        )()
        cfg = random_rollout_config_from_args()
        self.assertEqual(cfg.num_samples, 5)
        self.assertEqual(cfg.rollout_policy, "first_legal_fitting")


class TestFixedTailRandomRolloutMCTS(unittest.TestCase):
    def setUp(self):
        self.converted = _build_split_problem(duration=2, deadline=25)
        self.mdp = MDP(self.converted, discount_factor=1.0, reward_mode="terminal")
        self.stn = create_init_stn(self.mdp)
        self.state = self.mdp.initial_state()
        up.args = type(
            "A",
            (),
            {
                "fixed_tail_prefix_frac": 0.10,
                "fixed_tail_debug": False,
                "fixed_tail_rollout_samples": 1,
                "fixed_tail_rollout_policy": "first_legal_fitting",
            },
        )()

    def test_evaluate_leaf_does_not_call_create_snode(self):
        mcts = C_MCTS(
            self.mdp,
            None,
            self.state,
            4,
            1.0,
            self.stn,
            "avg",
            10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_random_rollout_eval",
        )
        prev = STNPlanNode(TimepointKind.GLOBAL_START)
        with mock.patch.object(mcts, "create_Snode") as mock_create:
            mcts._fixed_tail_random_rollout.evaluate_leaf(self.state, self.stn, prev)
        mock_create.assert_not_called()

    def test_selection_calls_evaluator_at_new_leaf(self):
        mcts = C_MCTS(
            self.mdp,
            None,
            self.state,
            4,
            1.0,
            self.stn,
            "avg",
            10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_random_rollout_eval",
        )
        with mock.patch.object(
            mcts._fixed_tail_random_rollout,
            "evaluate_leaf",
            return_value=0.33,
        ) as mock_eval:
            mcts.selection(mcts.root_node)
        self.assertGreaterEqual(mock_eval.call_count, 1)

    def test_mcts_blocks_expansion_past_tail_horizon(self):
        mcts = C_MCTS(
            self.mdp,
            None,
            self.state,
            40,
            1.0,
            self.stn,
            "avg",
            10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_random_rollout_eval",
        )
        tail_h = mcts._fixed_tail_ctx.tail_horizon
        with mock.patch.object(mcts, "_fixed_tail_leaf_eval_value", return_value=0.5) as mock_leaf:
            with mock.patch(
                "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.node_remaining",
                return_value=tail_h,
            ):
                val = mcts.selection(mcts.root_node)
        mock_leaf.assert_called()
        self.assertAlmostEqual(val, 0.5)

    def test_mcts_max_init_uses_k_action_evals(self):
        up.args.fixed_tail_rollout_samples = 2
        mcts = C_MCTS(
            self.mdp,
            None,
            self.state,
            40,
            1.0,
            self.stn,
            "max",
            3,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_random_rollout_eval",
        )
        action_names = ["a0", "a1", "a2", "a3"]
        fake_children = {}
        for name in action_names:
            an = mock.Mock()
            an.stn = self.stn
            an.STNNode = STNPlanNode(TimepointKind.GLOBAL_START)
            fake_children[name] = an
        snode = mock.Mock()
        snode.state = self.state
        snode.depth = 0
        snode.children = fake_children
        mcts.mdp.step = mock.Mock(
            return_value=(False, self.state, 0.0),
        )
        with mock.patch.object(
            mcts._fixed_tail_random_rollout,
            "evaluate_leaf",
            side_effect=[0.2, 0.5, 0.9],
        ) as mock_eval:
            best = mcts._fixed_tail_max_init_at_snode(snode, attach_children=False)
        self.assertEqual(mock_eval.call_count, 3)
        self.assertAlmostEqual(best, 0.9)

    def test_mcts_sampled_still_recurses_in_tree(self):
        up.args.fixed_tail_prefix_policy = "mcts_sampled"
        mcts = C_MCTS(
            self.mdp,
            None,
            self.state,
            4,
            1.0,
            self.stn,
            "avg",
            10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_mcts_sampled",
        )
        self.assertIsNone(mcts._fixed_tail_random_rollout)
        self.assertTrue(mcts._uses_fixed_tail_mcts_sampled())


if __name__ == "__main__":
    unittest.main()
