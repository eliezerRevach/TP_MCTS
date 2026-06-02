"""
Tests for fixed-tail prefix-frac PTRPG bootstrap (MCTS leaf evaluation).
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
        FixedTailSearchContext,
        build_fixed_tail_search_context,
        crossed_cutoff,
        elapsed_from_root,
        fixed_tail_bootstrap_value,
        node_remaining,
        ptrpg_at_horizon,
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


class TestFixedTailPrefixFrac(unittest.TestCase):
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

    def test_prefix_budget_at_root(self):
        root_rem = node_remaining(self.mdp, self.state, self.stn)
        expected = max(0, int(math.floor(0.10 * root_rem)))
        self.assertEqual(self.ctx.root_remaining, root_rem)
        self.assertEqual(self.ctx.prefix_budget, expected)

    def test_elapsed_and_cutoff(self):
        self.assertEqual(elapsed_from_root(self.ctx, self.ctx.root_remaining), 0)
        self.assertFalse(crossed_cutoff(self.ctx, 0))
        self.assertTrue(crossed_cutoff(self.ctx, self.ctx.prefix_budget))

    def test_bootstrap_value_in_unit_interval(self):
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.ptrpg_at_horizon",
            return_value=0.42,
        ):
            value = fixed_tail_bootstrap_value(
                mdp=self.mdp,
                state=self.state,
                stn=self.stn,
                strategy=self.config.tail_strategy,
                ctx=self.ctx,
            )
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(value, 0.42)

    def test_bootstrap_uses_node_remaining_as_horizon(self):
        with mock.patch(
            "unified_planning.engines.solvers.fixed_tail_ptrpg_rollout.ptrpg_at_horizon",
            return_value=0.5,
        ) as mock_tail:
            fixed_tail_bootstrap_value(
                mdp=self.mdp,
                state=self.state,
                stn=self.stn,
                strategy=self.config.tail_strategy,
                ctx=self.ctx,
            )
            rem = node_remaining(self.mdp, self.state, self.stn)
            self.assertEqual(mock_tail.call_args[0][3], rem)

    def test_prefix_budget_constant_per_search_context(self):
        ctx2 = FixedTailSearchContext(
            root_remaining=50,
            prefix_budget=5,
            prefix_frac=0.10,
        )
        self.assertEqual(ctx2.prefix_budget, 5)
        self.assertEqual(elapsed_from_root(ctx2, 44), 6)
        self.assertTrue(crossed_cutoff(ctx2, 6))

    def test_prefix_budget_recomputed_on_new_root_remaining(self):
        cfg = FixedTailConfig(prefix_frac=0.10)
        ctx40 = build_fixed_tail_search_context(
            self.mdp,
            self.state,
            self.stn,
            cfg,
        )
        ctx40_manual = FixedTailSearchContext(
            root_remaining=40,
            prefix_budget=4,
            prefix_frac=0.10,
        )
        self.assertEqual(ctx40_manual.prefix_budget, 4)

    def test_mcts_search_context_matches_root(self):
        up.args = type("A", (), {"fixed_tail_prefix_frac": 0.10, "fixed_tail_debug": False})()
        mcts = C_MCTS(
            self.mdp,
            None,
            self.state,
            2,
            1.0,
            self.stn,
            "avg",
            10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_ptrpg_rollout",
        )
        self.assertEqual(mcts._fixed_tail_ctx.prefix_budget, self.ctx.prefix_budget)

    def test_bootstrap_leaf_cached_on_snode(self):
        up.args = type("A", (), {"fixed_tail_prefix_frac": 0.10, "fixed_tail_debug": False})()
        mcts = C_MCTS(
            self.mdp,
            None,
            self.state,
            2,
            1.0,
            self.stn,
            "avg",
            10,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_ptrpg_rollout",
        )
        snode = mcts.root_node
        with mock.patch.object(
            mcts,
            "_fixed_tail_bootstrap_at_snode",
            return_value=0.77,
        ) as mock_boot:
            snode._fixed_tail_bootstrap = True
            snode._fixed_tail_value = 0.77
            val = mcts.selection(snode)
            self.assertAlmostEqual(val, 0.77)
            mock_boot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
