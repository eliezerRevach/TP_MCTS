"""
Unit tests for fixed-tail expectimax prefix evaluation.
"""

import sys
from pathlib import Path
import unittest
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import unified_planning as up
    from unified_planning.shortcuts import BoolType, OverallPreconditionTiming  # noqa: F401
    from unified_planning.engines.solvers.fixed_tail_expectimax import (
        FixedTailExpectimaxEvaluator,
        FixedTailExpectimaxGuards,
    )
    from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
        FixedTailSearchContext,
    )
    from unified_planning.engines.solvers.mcts import C_MCTS
    from unified_planning.engines.solvers.test_fixed_tail_ptrpg_rollout import (
        _build_split_problem,
    )
    from unified_planning.engines.mdp import MDP
    from unified_planning.engines.utils import create_init_stn
finally:
    sys.argv = _ORIGINAL_ARGV


class _MockAction:
    def __init__(self, name: str):
        self.name = name


class _MockState:
    def __init__(self, tag: str):
        self.tag = tag
        self.predicates = frozenset({tag})


class _MockSTN:
    def clone(self):
        return _MockSTN()

    def is_consistent(self):
        return True

    def get_current_end_time(self):
        return 0.0


class _MockSTNNode:
    pass


class TestFixedTailExpectimax(unittest.TestCase):
    def setUp(self):
        self.ctx = FixedTailSearchContext(
            root_remaining=100,
            prefix_budget=10,
            prefix_frac=0.10,
        )
        self.stn = _MockSTN()
        self.prev = _MockSTNNode()
        self.mdp = mock.MagicMock()
        self.mdp.deadline.return_value = 100
        self.guards = FixedTailExpectimaxGuards(max_nodes=10000, max_depth=32)
        self.evaluator = FixedTailExpectimaxEvaluator(
            mdp=self.mdp,
            ctx=self.ctx,
            strategy="atom_backtrack_exact_resolution",
            guards=self.guards,
        )

    def _patch_stn_fit(self):
        return mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax._fit_action_stn",
            return_value=(self.stn, self.prev),
        )

    def _patch_remaining(self, mapping):
        def _rem(mdp, state, stn):
            return mapping.get(state.tag, 50)

        return mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.node_remaining",
            side_effect=_rem,
        )

    def test_max_over_two_actions_not_average(self):
        good = _MockAction("good")
        bad = _MockAction("bad")
        root = _MockState("root")
        s_good = _MockState("good")
        s_bad = _MockState("bad")

        self.mdp.legal_actions.return_value = [good, bad]
        self.mdp.transition_function.side_effect = lambda state, action: {
            good: [(s_good, 1.0)],
            bad: [(s_bad, 1.0)],
        }[action]

        def goal(mdp, state):
            return state.tag == "good"

        with self._patch_stn_fit(), self._patch_remaining(
            {"root": 50, "good": 50, "bad": 50}
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax._goal_reached",
            side_effect=goal,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.fixed_tail_dead_end_value",
            return_value=False,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.crossed_cutoff",
            return_value=False,
        ):
            v = self.evaluator.value(root, self.stn, self.prev)

        self.assertAlmostEqual(v, 1.0)
        self.assertNotAlmostEqual(v, 0.5)

    def test_stochastic_expectation(self):
        only = _MockAction("only")
        root = _MockState("root")
        s_win = _MockState("win")
        s_lose = _MockState("lose")

        self.mdp.legal_actions.side_effect = lambda state: (
            [only] if state.tag == "root" else []
        )
        self.mdp.transition_function.return_value = [
            (s_win, 0.5),
            (s_lose, 0.5),
        ]

        def goal(mdp, state):
            return state.tag == "win"

        with self._patch_stn_fit(), self._patch_remaining(
            {"root": 50, "win": 50, "lose": 50}
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax._goal_reached",
            side_effect=goal,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.fixed_tail_dead_end_value",
            return_value=False,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.crossed_cutoff",
            return_value=False,
        ):
            q = self.evaluator.q_value(root, self.stn, self.prev, only)

        self.assertAlmostEqual(q, 0.5)

    def test_at_cutoff_calls_ptrpg_with_remaining_horizon(self):
        root = _MockState("root")
        self.mdp.legal_actions.return_value = []

        ptrpg = mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.ptrpg_at_horizon",
            return_value=0.42,
        )
        with self._patch_remaining({"root": 30}), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax._goal_reached",
            return_value=False,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.fixed_tail_dead_end_value",
            return_value=False,
        ), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax.crossed_cutoff",
            return_value=True,
        ), ptrpg as mock_ptrpg:
            v = self.evaluator.value(root, self.stn, self.prev)

        self.assertAlmostEqual(v, 0.42)
        mock_ptrpg.assert_called_once()
        _args, _kwargs = mock_ptrpg.call_args
        self.assertEqual(_args[3], 30)

    def test_value_cache_hit(self):
        root = _MockState("root")
        self.mdp.legal_actions.return_value = []

        with self._patch_remaining({"root": 30}), mock.patch(
            "unified_planning.engines.solvers.fixed_tail_expectimax._goal_reached",
            return_value=True,
        ):
            v1 = self.evaluator.value(root, self.stn, self.prev)
            nodes_after_first = self.evaluator._nodes_evaluated
            v2 = self.evaluator.value(root, self.stn, self.prev)

        self.assertAlmostEqual(v1, 1.0)
        self.assertAlmostEqual(v2, 1.0)
        self.assertEqual(nodes_after_first, 1)
        self.assertEqual(self.evaluator._nodes_evaluated, 1)


class TestFixedTailExpectimaxMCTSSeed(unittest.TestCase):
    def test_root_child_q_seeded(self):
        up.args = mock.MagicMock()
        up.args.fixed_tail_prefix_frac = 0.10
        up.args.fixed_tail_prefix_policy = "expectimax"
        up.args.fixed_tail_expectimax_max_nodes = 5000
        up.args.fixed_tail_expectimax_max_depth = 64
        up.args.fixed_tail_expectimax_max_time_sec = 0.0
        up.args.fixed_tail_debug = False

        converted = _build_split_problem(duration=2, deadline=25)
        mdp = MDP(converted, discount_factor=1.0, reward_mode="terminal")
        stn = create_init_stn(mdp)
        root_state = mdp.initial_state()

        mcts = C_MCTS(
            mdp=mdp,
            root_node=None,
            root_state=root_state,
            stn=stn,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=25,
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_ptrpg_rollout",
            exploration_constant=1.0,
            search_depth=1,
            selection_type="avg",
            k=10,
        )
        self.assertTrue(mcts._uses_fixed_tail_expectimax())
        snode, _ = mcts.create_Snode(root_state, 0, stn, None)
        self.assertGreater(len(snode.children), 0)
        mcts._fixed_tail_seed_expectimax_q(snode)
        for anode in snode.children.values():
            self.assertTrue(hasattr(anode, "_expectimax_q"))
            self.assertEqual(anode.count, 0)

    def test_max_init_fixed_tail_has_root_stn_during_create(self):
        up.args = mock.MagicMock()
        up.args.fixed_tail_prefix_frac = 0.20
        up.args.fixed_tail_prefix_policy = "expectimax"
        up.args.fixed_tail_expectimax_max_nodes = 5000
        up.args.fixed_tail_expectimax_max_depth = 64
        up.args.fixed_tail_expectimax_max_time_sec = 0.0
        up.args.fixed_tail_debug = False

        converted = _build_split_problem(duration=2, deadline=25)
        mdp = MDP(converted, discount_factor=1.0, reward_mode="terminal")
        stn = create_init_stn(mdp)
        root_state = mdp.initial_state()

        mcts = C_MCTS(
            mdp=mdp,
            root_node=None,
            root_state=root_state,
            stn=stn,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=25,
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_ptrpg_rollout",
            exploration_constant=0.5,
            search_depth=4,
            selection_type="max",
            k=9999,
        )
        self.assertIs(mcts.stn, stn)
        self.assertGreater(mcts.root_node.count, 0)

    def test_uct_after_expectimax_seed_no_log_domain_error(self):
        up.args = mock.MagicMock()
        up.args.fixed_tail_prefix_frac = 0.20
        up.args.fixed_tail_prefix_policy = "expectimax"
        up.args.fixed_tail_expectimax_max_nodes = 5000
        up.args.fixed_tail_expectimax_max_depth = 64
        up.args.fixed_tail_expectimax_max_time_sec = 0.0
        up.args.fixed_tail_debug = False

        converted = _build_split_problem(duration=2, deadline=25)
        mdp = MDP(converted, discount_factor=1.0, reward_mode="terminal")
        stn = create_init_stn(mdp)
        root_state = mdp.initial_state()

        mcts = C_MCTS(
            mdp=mdp,
            root_node=None,
            root_state=root_state,
            stn=stn,
            heuristic_name="temporal_probabilistic_rpg",
            temporal_heuristic_depth=25,
            temporal_heuristic_strategy="atom_backtrack_exact_resolution",
            value_mode="fixed_tail_ptrpg_rollout",
            exploration_constant=0.5,
            search_depth=4,
            selection_type="max",
            k=9999,
        )
        snode, _ = mcts.create_Snode(root_state, 0, stn, None)
        mcts._fixed_tail_seed_expectimax_q(snode)
        self.assertEqual(snode.count, 0)
        action = mcts.uct(snode, 0.5)
        self.assertIn(action, snode.possible_actions)


if __name__ == "__main__":
    unittest.main()
