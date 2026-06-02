"""Tests for frontier_aligned_option_a (global Option A selection)."""

import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from comdp_plus_no_deadline.engines.frontier_aligned_option_a import (
    OPTION_A_STRATEGIES,
    OptionAConfig,
    aligned_value_for_node,
    collect_global_frontier,
    compute_H_frontier,
    format_option_a_debug_row,
    is_option_a_strategy,
    option_a_ptrpg_suffix,
    remaining_horizon,
    select_frontier_node,
)
from comdp_plus_no_deadline.engines.rollout_aligned import RolloutAlignedEvaluator


@dataclass
class FakeState:
    name: str
    current_time: float = 0.0
    predicates: frozenset = frozenset()


class FakeANode:
    def __init__(self, count=0, children=None, stn=None):
        self.count = count
        self.children = children or {}
        self.stn = stn


class FakeSTN:
    def __init__(self, end_time: float):
        self._end = end_time

    def get_current_end_time(self):
        return self._end


class FakeSNode:
    def __init__(
        self,
        depth: int,
        possible_actions: Optional[List] = None,
        children: Optional[Dict] = None,
        parent=None,
        state=None,
        stn_end: float = 0.0,
    ):
        self.depth = depth
        self.possible_actions = possible_actions if possible_actions is not None else ["a"]
        self.children = children or {}
        self.parent = parent
        self.state = state or FakeState(f"d{depth}", current_time=stn_end)
        self._option_a_node_id = None
        self._option_a_parent_id = None


def _tree_mixed_elapsed():
    """Root elapsed=0 (expandable), child branch elapsed=10 (expandable)."""
    stn_child = FakeSTN(10.0)
    child_snode = FakeSNode(
        1,
        possible_actions=["b"],
        children={"b": FakeANode(count=0, children={})},
        stn_end=10.0,
    )
    child_snode.parent = FakeANode(count=1, children={"s": child_snode}, stn=stn_child)
    root = FakeSNode(
        0,
        possible_actions=["a", "c"],
        children={
            "a": FakeANode(count=1, children={"s": child_snode}, stn=FakeSTN(0.0)),
            "c": FakeANode(count=0, children={}),  # unvisited -> root still expandable
        },
        stn_end=0.0,
    )
    return root, child_snode


class TestOptionAStrategies(unittest.TestCase):
    def test_strategy_names(self):
        self.assertTrue(is_option_a_strategy("frontier_aligned_option_a"))
        self.assertFalse(is_option_a_strategy("frontier_aligned_baseline"))
        self.assertEqual(option_a_ptrpg_suffix("frontier_aligned_option_a"), "baseline")
        self.assertEqual(
            option_a_ptrpg_suffix("frontier_aligned_option_a_resolution"),
            "baseline_survival_resolution",
        )


class TestCollectFrontier(unittest.TestCase):
    def test_collect_expandable_nodes(self):
        root, child = _tree_mixed_elapsed()
        frontier = collect_global_frontier(root, search_depth=40, deadline=25.0)
        self.assertIn(root, frontier)
        self.assertIn(child, frontier)
        self.assertEqual(len(frontier), 2)

    def test_terminal_excluded(self):
        root = FakeSNode(0, possible_actions=[], children={})
        self.assertEqual(collect_global_frontier(root, search_depth=40), [])


class TestHFrontier(unittest.TestCase):
    def test_sanity_ab(self):
        root, child = _tree_mixed_elapsed()
        frontier = [root, child]
        deepest, H, emap = compute_H_frontier(frontier, 25.0)
        self.assertAlmostEqual(deepest, 10.0)
        self.assertAlmostEqual(H, 15.0)
        self.assertAlmostEqual(emap[id(root)], 0.0)
        self.assertAlmostEqual(emap[id(child)], 10.0)
        self.assertAlmostEqual(deepest - emap[id(root)], 10.0)
        self.assertAlmostEqual(deepest - emap[id(child)], 0.0)


class TestOptionASanity(unittest.TestCase):
    """A elapsed=0/R=25, B elapsed=10/R=15, H_frontier=15."""

    def test_ab_alignment_to_deepest(self):
        cfg = OptionAConfig()
        raw_calls = []
        rolled = []

        def raw_eval(state, horizon):
            raw_calls.append((getattr(state, "name", state), horizon))
            return 0.5

        def prefix(state, delta):
            rolled.append((getattr(state, "name", state), delta))
            return (
                FakeState("after_" + state.name, current_time=state.current_time + delta),
                False,
                int(delta),
                False,
            )

        ev = RolloutAlignedEvaluator(
            cfg.to_rollout_config(),
            raw_eval,
            prefix,
            state_hash_fn=lambda s: (frozenset(s.predicates), s.current_time),
        )
        H_frontier = 15.0
        v_a = aligned_value_for_node(
            ev,
            FakeState("A", current_time=0.0),
            remaining=25,
            H_frontier=H_frontier,
            deepest_elapsed=10.0,
            elapsed=0.0,
        )
        v_b = aligned_value_for_node(
            ev,
            FakeState("B", current_time=10.0),
            remaining=15,
            H_frontier=H_frontier,
            deepest_elapsed=10.0,
            elapsed=10.0,
        )
        self.assertEqual(rolled, [("A", 10)])
        self.assertEqual(raw_calls, [("after_A", 15), ("B", 15)])
        self.assertEqual(v_a, 0.5)
        self.assertEqual(v_b, 0.5)


class TestOptionAConfig(unittest.TestCase):
    def test_fixed_h_inert_in_dynamic_mode(self):
        cfg_hi = OptionAConfig(common_horizon_H=999)
        cfg_lo = OptionAConfig(common_horizon_H=1)
        raw_calls = []

        def raw_eval(state, horizon):
            raw_calls.append(horizon)
            return float(horizon)

        def prefix(state, delta):
            return state, False, 0, False

        ev_hi = RolloutAlignedEvaluator(cfg_hi.to_rollout_config(), raw_eval, prefix)
        ev_lo = RolloutAlignedEvaluator(cfg_lo.to_rollout_config(), raw_eval, prefix)
        v_hi = ev_hi.evaluate(FakeState("n"), 20, h_override=15)
        v_lo = ev_lo.evaluate(FakeState("n"), 20, h_override=15)
        self.assertEqual(v_hi, v_lo)
        self.assertEqual(raw_calls, [15, 15])


class TestSelectFrontier(unittest.TestCase):
    def test_argmax_aligned(self):
        a = FakeSNode(0)
        b = FakeSNode(1)
        a._option_a_node_id = 1
        b._option_a_node_id = 2
        best = select_frontier_node([a, b], {id(a): 0.3, id(b): 0.8})
        self.assertIs(best, b)


class TestDebugRow(unittest.TestCase):
    def test_format_contains_fields(self):
        row = format_option_a_debug_row(
            node_id=1,
            parent_id=None,
            depth=0,
            elapsed=0.0,
            remaining=25.0,
            deepest_elapsed=10.0,
            H_frontier=15.0,
            delta=10.0,
            raw_ptrpg=0.9,
            aligned_value=0.5,
            selected=True,
        )
        self.assertIn("node_id=1", row)
        self.assertIn("selected=yes", row)
        self.assertIn("H_frontier=15.0000", row)


class TestRemainingHorizon(unittest.TestCase):
    def test_deadline_remaining(self):
        self.assertEqual(remaining_horizon(25.0, 10.0, 40), 15)


class TestDebugTraceMixedElapsed(unittest.TestCase):
    def test_debug_trace_nonzero_delta(self):
        root, child = _tree_mixed_elapsed()
        root._option_a_node_id = 0
        root._option_a_parent_id = None
        child._option_a_node_id = 1
        child._option_a_parent_id = 0
        frontier = collect_global_frontier(root, search_depth=40, deadline=25.0)
        deepest, H, emap = compute_H_frontier(frontier, 25.0)
        cfg = OptionAConfig().to_rollout_config()
        rolled = []

        def raw_eval(state, horizon):
            return 0.42

        def prefix(state, delta):
            rolled.append((state.name, delta))
            return (
                FakeState("after_" + state.name, current_time=state.current_time + delta),
                False,
                int(delta),
                False,
            )

        ev = RolloutAlignedEvaluator(cfg, raw_eval, prefix)
        deltas = []
        for n in frontier:
            elapsed = emap[id(n)]
            rem = remaining_horizon(25.0, elapsed, 40)
            deltas.append(deepest - elapsed)
            aligned_value_for_node(
                ev, n.state, remaining=rem, H_frontier=H,
                deepest_elapsed=deepest, elapsed=elapsed,
            )
        self.assertEqual(rolled, [("d0", 10)])
        self.assertTrue(any(d > 0 for d in deltas))


class TestOptionANotInAlignedSuffix(unittest.TestCase):
    def test_option_a_not_in_rollout_or_legacy_frontier_aligned(self):
        legacy_aligned = {
            "rollout_aligned_baseline",
            "rollout_aligned_survival",
            "rollout_aligned_resolution_survival",
            "frontier_aligned_baseline",
            "frontier_aligned_survival",
            "frontier_aligned_resolution_survival",
        }
        for name in OPTION_A_STRATEGIES:
            self.assertNotIn(name, legacy_aligned)


if __name__ == "__main__":
    unittest.main()
