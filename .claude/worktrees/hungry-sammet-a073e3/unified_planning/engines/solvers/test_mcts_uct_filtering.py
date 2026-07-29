import unittest

import sys

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
try:
    import unified_planning as up
    import unified_planning.domains
    from unified_planning.engines.solvers.mcts import C_MCTS
finally:
    sys.argv = _ORIGINAL_ARGV


class _FakeAction:
    def __init__(self, name: str):
        self.name = name


class _FakeANode:
    def __init__(self, count: float, value: float):
        self.count = count
        self.value = value


class _FakeSNode:
    def __init__(self, actions, children, count: float):
        self.possible_actions = actions
        self.children = children
        self.count = count


class TestMCTSUctFiltering(unittest.TestCase):
    def _make_mcts(self, mode: str, initial_k: int, score_by_name: dict[str, float]) -> C_MCTS:
        mcts = C_MCTS.__new__(C_MCTS)
        mcts._uct_filter_mode = mode
        mcts._uct_initial_k = initial_k
        mcts._greedy_matched_action_target = (
            lambda snode, action: score_by_name[action.name]
        )
        return mcts

    def test_avg_topk_uct_never_uses_outside_actions(self):
        actions = [_FakeAction("a"), _FakeAction("b"), _FakeAction("c"), _FakeAction("d")]
        children = {
            actions[0]: _FakeANode(count=1, value=1.0),
            actions[1]: _FakeANode(count=1, value=0.5),
            actions[2]: _FakeANode(count=1, value=100.0),
            actions[3]: _FakeANode(count=1, value=90.0),
        }
        snode = _FakeSNode(actions=actions, children=children, count=10)
        mcts = self._make_mcts(
            mode="topk",
            initial_k=2,
            score_by_name={"a": 10.0, "b": 9.0, "c": 2.0, "d": 1.0},
        )

        chosen = mcts.uct(snode, explore_constant=0.0)
        self.assertEqual(chosen.name, "a")

    def test_avg_pw_uct_widens_with_snode_count(self):
        actions = [_FakeAction("a"), _FakeAction("b"), _FakeAction("c"), _FakeAction("d")]
        children = {
            actions[0]: _FakeANode(count=1, value=1.0),
            actions[1]: _FakeANode(count=1, value=2.0),
            actions[2]: _FakeANode(count=1, value=3.0),
            actions[3]: _FakeANode(count=1, value=100.0),
        }
        # initial_k=1 and sqrt(4)=2 -> allowed_n=3 (a,b,c), so d is still excluded.
        snode = _FakeSNode(actions=actions, children=children, count=4)
        mcts = self._make_mcts(
            mode="pw",
            initial_k=1,
            score_by_name={"a": 10.0, "b": 9.0, "c": 8.0, "d": 1.0},
        )

        chosen = mcts.uct(snode, explore_constant=0.0)
        self.assertEqual(chosen.name, "c")


if __name__ == "__main__":
    unittest.main()
