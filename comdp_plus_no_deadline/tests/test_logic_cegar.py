"""Logic CEGAR (``logic_cegar``): counterexamples from the initial state AND rollout states.

Toy hand: point_a / point_b need `free` and take it (point_a not when already at a); sample needs `at_a`, gives `g` and frees the
hand. Pointed at b, the hand is stuck. From the initial state the hand is free, so the route
point_a -> sample never fails there: `free` is only found from a state the rollouts visit."""

import os
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.logic_cegar import logic_cegar_patterns, rollout_states  # noqa: E402
from comdp_plus_no_deadline.engines.temporal_stn_pdb import DurativeOp  # noqa: E402


def event(pre=(), neg=(), adds=(), dels=()):
    return SimpleNamespace(pos_preconditions=set(pre), neg_preconditions=set(neg), add_effects=set(adds),
                           del_effects=set(dels), probabilistic_effects=[])


def durative(name, d, pre=(), neg=(), adds=(), dels=()):
    running = f"inExecution(start-{name})"
    start = event(pre=pre, neg=tuple(neg) + (running,), adds=(running,))
    end = event(pre=(running,), adds=adds, dels=tuple(dels) + (running,))
    return DurativeOp(key=name, duration=d, start_action=start, end_action=end)


OPS = [
    durative("point_a", 1, pre=("free",), neg=("at_a",), adds=("at_a",), dels=("free",)),
    durative("point_b", 1, pre=("free",), adds=("at_b",), dels=("free",)),
    durative("sample", 2, pre=("at_a",), adds=("g", "free"), dels=("at_a",)),
]
INIT = {"free"}


class Base:
    def __init__(self, ops):
        self._ops = {op.key: op for op in ops}

    def ops(self):
        return self._ops

    def outcomes(self, action, facts):
        return ((frozenset((set(facts) - action.del_effects) | action.add_effects), 1.0),)

    def start_is_inert(self, key):
        return False

    def deadline(self):
        return 10


class State:
    def __init__(self, facts):
        self.predicates = frozenset(facts)


class MDP:
    """Untimed engine-like MDP over the toy ops (starts and ends as actions)."""

    def __init__(self, base, init, goals):
        self.base, self.init, self.goals = base, frozenset(init), set(goals)

    def initial_state(self):
        return State(self.init)

    def is_terminal(self, s):
        return self.goals <= s.predicates

    def _events(self):
        for k, op in self.base.ops().items():
            yield SimpleNamespace(name="start_" + k), op.start_action
            yield SimpleNamespace(name="end_" + k), op.end_action

    def legal_actions(self, s):
        f = s.predicates
        return [a for a, ev in self._events() if ev.pos_preconditions <= f and not (ev.neg_preconditions & f)]

    def transition_function(self, s, a):
        ev = dict((x.name, e) for x, e in self._events())[a.name]
        return [(State((s.predicates - ev.del_effects) | ev.add_effects), 1.0)]

    def step(self, s, a):
        nxt = self.transition_function(s, a)[0][0]
        return self.is_terminal(nxt), nxt, 0.0


def run(**kw):
    base = Base(OPS)
    return logic_cegar_patterns(MDP(base, INIT, {"g"}), base, [["g"]], INIT, max_facts=6, **kw)[0]


def test_initial_state_alone_never_finds_the_hand():
    res = run(rollouts=0)
    assert res["facts"] == ["g", "at_a"] and res["stop"] == "no flaw"      # at_a: the achiever's precondition


def test_random_rollouts_find_the_taken_hand():
    res = run(rollouts=20, depth=6, seed=1)
    assert "free" in res["facts"]
    assert res["facts"][0] == "g"


def test_greedy_sources_run_and_keep_the_goals():
    res = run(sources="greedy", rollouts=5, depth=6, seed=1)
    assert res["facts"][0] == "g" and res["stop"] in ("no flaw", "cap", "rounds")


def test_cap_is_respected():
    base = Base(OPS)
    res = logic_cegar_patterns(MDP(base, INIT, {"g"}), base, [["g"]], INIT, max_facts=1, rollouts=20, depth=6,
                               seed=1)[0]
    assert res["facts"] == ["g"] and res["stop"] == "cap"


def test_rollouts_leave_the_global_random_state_alone():
    base = Base(OPS)
    random.seed(7)
    expected = random.random()
    random.seed(7)
    rollout_states(MDP(base, INIT, {"g"}), 10, 6, random.Random(3))
    assert random.random() == expected


def test_unknown_sources_are_rejected():
    with pytest.raises(ValueError):
        run(sources="bogus")


def test_heuristic_knobs_are_checked(monkeypatch):
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic
    monkeypatch.setenv("TP_MCTS_WILAO_REPORT", "0")
    monkeypatch.setenv("TP_MCTS_WILAO_PATTERN_GROWTH", "logic")
    monkeypatch.setenv("TP_MCTS_WILAO_LOGIC_SOURCES", "greedy")
    h = WindowsILAOPDBHeuristic(None)
    assert (h.growth, h.logic_sources, h.logic_rollouts) == ("logic", "greedy", 20)
    monkeypatch.setenv("TP_MCTS_WILAO_PATTERN_GROWTH", "bogus")
    with pytest.raises(ValueError):
        WindowsILAOPDBHeuristic(None)
    monkeypatch.setenv("TP_MCTS_WILAO_PATTERN_GROWTH", "logic")
    monkeypatch.setenv("TP_MCTS_WILAO_LOGIC_SOURCES", "bogus")
    with pytest.raises(ValueError):
        WindowsILAOPDBHeuristic(None)
