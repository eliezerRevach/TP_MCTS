"""Tests for counterexample-guided pattern growth (``cegar_pattern``)."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.cegar_pattern import cegar_pattern  # noqa: E402
from comdp_plus_no_deadline.engines.temporal_stn_pdb import DurativeOp  # noqa: E402


def event(pre=(), neg=(), adds=(), dels=()):
    return SimpleNamespace(pos_preconditions=set(pre), neg_preconditions=set(neg), add_effects=set(adds),
                           del_effects=set(dels), probabilistic_effects=[])


def durative(name, d, pre=(), neg=(), adds=(), dels=()):
    running = f"inExecution({name})"
    start = event(pre=pre, neg=tuple(neg) + (running,), adds=(running,))
    end = event(pre=(running,), adds=adds, dels=tuple(dels) + (running,))
    return DurativeOp(key=name, duration=d, start_action=start, end_action=end)


class Base:
    """Deterministic engine-like model: the interface FactPatternModel reads."""

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


# g <- a (needs p, needs NOT busy);  p <- b;  busy is true initially and u removes it
OPS = [
    durative("a", 2, pre=("p",), neg=("busy",), adds=("g",)),
    durative("b", 1, adds=("p",)),
    durative("u", 1, dels=("busy",)),
]


def test_adds_the_positive_then_the_negative_flaw_then_stops():
    res = cegar_pattern(Base(OPS), ["g"], {"busy"}, max_facts=5, deadline=10)
    assert res["facts"] == ["g", "p", "busy"]
    assert [e.get("added") for e in res["log"][:-1]] == ["p", "busy"]
    assert res["log"][-1]["stop"] == "no flaw"
    assert res["table"].lookup(res["model"].project({"busy"}), 10)[0] == 1.0


def test_the_deadline_is_respected_by_the_final_table():
    res = cegar_pattern(Base(OPS), ["g"], {"busy"}, max_facts=5, deadline=10)
    root = res["model"].project({"busy"})
    assert res["table"].lookup(root, 4)[0] == 1.0          # u || b (1), then a (2): done by 3
    assert res["table"].lookup(root, 2)[0] == 0.0


def test_the_cap_stops_the_loop():
    res = cegar_pattern(Base(OPS), ["g"], {"busy"}, max_facts=2, deadline=10)
    assert res["facts"] == ["g", "p"]
    assert res["log"][-1]["stop"] == "cap"
