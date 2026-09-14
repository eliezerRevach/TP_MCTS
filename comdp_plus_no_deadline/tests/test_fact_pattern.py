"""Tests for fact-capped pattern growth (``fact_pattern.grow_pattern``)."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.fact_pattern import grow_pattern  # noqa: E402


def op(pre=(), adds=(), dels=()):
    start = SimpleNamespace(pos_preconditions=set(pre), neg_preconditions=set(), add_effects=set(),
                            del_effects=set(dels), probabilistic_effects=[])
    end = SimpleNamespace(pos_preconditions=set(), neg_preconditions=set(), add_effects=set(adds),
                          del_effects=set(), probabilistic_effects=[])
    return SimpleNamespace(start_action=start, end_action=end)


# goal <- a (needs x, y) ; x <- b (needs z) ; y true initially and deleted by c ; w true, never deleted
OPS = {
    "a": op(pre=("x", "y", "w"), adds=("goal",)),
    "b": op(pre=("z",), adds=("x",)),
    "c": op(pre=(), adds=("q",), dels=("y",)),
    "d": op(pre=(), adds=("z",)),
}
INITIAL = {"y", "w"}


def test_needed_preconditions_come_first():
    assert grow_pattern(OPS, ["goal"], INITIAL, 2) == ["goal", "x"]          # x is not true initially


def test_chain_until_reachable_then_threatened_then_the_rest():
    assert grow_pattern(OPS, ["goal"], INITIAL, 5) == ["goal", "x", "z", "y", "w"]
    # x's achiever needs z (not initial) -> added before y (initial but deleted by c) -> then w


def test_stops_when_nothing_is_left():
    assert grow_pattern(OPS, ["goal"], INITIAL, 50) == ["goal", "x", "z", "y", "w"]


def test_cap_below_the_goal_count_keeps_the_goals():
    assert grow_pattern(OPS, ["goal", "q"], INITIAL, 1) == ["goal", "q"]
