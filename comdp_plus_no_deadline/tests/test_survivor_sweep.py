"""Tests for ``survivor_sweep`` (the v18 occupancy sweep).

Closed forms where the relaxation is exact, the v18 attempt count, the
"failure still reaches the goal" case, and a cross-check against the pinned
solver of ``temporal_stn_pdb`` (a lower bound on the same model).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.survivor_sweep import (  # noqa: E402
    SurvivorSweep,
    lookup,
    relaxed_actions_from_toy,
)
from comdp_plus_no_deadline.engines.temporal_stn_pdb import TemporalSTNPDB, ToyModel  # noqa: E402


def sweep_for(ops, goals):
    return SurvivorSweep(relaxed_actions_from_toy(ToyModel(ops, goals, 0)), goals)


@pytest.mark.parametrize("d,p,D", [(1, 0.5, 3), (2, 0.5, 5), (2, 0.6, 6), (3, 0.3, 9)])
def test_one_retryable_action(d, p, D):
    sweep = sweep_for([{"name": "try", "duration": d, "end": [(("g",), (), p), ((), (), 1 - p)]}], {"g"})
    curve = sweep.curve(set(), D)
    for r in range(D + 1):
        assert lookup(curve, r) == pytest.approx(1 - (1 - p) ** (r // d), abs=1e-12), r


def test_attempts_count_from_the_last_precondition():
    """v18: the fact arrives at 4, a lasts 2, D = 10 -> 3 attempts, not 5."""
    ops = [
        {"name": "setup", "duration": 4, "end": [(("ready",), (), 1.0)]},
        {"name": "a", "duration": 2, "pre": ("ready",), "end": [(("g",), (), 0.5), ((), (), 0.5)]},
    ]
    assert sweep_for(ops, {"g"}).value(set(), 10) == pytest.approx(1 - 0.5 ** 3, abs=1e-12)


def test_two_independent_attempts():
    ops = [
        {"name": "a1", "duration": 2, "end": [(("g",), (), 0.6), ((), (), 0.4)]},
        {"name": "a2", "duration": 3, "end": [(("g",), (), 0.5), ((), (), 0.5)]},
    ]
    sweep = sweep_for(ops, {"g"})
    assert sweep.value(set(), 3) == pytest.approx(0.8, abs=1e-12)
    assert sweep.value(set(), 2) == pytest.approx(0.6, abs=1e-12)


def test_a_failure_can_still_reach_the_goal():
    """The question is a deadline, not a path: a's failure adds k, and b turns
    k into the goal. Two ways in, 1.0 in total."""
    ops = [
        {"name": "a", "duration": 2, "end": [(("g",), (), 0.6), (("k",), (), 0.4)]},
        {"name": "b", "duration": 1, "pre": ("k",), "end": [(("g",), (), 1.0)]},
    ]
    sweep = sweep_for(ops, {"g"})
    assert sweep.value(set(), 2) == pytest.approx(0.6, abs=1e-12)
    assert sweep.value(set(), 3) == pytest.approx(1.0, abs=1e-12)


def test_start_effect_counts_immediately():
    ops = [{"name": "long", "duration": 9, "start": [(("g",), (), 1.0)]}]
    assert sweep_for(ops, {"g"}).value(set(), 0) == pytest.approx(1.0)


def test_running_action_completes_at_its_earliest_end_then_retries():
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 0.5), ((), (), 0.5)]}]
    sweep = sweep_for(ops, {"g"})
    curve = sweep.curve(set(), 5, running=[("try", 1)])      # completions at 1, 3, 5
    assert lookup(curve, 0) == pytest.approx(0.0)
    assert lookup(curve, 1) == pytest.approx(0.5)
    assert lookup(curve, 5) == pytest.approx(1 - 0.5 ** 3)


def test_running_action_later_than_its_duration():
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 1.0)]}]
    sweep = sweep_for(ops, {"g"})
    curve = sweep.curve(set(), 6, running=[("try", 3)])      # lo = 3 > d = 2
    assert lookup(curve, 2) == pytest.approx(0.0)
    assert lookup(curve, 3) == pytest.approx(1.0)


def test_goal_conjunction_is_read_off_the_joint():
    ops = [
        {"name": "a", "duration": 1, "end": [(("x",), (), 0.5), ((), (), 0.5)]},
        {"name": "b", "duration": 1, "end": [(("y",), (), 0.5), ((), (), 0.5)]},
    ]
    sweep = sweep_for(ops, {"x", "y"})
    assert sweep.value(set(), 2) == pytest.approx((1 - 0.5 ** 2) ** 2, abs=1e-12)


@pytest.mark.parametrize("deadline", [2, 3, 4, 5, 6, 7])
def test_never_below_the_pinned_lower_bound(deadline):
    ops = [
        {"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
        {"name": "b", "duration": 3, "pre": ("p",), "end": [(("g",), (), 0.8), ((), (), 0.2)]},
    ]
    solver = TemporalSTNPDB(ToyModel(ops, {"g"}, deadline), pin_starts_to_clock=True)
    lower = solver.solve(solver.root(set(), deadline))
    assert sweep_for(ops, {"g"}).value(set(), deadline) >= lower.hi - 1e-12


if __name__ == "__main__":
    pytest.main([__file__])
