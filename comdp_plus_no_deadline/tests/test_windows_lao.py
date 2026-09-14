"""Tests for ILAO* over the time-left windows MDP (``windows_lao``)."""

import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.temporal_stn_pdb import TemporalSTNPDB, ToyModel, in_execution  # noqa: E402
from comdp_plus_no_deadline.engines.windows_lao import (  # noqa: E402
    WindowsLAO,
    end_windows,
    start_windows,
)

F = Fraction


def solve(ops, goals, r, facts=(), running=(), **kwargs):
    lao = WindowsLAO(ToyModel(ops, goals, r), **kwargs)
    return lao.solve(set(facts), r, running), lao


# ---------------------------------------------------------------- window rules
A1 = ((F(5), F(5), F(5)),)                               # (lo, hi, e) for a1


def test_sequential_inside_a1_sums():
    w = start_windows(A1, F(2), 0)                       # a2 before a1
    assert w == ((F(2), F(5), F(2)), (F(5), F(5), F(5)))
    w, charged = end_windows(w, [F(5)])
    assert (w, charged) == (((F(0), F(3), F(3)),), F(2))
    w = start_windows(w, F(3), 0)                        # a3 before a1
    assert w == ((F(3), F(3), F(3)), (F(3), F(3), F(3)))
    w, charged = end_windows(w, [F(5)])
    assert (w, charged) == (((F(0), F(0), F(0)),), F(3))  # charged 2 + 3, a1 has 5 - 2 - 3


def test_overlap_inside_a1_charges_the_max():
    w = start_windows(A1, F(2), 0)                       # a2 before a1
    w = start_windows(w, F(3), 1)                        # a3 after a2, before a1
    w, first = end_windows(w, [F(3), F(5)])
    assert (w, first) == (((F(0), F(3), F(1)), (F(0), F(3), F(3))), F(2))
    w, second = end_windows(w, [F(5)])
    assert first + second == F(3)                        # max(2, 3)
    assert w == ((F(0), F(3), F(2)),)                    # e_a1 = 5 - max(2, 3); hi stays 3


def test_finish_together_is_kept():
    w = start_windows(((F(4), F(4), F(4)),), F(2), 1)    # a2 after a1
    assert w[1] == (F(4), F(6), F(4))


# ---------------------------------------------------------------- values
TWO = [
    {"name": "a1", "duration": 2, "end": [(("g",), (), 0.6), ((), (), 0.4)]},
    {"name": "a2", "duration": 3, "end": [(("g",), (), 0.5), ((), (), 0.5)]},
]


@pytest.mark.parametrize("heuristic", ["sweep", "none"])
def test_two_independent_attempts(heuristic):
    assert solve(TWO, {"g"}, 3, heuristic=heuristic)[0] == pytest.approx(0.8, abs=1e-9)
    assert solve(TWO, {"g"}, 2, heuristic=heuristic)[0] == pytest.approx(0.6, abs=1e-9)


@pytest.mark.parametrize("d,p,D", [(1, 0.5, 3), (2, 0.5, 5), (2, 0.6, 6), (3, 0.3, 9)])
def test_one_retryable_action(d, p, D):
    ops = [{"name": "try", "duration": d, "end": [(("g",), (), p), ((), (), 1 - p)]}]
    assert solve(ops, {"g"}, D)[0] == pytest.approx(1 - (1 - p) ** (D // d), abs=1e-9)


def _required_concurrency_ops():
    return [
        {"name": "a1", "duration": 3, "end": [(("done1",), (), 1.0)]},
        {"name": "a2", "duration": 2, "pre": ("a2_fresh",),
         "start": [(("a2_running",), ("a2_fresh",), 1.0)],
         "end": [((), ("a2_running",), 1.0)]},
        {"name": "a3", "duration": 1, "pre": ("done1", "a2_running"),
         "end_pre": ("a2_running",), "end": [(("goal",), (), 1.0)]},
    ]


def test_required_concurrency():
    ops = _required_concurrency_ops()
    assert solve(ops, {"goal"}, 4, facts={"a2_fresh"})[0] == pytest.approx(1.0)
    assert solve(ops, {"goal"}, 3, facts={"a2_fresh"})[0] == pytest.approx(0.0)


def test_retry_inside_a_running_action_stops_at_three():
    ops = [
        {"name": "a1", "duration": 3, "pre": ("fresh",),
         "start": [(("door",), ("fresh",), 1.0)], "end": [((), ("door",), 1.0)]},
        {"name": "a2", "duration": 1, "pre": ("door",), "end_pre": ("door",),
         "end": [(("through",), (), 0.5), ((), (), 0.5)]},
    ]
    value, _ = solve(ops, {"through"}, 10, facts={"fresh"})
    assert value == pytest.approx(0.875, abs=1e-9)


def test_failure_can_still_reach_the_goal():
    ops = [
        {"name": "a", "duration": 2, "end": [(("g",), (), 0.6), (("k",), (), 0.4)]},
        {"name": "b", "duration": 1, "pre": ("k",), "end": [(("g",), (), 1.0)]},
    ]
    assert solve(ops, {"g"}, 2)[0] == pytest.approx(0.6, abs=1e-9)
    assert solve(ops, {"g"}, 3)[0] == pytest.approx(1.0, abs=1e-9)


def test_running_action_at_the_root():
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 1.0)]}]
    facts = {in_execution("try")}
    assert solve(ops, {"g"}, 1, facts=facts, running=[("try", 1)])[0] == pytest.approx(1.0)
    assert solve(ops, {"g"}, F(1, 2), facts=facts, running=[("try", 1)])[0] == pytest.approx(0.0)


A_B = [
    {"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
    {"name": "b", "duration": 3, "pre": ("p",), "end": [(("g",), (), 0.8), ((), (), 0.2)]},
]
UNRELATED_C = {"name": "c", "duration": 1, "end": [(("q",), (), 0.5), ((), (), 0.5)]}


def test_heuristic_changes_work_not_value():
    """a then b, D = 8: exact 0.84 (a succeeds by 2 or 4, then b gets 2 or 1 tries)."""
    with_h, lao_h = solve(A_B, {"g"}, 8)
    without, lao_n = solve(A_B, {"g"}, 8, heuristic="none")
    assert with_h == pytest.approx(0.84, abs=1e-9)
    assert without == pytest.approx(0.84, abs=1e-9)
    assert lao_h.stats["expansions"] < lao_n.stats["expansions"]


def test_unrelated_action_no_longer_hides_time():
    """An unrelated c cannot help, so the true value stays 0.84. Charging lo per
    end used to give 0.9964 (c's end shrank b's lo by c's hi while r paid only
    c's lo). With e the charge is kept: 0.868. What remains is not a charging
    leak -- every goal branch of the best policy fits the deadline on its own --
    so it comes from how outcome branches combine. With the sweep as h: exact."""
    without, _ = solve(A_B + [UNRELATED_C], {"g"}, 8, heuristic="none")
    with_h, _ = solve(A_B + [UNRELATED_C], {"g"}, 8)
    assert without == pytest.approx(0.868, abs=1e-9)
    assert with_h == pytest.approx(0.84, abs=1e-9)


@pytest.mark.parametrize("deadline", [2, 3, 4, 5, 6, 7])
def test_never_below_the_pinned_lower_bound(deadline):
    ops = [
        {"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
        {"name": "b", "duration": 3, "pre": ("p",), "end": [(("g",), (), 0.8), ((), (), 0.2)]},
    ]
    solver = TemporalSTNPDB(ToyModel(ops, {"g"}, deadline), pin_starts_to_clock=True)
    lower = solver.solve(solver.root(set(), deadline))
    assert solve(ops, {"g"}, deadline)[0] >= lower.hi - 1e-9


def test_lookup_after_offline_solve_is_exact():
    lao = WindowsLAO(ToyModel(A_B, {"g"}, 8))
    assert lao.solve(set(), 8) == pytest.approx(0.84, abs=1e-9)
    value, kind = lao.lookup(set(), 8)
    assert (kind, value) == ("exact", pytest.approx(0.84, abs=1e-9))


def test_lookup_uses_a_covering_solved_state():
    """Offline the running action's remaining time was unknown ([0, d], e = 0);
    at the leaf it is exactly 1. The looser solved state covers the leaf."""
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 1.0)]}]
    lao = WindowsLAO(ToyModel(ops, {"g"}, 3))
    facts = {in_execution("try")}
    lao.solve(facts, 3, running=[("try", None)])
    value, kind = lao.lookup(facts, 2, running=[("try", 1)])
    assert kind == "cover"
    assert value == pytest.approx(1.0)
    assert lao.lookup(facts, 4, running=[("try", 1)]) == (None, "miss")     # more time than solved


def test_lazy_miss_reuses_solved_states():
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 0.5), ((), (), 0.5)]}]
    lao = WindowsLAO(ToyModel(ops, {"g"}, 8))
    assert lao.solve(set(), 8) == pytest.approx(1 - 0.5 ** 4, abs=1e-9)
    before = lao.stats["expansions"]
    facts = {in_execution("try")}
    assert lao.lookup(facts, 7, running=[("try", 1)]) == (None, "miss")
    lazy = lao.solve(facts, 7, running=[("try", 1)], lazy=True)
    assert lazy == pytest.approx(1 - 0.5 ** 4, abs=1e-9)                  # completions at 1, 3, 5, 7
    assert lao.stats["expansions"] - before <= 2                            # stops at solved states
    assert lao.lookup(facts, 7, running=[("try", 1)])[1] == "exact"


def test_extend_solves_the_branches_the_optimal_policy_skipped():
    """After optimal, extend solves the other branches without changing the root value."""
    ops = A_B + [UNRELATED_C]
    lao = WindowsLAO(ToyModel(ops, {"g"}, 8))
    value = lao.solve(set(), 8)
    solved_at_optimal = len(lao.solved)
    lao.extend(5.0)
    assert len(lao.solved) > solved_at_optimal
    root_value, kind = lao.lookup(set(), 8)
    assert kind == "exact" and root_value == pytest.approx(value, abs=1e-9)
    # a branch the optimal policy did not take is now a lookup hit
    first_choices = [child for _label, branches in lao.options[(frozenset(), (), (), F(8))]
                     for _p, child in branches]
    assert all(child in lao.solved for child in first_choices)


def test_budget_cut_stays_an_upper_bound():
    full, _ = solve(TWO, {"g"}, 3)
    for budget in (1, 2, 3):
        cut, lao = solve(TWO, {"g"}, 3, max_expansions=budget)
        assert cut >= full - 1e-9


if __name__ == "__main__":
    pytest.main([__file__])
