"""Tests for ILAO* over the time-left windows MDP (``windows_lao``)."""

import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.temporal_stn_pdb import TemporalSTNPDB, ToyModel, in_execution  # noqa: E402
from comdp_plus_no_deadline.engines.windows_lao import (  # noqa: E402
    WindowsLAO,
    WindowsTable,
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
    c's lo). With e the charge is kept and the windows MDP is exact here: 0.84.
    (0.868 was once read as a gap in how outcome branches combine; it was ILAO*
    stopping on a stale value after a tie switched its best choice -- a solve of
    every state from 0 gives 0.84.)"""
    without, _ = solve(A_B + [UNRELATED_C], {"g"}, 8, heuristic="none")
    with_h, _ = solve(A_B + [UNRELATED_C], {"g"}, 8)
    assert without == pytest.approx(0.84, abs=1e-9)
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


# ---------------------------------------------------------------- zero-time loops
NOOP = {"name": "fiddle", "duration": 0, "start": [((), (), 1.0)]}
TRY_HALF = {"name": "try", "duration": 2, "end": [(("g",), (), 0.5), ((), (), 0.5)]}


def test_instant_noop_no_longer_traps_ilao():
    """fiddle changes nothing, so V(s) = 1 * V(s) holds any value; ILAO* started at
    h = 1 used to stay there. Dropped, the value is the real one."""
    value, lao = solve([TRY_HALF, NOOP], {"g"}, 3, heuristic="none")
    assert value == pytest.approx(0.5, abs=1e-9)
    assert lao.stats["pruned_noop"] > 0


LIGHT = [
    {"name": "turn_on", "duration": 0, "pre": ("dark",),
     "start": [(("light",), ("dark",), 0.8), ((), (), 0.1), (("broken",), ("dark",), 0.1)]},
    {"name": "read", "duration": 3, "pre": ("light",), "end": [(("goal",), (), 1.0)]},
]


@pytest.mark.parametrize("heuristic", ["sweep", "none"])
def test_instant_retry_is_folded(heuristic):
    """Staying dark (0.1) comes back at zero time: retried until it leaves, 0.8 / 0.9."""
    assert solve(LIGHT, {"goal"}, 3, facts={"dark"}, heuristic=heuristic)[0] == pytest.approx(8 / 9, abs=1e-9)
    assert solve(LIGHT, {"goal"}, 2, facts={"dark"}, heuristic=heuristic)[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- the full table
def table(ops, goals, D, facts=()):
    t = WindowsTable(WindowsLAO(ToyModel(ops, goals, D), heuristic="none"), D)
    assert t.build(set(facts))
    return t


@pytest.mark.parametrize("ops,goals,D,facts", [
    (TWO, {"g"}, 3, ()),
    (A_B, {"g"}, 8, ()),
    (A_B + [UNRELATED_C], {"g"}, 8, ()),
    (_required_concurrency_ops(), {"goal"}, 4, {"a2_fresh"}),
    (LIGHT, {"goal"}, 5, {"dark"}),
])
def test_table_is_the_exact_windows_mdp_at_every_r(ops, goals, D, facts):
    """One build at D answers every r <= D with the optimum of the same MDP
    (ILAO* with h = 1 converges to it)."""
    t = table(ops, goals, D, facts)
    for r in range(D + 1):
        exact = solve(ops, goals, r, facts=facts, heuristic="none")[0]
        assert t.lookup(set(facts), r) == (pytest.approx(exact, abs=1e-9), "exact")


def test_table_retry_curve():
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 0.6), ((), (), 0.4)]}]
    t = table(ops, {"g"}, 9)
    assert t.stats["backward"] == "layers"
    for r in range(10):
        assert t.lookup(set(), r)[0] == pytest.approx(1 - 0.4 ** (r // 2), abs=1e-12)
    assert t.lookup(set(), 4.5)[0] == pytest.approx(1 - 0.4 ** 2, abs=1e-12)    # r off the grid


LIGHT_SWITCH = [LIGHT[0], {"name": "turn_off", "duration": 0, "pre": ("light",),
                           "start": [(("dark",), ("light",), 1.0)]}]


def test_table_zero_time_cycle_over_two_states():
    """dark -> light -> dark at zero time. Goal broken: 0.1 leaks per lap, so it is
    reached with probability 1. With a deterministic turn_on nothing leaks: 0."""
    t = table(LIGHT_SWITCH, {"broken"}, 2, {"dark"})
    assert t.stats["backward"] == "worklist"
    assert t.lookup({"dark"}, 2)[0] == pytest.approx(1.0, abs=1e-9)
    sure = [dict(LIGHT_SWITCH[0], start=[(("light",), ("dark",), 1.0)]), LIGHT_SWITCH[1]]
    assert table(sure, {"broken"}, 2, {"dark"}).lookup({"dark"}, 2)[0] == 0.0


def test_table_cover_for_an_exact_remaining_time():
    """a (3), then b (2) placed after a's end: when a ends, b has [0, 2] left, e = 0.
    A leaf where b is exactly 1 from its end is covered by that looser state."""
    ops = [{"name": "a", "duration": 3, "end": [(("pa",), (), 1.0)]},
           {"name": "b", "duration": 2, "end": [(("g",), (), 1.0)]}]
    t = table(ops, {"g"}, 6)
    value, kind = t.lookup({"pa", in_execution("b")}, 2, running=[("b", 1)])
    assert (kind, value) == ("cover", pytest.approx(1.0))
    assert t.lookup(set(), 7) == (None, "miss")                                   # beyond the horizon


def test_table_budget_cut_is_not_used():
    t = WindowsTable(WindowsLAO(ToyModel(A_B + [UNRELATED_C], {"g"}, 8), heuristic="none"), 8)
    assert t.build(set(), time_budget=0.0) is False
    assert t.lookup(set(), 8) == (None, "miss")


def test_table_covers_a_leaf_in_the_middle_of_an_action():
    """try (2) started at the last event; one unit later the leaf has exactly 1 left.
    The table holds [2, 2] at the event: covered after d = 1, read at r + 1."""
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 1.0)]}]
    t = table(ops, {"g"}, 4)
    facts = {in_execution("try")}
    assert t.lookup(facts, 1, running=[("try", 1)]) == (pytest.approx(1.0), "cover")
    assert t.lookup(facts, F(1, 2), running=[("try", 1)]) == (pytest.approx(0.0), "cover")


@pytest.mark.parametrize("ops,goals,D,facts", [
    (A_B + [UNRELATED_C], {"g"}, 8, ()),
    (_required_concurrency_ops(), {"goal"}, 5, {"a2_fresh"}),
    ([{"name": "a", "duration": 3, "end": [(("pa",), (), 0.7), ((), (), 0.3)]},
      {"name": "b", "duration": 2, "pre": ("pa",), "end": [(("g",), (), 0.6), ((), (), 0.4)]},
      {"name": "c", "duration": 1, "end": [(("q",), (), 0.5), ((), (), 0.5)]}], {"g"}, 7, ()),
])
def test_shifted_cover_is_never_below_the_leaf(ops, goals, D, facts):
    """Every table state with running ends, every time d after its event that the
    windows allow, every r: EVERY fitting table state, on its own, is at least the
    exact value of that leaf (ILAO* with h = 1 from the leaf itself) -- so the
    smallest of them, which lookup returns, is an upper bound."""
    t = table(ops, goals, D, facts)
    exact_solver = WindowsLAO(ToyModel(ops, goals, D), heuristic="none")   # h = 1: converges to the optimum
    checked = 0
    for i, (f, queue, windows) in enumerate(t.states[:400]):
        if not queue or goals <= f:
            continue
        lo_max = min(hi for _lo, hi, _e in windows)
        for d in sorted({F(0), lo_max / 2, lo_max}):
            rem = [max(lo - d, e - d, F(0)) for lo, _hi, e in windows]
            if any(x > hi - d for x, (_lo, hi, _e) in zip(rem, windows)) or len(set(rem)) < len(rem):
                continue
            for r in range(D + 1):
                if r + d > D - t.tmin[i]:
                    continue                        # the event would lie before this state can be reached
                running = list(zip(queue, rem))
                value, kind = t.lookup(f, r, running=running)
                exact = exact_solver.solve(f, r, running)
                assert value is not None and value >= exact - 1e-9, (f, queue, windows, d, r, value, exact)
                for rf, rq, rw, _r in t.lao._roots(f, F(r), running):
                    if (rf, rq, rw) not in t.index:
                        fits = t.fitting_values(rf, rq, rw, F(r))
                        assert fits and min(fits) >= exact - 1e-9, (f, queue, windows, d, r, fits, exact)
                checked += 1
    assert checked > 20


if __name__ == "__main__":
    pytest.main([__file__])
