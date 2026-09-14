"""Tests for the first (upper-bound) version of ``critical_path_pdb``.

Anchors are the doc's own numbers (``Temporal_PDB_Backward_Critical_Path5.docx``:
the section 3 trace, section 7 retries, section 11 validation plan) plus a
cross-check against ``temporal_stn_pdb``'s pinned solver, which is a LOWER bound
on the same model.
"""

import os
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.critical_path_pdb import (  # noqa: E402
    NO_TAIL,
    CriticalPathPDB,
    Entry,
    dominates,
    gap_is_consistent,
)
from comdp_plus_no_deadline.engines.temporal_stn_pdb import (  # noqa: E402
    TemporalSTNPDB,
    ToyModel,
    in_execution,
)


def pdb_for(ops, goals, horizon, **kwargs):
    return CriticalPathPDB(ToyModel(ops, goals, horizon), horizon, **kwargs)


# ------------------------------------------------------------ section 3 trace
def test_interval_node_gaps_match_the_section_3_trace():
    """d(a1) = 4 running. d(a2) = 2: both gaps. d(a2) = 5: only after E_a1."""
    assert gap_is_consistent((4,), 2, 0)
    assert gap_is_consistent((4,), 2, 1)
    assert not gap_is_consistent((4,), 5, 0)
    assert gap_is_consistent((4,), 5, 1)


def test_closed_gap_admits_the_tie():
    """E_a = E_1 is reachable from both sides: d(a) = d(r) with r just started."""
    assert gap_is_consistent((3,), 3, 0)
    assert gap_is_consistent((3,), 3, 1)


# ---------------------------------------------------------- section 11 checks
TWO_ATTEMPTS = [
    {"name": "a1", "duration": 2, "end": [(("g",), (), 0.6), ((), (), 0.4)]},
    {"name": "a2", "duration": 3, "end": [(("g",), (), 0.5), ((), (), 0.5)]},
]


def test_two_independent_attempts():
    """Section 11: d = 2, 3; p = 0.6, 0.5; D = 3 -> 0.8. One build answers D = 2 too."""
    pdb = pdb_for(TWO_ATTEMPTS, {"g"}, 3)
    assert pdb.value(set(), 3) == pytest.approx(0.8, abs=1e-12)
    assert pdb.value(set(), 2) == pytest.approx(0.6, abs=1e-12)
    assert pdb.complete and pdb.converged


def test_dominance_keeps_pairs_that_differ_only_in_tails():
    """Section 10. The two-attempt split holds both; neither may be dropped."""
    a = Entry(0, (0, NO_TAIL), 0.6)
    b = Entry(0, (0, 0), 0.8)
    assert not dominates(a, b)
    assert not dominates(b, a)


@pytest.mark.parametrize("d,p,H", [(1, 0.5, 3), (2, 0.5, 5), (2, 0.6, 6), (3, 0.3, 9)])
def test_one_retryable_action_closed_form(d, p, H):
    """Section 11: 1 - (1 - p)^floor(D/d), for every D <= H from ONE build."""
    ops = [{"name": "try", "duration": d, "end": [(("g",), (), p), ((), (), 1 - p)]}]
    pdb = pdb_for(ops, {"g"}, H)
    for D in range(H + 1):
        assert pdb.value(set(), D) == pytest.approx(1.0 - (1.0 - p) ** (D // d), abs=1e-12), D


def test_section_7_numbers():
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 0.6), ((), (), 0.4)]}]
    pdb = pdb_for(ops, {"g"}, 6)
    assert [pdb.value(set(), r) for r in (2, 4, 6)] == pytest.approx([0.6, 0.84, 0.936], abs=1e-12)


def _required_concurrency_ops():
    """a3 needs a1 finished and a2 still running; a2 is one-shot."""
    return [
        {"name": "a1", "duration": 3, "end": [(("done1",), (), 1.0)]},
        {"name": "a2", "duration": 2, "pre": ("a2_fresh",),
         "start": [(("a2_running",), ("a2_fresh",), 1.0)],
         "end": [((), ("a2_running",), 1.0)]},
        {"name": "a3", "duration": 1, "pre": ("done1", "a2_running"),
         "end_pre": ("a2_running",), "end": [(("goal",), (), 1.0)]},
    ]


def test_required_concurrency_start_order():
    """Section 11: a1 before a2 -> 1.0, a2 before a1 -> 0.0."""
    pdb = pdb_for(_required_concurrency_ops(), {"goal"}, 6)
    assert pdb.value({"a2_fresh"}, 6) == pytest.approx(1.0)
    a1_first = (frozenset({in_execution("a1"), "a2_fresh"}), ("a1",))
    a2_first = (frozenset({in_execution("a2"), "a2_running"}), ("a2",))
    assert pdb.state_value(a1_first, 6, (3,)) == pytest.approx(1.0)
    assert pdb.state_value(a2_first, 6, (2,)) == pytest.approx(0.0)


def test_required_concurrency_deadline():
    """a1 [0,3], a2 [2,4], a3 [3,4]: needs D >= 4."""
    pdb = pdb_for(_required_concurrency_ops(), {"goal"}, 6)
    assert pdb.value({"a2_fresh"}, 4) == pytest.approx(1.0)
    assert pdb.value({"a2_fresh"}, 3) == pytest.approx(0.0)


# ---------------------------------------------------------- section 5 / 8
@pytest.mark.parametrize("d1,d2", [(3, 2), (2, 3)])
def test_critical_path_is_the_longest_chain(d1, d2):
    """a3 needs both ends: T = max(d1 + d3, d2 + d3)."""
    ops = [
        {"name": "a1", "duration": d1, "end": [(("done1",), (), 1.0)]},
        {"name": "a2", "duration": d2, "end": [(("done2",), (), 1.0)]},
        {"name": "a3", "duration": 1, "pre": ("done1", "done2"), "end": [(("goal",), (), 1.0)]},
    ]
    pdb = pdb_for(ops, {"goal"}, 6)
    T = max(d1, d2) + 1
    assert pdb.value(set(), T) == pytest.approx(1.0)
    assert pdb.value(set(), T - 1) == pytest.approx(0.0)


def test_running_action_pushes_the_requirement_by_its_remaining_time():
    """Section 8: a running action that must still finish adds r_a + tail_a."""
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 1.0)]}]
    pdb = pdb_for(ops, {"g"}, 4)
    facts = {in_execution("try")}
    assert pdb.value(facts, 1, running=[("try", 1)]) == pytest.approx(1.0)
    assert pdb.value(facts, Fraction(1, 2), running=[("try", 1)]) == pytest.approx(0.0)


def test_start_effect_can_achieve_the_goal_while_running():
    ops = [{"name": "long", "duration": 9, "start": [(("g",), (), 1.0)]}]
    assert pdb_for(ops, {"g"}, 3).value(set(), 3) == pytest.approx(1.0)


def test_goal_already_true():
    assert pdb_for([{"name": "x", "duration": 1}], {"g"}, 5).value({"g"}, 5) == 1.0


def test_lookup_past_the_horizon_is_refused():
    with pytest.raises(ValueError):
        pdb_for(TWO_ATTEMPTS, {"g"}, 3).value(set(), 4)


# ---------------------------------------------------------- the upper bound
def test_documents_the_upper_bound_retries_inside_a_running_action():
    """The known loose case. a1 (one-shot, d = 3) holds the door; a2 (d = 1,
    p = 0.5) needs it open until its end. Exactly 3 tries fit: 0.875. The
    table only knows a1 is running, so every retry fits and the deadline caps
    them: 1 - 0.5^10."""
    ops = [
        {"name": "a1", "duration": 3, "pre": ("fresh",),
         "start": [(("door",), ("fresh",), 1.0)], "end": [((), ("door",), 1.0)]},
        {"name": "a2", "duration": 1, "pre": ("door",), "end_pre": ("door",),
         "end": [(("through",), (), 0.5), ((), (), 0.5)]},
    ]
    pdb = pdb_for(ops, {"through"}, 10)
    value = pdb.value({"fresh"}, 10)
    assert value == pytest.approx(1 - 0.5 ** 10, abs=1e-12)
    assert value >= 0.875


def test_budget_cut_stays_an_upper_bound():
    full = pdb_for(TWO_ATTEMPTS, {"g"}, 3).value(set(), 3)
    for budget in (1, 2, 3, 4):                   # the full table expands 5 states
        pdb = pdb_for(TWO_ATTEMPTS, {"g"}, 3, max_states=budget)
        cut = pdb.value(set(), 3)
        assert not pdb.complete
        assert cut >= full - 1e-12, (budget, cut, full)


@pytest.mark.parametrize("deadline", [2, 3, 4, 5, 6, 7])
def test_never_below_the_pinned_lower_bound(deadline):
    """temporal_stn_pdb with pinned starts is exact for a restricted policy
    class, so V_pinned <= V* <= this table."""
    ops = [
        {"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
        {"name": "b", "duration": 3, "pre": ("p",), "end": [(("g",), (), 0.8), ((), (), 0.2)]},
    ]
    lower_solver = TemporalSTNPDB(ToyModel(ops, {"g"}, deadline), pin_starts_to_clock=True)
    lower = lower_solver.solve(lower_solver.root(set(), deadline))
    assert lower.complete
    upper = pdb_for(ops, {"g"}, deadline).value(set(), deadline)
    assert upper >= lower.hi - 1e-12, (deadline, lower, upper)


if __name__ == "__main__":
    pytest.main([__file__])
