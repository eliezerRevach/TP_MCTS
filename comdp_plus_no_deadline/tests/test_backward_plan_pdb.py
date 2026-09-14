"""Tests for the backward plan PDB (``backward_plan_pdb``).

No temporal constraints anywhere: the table is built by regression from the
goal, a state is ``(subgoal facts, running actions)``, and a lookup returns an
ordered PLAN. Time never enters.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.backward_plan_pdb import (  # noqa: E402
    BackwardPlanPDB,
    RegressionState,
)
from comdp_plus_no_deadline.engines.temporal_stn_pdb import ToyModel  # noqa: E402


def build(ops, goals, **kw):
    kw.setdefault("max_events", 10)
    kw.setdefault("max_plans_per_state", 8)
    return BackwardPlanPDB(ToyModel(ops, goals, 99), goals, **kw).build()


CHAIN = [
    {"name": "a", "duration": 1, "end": [(("p",), (), 0.6), ((), (), 0.4)]},
    {"name": "b", "duration": 1, "pre": ("p",), "end": [(("g",), (), 0.5), ((), (), 0.5)]},
]

TWO_ACHIEVERS = [
    {"name": "a1", "duration": 1, "end": [(("p",), (), 0.6), ((), (), 0.4)]},
    {"name": "a2", "duration": 1, "end": [(("p",), (), 0.9), ((), (), 0.1)]},
    {"name": "b", "duration": 1, "pre": ("p",), "end": [(("g",), (), 0.5), ((), (), 0.5)]},
]


# ------------------------------------------------------------------ filter 1
@pytest.mark.parametrize("candidate,keep", [
    ({"f1", "f2"}, True),
    ({"f1", "f3"}, True),
    ({"f1", "f2", "f3"}, True),
    (set(), True),
    ({"f1", "f2", "f3", "f4"}, False),
    ({"f4"}, False),
])
def test_fact_filter_is_subset_of_the_query(candidate, keep):
    """A candidate may need FEWER facts than the query has, never one it lacks.

    Two values, not three: "f4 is false" is not represented in the state -- the
    query simply fails to supply it.
    """
    origin = frozenset({"f1", "f2", "f3"})
    state = RegressionState(frozenset(candidate), frozenset())
    assert state.matches(origin, frozenset()) is keep


# ------------------------------------------------------------------ plans
def test_plan_is_an_ordered_event_sequence_with_its_probability():
    pdb = build(CHAIN, {"g"})
    best, _ = pdb.lookup(set())
    assert best.events == (("a", "S"), ("a", "E"), ("b", "S"), ("b", "E"))
    assert best.probability == pytest.approx(0.6 * 0.5)


def test_already_true_facts_shorten_the_plan():
    pdb = build(CHAIN, {"g"})
    best, _ = pdb.lookup({"p"})
    assert best.events == (("b", "S"), ("b", "E"))
    assert best.probability == pytest.approx(0.5)


def test_running_action_is_matched_and_not_restarted():
    """A running action is resumed (its END is next), never started again."""
    pdb = build(CHAIN, {"g"})
    best, _ = pdb.lookup(set(), running={"a"})
    assert best.events[0] == ("a", "E")
    assert ("a", "S") not in best.events


def test_multiple_plans_per_state_and_lookup_takes_the_best():
    """Two achievers for the same fact = two plans at one state."""
    pdb = build(TWO_ACHIEVERS, {"g"})
    assert pdb.stats()["multi_plan_states"] >= 1
    best, matched = pdb.lookup(set())
    assert len(matched) >= 2
    assert best.probability == pytest.approx(0.9 * 0.5)     # a2, the better one
    assert ("a2", "S") in best.events


# ------------------------------------------------------------------ the DAG
def test_table_is_a_dag_not_a_plan_list():
    """States are expanded once; plans are PATHS through them.

    Storing whole sequences per state re-expands one state once per plan prefix
    and recreates the factorial the backward direction exists to avoid.
    """
    pdb = build(TWO_ACHIEVERS, {"g"})
    stats = pdb.stats()
    # Every key is expanded exactly once, so states <= edges and both stay small.
    assert stats["states"] <= stats["edges"]
    assert stats["states"] < 20
    for state in pdb.edges:
        assert len(pdb.edges[state]) <= pdb.max_plans_per_state


def test_irrelevant_actions_never_enter_the_table():
    """The whole point of going backward: clutter costs nothing.

    ``junk*`` achieves nothing any goal needs, so regression never reaches it.
    """
    clutter = [
        {"name": f"junk{i}", "duration": 1, "end": [((f"z{i}",), (), 1.0)]}
        for i in range(40)
    ]
    bare = build(CHAIN, {"g"})
    noisy = build(CHAIN + clutter, {"g"})
    assert noisy.stats()["states"] == bare.stats()["states"]
    assert noisy.stats()["edges"] == bare.stats()["edges"]
    best, _ = noisy.lookup(set())
    assert all(not k.startswith("junk") for k, _w in best.events)


# ------------------------------------------------------------------ validation
def test_replay_rejects_a_physically_impossible_query():
    """Claiming an action runs without its inExecution fact must score 0.

    Regression only PROPOSES; the forward replay through the real model is what
    decides, so a malformed query yields no plan instead of a confident number.
    """
    model = ToyModel(CHAIN, {"g"}, 99)
    pdb = BackwardPlanPDB(model, {"g"}, max_events=10).build()
    # "a is running" but the inExecution fact is absent from the facts set.
    assert pdb.replay((("a", "E"), ("b", "S"), ("b", "E")), frozenset()) == pytest.approx(0.0)


def test_replay_uses_the_models_own_outcomes():
    pdb = build(CHAIN, {"g"})
    p = pdb.replay((("a", "S"), ("a", "E"), ("b", "S"), ("b", "E")), frozenset())
    assert p == pytest.approx(0.6 * 0.5)


def test_unreachable_goal_yields_no_plan():
    ops = [{"name": "a", "duration": 1, "pre": ("never",), "end": [(("g",), (), 1.0)]}]
    pdb = build(ops, {"g"})
    best, matched = pdb.lookup(set())
    assert best is None and matched == []


# ------------------------------------------------------- partial completion
THREE = [
    {"name": f"a{i}", "duration": 1, "end": [((f"f{i}",), (), 0.8), ((), (), 0.2)]}
    for i in (1, 2, 3)
]


def test_every_running_subset_is_generated():
    """Regression reaches all 2^k running combinations on its own.

    Regressing an END puts that action into ``running``, and it does so for
    every action independently, so the table covers the mid-execution states
    without anyone enumerating them.
    """
    pdb = build(THREE, {"f1", "f2", "f3"}, max_events=12)
    runsets = {tuple(sorted(st.running)) for st in pdb.edges}
    assert runsets == {
        (), ("a1",), ("a2",), ("a3",),
        ("a1", "a2"), ("a1", "a3"), ("a2", "a3"),
        ("a1", "a2", "a3"),
    }


def test_partially_completed_execution_finds_its_exact_state():
    """Executed a1, a2, a3; a2 finished first. Query is (facts={f2}, running={a1,a3}).

    The plan must RESUME the two that are running and not redo a2.
    """
    pdb = build(THREE, {"f1", "f2", "f3"}, max_events=12)
    best, matched = pdb.lookup({"f2"}, {"a1", "a3"})
    assert best is not None
    assert set(best.events) == {("a1", "E"), ("a3", "E")}
    assert best.probability == pytest.approx(0.8 * 0.8)
    assert any(st.subgoal == {"f2"} and st.running == {"a1", "a3"} for st, _ in matched)


@pytest.mark.parametrize("max_events,max_plans", [(12, 8), (12, 1), (3, 8), (2, 8)])
def test_truncation_never_removes_a_state(max_events, max_plans):
    """Budgets truncate PLANS, not the state set.

    ``max_plans_per_state`` caps edges and ``max_events`` caps enumeration at
    lookup; neither drops a state. So coverage of mid-execution queries does not
    degrade with the budget -- only the number of alternatives on offer does.
    """
    pdb = build(THREE, {"f1", "f2", "f3"}, max_events=max_events, max_plans_per_state=max_plans)
    assert any(st.subgoal == {"f2"} and st.running == {"a1", "a3"} for st in pdb.edges)
    best, _ = pdb.lookup({"f2"}, {"a1", "a3"})
    assert best is not None and best.probability == pytest.approx(0.64)


def test_extra_running_actions_do_not_break_the_match():
    """An action running that no plan needs is harmless: ``state.running`` is a
    SUBSET requirement, so unrelated execution does not hide a plan."""
    ops = THREE + [{"name": "junk", "duration": 1, "end": [(("z",), (), 1.0)]}]
    pdb = build(ops, {"f1", "f2", "f3"}, max_events=12)
    best, _ = pdb.lookup({"f2"}, {"a1", "a3", "junk"})
    assert best is not None
    assert set(best.events) == {("a1", "E"), ("a3", "E")}


def test_only_physically_reachable_running_sets_are_generated():
    """Ordering constraints prune the running-sets automatically.

    With a1 -> a2 -> a3 a strict chain (each needs the previous END effect), a1
    and a2 can NEVER overlap, so no state pairs them. Regression discovers that
    on its own: it refuses to regress an END for an action already running, and
    the precondition it propagates is only satisfiable once the earlier action
    has finished.

    So "does the state I need exist?" answers itself -- a state exists exactly
    when the execution it describes is possible. No enumeration, no coverage
    gap, and no invalid combinations wasting table space.
    """
    chain = [
        {"name": "a1", "duration": 1, "end": [(("f1",), (), 0.8), ((), (), 0.2)]},
        {"name": "a2", "duration": 1, "pre": ("f1",),
         "end": [(("f2",), (), 0.8), ((), (), 0.2)]},
        {"name": "a3", "duration": 1, "pre": ("f2",),
         "end": [(("f3",), (), 0.8), ((), (), 0.2)]},
    ]
    pdb = build(chain, {"f3"}, max_events=12)
    runsets = {tuple(sorted(st.running)) for st in pdb.edges}
    assert runsets == {(), ("a1",), ("a2",), ("a3",)}       # no pairs: impossible


def test_partial_ordering_keeps_exactly_the_possible_pairs():
    """a1 -> a2 chained, a3 independent: (a1,a3) and (a2,a3) are possible,
    (a1,a2) is not, and the table reflects precisely that."""
    mixed = [
        {"name": "a1", "duration": 1, "end": [(("f1",), (), 0.8), ((), (), 0.2)]},
        {"name": "a2", "duration": 1, "pre": ("f1",),
         "end": [(("f2",), (), 0.8), ((), (), 0.2)]},
        {"name": "a3", "duration": 1, "end": [(("f3",), (), 0.8), ((), (), 0.2)]},
    ]
    pdb = build(mixed, {"f2", "f3"}, max_events=12)
    runsets = {tuple(sorted(st.running)) for st in pdb.edges}
    assert ("a1", "a3") in runsets
    assert ("a2", "a3") in runsets
    assert ("a1", "a2") not in runsets                      # chained: impossible

    best, _ = pdb.lookup(set(), {"a1", "a3"})
    assert best.events == (("a1", "E"), ("a2", "S"), ("a2", "E"), ("a3", "E"))
    assert best.probability == pytest.approx(0.8 ** 3)

    best, _ = pdb.lookup({"f1"}, {"a2", "a3"})
    assert set(best.events) == {("a2", "E"), ("a3", "E")}
    assert best.probability == pytest.approx(0.8 ** 2)


# ------------------------------------------------------------------ known gap
def test_regression_is_currently_outcome_blind():
    """DOCUMENTS A KNOWN GAP, it does not bless it.

    ``_regress`` unions the adds of ALL outcomes, so ``a`` is regressed as if it
    always succeeds and the 0.4 failure branch never enters the graph. The
    consequence is that ``probability`` is "this exact sequence works first
    try", NOT a value: there is no retry self-loop and no fixpoint.

    Per-outcome regression is the pending change; when it lands this test should
    be replaced by one asserting V = 0.6 + 0.4*V.
    """
    pdb = build(CHAIN, {"g"})
    best, _ = pdb.lookup(set())
    assert best.probability == pytest.approx(0.30)     # 0.6*0.5, a product
    # A true value with retries allowed and no deadline would be 1.0.
    assert best.probability < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
