"""Tests for Part I of the symbolic-STN temporal PDB (``temporal_stn_pdb``).

Anchors are section 9 of ``artifacts/Temporal_STN_PDB_Exact_Draft.docx`` -- the
spec states the expected numbers, so a disagreement points here rather than at a
re-derived domain -- plus closed forms that no other part of this codebase
supplies.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from comdp_plus_no_deadline.engines.temporal_stn_pdb import (  # noqa: E402
    Instance,
    TemporalSTNPDB,
    ToyModel,
)


def solve(ops, goals, deadline, facts=(), running=(), **kwargs):
    model = ToyModel(ops, goals, deadline)
    pdb = TemporalSTNPDB(model, **kwargs)
    return pdb.solve(pdb.root(facts, deadline, running))


# ------------------------------------------------------------------ section 9
def test_probability_check_two_independent_attempts():
    """Section 9: d = 2 and 3, D = 3, p = 0.6 and 0.5, either adds g => 0.8."""
    res = solve(
        [
            {"name": "a1", "duration": 2,
             "end": [(("g",), (), 0.6), ((), (), 0.4)]},
            {"name": "a2", "duration": 3,
             "end": [(("g",), (), 0.5), ((), (), 0.5)]},
        ],
        goals={"g"},
        deadline=3,
    )
    assert res.complete, res
    assert res.hi == pytest.approx(0.8, abs=1e-12)
    assert res.lo == pytest.approx(0.8, abs=1e-12)


def test_probability_check_each_alone():
    """The same instance with only one attempt available: 0.6 and 0.5."""
    a1 = {"name": "a1", "duration": 2, "end": [(("g",), (), 0.6), ((), (), 0.4)]}
    a2 = {"name": "a2", "duration": 3, "end": [(("g",), (), 0.5), ((), (), 0.5)]}
    assert solve([a1], {"g"}, 3).hi == pytest.approx(0.6)
    assert solve([a2], {"g"}, 3).hi == pytest.approx(0.5)


def test_delete_check_start_effect_removes_the_key():
    """Section 9: consume's START removes key; a key-consumer cannot start then.

    ``blocked`` can only be reached if ``use_key`` starts while ``key`` still
    holds, and consume's start effect deletes it. So the value is 0 once consume
    has started, and the only way to get ``blocked`` is to start ``use_key``
    first -- which the solver must find.
    """
    ops = [
        {"name": "consume", "duration": 2, "pre": ("key",),
         "start": [((), ("key",), 1.0)],
         "end": [(("used",), (), 1.0)]},
        {"name": "use_key", "duration": 1, "pre": ("key",),
         "end": [(("blocked",), (), 1.0)]},
    ]
    res = solve(ops, {"blocked", "used"}, 6, facts={"key"})
    assert res.complete, res
    assert res.hi == pytest.approx(1.0)

    # D = 2 is the tight case and it is still 1.0, but ONLY via a same-instant
    # start pair processed in the right order: use_key starts at 0 while key
    # still holds, consume starts at 0 too and deletes it. Both ends land at 1
    # and 2. This is section 2's non-commuting simultaneous events -- swap the
    # processing order and use_key can never start at all.
    tight = solve(ops, {"blocked", "used"}, 2, facts={"key"})
    assert tight.complete, tight
    assert tight.hi == pytest.approx(1.0)

    # D = 1 is where the delete genuinely bites: consume cannot finish in time.
    short = solve(ops, {"blocked", "used"}, 1, facts={"key"})
    assert short.complete, short
    assert short.hi == pytest.approx(0.0)


def test_key_is_false_while_the_action_is_still_running():
    """Section 9: "After the start, key is false although the action is still
    running." Checked directly on the derived state of the successor."""
    model = ToyModel(
        [{"name": "consume", "duration": 2, "pre": ("key",),
          "start": [((), ("key",), 1.0)],
          "end": [(("used",), (), 1.0)]}],
        goals={"used"},
        deadline=5,
    )
    pdb = TemporalSTNPDB(model)
    root = pdb.root({"key"}, 5)
    choices = list(pdb._choices(root))
    assert len(choices) == 1
    (_label, successors), = choices
    (prob, child), = successors
    assert prob == pytest.approx(1.0)
    assert "key" not in child.facts          # deleted by the START effect
    assert child.running == (Instance("consume", 0),)   # yet still running


def test_end_condition_failure_blocks_the_completion():
    """Section 9's overall-condition shape, in the form this codebase HAS one.

    ``Convert_problem`` puts OVERALL conditions on both snap actions, so an
    invariant broken by an intervening event and not restored is caught when the
    end fires. Here ``hold``'s end requires ``light``, and ``douse`` deletes it.
    """
    ops = [
        {"name": "hold", "duration": 4, "pre": ("light",), "end_pre": ("light",),
         "end": [(("held",), (), 1.0)]},
        {"name": "douse", "duration": 1, "pre": ("light",),
         "start": [((), ("light",), 1.0)]},
    ]
    # Without douse the invariant survives and hold completes.
    assert solve([ops[0]], {"held"}, 6, facts={"light"}).hi == pytest.approx(1.0)
    # douse is available but starting it can only destroy the invariant, so the
    # optimal policy simply does not use it -- the value is unchanged.
    both = solve(ops, {"held"}, 6, facts={"light"})
    assert both.complete, both
    assert both.hi == pytest.approx(1.0)
    # Force the conflict: make douse mandatory for the goal.
    forced = solve(ops, {"held", "doused"}, 6, facts={"light"})
    assert forced.hi == pytest.approx(0.0)


# ------------------------------------------------------------------ closed forms
@pytest.mark.parametrize("d,p,D", [(1, 0.5, 1), (1, 0.5, 3), (2, 0.5, 5), (2, 0.5, 4), (3, 0.3, 9)])
def test_sequential_retries_closed_form(d, p, D):
    """One retryable action: V = 1 - (1-p)^floor(D/d).

    ``inExecution`` forbids a second concurrent copy, so the retries serialise.
    """
    res = solve(
        [{"name": "try", "duration": d, "end": [(("g",), (), p), ((), (), 1 - p)]}],
        goals={"g"},
        deadline=D,
    )
    assert res.complete, res
    assert res.hi == pytest.approx(1.0 - (1.0 - p) ** (D // d), abs=1e-12)


def test_action_longer_than_the_deadline_is_worthless():
    ops = [{"name": "try", "duration": 5, "end": [(("g",), (), 1.0)]}]
    assert solve(ops, {"g"}, 4).hi == pytest.approx(0.0)
    assert solve(ops, {"g"}, 5).hi == pytest.approx(1.0)


def test_goal_already_true_is_value_one():
    res = solve([{"name": "x", "duration": 1}], {"g"}, 5, facts={"g"})
    assert res.hi == pytest.approx(1.0)


def test_start_effect_can_achieve_the_goal_without_completing():
    """Section 2: "Do not impose E_a <= D ... a start effect may achieve the
    goal while the action continues." A duration-9 action started at 0 with a
    goal-achieving START effect must score 1.0 under a deadline of 3."""
    res = solve(
        [{"name": "long", "duration": 9, "start": [(("g",), (), 1.0)]}],
        goals={"g"},
        deadline=3,
    )
    assert res.complete, res
    assert res.hi == pytest.approx(1.0)


# ------------------------------------------------------------------ machinery
def test_unknown_leaf_is_not_zero():
    """Section 5: "Budget limits computation, not the domain: UNKNOWN is not
    value zero." A budget of 1 must give the vacuous interval, not 0."""
    res = solve(
        [{"name": "try", "duration": 1, "end": [(("g",), (), 0.5), ((), (), 0.5)]}],
        goals={"g"},
        deadline=20,
        node_budget=1,
    )
    assert not res.complete
    assert not res.exact
    assert res.unknown_leaves > 0
    assert res.lo == pytest.approx(0.0)
    assert res.hi == pytest.approx(1.0)


def test_interval_brackets_the_complete_answer():
    """Whatever the budget, [lo, hi] must contain the fully expanded value."""
    ops = [
        {"name": "a", "duration": 2, "end": [(("g",), (), 0.6), ((), (), 0.4)]},
        {"name": "b", "duration": 3, "end": [(("g",), (), 0.5), ((), (), 0.5)]},
    ]
    truth = solve(ops, {"g"}, 6)
    assert truth.complete
    for budget in (1, 2, 3, 5, 10, 25, 60):
        cut = solve(ops, {"g"}, 6, node_budget=budget)
        assert cut.lo <= truth.hi + 1e-12
        assert cut.hi >= truth.lo - 1e-12


def test_eager_placement_agrees_with_deferred_placement():
    """Section 3 enumerated at START time must give the same value as letting
    the END group choices generate the same order-regions."""
    ops = [
        {"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
        {"name": "b", "duration": 3, "end": [(("q",), (), 0.6), ((), (), 0.4)]},
    ]
    for deadline in (3, 4, 5):
        lazy = solve(ops, {"p", "q"}, deadline)
        eager = solve(ops, {"p", "q"}, deadline, eager_placement=True)
        assert lazy.complete and eager.complete, (lazy, eager)
        assert lazy.hi == pytest.approx(eager.hi, abs=1e-12)
        # Eager mode splits the same order-regions earlier, so it must visit at
        # least as many nodes for the identical answer -- that is the cost of
        # taking section 3 literally on top of section 4.
        assert eager.nodes_expanded >= lazy.nodes_expanded


def test_repeated_instances_of_one_type_are_distinct():
    """Section 8: "Repeated instances of one type are distinct." Two retries
    must occupy different STN variables or their durations collapse."""
    model = ToyModel(
        [{"name": "try", "duration": 2, "end": [(("g",), (), 0.5), ((), (), 0.5)]}],
        goals={"g"},
        deadline=6,
    )
    pdb = TemporalSTNPDB(model)
    node = pdb.root(set(), 6)
    seen = set()
    for _ in range(3):
        (_label, successors), = [c for c in pdb._choices(node) if c[0][0] == "START"]
        (_p, node) = successors[0]
        seen.add(node.running[0].start_var if node.running else None)
        ends = [c for c in pdb._choices(node) if c[0][0] == "END"]
        if not ends:
            break
        # take the failure branch so the action can be retried
        _label, successors = ends[0]
        node = min(successors, key=lambda ps: len(ps[1].facts))[1]
    assert len(seen) >= 2
    assert "S:try#0" in seen and "S:try#1" in seen


# ------------------------------------------------------ section 6: the bracket
def test_pinned_starts_certify_exactness_and_symbolic_starts_do_not():
    """``pin_starts_to_clock`` is what discharges section 6's obligation.

    With ``S_a = C`` every event time is a rational point, so a node is one
    ground state and the per-node max IS the MDP backup. With symbolic starts a
    node covers an interval and the same max is a relaxation -- which the
    certificate must refuse to bless.
    """
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 0.5), ((), (), 0.5)]}]
    pinned = solve(ops, {"g"}, 5, pin_starts_to_clock=True)
    symbolic = solve(ops, {"g"}, 5)
    assert pinned.starts_pinned and pinned.complete
    assert pinned.exact, pinned
    assert symbolic.complete and not symbolic.exact
    assert symbolic.branchwise_gap_witnessed


@pytest.mark.parametrize("deadline", [2, 3, 4, 5, 6, 7])
def test_pinned_value_never_exceeds_the_symbolic_bound(deadline):
    """V_pinned <= V* <= V_symbolic.

    Pinning removes policies (starts may only happen at event boundaries), so it
    can never score higher than the relaxed bound. A violation would mean the
    'upper bound' is not one.
    """
    ops = [
        {"name": "a", "duration": 2, "end": [(("p",), (), 0.7), ((), (), 0.3)]},
        {"name": "b", "duration": 3, "pre": ("p",), "end": [(("g",), (), 0.8), ((), (), 0.2)]},
    ]
    pinned = solve(ops, {"g"}, deadline, pin_starts_to_clock=True)
    symbolic = solve(ops, {"g"}, deadline)
    assert pinned.complete and symbolic.complete
    assert pinned.hi <= symbolic.hi + 1e-12, (pinned, symbolic)


def test_the_bracket_collapses_when_waiting_cannot_help():
    """With no reason to delay, the two sides agree -- so the relaxation is not
    automatically loose, it is loose only where slack is exploitable."""
    ops = [{"name": "try", "duration": 2, "end": [(("g",), (), 0.5), ((), (), 0.5)]}]
    for deadline in (2, 4, 6, 8):
        pinned = solve(ops, {"g"}, deadline, pin_starts_to_clock=True)
        symbolic = solve(ops, {"g"}, deadline)
        assert pinned.hi == pytest.approx(symbolic.hi, abs=1e-12), (deadline, pinned, symbolic)


def test_inert_unfinishable_pruning_never_changes_a_value():
    """The dominance prune must be a PRUNE, not an approximation.

    An instance whose start does nothing but occupy its own slot can only pay
    off at its end, so an end past D makes it dominated by not starting it. The
    values must be identical with the prune on and off.
    """
    ops = [
        {"name": "quick", "duration": 1, "end": [(("p",), (), 0.6), ((), (), 0.4)]},
        {"name": "slow", "duration": 9, "end": [(("p",), (), 1.0)]},
        {"name": "finish", "duration": 1, "pre": ("p",), "end": [(("g",), (), 1.0)]},
    ]
    agreed = 0
    # Deadlines where "slow" (d=9) cannot finish, so the prune fires. That it
    # does NOT fire once slow becomes finishable is covered separately.
    for deadline in (1, 2, 3, 4, 5, 6):
        on = solve(ops, {"g"}, deadline, node_budget=20000)
        off = solve(ops, {"g"}, deadline, prune_inert_unfinishable=False, node_budget=20000)
        assert on.complete, (deadline, on)
        assert on.nodes_expanded <= off.nodes_expanded
        if off.complete:
            # Both closed: the values must be identical, not merely close.
            assert on.hi == pytest.approx(off.hi, abs=1e-12), (deadline, on, off)
            agreed += 1
        else:
            # The unpruned side ran out of budget -- which is the point of the
            # prune -- so it only gives an interval, and the pruned answer must
            # lie inside it.
            assert off.lo - 1e-12 <= on.hi <= off.hi + 1e-12, (deadline, on, off)
    assert agreed >= 3, "expected several deadlines where both sides close"


def test_pruning_keeps_a_start_effect_that_achieves_the_goal():
    """Section 2's caveat: "a start effect may achieve the goal while the action
    continues". Such an op is NOT inert, so the prune must leave it alone even
    though its end lands well past the deadline."""
    ops = [{"name": "long", "duration": 9, "start": [(("g",), (), 1.0)]}]
    res = solve(ops, {"g"}, 3)
    assert res.complete
    assert res.hi == pytest.approx(1.0)


# --------------------------------------------- required concurrency / start order
def _required_concurrency_ops():
    """a3 needs a1 FINISHED and a2 STILL RUNNING -- the match/fuse structure.

    a1 (d=3) produces ``done1`` at its end. a2 (d=2) opens a window
    ``a2_running`` at its start and closes it at its end, and is ONE-SHOT (its
    start consumes ``a2_fresh``). a3 (d=1) needs ``done1`` and the window, at
    both its start and its end -- which is how ``Convert_problem`` compiles an
    OVERALL condition.

    So a2 must be DELAYED until a1 has finished. "Start everything as early as
    possible" fails here, which is the whole point of the example.
    """
    return [
        {"name": "a1", "duration": 3, "end": [(("done1",), (), 1.0)]},
        {"name": "a2", "duration": 2, "pre": ("a2_fresh",),
         "start": [(("a2_running",), ("a2_fresh",), 1.0)],
         "end": [((), ("a2_running",), 1.0)]},
        {"name": "a3", "duration": 1, "pre": ("done1", "a2_running"),
         "end_pre": ("a2_running",), "end": [(("goal",), (), 1.0)]},
    ]


def _first_two_starts(pdb, root, first, second):
    def start(node, name):
        for label, successors in pdb._choices(node):
            if label[0] == "START" and label[1] == name:
                return successors[0][1]
        raise AssertionError(f"{name} not startable")
    return start(start(root, first), second)


def test_start_order_is_not_interchangeable():
    """``a1 < a2`` and ``a2 < a1`` are NOT the same choice.

    The singleton-START decomposition imposes ``S_first <= S_second``, and here
    that single bound is worth the entire value range: starting a2 first forces
    its one window to close before a1 can finish, so a3 can never run. This is
    why start orders may not be collapsed by a partial-order reduction, however
    "independent" the actions look -- they interact through TIME, not facts.
    """
    ops = _required_concurrency_ops()
    model = ToyModel(ops, {"goal"}, 6)
    pdb = TemporalSTNPDB(model)
    root = pdb.root({"a2_fresh"}, 6)

    good = _first_two_starts(pdb, root, "a1", "a2")
    bad = _first_two_starts(pdb, root, "a2", "a1")
    assert good.facts == bad.facts          # same facts...
    assert good.z.canonical() != bad.z.canonical()   # ...different time regions

    good_lo, good_hi, _ = TemporalSTNPDB(ToyModel(ops, {"goal"}, 6))._value(good)
    bad_lo, bad_hi, _ = TemporalSTNPDB(ToyModel(ops, {"goal"}, 6))._value(bad)
    assert good_lo == pytest.approx(1.0) and good_hi == pytest.approx(1.0)
    assert bad_lo == pytest.approx(0.0) and bad_hi == pytest.approx(0.0)

    # The solver must pick the order that works.
    assert solve(ops, {"goal"}, 6, facts={"a2_fresh"}).hi == pytest.approx(1.0)


def test_required_concurrency_needs_a_deliberate_delay():
    """The winning schedule delays a2 until a1 is done: a1 [0,3], a2 [3,5],
    a3 inside [3,4]. Too short a deadline and it stops fitting."""
    ops = _required_concurrency_ops()
    assert solve(ops, {"goal"}, 5, facts={"a2_fresh"}).hi == pytest.approx(1.0)
    assert solve(ops, {"goal"}, 3, facts={"a2_fresh"}).hi == pytest.approx(0.0)


def test_retries_dissolve_the_start_order_counterexample():
    """The order only matters because a2 is ONE-SHOT.

    Drop the ``a2_fresh`` consumable and a2 can simply be restarted after a1
    finishes, so both orders reach 1.0. Worth pinning down: it says the hazard
    is CONSUMABLE windows, not concurrency as such.
    """
    ops = _required_concurrency_ops()
    ops[1] = dict(ops[1])
    ops[1]["pre"] = ()
    ops[1]["start"] = [(("a2_running",), (), 1.0)]
    model = ToyModel(ops, {"goal"}, 8)
    pdb = TemporalSTNPDB(model)
    root = pdb.root(set(), 8)
    for first, second in (("a1", "a2"), ("a2", "a1")):
        node = _first_two_starts(pdb, root, first, second)
        lo, hi, _ = TemporalSTNPDB(ToyModel(ops, {"goal"}, 8))._value(node)
        assert hi == pytest.approx(1.0), (first, second, lo, hi)


def test_probability_mass_is_conserved_across_outcomes():
    ops = [{"name": "a", "duration": 1,
            "end": [(("x",), (), 0.2), (("y",), (), 0.3), ((), (), 0.5)]}]
    model = ToyModel(ops, {"x"}, 4)
    pdb = TemporalSTNPDB(model)
    node = pdb.root(set(), 4)
    (_label, successors), = [c for c in pdb._choices(node) if c[0][0] == "START"]
    (_p, started) = successors[0]
    (_label, ends), = [c for c in pdb._choices(started) if c[0][0] == "END"]
    assert sum(p for p, _ in ends) == pytest.approx(1.0, abs=1e-12)


def test_duplicate_achievers_fold_into_one_successor():
    """Section 4: "Sum outcomes reaching an identical annotated successor."
    Two branches with the same projected effect must not become two plans."""
    ops = [{"name": "a", "duration": 1,
            "end": [(("g",), (), 0.3), (("g",), (), 0.4), ((), (), 0.3)]}]
    res = solve(ops, {"g"}, 1)
    assert res.complete, res
    assert res.hi == pytest.approx(0.7, abs=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
