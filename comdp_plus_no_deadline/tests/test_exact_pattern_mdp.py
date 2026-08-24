"""Tests for the exact pattern-MDP heuristic (``exact_pattern_mdp``).

The anchors are chosen so that none of them depends on any other part of this
codebase being right:

* **A1 closed form** -- one action retried sequentially has value
  ``1 - (1-p)^floor(D/d)``. Catches every timing off-by-one (whether ``r=0`` is
  inclusive, whether an action finishing exactly at the deadline counts,
  whether the last retry fits).
* **A2 deterministic** -- with all probabilities 1 the value is 0/1 and is
  checked against an independently written exhaustive schedule simulator.
* **consumable** -- the reason this module exists. Under delete relaxation the
  pin can never break, so PTRPG/survivor-PDB saturate to 1.0; here ``has_pin``
  is genuinely consumed and the value is capped no matter how long the deadline.
"""

import itertools
from types import SimpleNamespace

import pytest

from comdp_plus_no_deadline.engines.exact_pattern_mdp import (
    ExactPatternMDPHeuristic,
    Pattern,
    PatternOp,
    PatternSolver,
    _conflict_table,
    _execution_conflict_table,
    solve_pattern,
)


def _pattern(facts, goal, ops):
    index = {f: i for i, f in enumerate(facts)}

    def mask(fs):
        m = 0
        for f in fs:
            m |= 1 << index[f]
        return m

    built = []
    for name, dur, pre, start_out, end_out in ops:
        s = tuple((mask(a), mask(d), p) for a, d, p in start_out)
        e = tuple((mask(a), mask(d), p) for a, d, p in end_out)
        fp = 0
        for a, d, _ in s + e:
            fp |= a | d
        built.append(PatternOp(
            name=name, duration=dur, pre_mask=mask(pre),
            start_outcomes=s, end_outcomes=e,
            touches_end=any(a or d for a, d, _ in e), footprint=fp,
        ))
    return Pattern(
        facts=tuple(facts), seed_goal=goal, ops=tuple(built),
        goal_mask=mask({goal}), conflicts=_conflict_table(built),
    )


# ---------------------------------------------------------------- A1
@pytest.mark.parametrize("d,p,D", [
    (1, 0.5, 1), (1, 0.5, 3), (2, 0.5, 5), (2, 0.5, 4),
    (3, 0.3, 9), (3, 0.3, 8), (4, 0.7, 12), (1, 0.2, 7),
])
def test_single_action_closed_form(d, p, D):
    """V = 1 - (1-p)^floor(D/d): sequential retries, no overlap of one action."""
    pat = _pattern(
        ("goal",), "goal",
        [("try", d, (), [((), (), 1.0)], [(("goal",), (), p), ((), (), 1 - p)])],
    )
    expected = 1.0 - (1.0 - p) ** (D // d)
    assert solve_pattern(pat, set(), D) == pytest.approx(expected, abs=1e-12)


def test_action_longer_than_deadline_is_worthless():
    pat = _pattern(
        ("goal",), "goal",
        [("try", 5, (), [((), (), 1.0)], [(("goal",), (), 1.0)])],
    )
    assert solve_pattern(pat, set(), 4) == pytest.approx(0.0)
    assert solve_pattern(pat, set(), 5) == pytest.approx(1.0)


# ---------------------------------------------------------------- concurrency
def test_two_achievers_run_concurrently_with_independent_coins():
    """Both may start at the same instant; only the max over policies is taken."""
    pat = _pattern(
        ("goal",), "goal",
        [
            ("a", 1, (), [((), (), 1.0)], [(("goal",), (), 0.3), ((), (), 0.7)]),
            ("b", 1, (), [((), (), 1.0)], [(("goal",), (), 0.6), ((), (), 0.4)]),
        ],
    )
    assert solve_pattern(pat, set(), 1) == pytest.approx(1 - 0.7 * 0.4)


def test_exclusive_outcomes_are_not_decorrelated():
    """a1 reaches the goal w.p. .4 and otherwise lands in s1, from which a2 is
    certain. Truth at horizon 2 is exactly 1.0; noisy-OR over the two achievers
    returns 0.856 because a FAILED path is what ENABLES the other one."""
    pat = _pattern(
        ("s1", "goal"), "goal",
        [
            ("a1", 1, (), [((), (), 1.0)], [(("goal",), (), 0.4), (("s1",), (), 0.6)]),
            ("a2", 1, ("s1",), [((), (), 1.0)], [(("goal",), (), 1.0)]),
        ],
    )
    assert solve_pattern(pat, set(), 1) == pytest.approx(0.4)
    assert solve_pattern(pat, set(), 2) == pytest.approx(1.0)


def test_partial_dispatch_groups_do_not_become_pdb_rows():
    """Redundant starts disappear and useful groups jump to their boundary."""
    pat = _pattern(
        ("f1", "goal"), "goal",
        [
            ("a1", 1, (), [((), (), 1.0)], [(("f1",), (), 1.0)]),
            ("a2", 2, (), [((), (), 1.0)], [(("goal",), (), 1.0)]),
        ],
    )
    solver = PatternSolver(pat)
    assert solve_pattern(pat, {"f1"}, 3, solver=solver) == pytest.approx(1.0)

    at_r3 = [key for key in solver.memo if key[2] == 3]
    assert at_r3 == [(1, (), 3)]
    at_r2 = {(m, rho) for m, rho, r in solver.memo if r == 2}
    assert at_r2 == {(1, ())}
    assert all(all(op_index != 0 for op_index, _rem in rho) for _m, rho, _r in solver.memo)


def test_late_noop_starts_do_not_create_queue_subsets():
    """An identity start whose end misses the deadline is never dispatched."""
    pat = _pattern(
        ("goal",), "goal",
        [
            (f"slow_{i}", 5, (), [((), (), 1.0)], [(("goal",), (), 0.5), ((), (), 0.5)])
            for i in range(12)
        ],
    )
    solver = PatternSolver(pat)
    assert solve_pattern(pat, set(), 4, solver=solver) == pytest.approx(0.0)
    assert list(solver.memo) == [(0, (), 4)]


def test_projected_identical_actions_use_a_remaining_time_multiset():
    """Swapping indistinguishable ground-action identities is one PDB state."""
    pat = _pattern(
        ("goal",), "goal",
        [
            ("try_store_0", 2, (), [((), (), 1.0)],
             [(("goal",), (), 0.5), ((), (), 0.5)]),
            ("try_store_1", 2, (), [((), (), 1.0)],
             [(("goal",), (), 0.5), ((), (), 0.5)]),
        ],
    )
    solver = PatternSolver(pat)
    assert solver.symmetry_classes == ((0, 1),)
    assert solver._canonical_rho(((1, 2),)) == ((0, 2),)
    assert solver._canonical_rho(((0, 1), (1, 2))) == ((0, 1), (1, 2))
    assert solver._canonical_rho(((0, 2), (1, 1))) == ((0, 1), (1, 2))
    assert solve_pattern(pat, set(), 2, solver=solver) == pytest.approx(0.75)


def test_deadline_aligned_grid_keeps_useful_non_event_start():
    """Start-only conditions do not make completion epochs alone sufficient.

    C produces q at t=5. A must start at the unchanged-world time t=4 so its
    delete of f lands at the deadline, after B has used f and q at t=5.
    """
    pat = _pattern(
        ("f", "q", "goal"), "goal",
        [
            ("C", 5, (), [((), (), 1.0)], [(("q",), (), 1.0)]),
            ("A", 2, ("f",), [((), (), 1.0)],
             [(("goal",), ("f",), 0.5), ((), ("f",), 0.5)]),
            ("B", 1, ("f", "q"), [((), (), 1.0)],
             [(("goal",), (), 0.5), ((), (), 0.5)]),
        ],
    )
    assert solve_pattern(pat, {"f"}, 6) == pytest.approx(0.75)


# ---------------------------------------------------------------- deletes
def test_consumable_caps_the_value_forever():
    """The pin/lockpick domain -- the whole point of leaving delete relaxation.

    insert_pin puts the pin in at start and BREAKS it at end (deleting both
    pin_in_hole and has_pin), so it can run exactly once. pick_lock needs the pin
    in the hole to start. Under delete relaxation the pin never breaks and the
    estimate climbs to 1.0 with the horizon; here it must plateau.
    """
    pat = _pattern(
        ("has_lockpick", "has_pin", "pin_in_hole", "door"), "door",
        [
            ("insert_pin", 3, ("has_pin",),
             [(("pin_in_hole",), (), 1.0)],
             [((), ("pin_in_hole", "has_pin"), 1.0)]),
            ("pick_lock", 2, ("has_lockpick", "pin_in_hole"),
             [((), (), 1.0)],
             [(("door",), (), 0.6), ((), (), 0.4)]),
        ],
    )
    init = {"has_lockpick", "has_pin"}
    assert solve_pattern(pat, init, 2) == pytest.approx(0.6)
    assert solve_pattern(pat, init, 3) == pytest.approx(0.6)
    # two sequential attempts fit inside the pin's lifetime, then it is gone
    assert solve_pattern(pat, init, 4) == pytest.approx(1 - 0.4 ** 2)
    plateau = solve_pattern(pat, init, 4)
    for horizon in (8, 20, 40):
        assert solve_pattern(pat, init, horizon) == pytest.approx(plateau), (
            "value kept growing with the horizon -- the consumable is not biting"
        )
    assert plateau < 1.0


def test_delete_blocks_a_second_use():
    """One-shot action: its end effect removes its own precondition."""
    pat = _pattern(
        ("token", "goal"), "goal",
        [("spend", 1, ("token",), [((), (), 1.0)],
          [(("goal",), ("token",), 0.5), ((), ("token",), 0.5)])],
    )
    for horizon in (1, 2, 10, 50):
        assert solve_pattern(pat, {"token"}, horizon) == pytest.approx(0.5)


def test_contradictory_effects_may_not_overlap():
    """Paper's third mutex clause, enforced inside the pattern for free."""
    pat = _pattern(
        ("f", "goal"), "goal",
        [
            ("setter", 4, (), [(("f",), (), 1.0)], [((), (), 1.0)]),
            ("clearer", 4, (), [((), ("f",), 1.0)], [((), (), 1.0)]),
            ("use", 1, ("f",), [((), (), 1.0)], [(("goal",), (), 1.0)]),
        ],
    )
    assert pat.conflicts[0] & (1 << 1), "setter/clearer must be mutex"
    assert solve_pattern(pat, set(), 3) == pytest.approx(1.0)


def test_converted_execution_markers_preserve_static_mutex():
    class Marker:
        def __init__(self, name):
            self.name = name

        def is_fluent_exp(self):
            return True

        def fluent(self):
            return SimpleNamespace(name="inExecution")

    marker_a = Marker("a")
    marker_b = Marker("b")
    actions = [
        SimpleNamespace(add_effects={marker_a}, neg_preconditions={marker_a}),
        SimpleNamespace(add_effects={marker_b}, neg_preconditions={marker_a, marker_b}),
    ]
    table = _execution_conflict_table(actions)
    assert table[0] == 1 << 1
    assert table[1] == 1 << 0


# ---------------------------------------------------------------- A2
def _exhaustive_deterministic(pat, init_facts, D, max_instances=4):
    """Independent oracle: enumerate schedules, simulate the timeline.

    Deliberately written as a different algorithm (assign start times, replay
    events) rather than a second dynamic program, so it cannot share a bug with
    ``solve_pattern``.
    """
    index = {f: i for i, f in enumerate(pat.facts)}
    m0 = 0
    for f in init_facts:
        if f in index:
            m0 |= 1 << index[f]
    times = range(0, D + 1)
    ops = list(enumerate(pat.ops))

    for count in range(max_instances + 1):
        for combo in itertools.product(ops, repeat=count):
            for starts in itertools.product(times, repeat=count):
                # no overlapping copies of one ground action
                bad = False
                for x in range(count):
                    for y in range(x + 1, count):
                        if combo[x][0] != combo[y][0]:
                            continue
                        sx, dx = starts[x], combo[x][1].duration
                        sy, dy = starts[y], combo[y][1].duration
                        if not (sx + dx <= sy or sy + dy <= sx):
                            bad = True
                if bad:
                    continue
                events = []
                for k in range(count):
                    i, op = combo[k]
                    # ENDS must be processed before STARTS at the same instant:
                    # an action beginning at t may rely on a fact produced by
                    # another action finishing at t.
                    events.append((starts[k], 1, "S", i, op))
                    events.append((starts[k] + op.duration, 0, "E", i, op))
                events.sort(key=lambda e: (e[0], e[1]))
                m = m0
                ok = True
                if (m & pat.goal_mask) == pat.goal_mask:
                    return True
                for t, _o, kind, _i, op in events:
                    if t > D:
                        break
                    if kind == "S":
                        if m & op.pre_mask != op.pre_mask:
                            ok = False
                            break
                        add, dele, _p = op.start_outcomes[0]
                    else:
                        add, dele, _p = op.end_outcomes[0]
                    m = (m & ~dele) | add
                    if (m & pat.goal_mask) == pat.goal_mask:
                        return True
                if not ok:
                    continue
    return False


@pytest.mark.parametrize("D", [1, 2, 3, 4, 5, 6])
def test_deterministic_matches_exhaustive_schedule_search(D):
    pat = _pattern(
        ("a", "b", "goal"), "goal",
        [
            ("mk_a", 2, (), [((), (), 1.0)], [(("a",), (), 1.0)]),
            ("mk_b", 1, ("a",), [((), (), 1.0)], [(("b",), (), 1.0)]),
            ("finish", 2, ("b",), [((), (), 1.0)], [(("goal",), (), 1.0)]),
        ],
    )
    dp = solve_pattern(pat, set(), D)
    truth = _exhaustive_deterministic(pat, set(), D)
    assert (dp > 0.5) == truth, f"D={D}: dp={dp} exhaustive={truth}"
    assert dp in (pytest.approx(0.0), pytest.approx(1.0))


def test_goal_already_true_is_one():
    pat = _pattern(("goal",), "goal", [("x", 1, (), [((), (), 1.0)], [((), (), 1.0)])])
    assert solve_pattern(pat, {"goal"}, 0) == pytest.approx(1.0)


def test_from_problem_keeps_only_true_initial_facts():
    true = SimpleNamespace(bool_constant_value=lambda: True)
    false = SimpleNamespace(bool_constant_value=lambda: False)
    problem = SimpleNamespace(
        initial_values={"initially_true": true, "initially_false": false},
        goals={"goal"},
        actions=[],
    )
    heuristic = ExactPatternMDPHeuristic.from_problem(problem)
    assert heuristic._initial_facts == {"initially_true"}


# ---------------------------------------------------------------- relevance
def test_relevance_filter_drops_irrelevant_ops_without_changing_the_value():
    """A pure-delete op is never chosen by a maximiser, so removing it is EXACT."""
    from comdp_plus_no_deadline.engines.exact_pattern_mdp import relevant_ops

    kept = _pattern(
        ("token", "goal"), "goal",
        [("spend", 1, ("token",), [((), (), 1.0)],
          [(("goal",), (), 0.5), ((), (), 0.5)])],
    )
    with_junk = _pattern(
        ("token", "goal"), "goal",
        [
            ("spend", 1, ("token",), [((), (), 1.0)],
             [(("goal",), (), 0.5), ((), (), 0.5)]),
            # only ever removes a tracked fact: contributes nothing
            ("waste", 3, (), [((), ("token",), 1.0)], [((), (), 1.0)]),
        ],
    )
    filtered = relevant_ops(with_junk.ops, with_junk.goal_mask)
    assert [o.name for o in filtered] == ["spend"]
    for horizon in (1, 3, 10):
        assert (solve_pattern(with_junk, {"token"}, horizon)
                == pytest.approx(solve_pattern(kept, {"token"}, horizon)))


def test_relevance_filter_keeps_every_achiever_of_a_needed_fact():
    """Dropping one achiever would be an OR-drop; the filter must not do that."""
    from comdp_plus_no_deadline.engines.exact_pattern_mdp import relevant_ops

    pat = _pattern(
        ("f", "goal"), "goal",
        [
            ("mk_f_a", 1, (), [((), (), 1.0)], [(("f",), (), 0.3), ((), (), 0.7)]),
            ("mk_f_b", 1, (), [((), (), 1.0)], [(("f",), (), 0.6), ((), (), 0.4)]),
            ("use", 1, ("f",), [((), (), 1.0)], [(("goal",), (), 1.0)]),
        ],
    )
    names = {o.name for o in relevant_ops(pat.ops, pat.goal_mask)}
    assert names == {"mk_f_a", "mk_f_b", "use"}
