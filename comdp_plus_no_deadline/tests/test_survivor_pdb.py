"""Tests for the survivor-DP pattern database (baseline_admissible_survivor_pdb).

Two hand-computed reference problems anchor the whole design:

* the ESCAPE-HOUSE example (search for a key p=0.3/step and a lockpick
  p=0.6/step, open the door with either, walk out) — its exact curve is known
  in closed form and by Monte Carlo. The marginal-only bound saturates to 1.0;
  the pattern DP must not.
* the RECOVERABLE-FAILURE example (``a1`` reaches the goal w.p. 0.4 and
  otherwise lands in ``s1``, from which ``a2`` reaches the goal w.p. 1). The
  true value at horizon 2 is exactly 1.0. Noisy-OR over the two achievers
  returns 0.856 — inadmissible — because a FAILED path is what ENABLES the
  other one. The joint sweep has to get this right by construction.
"""

from types import SimpleNamespace

import pytest

from comdp_plus_no_deadline.engines.survivor_pdb import (
    SurvivorAchiever,
    SurvivorPattern,
    build_patterns,
    compute_gate,
    conditional_hazards,
    first_positive_layer,
    joint_add_distribution,
    solve_survivor_pattern,
)


def _achiever(name, pattern_pre, outcomes, *, gate=1, delay=1, strength=1.0):
    return SurvivorAchiever(
        name=name,
        pattern_pre=frozenset(pattern_pre),
        gate=gate,
        delay=delay,
        strength=strength,
        outcomes=tuple((frozenset(add), p) for add, p in outcomes),
    )


def _escape_house_pattern():
    """{key, lockpick, door} with both openers, unit durations."""
    return SurvivorPattern(
        facts=("key", "lockpick", "door"),
        seed_goal="door",
        achievers=(
            _achiever("find_key", (), [(("key",), 0.3), ((), 0.7)]),
            _achiever("find_lockpick", (), [(("lockpick",), 0.6), ((), 0.4)]),
            _achiever("open_with_key", ("key",), [(("door",), 1.0)]),
            _achiever(
                "open_with_lockpick", ("lockpick",), [(("door",), 0.6), ((), 0.4)]
            ),
        ),
    )


def test_escape_house_matches_exact_curve():
    curves = solve_survivor_pattern(_escape_house_pattern(), set(), 6)
    # Independently verified by Monte Carlo (400k runs) on the same process.
    expected = {2: 0.5520, 3: 0.8275, 4: 0.9385, 5: 0.9791, 6: 0.9931}
    for layer, value in expected.items():
        assert curves["door"][layer] == pytest.approx(value, abs=5e-4)


def test_escape_house_does_not_saturate():
    """The whole point: no cap is ever hit, so deep layers stay comparable."""
    curves = solve_survivor_pattern(_escape_house_pattern(), set(), 12)
    assert curves["door"][12] < 1.0
    # min(1, sum H_t) reaches 1.0 by layer 3 on this problem; the joint DP must
    # keep discriminating well past that.
    assert curves["door"][6] < curves["door"][9] < curves["door"][12]


def test_recoverable_failure_is_admissible():
    """Noisy-OR returns 0.856 here; the true value is 1.0."""
    pattern = SurvivorPattern(
        facts=("s1", "goal"),
        seed_goal="goal",
        achievers=(
            # One action, two EXCLUSIVE outcomes: this is the joint that
            # marginal-level operators cannot represent.
            _achiever("a1", (), [(("goal",), 0.4), (("s1",), 0.6)]),
            _achiever("a2", ("s1",), [(("goal",), 1.0)]),
        ),
    )
    curves = solve_survivor_pattern(pattern, set(), 2)
    assert curves["goal"][1] == pytest.approx(0.4)
    assert curves["goal"][2] == pytest.approx(1.0)
    assert curves["goal"][2] >= 1.0 - 1e-12  # would be 0.856 under noisy-OR


def test_curves_are_monotone_and_bounded():
    curves = solve_survivor_pattern(_escape_house_pattern(), set(), 8)
    for curve in curves.values():
        assert all(0.0 <= value <= 1.0 for value in curve)
        assert all(a <= b + 1e-12 for a, b in zip(curve, curve[1:]))


def test_conditional_hazard_is_a_ratio_of_stored_values():
    curves = solve_survivor_pattern(_escape_house_pattern(), set(), 6)
    door = curves["door"]
    hazards = conditional_hazards(door)
    for layer in range(1, 7):
        survivors = 1.0 - door[layer - 1]
        expected = (door[layer] - door[layer - 1]) / survivors if survivors > 0 else 0.0
        assert hazards[layer] == pytest.approx(expected)
    # The hazard is NOT constant: conditioning on failure tilts the posterior
    # toward the states that have not been lucky yet.
    assert hazards[3] != pytest.approx(hazards[2], abs=1e-3)


def test_initial_facts_seed_the_joint():
    curves = solve_survivor_pattern(_escape_house_pattern(), {"key"}, 3)
    assert curves["key"][0] == 1.0
    # Key in hand at t=0 -> the deterministic opener fires immediately.
    assert curves["door"][1] == pytest.approx(1.0)


def test_duration_respected_exactly():
    """An achiever of duration d needs its precondition true at t - d, not t-1."""
    pattern = SurvivorPattern(
        facts=("p", "g"),
        seed_goal="g",
        achievers=(
            _achiever("make_p", (), [(("p",), 1.0)], gate=1, delay=1),
            _achiever("use_p", ("p",), [(("g",), 1.0)], gate=4, delay=3),
        ),
    )
    curves = solve_survivor_pattern(pattern, set(), 6)
    # p becomes true at layer 1; use_p takes 3 layers, so g cannot land before 4.
    assert curves["g"][3] == pytest.approx(0.0)
    assert curves["g"][4] == pytest.approx(1.0)


def test_age_collapse_stays_admissible():
    """The bounded fallback may only raise the value, never lower it."""
    pattern = SurvivorPattern(
        facts=("p", "g"),
        seed_goal="g",
        achievers=(
            _achiever("make_p", (), [(("p",), 0.5), ((), 0.5)], gate=1, delay=1),
            _achiever("use_p", ("p",), [(("g",), 1.0)], gate=4, delay=3),
        ),
    )
    exact = solve_survivor_pattern(pattern, set(), 8, max_states=4096)
    collapsed = solve_survivor_pattern(pattern, set(), 8, max_states=1)
    for layer in range(9):
        assert collapsed["g"][layer] >= exact["g"][layer] - 1e-12


def test_first_positive_layer_and_gate():
    marginals = {"a": [0.0, 0.0, 0.3, 0.5], "b": [0.0, 0.7, 0.9, 1.0]}
    assert first_positive_layer(marginals["a"]) == 2
    assert first_positive_layer([0.0, 0.0]) is None
    # gate = latest precondition arrival + duration
    assert compute_gate(["a", "b"], 2, marginals) == 4
    assert compute_gate(["a", "missing"], 1, marginals) is None


class _Effect:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    def probability_function(self, state, _):
        return self._outcomes


def test_joint_add_distribution_keeps_outcomes_exclusive():
    """Marginals would lose that {goal} and {s1} cannot both happen."""
    action = SimpleNamespace(
        name="a1",
        add_effects=set(),
        probabilistic_effects=[_Effect({0.4: {"goal": True}, 0.6: {"s1": True}})],
    )
    dist = joint_add_distribution(action, set())
    assert dist[frozenset({"goal"})] == pytest.approx(0.4)
    assert dist[frozenset({"s1"})] == pytest.approx(0.6)
    assert frozenset({"goal", "s1"}) not in dist


def test_build_patterns_grows_toward_the_biggest_lie():
    """The freed precondition with the largest marginal gap is attached first."""
    open_with_key = SimpleNamespace(
        name="open_with_key", add_effects={"door"}, probabilistic_effects=[]
    )
    find_key = SimpleNamespace(
        name="find_key", add_effects={"key"}, probabilistic_effects=[]
    )
    horizon = 5
    marginals = {
        "door": [0.0] * (horizon + 1),
        # key is far from certain by the horizon -> "assume free" is a big lie
        "key": [0.0, 0.3, 0.5, 0.6, 0.65, 0.7],
    }
    patterns = build_patterns(
        ["door"],
        actions_by_effect_fact={"door": [open_with_key], "key": [find_key]},
        action_preconditions={
            "open_with_key": frozenset({"key"}),
            "find_key": frozenset(),
        },
        action_delays={"open_with_key": 1, "find_key": 1},
        action_add_probabilities={
            "open_with_key": {"door": 1.0},
            "find_key": {"key": 1.0},
        },
        marginals=marginals,
        initial_facts=set(),
        probe_universe=set(),
        horizon=horizon,
        max_facts=3,
    )
    assert len(patterns) == 1
    assert "key" in patterns[0].facts
    assert {a.name for a in patterns[0].achievers} == {"open_with_key", "find_key"}


class _SyntheticProbabilisticEffect:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.fluents = [
            fact for assignments in outcomes.values() for fact in assignments
        ]

    def probability_function(self, state, env):
        del state, env
        return self.outcomes


class _SyntheticAction:
    def __init__(self, name, pre, duration=1, adds=(), probabilistic=()):
        self.name = name
        self.pos_preconditions = frozenset(pre)
        self.add_effects = frozenset(adds)
        self.duration_steps = duration
        self.probabilistic_effects = tuple(probabilistic)

    def duration_int(self):
        return self.duration_steps


def _escape_house_actions():
    return [
        _SyntheticAction(
            "find_key", (), probabilistic=[_SyntheticProbabilisticEffect({0.3: {"key": True}})]
        ),
        _SyntheticAction(
            "find_lockpick",
            (),
            probabilistic=[_SyntheticProbabilisticEffect({0.6: {"lockpick": True}})],
        ),
        _SyntheticAction("open_with_key", ("key",), adds=("door",)),
        _SyntheticAction(
            "open_with_lockpick",
            ("lockpick",),
            probabilistic=[_SyntheticProbabilisticEffect({0.6: {"door": True}})],
        ),
    ]


def _heuristic_for(actions, goals):
    from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
        TemporalProbabilisticRPGHeuristic,
    )

    facts = set(goals)
    for action in actions:
        facts |= set(action.pos_preconditions)
        facts |= set(action.add_effects)
        for effect in action.probabilistic_effects:
            facts |= set(effect.fluents)
    return TemporalProbabilisticRPGHeuristic(
        actions, facts=facts, initial_facts=set(), goal_facts=set(goals)
    )


@pytest.mark.parametrize("depth", [3, 5, 8, 14, 25])
def test_strategy_never_exceeds_baseline_admissible(depth):
    """The clamp is min(marginal, pattern), so it can only tighten."""
    heuristic = _heuristic_for(_escape_house_actions(), ["door"])
    admissible = heuristic.heuristic_score(
        {}, ["door"], aggregation="product", fixed_depth=depth,
        strategy="baseline_admissible",
    )
    survivor = heuristic.heuristic_score(
        {}, ["door"], aggregation="product", fixed_depth=depth,
        strategy="baseline_admissible_survivor_pdb",
    )
    assert float(survivor) <= float(admissible) + 1e-9


def test_strategy_beats_saturation_on_the_escape_house():
    """The marginal bound pins to 1.0; the pattern keeps discriminating."""
    heuristic = _heuristic_for(_escape_house_actions(), ["door"])
    values = {}
    for depth in (4, 6, 10):
        result = heuristic.heuristic_propagate(
            {}, goal_facts=["door"], fixed_depth=depth,
            strategy="baseline_admissible_survivor_pdb",
        )
        values[depth] = result.probabilities_by_layer[depth]["door"]
        admissible = heuristic.heuristic_propagate(
            {}, goal_facts=["door"], fixed_depth=depth, strategy="baseline_admissible",
        )
        assert admissible.probabilities_by_layer[depth]["door"] == pytest.approx(1.0)
    assert values[4] < values[6] < values[10] < 1.0


def test_pure_strategy_matches_the_rpg_wrapped_one():
    """The RPG sweep is redundant once every goal fact carries a pattern."""
    heuristic = _heuristic_for(_escape_house_actions(), ["door"])
    for depth in (3, 5, 8, 14):
        wrapped = heuristic.heuristic_score(
            {}, ["door"], aggregation="product", fixed_depth=depth,
            strategy="baseline_admissible_survivor_pdb",
        )
        pure = heuristic.heuristic_score(
            {}, ["door"], aggregation="product", fixed_depth=depth,
            strategy="survivor_pdb_pure",
        )
        assert float(pure) == pytest.approx(float(wrapped), abs=1e-9)


def test_earliest_times_are_a_reachability_fixpoint():
    from comdp_plus_no_deadline.engines.survivor_pdb import compute_earliest_times

    specs = [
        ("make_a", frozenset(), 2, ("a",)),
        ("make_b", frozenset({"a"}), 3, ("b",)),
        ("unreachable", frozenset({"never"}), 1, ("c",)),
    ]
    earliest = compute_earliest_times({"start"}, specs, horizon=10)
    assert earliest["start"] == 0
    assert earliest["a"] == 2
    assert earliest["b"] == 5  # 2 + 3
    assert "c" not in earliest  # precondition never achievable


def test_earliest_times_respect_the_horizon():
    from comdp_plus_no_deadline.engines.survivor_pdb import compute_earliest_times

    specs = [("slow", frozenset(), 9, ("late",))]
    assert "late" not in compute_earliest_times(set(), specs, horizon=5)
    assert compute_earliest_times(set(), specs, horizon=9)["late"] == 9


def test_gates_recomputed_per_state_get_earlier_not_later():
    """A deeper state has more facts true, so its gates must not be LATER.

    Caching a root gate for a descendant would suppress achievers that really
    are available, which would push the bound below the true value.
    """
    from comdp_plus_no_deadline.engines.survivor_pdb import compute_earliest_times

    specs = [
        ("make_a", frozenset(), 4, ("a",)),
        ("use_a", frozenset({"a"}), 1, ("g",)),
    ]
    root = compute_earliest_times(set(), specs, horizon=20)
    deeper = compute_earliest_times({"a"}, specs, horizon=20)
    assert root["g"] == 5
    assert deeper["g"] == 1
    assert deeper["g"] <= root["g"]


def test_state_dependent_branch_is_not_silently_dropped():
    """Admissibility guard for probe-invisible achievers.

    An effect whose probability function only fires on a state neither probe
    reaches looks like "no achiever" to the joint extraction. The rest of the
    heuristic still credits it (declared-fluents safety net), so the pattern
    must top the fact up or it under-counts and the bound stops being an upper
    bound. This is the shape that once dropped machine_shop's free(machine)
    re-achiever.
    """
    invisible = SimpleNamespace(
        name="move",
        add_effects=set(),
        # Probes are the empty state and the full universe; this fires on
        # neither, so the extracted joint carries no add at all.
        probabilistic_effects=[_Effect({})],
    )
    patterns = build_patterns(
        ["free_m0"],
        actions_by_effect_fact={"free_m0": [invisible]},
        action_preconditions={"move": frozenset()},
        action_delays={"move": 1},
        # ... but the action model credits it, so the pattern has to as well.
        action_add_probabilities={"move": {"free_m0": 1.0}},
        marginals={"free_m0": [0.0, 1.0, 1.0]},
        initial_facts=set(),
        probe_universe=set(),
        horizon=2,
        max_facts=2,
    )
    assert len(patterns) == 1, "achiever was dropped entirely"
    curves = solve_survivor_pattern(patterns[0], set(), 2)
    assert curves["free_m0"][1] == pytest.approx(1.0)


def test_pattern_with_no_usable_achiever_is_dropped():
    patterns = build_patterns(
        ["unreachable"],
        actions_by_effect_fact={},
        action_preconditions={},
        action_delays={},
        action_add_probabilities={},
        marginals={"unreachable": [0.0, 0.0]},
        initial_facts=set(),
        probe_universe=set(),
        horizon=1,
    )
    assert patterns == []
