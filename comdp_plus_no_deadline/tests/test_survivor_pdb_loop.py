"""Tests for the phase-augmented ("time loop") survivor-PDB (survivor_pdb_loop).

The mechanism under test is one line of physics: an action runs ONE instance at
a time, so in a window of length ``W`` it can complete at most ``W / d(a)``
times, not ``W``. ``solve_survivor_pattern`` grants a fresh coin every layer;
``solve_survivor_loop_pattern`` grants one per completion.

Three anchors:

* a UNIT-DURATION pattern must return the old numbers to the last bit — the
  phase is vacuous there, so the module has to be a no-op;
* a single duration-3 achiever with ``p = 0.5`` has a closed-form curve
  (``1 - 0.5^k`` with ``k`` completions by ``t``), which the sweep must hit
  exactly, and which the un-phased sweep overshoots by a factor that grows with
  the horizon;
* a MISALIGNED gate (``gate`` not congruent to ``tau + d`` modulo ``d``) is
  where a residue-only phase would lose the tightening; the anchor
  ``A = max(tau + d, gate)`` has to be pinned exactly.
"""

import random

import pytest

from comdp_plus_no_deadline.engines.survivor_pdb import (
    SurvivorAchiever,
    SurvivorPattern,
    solve_survivor_pattern,
)
from comdp_plus_no_deadline.engines.survivor_pdb_loop import (
    PHASE_ANY,
    advance_phase,
    anchor_residue,
    attempts_in_step,
    build_interest_points,
    completions_in_step,
    decode_phase,
    encode_phase,
    solve_survivor_loop_pattern,
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
    """{key, lockpick, door} with both openers, ALL unit durations."""
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


def _slow_pattern(delay=3, gate=3, probability=0.5):
    """One achiever, no pattern preconditions: completions at gate, gate+d, ..."""
    return SurvivorPattern(
        facts=("goal",),
        seed_goal="goal",
        achievers=(
            _achiever(
                "slow",
                (),
                [(("goal",), probability), ((), 1.0 - probability)],
                gate=gate,
                delay=delay,
            ),
        ),
    )


# --------------------------------------------------------------------------
# 1. no-op on unit durations
# --------------------------------------------------------------------------


def test_unit_durations_reproduce_the_unphased_sweep_exactly():
    """A duration-1 achiever completes once per unit layer either way."""
    pattern = _escape_house_pattern()
    old = solve_survivor_pattern(pattern, set(), 12)
    new = solve_survivor_loop_pattern(pattern, set(), 12)
    assert set(old) == set(new)
    for fact in old:
        assert new[fact] == pytest.approx(old[fact], abs=1e-12)


def test_unit_duration_pattern_carries_no_counter():
    """No slot means no extra state dimension: the module has to be free here."""
    pattern = _escape_house_pattern()
    curves = solve_survivor_loop_pattern(pattern, set(), 6)
    # Known exact curve of the escape house (Monte Carlo verified in
    # test_survivor_pdb); reproduced here to pin the no-op claim to numbers.
    expected = {2: 0.5520, 3: 0.8275, 4: 0.9385, 5: 0.9791, 6: 0.9931}
    for layer, value in expected.items():
        assert curves["door"][layer] == pytest.approx(value, abs=5e-4)


# --------------------------------------------------------------------------
# 2. the closed-form duration-3 curve
# --------------------------------------------------------------------------


def test_duration_three_achiever_fires_once_every_three_layers():
    """p=0.5, d=3, gate=3 -> completions at 3, 6, 9 and nowhere else."""
    curves = solve_survivor_loop_pattern(_slow_pattern(), set(), 10)
    expected = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.875, 0.875]
    assert curves["goal"] == pytest.approx(expected, abs=1e-12)


def test_unphased_sweep_overshoots_and_the_gap_grows_with_the_horizon():
    """The bug this module exists for: H draws where only H/d can complete."""
    horizon = 12
    old = solve_survivor_pattern(_slow_pattern(), set(), horizon)["goal"]
    new = solve_survivor_loop_pattern(_slow_pattern(), set(), horizon)["goal"]
    assert old[3] == pytest.approx(new[3])  # first completion agrees
    assert old[6] - new[6] == pytest.approx(0.1875, abs=1e-9)
    assert old[12] - new[12] == pytest.approx(0.0615, abs=1e-3)
    # ... and the un-phased curve is the one that saturates.
    assert old[12] > 0.999
    assert new[12] < 0.95


def test_probability_compounds_as_one_minus_one_minus_p_to_the_n():
    """peff(a, Delta) = 1 - (1-p)^n, with n the COMPLETIONS, not the layers."""
    curves = solve_survivor_loop_pattern(
        _slow_pattern(delay=4, gate=4, probability=0.3), set(), 16
    )["goal"]
    for completions in range(1, 5):
        layer = 4 * completions
        assert curves[layer] == pytest.approx(1.0 - 0.7 ** completions, abs=1e-12)


# --------------------------------------------------------------------------
# 3. the anchor A = max(tau + d, gate)
# --------------------------------------------------------------------------


def test_anchor_follows_the_precondition_arrival():
    """mid lands at 2; the d=2 consumer then completes at 4, 6, 8, ..."""
    pattern = SurvivorPattern(
        facts=("mid", "goal"),
        seed_goal="goal",
        achievers=(
            _achiever("make_mid", (), [(("mid",), 1.0)], gate=2, delay=2),
            _achiever(
                "use_mid", ("mid",), [(("goal",), 0.5), ((), 0.5)], gate=4, delay=2
            ),
        ),
    )
    goal = solve_survivor_loop_pattern(pattern, set(), 10)["goal"]
    assert goal == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.75, 0.75, 0.875, 0.875, 0.9375], abs=1e-12
    )


def test_misaligned_gate_keeps_the_tightening():
    """tau = 3, d = 2, gate = 4 -> A = 5, so completions are 5, 7, 9, 11.

    A phase that only knew ``tau mod d`` would have to admit the gate-anchored
    progression 4, 6, 8, ... as well, and the union of the two is every layer —
    i.e. the whole tightening would evaporate exactly when the gate is odd
    against the arrival. Storing ``A mod d`` instead keeps it.
    """
    pattern = SurvivorPattern(
        facts=("mid", "goal"),
        seed_goal="goal",
        achievers=(
            _achiever("make_mid", (), [(("mid",), 1.0)], gate=3, delay=3),
            _achiever(
                "use_mid", ("mid",), [(("goal",), 0.5), ((), 0.5)], gate=4, delay=2
            ),
        ),
    )
    goal = solve_survivor_loop_pattern(pattern, set(), 12)["goal"]
    fires_at = [t for t in range(1, 13) if goal[t] > goal[t - 1] + 1e-12]
    assert fires_at == [5, 7, 9, 11]


def test_gate_can_delay_past_the_precondition_arrival():
    """A freed precondition that lands late pushes the whole progression."""
    pattern = SurvivorPattern(
        facts=("goal",),
        seed_goal="goal",
        achievers=(
            _achiever("late", (), [(("goal",), 0.5), ((), 0.5)], gate=8, delay=2),
        ),
    )
    goal = solve_survivor_loop_pattern(pattern, set(), 12)["goal"]
    assert goal[7] == pytest.approx(0.0)
    assert [t for t in range(1, 13) if goal[t] > goal[t - 1] + 1e-12] == [8, 10, 12]


def test_anchor_residue_matches_its_definition():
    for tau in range(8):
        for gate in range(1, 12):
            for delay in (2, 3, 5):
                assert anchor_residue(tau, gate, delay) == max(tau + delay, gate) % delay


# --------------------------------------------------------------------------
# 4. the counter encoding
# --------------------------------------------------------------------------


def test_phase_round_trips_and_elapsed_saturates():
    for delay in (2, 3, 5):
        for residue in range(delay):
            for elapsed in range(0, delay + 4):
                code = encode_phase(residue, elapsed, delay)
                back_residue, back_elapsed = decode_phase(code, delay)
                assert back_residue == residue
                assert back_elapsed == min(elapsed, delay)
                assert 0 <= code < delay * (delay + 1)


def test_advance_never_moves_the_anchor():
    for delay in (2, 3, 4):
        for residue in range(delay):
            phase = encode_phase(residue, 0, delay)
            for _ in range(10):
                phase = advance_phase(phase, 1, delay)
                assert decode_phase(phase, delay)[0] == residue
            assert decode_phase(phase, delay)[1] == delay  # saturated


def test_steady_state_count_is_the_v18_formula():
    """Once the first completion is past, completions_in_step IS floor((c+D)/d)."""
    for delay in (2, 3, 5):
        for residue in range(delay):
            saturated = encode_phase(residue, delay, delay)
            for previous in range(20, 30):
                for delta in (1, 2, 3, 7):
                    counted = completions_in_step(
                        saturated, delay, 0, previous, previous + delta
                    )
                    phase = (previous - residue) % delay
                    assert counted == attempts_in_step(phase, delta, delay)


def test_attempts_in_step_matches_the_doc():
    assert attempts_in_step(0, 1, 3) == 0
    assert attempts_in_step(2, 1, 3) == 1
    assert attempts_in_step(0, 3, 3) == 1
    assert attempts_in_step(0, 7, 3) == 2
    assert attempts_in_step(0, 5, 1) == 5  # unit duration: one per layer


# --------------------------------------------------------------------------
# 5. admissibility
# --------------------------------------------------------------------------


def _random_pattern(rng, facts=("a", "b", "c", "goal")):
    achievers = []
    for i in range(rng.randint(2, 5)):
        target = rng.choice(facts)
        pre = tuple(
            f for f in rng.sample(facts, rng.randint(0, 2)) if f != target
        )
        delay = rng.randint(1, 4)
        probability = round(rng.uniform(0.1, 0.9), 2)
        achievers.append(
            _achiever(
                f"a{i}",
                pre,
                [((target,), probability), ((), 1.0 - probability)],
                gate=rng.randint(delay, delay + 4),
                delay=delay,
            )
        )
    return SurvivorPattern(facts=facts, seed_goal="goal", achievers=tuple(achievers))


def test_never_exceeds_the_unphased_bound():
    """loop <= lazy pointwise: the completion set is a subset of every layer."""
    rng = random.Random(7)
    facts = ("a", "b", "c", "goal")
    horizon = 14
    tighter = 0
    for _ in range(150):
        pattern = _random_pattern(rng, facts)
        initial = set(rng.sample(facts, rng.randint(0, 2)))
        old = solve_survivor_pattern(pattern, initial, horizon)
        new = solve_survivor_loop_pattern(pattern, initial, horizon)
        for fact in facts:
            for layer in range(horizon + 1):
                assert new[fact][layer] <= old[fact][layer] + 1e-12
                if new[fact][layer] < old[fact][layer] - 1e-9:
                    tighter += 1
    assert tighter > 0, "the phase never bit — the test lost its teeth"


def test_curves_are_monotone_and_bounded():
    rng = random.Random(11)
    for _ in range(40):
        pattern = _random_pattern(rng)
        curves = solve_survivor_loop_pattern(pattern, set(), 12)
        for curve in curves.values():
            assert all(0.0 <= value <= 1.0 for value in curve)
            assert all(a <= b + 1e-12 for a, b in zip(curve, curve[1:]))


def test_collapse_fallback_degrades_to_the_unphased_sweep():
    """Forgetting the counters may only RAISE the value, never lower it."""
    pattern = _slow_pattern()
    exact = solve_survivor_loop_pattern(pattern, set(), 12, max_states=4096)
    collapsed = solve_survivor_loop_pattern(pattern, set(), 12, max_states=0)
    unphased = solve_survivor_pattern(pattern, set(), 12)
    for layer in range(13):
        assert collapsed["goal"][layer] >= exact["goal"][layer] - 1e-12
        assert collapsed["goal"][layer] == pytest.approx(
            unphased["goal"][layer], abs=1e-12
        )


def test_phase_any_is_the_forget_marker():
    assert PHASE_ANY < 0
    assert advance_phase(PHASE_ANY, 3, 4) == PHASE_ANY


def test_concurrency_only_ever_adds_completions():
    """The knob for domains where one grounded action may overlap itself."""
    pattern = _slow_pattern()
    serial = solve_survivor_loop_pattern(pattern, set(), 12)["goal"]
    overlapped = solve_survivor_loop_pattern(pattern, set(), 12, concurrency=2)["goal"]
    assert all(b >= a - 1e-12 for a, b in zip(serial, overlapped))
    # Two instances completing together at layer 3: 1 - 0.5^2.
    assert overlapped[3] == pytest.approx(0.75)


# --------------------------------------------------------------------------
# 6. interest points (v18 section 1)
# --------------------------------------------------------------------------


def test_interest_points_close_durations_and_seeds():
    assert build_interest_points([2, 3], 10) == [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert build_interest_points([4], 12) == [0, 4, 8, 12]
    # A gate that is not a sum of durations has to be seeded in, or a
    # completion anchored on it would fall between two points.
    assert 5 in build_interest_points([4], 12, seeds=[5])


def test_interest_point_sweep_equals_the_per_layer_sweep():
    """Restricting to T is exact, not an approximation, once gates are seeded."""
    rng = random.Random(7)
    facts = ("a", "b", "c", "goal")
    horizon = 14
    for _ in range(150):
        pattern = _random_pattern(rng, facts)
        initial = set(rng.sample(facts, rng.randint(0, 2)))
        dense = solve_survivor_loop_pattern(pattern, initial, horizon)
        sparse = solve_survivor_loop_pattern(
            pattern, initial, horizon, timestamps="interest"
        )
        for fact in facts:
            assert sparse[fact] == pytest.approx(dense[fact], abs=1e-12)


def test_unknown_timestamps_mode_is_rejected():
    with pytest.raises(ValueError):
        solve_survivor_loop_pattern(_slow_pattern(), set(), 6, timestamps="nope")


# --------------------------------------------------------------------------
# 7. wiring
# --------------------------------------------------------------------------


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


def _durative_escape_house():
    """The escape house with SLOW searches — where the phase actually bites."""
    return [
        _SyntheticAction(
            "find_key",
            (),
            duration=3,
            probabilistic=[_SyntheticProbabilisticEffect({0.3: {"key": True}})],
        ),
        _SyntheticAction(
            "find_lockpick",
            (),
            duration=2,
            probabilistic=[_SyntheticProbabilisticEffect({0.6: {"lockpick": True}})],
        ),
        _SyntheticAction("open_with_key", ("key",), duration=2, adds=("door",)),
        _SyntheticAction(
            "open_with_lockpick",
            ("lockpick",),
            duration=2,
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


def test_strategy_name_is_accepted():
    from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
        TemporalProbabilisticRPGHeuristic,
    )

    assert (
        TemporalProbabilisticRPGHeuristic._normalize_strategy("survivor_pdb_loop")
        == "survivor_pdb_loop"
    )


def test_alias_and_cli_wiring_resolve():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from experiment_common import HEURISTIC_ALIASES

    for alias in ("survivor_pdb_loop", "survivor_pdb_rpg_loop", "survivor_loop"):
        assert (
            HEURISTIC_ALIASES[alias]["temporal_heuristic_strategy"]
            == "survivor_pdb_loop"
        )

    from unified_planning.parser import _parse_temporal_heuristic_strategy

    assert _parse_temporal_heuristic_strategy("36") == "survivor_pdb_loop"
    assert (
        _parse_temporal_heuristic_strategy("survivor_pdb_loop") == "survivor_pdb_loop"
    )


@pytest.mark.parametrize("depth", [4, 6, 10, 16, 25])
def test_end_to_end_never_exceeds_the_lazy_strategy(depth):
    """Same patterns, same gates — only the sweep changes, and only downward."""
    lazy = _heuristic_for(_durative_escape_house(), ["door"]).heuristic_score(
        {}, ["door"], aggregation="product", fixed_depth=depth,
        strategy="survivor_pdb_lazy",
    )
    loop = _heuristic_for(_durative_escape_house(), ["door"]).heuristic_score(
        {}, ["door"], aggregation="product", fixed_depth=depth,
        strategy="survivor_pdb_loop",
    )
    assert float(loop) <= float(lazy) + 1e-9


def test_end_to_end_is_strictly_tighter_on_a_durative_domain():
    heuristic = _heuristic_for(_durative_escape_house(), ["door"])
    lazy = heuristic.heuristic_propagate(
        {}, goal_facts=["door"], fixed_depth=20, strategy="survivor_pdb_lazy"
    ).probabilities_by_layer[20]["door"]
    loop = _heuristic_for(_durative_escape_house(), ["door"]).heuristic_propagate(
        {}, goal_facts=["door"], fixed_depth=20, strategy="survivor_pdb_loop"
    ).probabilities_by_layer[20]["door"]
    assert loop < lazy - 1e-6


def test_end_to_end_keeps_discriminating_across_depths():
    """The point of the family: deeper layers must stay comparable, not pin."""
    values = {}
    for depth in (8, 12, 20):
        heuristic = _heuristic_for(_durative_escape_house(), ["door"])
        values[depth] = heuristic.heuristic_propagate(
            {}, goal_facts=["door"], fixed_depth=depth, strategy="survivor_pdb_loop"
        ).probabilities_by_layer[depth]["door"]
    assert values[8] < values[12] < values[20] < 1.0


def test_loop_and_lazy_do_not_share_a_pattern_cache():
    """Cross-strategy isolation. The solve memos are keyed on ``id(pattern)``.

    If both strategies shared one pattern cache, the first to grow its horizon
    would clear it, freeing pattern objects whose ids the OTHER strategy's memo
    still holds — and CPython reuses ids, so a stale entry could serve one
    pattern's curves for another. The notebook's lazy-vs-loop A/B is exactly
    that configuration.
    """
    heuristic = _heuristic_for(_durative_escape_house(), ["door"])
    for depth in (6, 12, 20):
        for strategy in ("survivor_pdb_lazy", "survivor_pdb_loop"):
            heuristic.heuristic_score(
                {}, ["door"], aggregation="product", fixed_depth=depth,
                strategy=strategy,
            )
    assert (
        heuristic._survivor_pdb_loop_pattern_cache
        is not heuristic._survivor_pdb_lazy_pattern_cache
    )
    assert heuristic._survivor_pdb_loop_pattern_cache
    assert heuristic._survivor_pdb_lazy_pattern_cache


def test_interleaving_the_two_strategies_matches_running_them_alone():
    """Values must not depend on whether the other strategy shares the instance."""
    shared = _heuristic_for(_durative_escape_house(), ["door"])
    for depth in (6, 12, 20):
        for strategy in ("survivor_pdb_lazy", "survivor_pdb_loop"):
            shared.heuristic_score(
                {}, ["door"], aggregation="product", fixed_depth=depth,
                strategy=strategy,
            )
    for depth in (6, 12, 20):
        for strategy in ("survivor_pdb_lazy", "survivor_pdb_loop"):
            alone = _heuristic_for(_durative_escape_house(), ["door"])
            assert float(
                shared.heuristic_score(
                    {}, ["door"], aggregation="product", fixed_depth=depth,
                    strategy=strategy,
                )
            ) == pytest.approx(
                float(
                    alone.heuristic_score(
                        {}, ["door"], aggregation="product", fixed_depth=depth,
                        strategy=strategy,
                    )
                ),
                abs=1e-12,
            )


# --------------------------------------------------------------------------
# 8. direct-value variants: NOT the PTRPG
# --------------------------------------------------------------------------


def _two_goal_house():
    """Two goals behind ONE key — maximally correlated, so product is unsound."""
    return [
        _SyntheticAction(
            "find_key",
            (),
            duration=3,
            probabilistic=[_SyntheticProbabilisticEffect({0.3: {"key": True}})],
        ),
        _SyntheticAction("open_door", ("key",), duration=2, adds=("door",)),
        _SyntheticAction("open_window", ("key",), duration=2, adds=("window",)),
    ]


def test_direct_strategy_names_are_accepted():
    from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
        DIRECT_VALUE_BASE_STRATEGY,
        TemporalProbabilisticRPGHeuristic,
    )

    for name, base in DIRECT_VALUE_BASE_STRATEGY.items():
        assert TemporalProbabilisticRPGHeuristic._normalize_strategy(name) == name
        assert TemporalProbabilisticRPGHeuristic._normalize_strategy(base) == base


def test_direct_propagation_is_identical_to_its_base():
    """Only the goal aggregation differs — the DP itself must be untouched."""
    for strategy, base in (
        ("survivor_pdb_loop_direct", "survivor_pdb_loop"),
        ("survivor_pdb_lazy_direct", "survivor_pdb_lazy"),
    ):
        a = _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_propagate(
            {}, goal_facts=["door", "window"], fixed_depth=14, strategy=strategy
        )
        b = _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_propagate(
            {}, goal_facts=["door", "window"], fixed_depth=14, strategy=base
        )
        assert a.probabilities_by_layer == b.probabilities_by_layer


def test_direct_score_is_the_min_and_the_product_falls_below_it():
    """min(U(A), U(B)) >= min(P(A), P(B)) >= P(A and B); a product need not."""
    heuristic = _heuristic_for(_two_goal_house(), ["door", "window"])
    layers = heuristic.heuristic_propagate(
        {}, goal_facts=["door", "window"], fixed_depth=14,
        strategy="survivor_pdb_loop",
    ).probabilities_by_layer[14]
    per_goal = [layers["door"], layers["window"]]

    direct = float(
        _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_score(
            {}, ["door", "window"], aggregation="product", fixed_depth=14,
            strategy="survivor_pdb_loop_direct",
        )
    )
    product = float(
        _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_score(
            {}, ["door", "window"], aggregation="product", fixed_depth=14,
            strategy="survivor_pdb_loop",
        )
    )
    assert direct == pytest.approx(min(per_goal), abs=1e-12)
    assert product == pytest.approx(per_goal[0] * per_goal[1], abs=1e-12)
    # Both goals need the same key, so the product is far below the conjunction.
    assert product < direct - 1e-6


def test_direct_ignores_a_requested_product_aggregation():
    """The product is not offered for these — it is not an upper bound."""
    for aggregation in ("product", "min", "area"):
        value = float(
            _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_score(
                {}, ["door", "window"], aggregation=aggregation, fixed_depth=14,
                strategy="survivor_pdb_loop_direct",
            )
        )
        assert value == pytest.approx(
            float(
                _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_score(
                    {}, ["door", "window"], aggregation="min", fixed_depth=14,
                    strategy="survivor_pdb_loop",
                )
            ),
            abs=1e-12,
        )


def _load_aggregation_for_strategy():
    """Import the MCTS helper without tripping the package's circular import.

    ``unified_planning.shortcuts`` has to be fully imported first, and
    ``unified_planning.parser`` parses ``sys.argv`` at import time — both are
    the same dance the scripts in ``scripts/`` do.
    """
    import sys

    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        import unified_planning.shortcuts  # noqa: F401
        from unified_planning.engines.solvers.mcts import (
            _aggregation_for_strategy as helper,
        )
    finally:
        sys.argv = saved
    return helper


def test_env_aggregation_override_cannot_reintroduce_the_product(monkeypatch):
    _aggregation_for_strategy = _load_aggregation_for_strategy()

    monkeypatch.setenv("TP_MCTS_HEURISTIC_AGGREGATION", "product")
    assert _aggregation_for_strategy("survivor_pdb_loop_direct") == "min"
    assert _aggregation_for_strategy("survivor_pdb_loop") == "product"
    monkeypatch.delenv("TP_MCTS_HEURISTIC_AGGREGATION")
    assert _aggregation_for_strategy("survivor_pdb_loop_direct") == "min"

    value = float(
        _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_score(
            {}, ["door", "window"], aggregation="product", fixed_depth=14,
            strategy="survivor_pdb_loop_direct",
        )
    )
    layers = _heuristic_for(_two_goal_house(), ["door", "window"]).heuristic_propagate(
        {}, goal_facts=["door", "window"], fixed_depth=14,
        strategy="survivor_pdb_loop",
    ).probabilities_by_layer[14]
    assert value == pytest.approx(min(layers["door"], layers["window"]), abs=1e-12)


def test_direct_aliases_and_cli_wiring_resolve():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from experiment_common import HEURISTIC_ALIASES
    from unified_planning.parser import _parse_temporal_heuristic_strategy

    for alias, code in (
        ("survivor_pdb_loop_direct", "37"),
        ("survivor_pdb_lazy_direct", "38"),
        ("survivor_pdb_pure_direct", "39"),
    ):
        assert HEURISTIC_ALIASES[alias]["temporal_heuristic_strategy"] == alias
        assert _parse_temporal_heuristic_strategy(code) == alias
        assert _parse_temporal_heuristic_strategy(alias) == alias
