"""Phase-augmented ("time loop") survivor-PDB sweep.

Strategy ``survivor_pdb_loop``. Same pattern abstraction as
:mod:`comdp_plus_no_deadline.engines.survivor_pdb` -- same patterns, same
achievers, same gates, same joint distribution over the pattern's
configurations -- with ONE mechanism added: every achiever carries a PHASE, so
the sweep counts how many instances of it can actually complete inside a step
instead of granting it a fresh coin at every layer.

The bug it removes
------------------
``solve_survivor_pattern`` fires every applicable achiever at every layer. An
action of duration ``d`` therefore gets ``d`` independent draws in the ``d``
layers its single execution occupies. Over a horizon ``H`` it draws ``H`` times
where at most ``H / d`` executions can physically finish, so the estimate is
inflated by a factor that GROWS with the deadline -- precisely the saturation
that makes different MCTS layers look alike.

The fix (v18 section 3): one instance at a time
-----------------------------------------------
An action starts once its preconditions hold, occupies ``d(a)``, and only then
may start again. Let ``c(a)`` be the phase -- time since the last precondition
of ``a`` arrived, taken modulo ``d(a)``. The attempts that COMPLETE inside a
step of width ``Delta`` are then

    n(a, Delta)  =  floor( ( c(a) + Delta ) / d(a) )

and, the draws being independent,

    peff(a, Delta)  =  1 - ( 1 - p(a) ) ** n(a, Delta)

``n = 0`` removes the achiever from the step entirely; ``n = 1`` leaves ``p(a)``
unchanged. With unit steps (the default here) ``n`` is an indicator that is 1
once every ``d`` layers, so a duration-5 action draws 5x fewer coins than
before. Nothing else changes, so the value can only go DOWN: this is a pure
tightening of ``survivor_pdb_lazy``.

Rather than a per-fact ``p``, the achiever carries a JOINT distribution over the
add-sets one execution produces (exclusive branches must stay exclusive), so
``n`` attempts are modelled by folding that distribution ``n`` times -- which is
``1 - (1-p)^n`` per fact, without assuming the branches are independent coins.

The anchor
----------
An achiever's FIRST completion is at

    A(a)  =  max( tau + d(a),  gate(a) )

-- its pattern preconditions must have held for ``d`` layers, AND it cannot beat
the delete-relaxed gate of the preconditions the pattern freed (``gate`` is the
earliest layer at which its effect can land at all). Completions then repeat
every ``d``, so the completion set is exactly

    { A, A + d, A + 2d, ... }

and ``c(a) = (t - A) mod d`` is the phase the formula above wants. ``A`` is
known the instant the precondition set is satisfied, because ``tau`` IS that
layer, so the state stores ``A`` in the only two pieces the fire test reads:
its residue ``A mod d``, and enough of the elapsed time to decide ``t >= A``.

State
-----
``(mask, phases)``. ``mask`` is the pattern configuration, exactly as before.
``phases`` carries one counter per distinct ``(pattern preconditions, duration,
gate)`` triple -- v18 section 6: actions sharing a precondition set share its
arrival time, so one counter serves all of them and each ``c(a)`` is read off it
modulo ``d(a)``. Achievers of unit duration carry no counter at all (they
complete once per unit layer either way), so a unit-duration domain pays nothing
for this module and returns exactly the un-phased numbers.

Encoding of a counter:

    ``PHASE_UNSATISFIED`` (-1)  the precondition set is not yet satisfied
    ``PHASE_ANY``         (-2)  collapsed (see ``_collapse_phases``): fire freely
    ``r * (d + 1) + e``         ``r = A mod d`` (fixed once the set is satisfied)
                                and ``e = min(t - tau, d)``, an elapsed counter
                                that SATURATES at ``d``. ``e`` is only ever read
                                through ``t >= tau + d``, so saturating it loses
                                nothing, and it bounds the counter by
                                ``d * (d + 1)`` instead of by the deadline.

Admissibility
-------------
Every relaxation of ``survivor_pdb`` is inherited (freed preconditions gated by
relaxed reachability, delete effects ignored, distinct achievers independent),
and the phase only ever REMOVES completions that no single-instance execution
could have made, never adds one: the completion set above is exact given
``tau`` and ``gate``, and both are treated optimistically already (``gate`` is a
delete-relaxed lower bound on the freed preconditions' arrival, ``tau`` is the
pattern's own optimistic arrival). The bounded-state fallback sets counters to
``PHASE_ANY``, which fires whenever the gate allows -- i.e. degrades exactly to
``survivor_pdb``'s behaviour, itself already an upper bound.

So ``loop <= lazy`` pointwise (the completion set is a subset of "every layer
past the gate"), and the value remains an ADMISSIBLE upper bound under the same
delete-relaxed temporal envelope as the rest of the family.

Known gap (v18 section 9): an action permitted to run two instances
CONCURRENTLY is not covered by ``n(a, Delta)``, which counts serial completions.
Grounded actions do not self-overlap in this codebase's semantics (two objects
give two distinct grounded achievers, each with its own counter), so the default
``concurrency=1`` is right; the knob exists for domains that relax it.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from comdp_plus_no_deadline.engines.survivor_pdb import (
    SurvivorPattern,
    _clamp01,
)

Fact = Hashable

PHASE_UNSATISFIED = -1
PHASE_ANY = -2

DEFAULT_MAX_STATES = 4096


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def attempts_in_step(phase: int, delta: int, delay: int) -> int:
    """v18 section 3: ``n(a, Delta) = floor((c(a) + Delta) / d(a))``.

    ``phase`` is ``c(a)``, the time since the achiever's last completion, taken
    modulo ``d(a)``. This is the steady-state count -- every completion of the
    progression is in the past, so only the phase and the step width matter.
    ``completions_in_step`` is the same quantity with the FIRST completion's
    lower bound applied, and reduces to this once ``e`` saturates.
    """
    if delay <= 1:
        return max(0, int(delta))
    return ((phase % delay) + delta) // delay


def encode_phase(residue: int, elapsed: int, delay: int) -> int:
    """Pack ``(A mod d, min(t - tau, d))`` into one counter value."""
    return (residue % delay) * (delay + 1) + min(max(0, elapsed), delay)


def decode_phase(phase: int, delay: int) -> Tuple[int, int]:
    """Unpack a counter into ``(A mod d, min(t - tau, d))``."""
    return divmod(phase, delay + 1)


def anchor_residue(tau: int, gate: int, delay: int) -> int:
    """``A mod d`` for ``A = max(tau + d, gate)``, fixed once ``tau`` is known."""
    return max(tau + delay, gate) % delay


def completions_in_step(
    phase: int,
    delay: int,
    gate: int,
    previous_time: int,
    current_time: int,
) -> int:
    """How many executions of the achiever COMPLETE in ``(prev, current]``.

    The completion set is ``{A, A+d, A+2d, ...}`` with ``A = max(tau+d, gate)``.
    ``phase`` supplies ``A mod d`` and a saturating ``e = min(prev - tau, d)``,
    which together pin the set down: a completion is any ``x`` congruent to
    ``A`` modulo ``d`` that is at least ``A``, and ``x >= A`` unpacks into
    ``x >= gate`` and ``x >= tau + d`` -- the latter being ``x >= prev + d - e``
    while ``e`` is exact, and vacuous once it saturates.
    """
    residue, elapsed = decode_phase(phase, delay)
    low = max(gate, previous_time + 1, previous_time + delay - elapsed)
    if current_time < low:
        return 0
    first = low + ((residue - low) % delay)
    if first > current_time:
        return 0
    return (current_time - first) // delay + 1


def advance_phase(phase: int, delta: int, delay: int) -> int:
    """Move a counter forward by ``delta`` layers.

    The anchor residue never moves; only the elapsed counter does, and it
    saturates at ``d`` because that is the largest value the ``t >= tau + d``
    test can distinguish.
    """
    if phase < 0 or delay <= 1:
        return phase
    residue, elapsed = decode_phase(phase, delay)
    return encode_phase(residue, elapsed + delta, delay)


def build_interest_points(
    delays: Sequence[int], horizon: int, seeds: Sequence[int] = ()
) -> List[int]:
    """v18 section 1: the timestamps at which an effect can land, ``T``.

    ``T`` is ``{0} union seeds`` closed under ``t -> t + d(a)`` for every action
    duration, capped at the horizon. Every completion time of the pattern chain
    lies in ``T`` provided the GATES are seeded: a completion sits at
    ``max(tau + d, gate) + k*d``, ``tau`` is itself a completion (so already in
    the closure of ``{0}``), but a gate is an earliest-arrival read off the
    freed part of the problem and need not be a sum of the pattern's own
    durations. With the gates seeded, restricting the sweep to ``T`` and holding
    the curve constant in between is EXACT within the pattern's model, not an
    approximation -- nothing can land strictly between two points of ``T``.
    """
    horizon = max(0, int(horizon))
    positive = sorted({max(1, int(d)) for d in delays})
    if not positive:
        return list(range(horizon + 1))
    reachable = [False] * (horizon + 1)
    reachable[0] = True
    for seed in seeds:
        seed = int(seed)
        if 0 <= seed <= horizon:
            reachable[seed] = True
    for time in range(horizon + 1):
        if not reachable[time]:
            continue
        for delay in positive:
            landing = time + delay
            if landing <= horizon:
                reachable[landing] = True
    points = [time for time in range(horizon + 1) if reachable[time]]
    if points[-1] != horizon:
        points.append(horizon)
    return points


def solve_survivor_loop_pattern(
    pattern: SurvivorPattern,
    initial_facts: Set[Fact],
    horizon: int,
    max_states: int = DEFAULT_MAX_STATES,
    gates: Optional[Mapping[str, Optional[int]]] = None,
    *,
    timestamps: Optional[Union[str, Sequence[int]]] = None,
    concurrency: int = 1,
) -> Dict[Fact, List[float]]:
    """Forward sweep of the joint distribution, with per-achiever phases.

    Drop-in replacement for ``survivor_pdb.solve_survivor_pattern``: same
    arguments, same ``{fact: curve[0..horizon]}`` result.

    ``timestamps`` selects the layers actually evaluated. ``None`` (the default)
    is every integer layer, i.e. all ``Delta = 1``. The string ``"interest"``
    builds ``T`` from this pattern's own durations and RESOLVED gates (v18
    section 1), which is exact and cheaper; an explicit sequence is used as
    given, and is only exact if it contains every completion time.

    ``concurrency`` is how many instances of one grounded action may overlap in
    time; it multiplies the completion count. The default 1 is this codebase's
    semantics and is what makes the result dominated by
    ``solve_survivor_pattern``; raising it deliberately admits more completions
    than the un-phased sweep allowed.
    """
    facts = pattern.facts
    index = {fact: position for position, fact in enumerate(facts)}
    horizon = max(0, int(horizon))
    concurrency = max(1, int(concurrency))

    initial_mask = 0
    for fact, position in index.items():
        if fact in initial_facts:
            initial_mask |= 1 << position

    # ---- counter slots ------------------------------------------------------
    # One per distinct (pattern preconditions, duration, gate): the first two
    # fix tau and the period, the gate fixes the anchor. Unit-duration achievers
    # complete once per unit layer whatever the phase, so they never need a
    # slot; a unit-duration pattern pays nothing.
    slot_of: Dict[Tuple[frozenset, int, int], int] = {}
    slot_masks: List[int] = []
    slot_delays: List[int] = []
    slot_gates: List[int] = []
    prepared: List[Tuple[int, int, int, int, Tuple[Tuple[int, float], ...]]] = []
    for achiever in pattern.achievers:
        gate = achiever.gate if gates is None else gates.get(achiever.name, achiever.gate)
        if gate is None:
            continue
        gate = int(gate)
        delay = max(1, int(achiever.delay))
        pre_mask = 0
        for fact in achiever.pattern_pre:
            pre_mask |= 1 << index[fact]
        slot = -1
        if delay > 1:
            key = (achiever.pattern_pre, delay, gate)
            slot = slot_of.get(key, -1)
            if slot < 0:
                slot = len(slot_masks)
                slot_of[key] = slot
                slot_masks.append(pre_mask)
                slot_delays.append(delay)
                slot_gates.append(gate)
        outcome_masks: List[Tuple[int, float]] = []
        for add, probability in achiever.outcomes:
            mask = 0
            for fact in add:
                mask |= 1 << index[fact]
            outcome_masks.append((mask, probability))
        prepared.append((gate, pre_mask, delay, slot, tuple(outcome_masks)))

    # A precondition set already true in the initial state arrived at tau = 0
    # (v18 section 5: "c(a) = t mod d(a) when pre(a) is empty").
    initial_phases = tuple(
        encode_phase(anchor_residue(0, slot_gates[slot], slot_delays[slot]), 0, slot_delays[slot])
        if initial_mask & mask == mask
        else PHASE_UNSATISFIED
        for slot, mask in enumerate(slot_masks)
    )

    curves: Dict[Fact, List[float]] = {fact: [0.0] * (horizon + 1) for fact in facts}
    for fact, position in index.items():
        curves[fact][0] = 1.0 if initial_mask >> position & 1 else 0.0

    if timestamps is None:
        layers = list(range(horizon + 1))
    else:
        if isinstance(timestamps, str):
            if timestamps != "interest":
                raise ValueError(f"Unknown timestamps mode: {timestamps!r}")
            # Seeded with the resolved gates: a completion sits at
            # max(tau+d, gate) + k*d, and a gate need not be a sum of this
            # pattern's durations.
            timestamps = build_interest_points(
                [delay for _g, _p, delay, _s, _o in prepared],
                horizon,
                seeds=[gate for gate, _p, _d, _s, _o in prepared],
            )
        layers = sorted(
            {0, *(int(t) for t in timestamps if 0 <= int(t) <= horizon), horizon}
        )

    distribution: Dict[Tuple[int, Tuple[int, ...]], float] = {
        (initial_mask, initial_phases): 1.0
    }
    previous_time = layers[0]
    for current_time in layers[1:]:
        delta = current_time - previous_time
        if delta <= 0:
            continue
        advanced: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        for (state, phases), mass in distribution.items():
            current: Dict[int, float] = {state: mass}
            for gate, pre_mask, delay, slot, outcome_masks in prepared:
                if current_time < gate:
                    continue
                if state & pre_mask != pre_mask:
                    continue
                phase = PHASE_ANY if slot < 0 else phases[slot]
                if phase == PHASE_UNSATISFIED:
                    continue
                if delay <= 1:
                    # One completion per unit layer, but only from the gate on:
                    # with a wide step the gate can fall inside it.
                    attempts = current_time - max(previous_time, gate - 1)
                elif phase == PHASE_ANY:
                    # Collapsed counter: fall back to the un-phased behaviour,
                    # the most completions a period-d process can fit in the
                    # step. Optimistic, so still an upper bound.
                    attempts = _ceil_div(delta, delay)
                else:
                    attempts = completions_in_step(
                        phase, delay, gate, previous_time, current_time
                    )
                if attempts <= 0:
                    continue
                attempts *= concurrency
                # n attempts = the outcome distribution folded n times. For a
                # single-fact achiever this is exactly peff = 1 - (1-p)^n; for a
                # branching one it keeps the branches exclusive within each
                # attempt.
                for _ in range(attempts):
                    folded: Dict[int, float] = {}
                    for partial_state, partial_mass in current.items():
                        for outcome_mask, probability in outcome_masks:
                            if probability <= 0.0:
                                continue
                            successor = partial_state | outcome_mask
                            folded[successor] = (
                                folded.get(successor, 0.0) + partial_mass * probability
                            )
                    if not folded:
                        break
                    current = folded
            for successor, successor_mass in current.items():
                successor_phases = tuple(
                    _successor_phase(
                        phases[slot],
                        successor,
                        mask,
                        delta,
                        slot_delays[slot],
                        slot_gates[slot],
                        current_time,
                    )
                    for slot, mask in enumerate(slot_masks)
                )
                key = (successor, successor_phases)
                advanced[key] = advanced.get(key, 0.0) + successor_mass
        if len(advanced) > max_states:
            advanced = _collapse_phases(advanced)
        distribution = advanced
        for fact, position in index.items():
            bit = 1 << position
            value = _clamp01(
                sum(mass for (state, _phases), mass in distribution.items() if state & bit)
            )
            curve = curves[fact]
            # Layers skipped by ``timestamps`` hold the value of the PRECEDING
            # interest point: nothing lands strictly between two of them, so the
            # curve is genuinely flat there. Filling them from ``value`` instead
            # would credit this step's completions to layers before it.
            held = curve[previous_time]
            for layer in range(previous_time + 1, current_time):
                curve[layer] = held
            curve[current_time] = value
        previous_time = current_time

    for fact in facts:
        curve = curves[fact]
        for layer in range(1, horizon + 1):
            if curve[layer] < curve[layer - 1]:
                curve[layer] = curve[layer - 1]
    return curves


def _successor_phase(
    phase: int,
    successor: int,
    pre_mask: int,
    delta: int,
    delay: int,
    gate: int,
    tau: int,
) -> int:
    """The counter carried into the next layer (v18 section 5).

    An unsatisfied set that this step completes restarts its run from zero
    (``c'(a) = 0 if pre(a) intersects s' - s``) -- and ``tau`` is exactly the
    layer that happens on, which is what lets the anchor ``max(tau + d, gate)``
    be pinned down here and never recomputed. An already-satisfied set just
    advances. Facts persist, so a satisfied set never becomes unsatisfied.
    """
    if successor & pre_mask != pre_mask:
        return PHASE_UNSATISFIED
    if phase == PHASE_ANY:
        return PHASE_ANY
    if phase == PHASE_UNSATISFIED:
        return encode_phase(anchor_residue(tau, gate, delay), 0, delay)
    return advance_phase(phase, delta, delay)


def _collapse_phases(
    distribution: Mapping[Tuple[int, Tuple[int, ...]], float],
) -> Dict[Tuple[int, Tuple[int, ...]], float]:
    """Merge phase-augmented states sharing a mask, forgetting every counter.

    The bounded fallback for patterns whose phase dimension blows up. A
    forgotten counter fires whenever its gate allows -- exactly what
    ``survivor_pdb`` did before this module -- so the collapsed chain dominates
    the phased one and the bound stays admissible; it just gives up the
    one-instance-at-a-time tightening it was tracking.
    """
    width = 0
    for _state, phases in distribution:
        width = len(phases)
        break
    forgotten = (PHASE_ANY,) * width
    merged: Dict[Tuple[int, Tuple[int, ...]], float] = {}
    for (state, _phases), mass in distribution.items():
        key = (state, forgotten)
        merged[key] = merged.get(key, 0.0) + mass
    return merged
