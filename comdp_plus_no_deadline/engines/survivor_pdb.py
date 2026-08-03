"""Survivor-DP pattern database for the temporal probabilistic RPG.

Strategy ``baseline_admissible_survivor_pdb``. This is NOT the older
``pdb_correction`` module (that one replaces an action's precondition support
``R_t(a)`` by a backward max-DP over a projected pattern). This one attacks the
TEMPORAL direction instead: it replaces the per-fact cumulative recursion by an
exact joint distribution over a small pattern of facts.

Why a joint distribution at all
-------------------------------
The admissible cumulative update ``P_t = min(1, P_{t-1} + H_t)`` is loose for a
structural reason: the exact recursion is

    P_t(f) = P_{t-1}(f) + (1 - P_{t-1}(f)) * Pr[f achieved at t | NOT by t-1]

and the conditional in the second term is *not* a function of the stored
marginals ``P_.(f)`` — two processes with identical marginal curves can have
different conditionals. The marginal table forgets WHY ``f`` is still false
(preconditions never arrived? arrived but the effect coin kept failing?), and
that is exactly what the conditional needs. ``min(1, sum)`` is the tightest
bound derivable from marginals alone, which is why it saturates to 1.0 early
and stops discriminating between tree layers.

A joint distribution over a small set of facts does not forget. Tracking
``pi_t(s) = Pr[pattern facts are exactly configuration s at time t]`` makes the
conditional a ratio of stored numbers,

    h_t = (mass flowing into f-states at t) / (mass in non-f states at t-1)

so the recursion becomes exact *inside the pattern* with no independence
assumption at all: outcome-exclusivity of one action's branches, and shared
preconditions between achievers, are both carried by the joint. In particular
the two failure modes that break marginal-level operators are handled natively:

- noisy-OR breaks when a FAILED achiever ENABLES another (e.g. ``a1`` reaches
  the goal w.p. 0.4 and otherwise lands in ``s1``, from which ``a2`` reaches
  the goal w.p. 1 — the true value at horizon 2 is exactly 1, noisy-OR says
  0.856). Here both branches are one transition of the joint chain.
- the union bound is exact for that case but saturates whenever a fact has two
  individually strong achievers. The joint has no cap to hit.

Admissibility
-------------
Everything the pattern does NOT model is relaxed optimistically, so the result
is an upper bound under the same delete-relaxed temporal envelope as
``baseline_admissible``:

1. Preconditions outside the pattern are assumed FREE — but only from the
   achiever's ``gate``, the earliest layer at which its effect could land in
   the delete-relaxed graph. Dropping a constraint only raises the value;
   gating it back is a pure reachability tightening (no probability involved).
2. Pattern preconditions are checked at ``t - duration(a)``, exactly as the
   forward marginal sweep schedules its arrivals — an achiever landing at ``t``
   must have been applicable ``duration(a)`` layers earlier, not merely at
   ``t-1``. Facts persist, so "held at ``t - duration``" is recorded by giving
   each pattern fact an AGE (layers since it became true) and requiring
   ``age >= duration - 1`` in the previous state. Ages are only tracked for
   facts that gate an achiever with ``duration > 1``, so unit-duration patterns
   pay nothing. If the age-augmented state set exceeds ``max_states`` it is
   collapsed by taking the elementwise MAX age per mask, which only makes
   achievers fire earlier — optimistic, hence still admissible.
3. Every applicable achiever fires at every layer (relaxed re-execution) and
   distinct actions draw independent coins. Real executions may be mutex;
   assuming they all run is optimistic.
4. Delete effects are ignored, as everywhere else in this family.

Because of (1)-(4) the pattern value is an upper bound, but it is NOT
guaranteed to be below the plain marginal bound (freeing preconditions can cost
more than the exact temporal treatment gains). The caller therefore uses
``min(marginal bound, pattern bound)``, which is admissible because the minimum
of two upper bounds is an upper bound, and sound to inject mid-sweep because
every operator in the forward DP is monotone non-decreasing in its inputs.

Pattern selection (counterexample-guided)
-----------------------------------------
Patterns are grown from the goal backwards, one fact at a time, in the spirit
of "start with every precondition dropped, then re-attach whatever the
abstraction is lying about most":

- seed each pattern with one goal fact;
- collect the achievers of the pattern facts, and their freed preconditions;
- score each freed precondition by how much the free assumption inflates it
  (``1 - marginal(c)`` at the horizon) weighted by the strength of the achiever
  that needs it;
- attach the highest-scoring one and repeat until the size cap.

Re-attaching a real precondition only removes abstract-only behaviour, so the
bound is monotone non-increasing across refinements and admissible at every
prefix — the loop can be stopped at any point (anytime).

Not modelled in this version: MUTEX refinement (adding a shared consumable
resource as a pattern bit and switching the pattern solve to a max-over-action-
sets value iteration). Inside the delete-relaxed envelope a consumable is never
consumed, so the extra bit would change nothing; it becomes meaningful only
together with the delete/survival machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Fact = Hashable

DEFAULT_MAX_FACTS_PER_PATTERN = 4
DEFAULT_MAX_PATTERNS = 4
# Upper limit when the caller asks for "one pattern per unachieved goal fact".
# Patterns are independent and cheap (2^k states each), so the cap only exists
# to stop a pathological many-goal instance from paying per-node.
PATTERN_HARD_CAP = 12
DEFAULT_MAX_STATES = 4096
_EPS = 1e-12


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def joint_add_distribution(action, probe_predicates: Set[Fact]) -> Dict[frozenset, float]:
    """Joint distribution over the SETS of facts one execution makes true.

    The joint (rather than per-fact marginals) is the whole point: an action
    whose probabilistic effect branches 0.4 -> {goal} / 0.6 -> {s1} must not be
    read as two independent coins, or the "failure enables the other path"
    structure is lost and the bound becomes inadmissible.

    Deterministic adds combine with each probabilistic effect by cartesian
    product (independent effects), matching ``correlation_heuristic``.
    """
    base_add = set(getattr(action, "add_effects", set()) or set())
    dist: Dict[frozenset, float] = {frozenset(base_add): 1.0}

    for effect in getattr(action, "probabilistic_effects", []) or []:
        try:
            outcomes = effect.probability_function(
                SimpleNamespace(predicates=set(probe_predicates)), None
            )
        except Exception:
            outcomes = {}
        if not outcomes:
            continue
        # A probability function may be SUB-stochastic: the missing mass is the
        # implicit "this effect did not fire" outcome. Renormalising it away
        # would turn a 0.3 chance into a certainty (and quietly make every
        # pattern curve 1.0), so the residual is carried explicitly as the
        # empty add-set instead.
        residual = 1.0 - sum(_clamp01(float(p)) for p in outcomes)
        expanded: Dict[frozenset, float] = {}
        if residual > 1e-12:
            for previous_add, previous_p in dist.items():
                expanded[previous_add] = (
                    expanded.get(previous_add, 0.0) + previous_p * residual
                )
        for previous_add, previous_p in dist.items():
            for probability, assignments in outcomes.items():
                p = _clamp01(float(probability))
                if p <= 0.0:
                    continue
                add_set = set(previous_add)
                del_set: Set[Fact] = set()
                for fact, value in assignments.items():
                    is_positive = (
                        bool(value.bool_constant_value())
                        if hasattr(value, "bool_constant_value")
                        else bool(value)
                    )
                    if is_positive:
                        add_set.add(fact)
                    else:
                        del_set.add(fact)
                combined = frozenset(add_set - del_set)
                expanded[combined] = expanded.get(combined, 0.0) + previous_p * p
        if expanded:
            dist = expanded

    total = sum(dist.values())
    if total <= 0.0:
        return {frozenset(): 1.0}
    # Only ever normalise DOWN (a malformed super-stochastic function); scaling
    # up is what the residual handling above exists to avoid.
    if total > 1.0 + 1e-9:
        dist = {key: value / total for key, value in dist.items()}
    return dist


def best_joint_add_distribution(
    action,
    probe_universe: Set[Fact],
    targets: Set[Fact],
) -> Dict[frozenset, float]:
    """Pick the probe branch most favourable to ``targets``.

    Probability functions may branch on the state, and a single empty-state
    probe silently loses whole outcomes (this is what dropped machine_shop's
    ``free(machine)`` re-achiever once). Probe both extremes — nothing true and
    everything true — and keep the branch that gives the target facts the most
    mass, matching the optimistic "credit the most favourable reachable branch"
    rule used when the per-fact add probabilities are extracted.
    """
    best: Optional[Dict[frozenset, float]] = None
    best_score = -1.0
    for probe in (set(), set(probe_universe)):
        candidate = joint_add_distribution(action, probe)
        score = sum(p for add, p in candidate.items() if add & targets)
        if score > best_score + 1e-15:
            best_score = score
            best = candidate
    return best if best is not None else {frozenset(): 1.0}


@dataclass(frozen=True)
class SurvivorAchiever:
    """One action, seen through a pattern.

    ``pattern_pre`` are the preconditions the pattern tracks (checked against
    the joint state); every other precondition is freed and represented only by
    ``gate``, the earliest layer at which this effect can land.
    """

    name: str
    pattern_pre: frozenset
    gate: int
    delay: int
    strength: float
    outcomes: Tuple[Tuple[frozenset, float], ...]


@dataclass(frozen=True)
class SurvivorPattern:
    facts: Tuple[Fact, ...]
    seed_goal: Fact
    achievers: Tuple[SurvivorAchiever, ...]


def first_positive_layer(curve: Sequence[float]) -> Optional[int]:
    """Earliest layer with non-zero probability (delete-relaxed earliest time)."""
    for layer, value in enumerate(curve):
        if value > _EPS:
            return layer
    return None


def compute_gate(
    preconditions: Iterable[Fact],
    delay: int,
    marginals: Mapping[Fact, Sequence[float]],
) -> Optional[int]:
    """Earliest layer this achiever's effect can land, or None if unreachable.

    Read straight off the marginal table of a completed relaxed sweep: a fact's
    first positive layer IS its delete-relaxed earliest achievement time, so no
    separate reachability fixpoint is needed.
    """
    earliest_pre = 0
    for fact in preconditions:
        curve = marginals.get(fact)
        if curve is None:
            return None
        layer = first_positive_layer(curve)
        if layer is None:
            return None
        earliest_pre = max(earliest_pre, layer)
    return earliest_pre + max(1, int(delay))


def solve_survivor_pattern(
    pattern: SurvivorPattern,
    initial_facts: Set[Fact],
    horizon: int,
    max_states: int = DEFAULT_MAX_STATES,
) -> Dict[Fact, List[float]]:
    """Forward sweep of the joint distribution over the pattern's facts.

    Returns, per pattern fact, the curve ``P(fact achieved by t)`` for
    ``t = 0..horizon``. Reading a marginal out of the joint is just summing the
    states whose bit is set; the survivor mass (and hence the exact conditional
    hazard) is the complement, which is why nothing here needs a cap at 1.

    A state is ``(mask, ages)``: the facts currently true, plus how many layers
    each precondition-carrying fact has been true. The ages are what make the
    ``t - duration(a)`` timing exact — an achiever of duration ``d`` landing at
    ``t`` needs its pattern preconditions to have been true at ``t - d``, i.e.
    to carry age ``>= d - 1`` in the state at ``t - 1``.
    """
    facts = pattern.facts
    index = {fact: position for position, fact in enumerate(facts)}

    # Only facts that gate a multi-layer achiever need an age; with unit
    # durations everywhere the age vector is empty and this is a plain
    # bitmask chain.
    age_cap: Dict[Fact, int] = {}
    for achiever in pattern.achievers:
        if achiever.delay <= 1:
            continue
        for fact in achiever.pattern_pre:
            age_cap[fact] = max(age_cap.get(fact, 0), achiever.delay - 1)
    aged_facts: Tuple[Fact, ...] = tuple(f for f in facts if f in age_cap)
    aged_positions = tuple(index[f] for f in aged_facts)
    caps = tuple(age_cap[f] for f in aged_facts)

    initial_mask = 0
    for fact, position in index.items():
        if fact in initial_facts:
            initial_mask |= 1 << position
    # Facts true in the initial state have been true "forever".
    initial_ages = tuple(
        cap if initial_mask >> position & 1 else -1
        for position, cap in zip(aged_positions, caps)
    )

    prepared: List[Tuple[int, int, Tuple[int, ...], Tuple[Tuple[int, float], ...]]] = []
    for achiever in pattern.achievers:
        pre_mask = 0
        for fact in achiever.pattern_pre:
            pre_mask |= 1 << index[fact]
        # Per aged slot, the minimum age this achiever needs (-1 = no constraint
        # beyond the mask bit, which the pre_mask test already covers).
        required = tuple(
            achiever.delay - 1 if fact in achiever.pattern_pre else -1
            for fact in aged_facts
        )
        outcome_masks: List[Tuple[int, float]] = []
        for add, probability in achiever.outcomes:
            mask = 0
            for fact in add:
                mask |= 1 << index[fact]
            outcome_masks.append((mask, probability))
        prepared.append((achiever.gate, pre_mask, required, tuple(outcome_masks)))

    curves: Dict[Fact, List[float]] = {
        fact: [0.0] * (horizon + 1) for fact in facts
    }
    for fact, position in index.items():
        curves[fact][0] = 1.0 if initial_mask >> position & 1 else 0.0

    distribution: Dict[Tuple[int, Tuple[int, ...]], float] = {
        (initial_mask, initial_ages): 1.0
    }
    for layer in range(1, horizon + 1):
        advanced: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        for (state, ages), mass in distribution.items():
            current: Dict[int, float] = {state: mass}
            for gate, pre_mask, required, outcome_masks in prepared:
                if layer < gate:
                    continue
                if state & pre_mask != pre_mask:
                    continue
                # Exact t - duration(a) check: every pattern precondition must
                # already have been true long enough for this effect to land now.
                if any(
                    need >= 0 and ages[slot] < need
                    for slot, need in enumerate(required)
                ):
                    continue
                folded: Dict[int, float] = {}
                for partial_state, partial_mass in current.items():
                    for outcome_mask, probability in outcome_masks:
                        if probability <= 0.0:
                            continue
                        successor = partial_state | outcome_mask
                        folded[successor] = (
                            folded.get(successor, 0.0) + partial_mass * probability
                        )
                if folded:
                    current = folded
            for successor, successor_mass in current.items():
                successor_ages = tuple(
                    min(ages[slot] + 1, caps[slot])
                    if successor >> position & 1 and ages[slot] >= 0
                    else (0 if successor >> position & 1 else -1)
                    for slot, position in enumerate(aged_positions)
                )
                key = (successor, successor_ages)
                advanced[key] = advanced.get(key, 0.0) + successor_mass
        if len(advanced) > max_states:
            advanced = _collapse_ages(advanced)
        distribution = advanced
        for fact, position in index.items():
            bit = 1 << position
            curves[fact][layer] = _clamp01(
                sum(
                    mass
                    for (state, _ages), mass in distribution.items()
                    if state & bit
                )
            )

    for fact in facts:
        curve = curves[fact]
        for layer in range(1, horizon + 1):
            if curve[layer] < curve[layer - 1]:
                curve[layer] = curve[layer - 1]
    return curves


def _collapse_ages(
    distribution: Mapping[Tuple[int, Tuple[int, ...]], float],
) -> Dict[Tuple[int, Tuple[int, ...]], float]:
    """Merge age-augmented states sharing a mask, keeping the MAX age per slot.

    A bounded fallback for patterns whose age dimension blows up. Raising an age
    can only let achievers fire earlier, so the collapsed chain dominates the
    exact one and the bound stays admissible — it just loses the fine
    ``t - duration`` timing it was tracking.
    """
    merged: Dict[int, Tuple[List[int], float]] = {}
    for (state, ages), mass in distribution.items():
        entry = merged.get(state)
        if entry is None:
            merged[state] = ([*ages], mass)
            continue
        best_ages, total = entry
        for slot, age in enumerate(ages):
            if age > best_ages[slot]:
                best_ages[slot] = age
        merged[state] = (best_ages, total + mass)
    return {
        (state, tuple(ages)): mass for state, (ages, mass) in merged.items()
    }


def conditional_hazards(curve: Sequence[float]) -> List[float]:
    """``h_t = P(achieved at t | not achieved by t-1)`` from a pattern curve.

    The quantity the marginal recursion cannot recover; here it is a ratio of
    two stored numbers.
    """
    hazards = [0.0] * len(curve)
    for layer in range(1, len(curve)):
        survivors = 1.0 - curve[layer - 1]
        if survivors <= _EPS:
            hazards[layer] = 0.0
        else:
            hazards[layer] = _clamp01((curve[layer] - curve[layer - 1]) / survivors)
    return hazards


def build_pattern(
    seed_goal: Fact,
    *,
    actions_by_effect_fact: Mapping[Fact, Sequence[object]],
    action_preconditions: Mapping[str, frozenset],
    action_delays: Mapping[str, int],
    action_add_probabilities: Mapping[str, Mapping[Fact, float]],
    marginals: Mapping[Fact, Sequence[float]],
    initial_facts: Set[Fact],
    probe_universe: Set[Fact],
    horizon: int,
    max_facts: int,
) -> Optional[SurvivorPattern]:
    """Grow one pattern backwards from ``seed_goal``, then freeze its achievers.

    Growth is counterexample-guided: at each step the freed precondition whose
    "assume free" treatment inflates the bound most (largest remaining marginal
    gap, weighted by the strength of the achiever that needs it) is attached to
    the pattern. Every prefix is admissible, so the cap can cut the loop at any
    point.
    """
    pattern_facts: List[Fact] = [seed_goal]

    def achievers_of(facts: Sequence[Fact]) -> List[Tuple[Fact, object]]:
        collected: List[Tuple[Fact, object]] = []
        seen: Set[Tuple[Fact, int]] = set()
        for fact in facts:
            for action in actions_by_effect_fact.get(fact, ()) or ():
                key = (fact, id(action))
                if key in seen:
                    continue
                seen.add(key)
                collected.append((fact, action))
        return collected

    while len(pattern_facts) < max_facts:
        best_candidate: Optional[Fact] = None
        best_score = 0.0
        fallback_candidate: Optional[Fact] = None
        fallback_strength = 0.0
        for fact, action in achievers_of(pattern_facts):
            name = getattr(action, "name", repr(action))
            preconditions = action_preconditions.get(name, frozenset())
            gate = compute_gate(
                preconditions, action_delays.get(name, 1), marginals
            )
            if gate is None or gate > horizon:
                continue
            distribution = best_joint_add_distribution(
                action, probe_universe, {fact}
            )
            strength = sum(p for add, p in distribution.items() if fact in add)
            if strength <= _EPS:
                continue
            for candidate in preconditions:
                if candidate in pattern_facts or candidate in initial_facts:
                    continue
                curve = marginals.get(candidate)
                if curve is None:
                    continue
                # How much the "assume free" treatment inflates this achiever,
                # averaged over the window where the achiever can actually fire
                # (its gate to the horizon). Reading the gap at the HORIZON
                # alone is useless: the marginal bound saturates to 1.0 there,
                # so it reports "nothing is being inflated" precisely on the
                # long-deadline instances this heuristic exists to fix.
                start = min(max(0, gate), len(curve) - 1)
                stop = min(horizon, len(curve) - 1)
                span = stop - start + 1
                gap = (
                    sum(1.0 - _clamp01(curve[t]) for t in range(start, stop + 1)) / span
                    if span > 0
                    else 0.0
                )
                score = strength * gap
                if score > best_score + 1e-12:
                    best_score = score
                    best_candidate = candidate
                if strength > fallback_strength:
                    fallback_strength = strength
                    fallback_candidate = candidate
        # Even with no measurable gap, attaching a real precondition can only
        # remove abstract-only behaviour, so growing to the cap never loosens
        # the bound. Prefer the strongest achiever's precondition.
        if best_candidate is None:
            best_candidate = fallback_candidate
        if best_candidate is None:
            break
        pattern_facts.append(best_candidate)

    facts = tuple(pattern_facts)
    fact_set = set(facts)
    achievers: List[SurvivorAchiever] = []
    for fact, action in achievers_of(facts):
        name = getattr(action, "name", repr(action))
        preconditions = action_preconditions.get(name, frozenset())
        gate = compute_gate(preconditions, action_delays.get(name, 1), marginals)
        if gate is None or gate > horizon:
            continue
        pattern_pre = frozenset(preconditions & fact_set)
        delay = max(1, int(action_delays.get(name, 1)))
        distribution = best_joint_add_distribution(action, probe_universe, fact_set)
        outcomes: Dict[frozenset, float] = {}
        for add, probability in distribution.items():
            restricted = frozenset(add & fact_set)
            outcomes[restricted] = outcomes.get(restricted, 0.0) + probability

        # ADMISSIBILITY GUARD. The joint above is read by executing the effect's
        # probability function on two probe states. A STATE-DEPENDENT branch that
        # neither probe reaches is invisible to it, so the achiever silently
        # vanishes and the pattern under-counts — the same failure that once
        # dropped machine_shop's only free(machine) re-achiever. The rest of the
        # heuristic guards this with a declared-fluents safety net, giving a
        # per-fact add probability the joint must not fall below. Where it does,
        # top the fact up with an extra INDEPENDENT coin sized so the resulting
        # marginal matches: achievers combine as independent draws here, so
        # 1-(1-q_joint)(1-r) = q_model. This can only add mass, never remove it.
        model_adds = action_add_probabilities.get(name, {})
        top_ups: List[SurvivorAchiever] = []
        for target, model_q in model_adds.items():
            if target not in fact_set:
                continue
            joint_q = sum(p for add, p in outcomes.items() if target in add)
            if model_q <= joint_q + 1e-9:
                continue
            residual = (
                1.0 if joint_q >= 1.0 else (model_q - joint_q) / (1.0 - joint_q)
            )
            residual = _clamp01(residual)
            top_ups.append(
                SurvivorAchiever(
                    name=f"{name}#topup:{target!r}",
                    pattern_pre=pattern_pre,
                    gate=int(gate),
                    delay=delay,
                    strength=float(residual),
                    outcomes=(
                        (frozenset({target}), residual),
                        (frozenset(), 1.0 - residual),
                    ),
                )
            )

        if any(add for add in outcomes if add):
            strength = sum(p for add, p in outcomes.items() if fact in add)
            achievers.append(
                SurvivorAchiever(
                    name=name,
                    pattern_pre=pattern_pre,
                    gate=int(gate),
                    delay=delay,
                    strength=float(strength),
                    outcomes=tuple(
                        sorted(outcomes.items(), key=lambda kv: sorted(map(repr, kv[0])))
                    ),
                )
            )
        achievers.extend(top_ups)

    if not achievers:
        return None
    # De-duplicate: the same action reached through two pattern facts describes
    # one execution, and folding it twice would double-count its coin.
    unique: Dict[str, SurvivorAchiever] = {}
    for achiever in achievers:
        existing = unique.get(achiever.name)
        if existing is None or achiever.strength > existing.strength:
            unique[achiever.name] = achiever
    return SurvivorPattern(
        facts=facts,
        seed_goal=seed_goal,
        achievers=tuple(unique.values()),
    )


def build_patterns(
    goal_facts: Sequence[Fact],
    *,
    actions_by_effect_fact: Mapping[Fact, Sequence[object]],
    action_preconditions: Mapping[str, frozenset],
    action_delays: Mapping[str, int],
    action_add_probabilities: Mapping[str, Mapping[Fact, float]],
    marginals: Mapping[Fact, Sequence[float]],
    initial_facts: Set[Fact],
    probe_universe: Set[Fact],
    horizon: int,
    max_facts: int = DEFAULT_MAX_FACTS_PER_PATTERN,
    max_patterns: int = DEFAULT_MAX_PATTERNS,
) -> List[SurvivorPattern]:
    """One pattern per goal fact (capped), each grown independently."""
    patterns: List[SurvivorPattern] = []
    for goal in list(goal_facts)[: max(1, int(max_patterns))]:
        if goal in initial_facts:
            continue
        pattern = build_pattern(
            goal,
            actions_by_effect_fact=actions_by_effect_fact,
            action_preconditions=action_preconditions,
            action_delays=action_delays,
            action_add_probabilities=action_add_probabilities,
            marginals=marginals,
            initial_facts=initial_facts,
            probe_universe=probe_universe,
            horizon=horizon,
            max_facts=max_facts,
        )
        if pattern is not None:
            patterns.append(pattern)
    return patterns
