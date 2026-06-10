"""
Horizon-indexed Pattern Database (PDB) correction for the probabilistic
temporal heuristic — first prototype.

Motivation
----------
The PTRPG heuristic estimates the applicability of an action by the *product*
of its precondition marginals ``prod_f P_t(f)`` (the independence / AND
relaxation). When preconditions share achievers or block each other this
product is biased. A small pattern database replaces that product with an
*exact abstract* probability of jointly reaching the preconditions, computed by
a backward DP over a projection of the problem onto a handful of facts.

Scope of this prototype (intentionally narrow)
----------------------------------------------
* Core recurrence (NO survival / delete-effect decay term ``S(t)``):

      P_f(t) = P_f(t-1) + (1 - P_f(t-1)) * h_f(t)

  where ``h_f(t)`` is the prob ``f`` is *newly* achieved at ``t``. Survival /
  delete decay is explicitly out of scope here.
* Normal integer / unit layers. No resolution / anchor shrinking.
* The PDB tightens the AND / precondition layer only. The OR / fact layer keeps
  the existing noisy-OR ``h_f(t) = 1 - prod_a (1 - contribution(a,f,t))``.

Abstraction
-----------
A *pattern* ``P`` is a subset of facts. Projection drops everything outside
``P`` (so the abstraction is *optimistic* — ignored preconditions are assumed
satisfiable):

    alpha_P(s)        = s ∩ P                          (project a state)
    pre(a_P)          = pre(a) ∩ P
    add(o_P)          = add(o) ∩ P  for each outcome o
    del(o_P)          = del(o) ∩ P
    duration, outcome probabilities                    unchanged

Time-indexed concurrent DP
--------------------------

    V_P(x, H) = best abstract probability that target G_P holds within remaining
                time H, starting from abstract state x.

Durative actions run **concurrently** (this matches the forward temporal
heuristic — an action finishing at time t started at t − dur(a), and actions
overlap freely). The DP is therefore *event-driven over in-flight actions*
rather than charging each action's duration sequentially:

    V_P(x, inflight, H):
        if G_P ⊆ x:                        return 1
        start every *useful* applicable action (pre ⊆ x, adds a fact ∉ x, not
            already running) — concurrency = start a set, not one action
        if nothing in flight:              return 0
        δ = time to the next completion (min remaining over in-flight)
        if δ > H:                          return 0      # next effect past deadline
        for the action(s) completing at δ:
            expectation over their joint outcomes o:
                x' = (x ∪ add(o)) − del(o)
                accumulate p(o) · V_P(x', still_running, H − δ)

"Useful" (adds a not-yet-achieved fact) is what makes the stochastic chain
reproduce the sequential values: once an achiever's fact is obtained it is no
longer restarted, so there is no spurious retry boost; but in a *fail* branch
the achiever stays useful and is retried, and *independent* achievers overlap
instead of serializing. Durations may be fractional (e.g. 0.5); they are
rescaled to an integer grid internally. The target ``G_P`` is supplied per query
(the goal for reachability, or ``pre(a)`` for the applicability correction).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from types import SimpleNamespace
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

Fact = Hashable
Pattern = frozenset


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Action abstraction with *joint* (correlated) outcomes.
#
# Unlike TemporalRelaxedActionModel (which flattens each action to independent
# per-fact add probabilities), the PDB needs the full joint outcome
# distribution so the DP can do `sum_o p(o) * V(next(x,a,o), ...)`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDBOutcome:
    """One joint outcome of an action: fires with ``probability``."""

    probability: float
    add: frozenset
    delete: frozenset

    def project(self, pattern: Pattern) -> "PDBOutcome":
        return PDBOutcome(
            probability=self.probability,
            add=self.add & pattern,
            delete=self.delete & pattern,
        )


@dataclass(frozen=True)
class PDBAction:
    """Durative probabilistic action with explicit joint outcomes."""

    name: str
    preconditions: frozenset
    outcomes: Tuple[PDBOutcome, ...]
    duration: float = 1  # may be fractional (e.g. 0.5); rescaled to a grid in the DP

    def project(self, pattern: Pattern) -> "PDBAction":
        return PDBAction(
            name=self.name,
            preconditions=self.preconditions & pattern,
            outcomes=tuple(o.project(pattern) for o in self.outcomes),
            duration=self.duration,
        )

    def add_facts(self) -> frozenset:
        facts: Set[Fact] = set()
        for outcome in self.outcomes:
            facts |= outcome.add
        return frozenset(facts)

    def achievement_probability(self, fact: Fact) -> float:
        """Probability this action sets ``fact`` true in one application."""
        prob = 0.0
        for outcome in self.outcomes:
            if fact in outcome.add:
                prob += outcome.probability
        return _clamp_probability(prob)


# ---------------------------------------------------------------------------
# Adapter: build PDBActions from the codebase's action objects.
#
# Reads the same duck-typed interface SyntheticAction / unified_planning
# actions expose: pos_preconditions, add_effects, del_effects,
# probabilistic_effects (each with probability_function(state, env) ->
# {prob: {fact: value}}).
# ---------------------------------------------------------------------------


def _extract_duration(action) -> int:
    for getter in ("duration_int",):
        fn = getattr(action, getter, None)
        if callable(fn):
            try:
                return max(1, int(fn()))
            except Exception:
                pass
    for attr in ("duration_steps", "duration"):
        value = getattr(action, attr, None)
        if isinstance(value, int):
            return max(1, value)
    return 1


def _outcome_assignment(assignments) -> Tuple[frozenset, frozenset]:
    add: Set[Fact] = set()
    delete: Set[Fact] = set()
    for fact, value in assignments.items():
        if hasattr(value, "bool_constant_value"):
            positive = bool(value.bool_constant_value())
        else:
            positive = bool(value)
        (add if positive else delete).add(fact)
    return frozenset(add), frozenset(delete)


def _parse_probabilistic_effect(prob_effect) -> List[Tuple[float, frozenset, frozenset]]:
    """One probabilistic effect -> list of (prob, add, del) outcomes.

    A residual no-op outcome carries any missing probability mass (e.g. an
    effect listed only as ``0.8 -> X`` implies ``0.2 -> nothing``)."""
    try:
        outcomes = prob_effect.probability_function(SimpleNamespace(predicates=set()), None)
    except Exception:
        outcomes = None

    dist: List[Tuple[float, frozenset, frozenset]] = []
    if outcomes:
        total = 0.0
        for prob, assignments in outcomes.items():
            p = _clamp_probability(prob)
            total += p
            add, delete = _outcome_assignment(assignments)
            dist.append((p, add, delete))
        residual = max(0.0, 1.0 - total)
        if residual > 1e-12:
            dist.append((residual, frozenset(), frozenset()))
    else:
        # Opaque, state-dependent callable: register the structurally affected
        # fluents as an optimistic add (matches the heuristic's fallback).
        fluents = frozenset(getattr(prob_effect, "fluents", []))
        dist.append((1.0, fluents, frozenset()))
    return dist


def build_pdb_action(action) -> Optional[PDBAction]:
    """Convert one action object into a :class:`PDBAction` (None if it has no
    relaxed effects, e.g. a pure combination wrapper)."""
    # Skip combination wrappers (a list of sub-actions, no own effects).
    if hasattr(action, "actions") and not hasattr(action, "add_effects"):
        return None

    preconditions = frozenset(getattr(action, "pos_preconditions", set()))
    det_add = frozenset(getattr(action, "add_effects", set()))
    det_del = frozenset(getattr(action, "del_effects", set()))
    prob_effects = list(getattr(action, "probabilistic_effects", []) or [])

    if not preconditions and not det_add and not det_del and not prob_effects:
        return None

    # Base deterministic outcome; cross-product with each probabilistic effect
    # (effects assumed independent — matches the relaxed model).
    combined: List[Tuple[float, frozenset, frozenset]] = [(1.0, det_add, det_del)]
    for prob_effect in prob_effects:
        dist = _parse_probabilistic_effect(prob_effect)
        new_combined: List[Tuple[float, frozenset, frozenset]] = []
        for p_base, a_base, d_base in combined:
            for p_eff, a_eff, d_eff in dist:
                new_combined.append(
                    (p_base * p_eff, a_base | a_eff, d_base | d_eff)
                )
        combined = new_combined

    outcomes = tuple(
        PDBOutcome(probability=p, add=a, delete=d)
        for p, a, d in combined
        if p > 0.0
    )
    if not outcomes:
        outcomes = (PDBOutcome(1.0, frozenset(), frozenset()),)

    return PDBAction(
        name=getattr(action, "name", repr(action)),
        preconditions=preconditions,
        outcomes=outcomes,
        duration=_extract_duration(action),
    )


def build_pdb_actions(actions: Iterable[object]) -> List[PDBAction]:
    result: List[PDBAction] = []
    for action in actions:
        pdb_action = build_pdb_action(action)
        if pdb_action is not None:
            result.append(pdb_action)
    return result


# ---------------------------------------------------------------------------
# Pattern database: horizon-indexed backward DP over a projection.
# ---------------------------------------------------------------------------


class PatternDatabase:
    """Horizon-indexed PDB for one pattern.

    The DP uses ``max`` over *all* projected actions (not only the achievers
    chosen during pattern growth). The query target ``G_P`` is supplied per
    call, so this one object answers both goal-reachability and
    precondition-applicability queries.
    """

    def __init__(self, pattern: Iterable[Fact], pdb_actions: Sequence[PDBAction]):
        self.pattern: Pattern = frozenset(pattern)
        # Project every action onto the pattern; keep only those that can change
        # a pattern fact (others can never help and only waste DP work).
        projected: List[PDBAction] = []
        for action in pdb_actions:
            pa = action.project(self.pattern)
            if any(o.add or o.delete for o in pa.outcomes):
                projected.append(pa)
        self.projected_actions: List[PDBAction] = projected
        # Precompute, per action, the union of facts it can add (the "useful"
        # test) and an integer-rescaled duration (durations may be fractional).
        self._action_add_facts: List[frozenset] = [
            a.add_facts() for a in projected
        ]
        fractions = [
            (Fraction(a.duration).limit_denominator(1000) if a.duration > 0 else Fraction(1))
            for a in projected
        ]
        scale = 1
        for fr in fractions:
            scale = lcm(scale, fr.denominator)
        self._time_scale: int = scale
        self._scaled_durations: List[int] = [
            max(1, int(fr * scale)) for fr in fractions
        ]
        # memo key -> value; key = (target_proj, achieved_set, in_flight, H)
        self._memo: Dict[Tuple, float] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @property
    def table_size(self) -> int:
        return len(self._memo)

    def value(self, state_facts: Iterable[Fact], horizon: float, target: Iterable[Fact]) -> float:
        """V_P(alpha_P(state), horizon) for projected ``target`` under concurrent
        durative semantics. ``horizon`` may be fractional."""
        x = frozenset(state_facts) & self.pattern
        target_proj = frozenset(target) & self.pattern
        # Rescale the horizon to the integer grid. Floor (not round): time below
        # the next grid point is unusable, so 1.9 with unit durations -> 1.
        scaled_h = int(math.floor(float(horizon) * self._time_scale + 1e-9))
        return self._value(target_proj, x, frozenset(), scaled_h)

    def _value(
        self,
        target_proj: Pattern,
        achieved: Pattern,
        in_flight: frozenset,
        horizon: int,
    ) -> float:
        # ``in_flight`` = frozenset of (action_index, remaining_scaled_time).
        if target_proj <= achieved:
            return 1.0
        key = (target_proj, achieved, in_flight, horizon)
        cached = self._memo.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        # Start every USEFUL applicable action that is not already running
        # (concurrency: start a *set* of actions, not a single one).
        running_ids = {idx for idx, _ in in_flight}
        scheduled = set(in_flight)
        for idx, action in enumerate(self.projected_actions):
            if idx in running_ids:
                continue
            if not action.preconditions <= achieved:
                continue
            if not (self._action_add_facts[idx] - achieved):
                continue  # adds nothing new -> not useful, would only spin
            scheduled.add((idx, self._scaled_durations[idx]))

        if not scheduled:
            self._memo[key] = 0.0
            return 0.0

        delta = min(remaining for _, remaining in scheduled)
        if delta > horizon:
            # The next effect lands past the deadline -> nothing more achievable.
            self._memo[key] = 0.0
            return 0.0

        completing = [idx for idx, remaining in scheduled if remaining == delta]
        still_running = frozenset(
            (idx, remaining - delta) for idx, remaining in scheduled if remaining > delta
        )

        # Expectation over the joint outcomes of all actions completing now.
        value = 0.0
        outcome_lists = [self.projected_actions[idx].outcomes for idx in completing]
        for combo in itertools.product(*outcome_lists):
            prob = 1.0
            next_achieved = set(achieved)
            for outcome in combo:
                prob *= outcome.probability
                next_achieved |= outcome.add
                next_achieved -= outcome.delete
            if prob <= 0.0:
                continue
            value += prob * self._value(
                target_proj, frozenset(next_achieved), still_running, horizon - delta
            )

        value = _clamp_probability(value)
        self._memo[key] = value
        return value


# ---------------------------------------------------------------------------
# Pattern generation (goal-directed growth).
# ---------------------------------------------------------------------------


def _build_achiever_index(
    pdb_actions: Sequence[PDBAction],
) -> Dict[Fact, List[Tuple[PDBAction, float]]]:
    """fact -> list of (action, achievement_probability) that can add it."""
    index: Dict[Fact, List[Tuple[PDBAction, float]]] = {}
    for action in pdb_actions:
        for fact in action.add_facts():
            index.setdefault(fact, []).append(
                (action, action.achievement_probability(fact))
            )
    return index


def pattern_covers_any_action(pattern: Iterable[Fact], pdb_actions: Sequence[PDBAction]) -> bool:
    """True if some action's whole precondition set is inside ``pattern`` — i.e.
    the manager would actually *use* this pattern for that action (else it falls
    back to the independence product). A pattern that covers nothing is dead
    weight, which is exactly what made ``pdb_used == 0`` on nasa_rover."""
    pat = frozenset(pattern)
    return any(a.preconditions and a.preconditions <= pat for a in pdb_actions)


def grow_pattern(
    seed_facts: Iterable[Fact],
    pdb_actions: Sequence[PDBAction],
    *,
    max_facts_per_pattern: int,
    expansion_policy: str = "random",
    rng=None,
    achiever_index: Optional[Dict[Fact, List[Tuple[PDBAction, float]]]] = None,
    stop_test: Optional[Callable[[frozenset], bool]] = None,
    hard_cap: Optional[int] = None,
) -> Pattern:
    """Grow one pattern from ``seed_facts`` (typically a single goal fact).

    1. Start from the seed facts.
    2. Find (fact-in-pattern, achiever, missing-precondition) options.
    3. Choose an achiever:
         * ``random``   — uniformly among options (using ``rng``).
         * ``max_prob`` — the achiever with the highest achievement probability.
    4. Add one missing precondition fact of that achiever to the pattern.
    5. Repeat until ``max_facts_per_pattern`` is reached or no useful fact
       remains.

    If ``stop_test`` is given the pattern keeps growing *past*
    ``max_facts_per_pattern`` until ``stop_test(pattern)`` is satisfied (e.g.
    "covers at least one action's preconditions"), bounded by ``hard_cap``.

    NOTE: the random / max_prob choice only steers *growth*. The PDB DP itself
    always maximises over *all* projected actions.
    """
    import random as _random

    if rng is None:
        rng = _random.Random()
    if achiever_index is None:
        achiever_index = _build_achiever_index(pdb_actions)

    pattern: Set[Fact] = set(seed_facts)
    policy = (expansion_policy or "random").strip().lower()
    if policy not in ("random", "max_prob"):
        raise ValueError(f"Unknown expansion_policy: {expansion_policy!r}")

    cap = max_facts_per_pattern
    if stop_test is not None and hard_cap is not None:
        cap = max(max_facts_per_pattern, int(hard_cap))

    while len(pattern) < cap:
        # Stop once we are at least max_facts AND (no stop test, or it passes).
        if len(pattern) >= max_facts_per_pattern:
            if stop_test is None or stop_test(frozenset(pattern)):
                break

        # (fact, achiever_action, achievement_prob, missing_preconditions)
        options: List[Tuple[Fact, PDBAction, float, frozenset]] = []
        for fact in pattern:
            for action, prob in achiever_index.get(fact, []):
                missing = action.preconditions - pattern
                if missing:
                    options.append((fact, action, prob, frozenset(missing)))
        if not options:
            break  # no new useful facts

        if policy == "max_prob":
            # Highest-contribution achiever; tie-break deterministically.
            _, _, _, missing = max(
                options, key=lambda opt: (opt[2], str(opt[1].name))
            )
        else:
            missing = rng.choice(options)[3]

        missing_list = sorted(missing, key=lambda f: str(f))
        if policy == "random":
            new_fact = rng.choice(missing_list)
        else:
            new_fact = missing_list[0]
        pattern.add(new_fact)

    return frozenset(pattern)


def generate_patterns(
    goal_facts: Iterable[Fact],
    pdb_actions: Sequence[PDBAction],
    *,
    num_patterns: int,
    max_facts_per_pattern: int,
    expansion_policy: str = "random",
    seed: Optional[int] = None,
    seed_per_goal: bool = True,
    grow_until_covers: bool = False,
    cover_hard_cap: Optional[int] = None,
) -> List[Pattern]:
    """Generate up to ``num_patterns`` distinct patterns.

    ``seed_per_goal`` (default): seed each pattern from a *single* goal fact,
    cycling through the goals. This is essential for multi-goal problems — a
    pattern seeded from the *whole* goal set starts already large and cannot
    grow along any one goal's achiever chain, so it covers no action's
    preconditions (this is what produced ``pdb_used == 0`` on nasa_rover).

    ``grow_until_covers``: keep growing each pattern past
    ``max_facts_per_pattern`` until it covers at least one action's full
    precondition set (bounded by ``cover_hard_cap``), guaranteeing the pattern
    is actually usable by the manager rather than a dead fallback.
    """
    import random as _random

    rng = _random.Random(seed)
    achiever_index = _build_achiever_index(pdb_actions)
    goals = list(dict.fromkeys(goal_facts))  # de-dup, keep order

    stop_test = None
    if grow_until_covers:
        stop_test = lambda pat: pattern_covers_any_action(pat, pdb_actions)
        if cover_hard_cap is None:
            cover_hard_cap = max_facts_per_pattern + 12

    def seeds_for(i: int) -> Iterable[Fact]:
        if seed_per_goal and goals:
            return {goals[i % len(goals)]}
        return set(goals)

    patterns: List[Pattern] = []
    seen: Set[Pattern] = set()
    attempts = 0
    max_attempts = max(num_patterns * 5, num_patterns + 5, len(goals) + 5)
    while len(patterns) < num_patterns and attempts < max_attempts:
        idx = attempts
        attempts += 1
        pattern = grow_pattern(
            seeds_for(idx),
            pdb_actions,
            max_facts_per_pattern=max_facts_per_pattern,
            expansion_policy=expansion_policy,
            rng=rng,
            achiever_index=achiever_index,
            stop_test=stop_test,
            hard_cap=cover_hard_cap,
        )
        if pattern in seen:
            # Deterministic growth from the same seed repeats; with per-goal
            # seeding we have already cycled all goals once we have one each.
            if expansion_policy == "max_prob" and (
                not seed_per_goal or len(patterns) >= len(goals)
            ):
                break
            continue
        seen.add(pattern)
        patterns.append(pattern)
    return patterns


# ---------------------------------------------------------------------------
# Manager: selects a covering PDB for an action's preconditions, with fallback
# to the independence product, and logs usage statistics.
# ---------------------------------------------------------------------------


class PDBCorrection:
    """Holds a set of pattern databases and answers applicability queries.

    For an action ``a`` the manager looks for the *smallest* pattern that fully
    contains ``pre(a)`` (the tightest, cheapest covering abstraction) and
    queries it with ``target = pre(a)``. If no pattern covers the
    preconditions it falls back to the supplied independence estimate.
    """

    def __init__(
        self,
        patterns: Iterable[Iterable[Fact]],
        pdb_actions: Sequence[PDBAction],
    ):
        self.pdb_actions: List[PDBAction] = list(pdb_actions)
        self.databases: List[PatternDatabase] = [
            PatternDatabase(pattern, self.pdb_actions) for pattern in patterns
        ]
        self.pdb_used: int = 0
        self.fallbacks: int = 0
        # Cache pattern selection per precondition set.
        self._selection_cache: Dict[frozenset, Optional[PatternDatabase]] = {}

    @classmethod
    def from_actions(
        cls,
        actions: Iterable[object],
        goal_facts: Iterable[Fact],
        *,
        num_patterns: int = 4,
        max_facts_per_pattern: int = 4,
        expansion_policy: str = "max_prob",
        seed: Optional[int] = None,
        seed_per_goal: bool = True,
        grow_until_covers: bool = False,
        cover_hard_cap: Optional[int] = None,
    ) -> "PDBCorrection":
        pdb_actions = build_pdb_actions(actions)
        patterns = generate_patterns(
            goal_facts,
            pdb_actions,
            num_patterns=num_patterns,
            max_facts_per_pattern=max_facts_per_pattern,
            expansion_policy=expansion_policy,
            seed=seed,
            seed_per_goal=seed_per_goal,
            grow_until_covers=grow_until_covers,
            cover_hard_cap=cover_hard_cap,
        )
        return cls(patterns, pdb_actions)

    def select_database(self, preconditions: Iterable[Fact]) -> Optional[PatternDatabase]:
        pre = frozenset(preconditions)
        if not pre:
            return None
        if pre in self._selection_cache:
            return self._selection_cache[pre]
        best: Optional[PatternDatabase] = None
        for database in self.databases:
            if pre <= database.pattern:
                if best is None or len(database.pattern) < len(best.pattern):
                    best = database
        self._selection_cache[pre] = best
        return best

    def applicability(
        self,
        state_facts: Iterable[Fact],
        preconditions: Iterable[Fact],
        horizon: int,
        fallback: Callable[[], float],
    ) -> float:
        """PDB estimate of P(preconditions jointly reachable within horizon).

        Falls back to ``fallback()`` (the independence product) when no pattern
        covers ``preconditions``."""
        database = self.select_database(preconditions)
        if database is None:
            self.fallbacks += 1
            return _clamp_probability(fallback())
        self.pdb_used += 1
        return _clamp_probability(
            database.value(state_facts, horizon, target=preconditions)
        )

    # -- logging -----------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        return {
            "num_patterns": len(self.databases),
            "patterns": [sorted(map(str, db.pattern)) for db in self.databases],
            "pattern_sizes": [len(db.pattern) for db in self.databases],
            "pdb_table_sizes": [db.table_size for db in self.databases],
            "pdb_table_total": sum(db.table_size for db in self.databases),
            "cache_hits": sum(db.cache_hits for db in self.databases),
            "cache_misses": sum(db.cache_misses for db in self.databases),
            "pdb_used": self.pdb_used,
            "fallbacks": self.fallbacks,
        }

    def log_summary(self, printer: Callable[[str], None] = print) -> None:
        stats = self.stats()
        printer("[PDBCorrection] summary:")
        printer(f"  patterns ({stats['num_patterns']}):")
        for pattern, size, tbl in zip(
            stats["patterns"], stats["pattern_sizes"], stats["pdb_table_sizes"]
        ):
            printer(f"    - {pattern}  (|P|={size}, table={tbl})")
        printer(f"  pdb table entries total : {stats['pdb_table_total']}")
        printer(f"  cache hits / misses     : {stats['cache_hits']} / {stats['cache_misses']}")
        printer(f"  pdb used for applicability: {stats['pdb_used']}")
        printer(f"  fallbacks to independence : {stats['fallbacks']}")
