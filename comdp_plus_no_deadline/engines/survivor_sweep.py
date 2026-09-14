"""Survivor sweep: delete-relaxed occupancy over (facts, phases) -- ``rpg_exact_states_v18``.

Recreated from the v18 doc as the guide heuristic for search over the windows
MDP (``Temporal_PDB_Time_Left_Windows``). It runs on the SAME model interface
(``EngineModel`` / ``ToyModel``), so no bridge between encodings is needed.

    T      : {0}, closed under u + d(a) <= D for every a reachable by u        (sec. 1)
    n(a,D) : floor((c(a) + D) / d(a))   attempts completing in a step          (sec. 3)
    Tr     : every enabled achiever fires independently, n attempts each       (sec. 4)
    c'(a)  : (c(a) + D) mod d(a);  0 when pre(a) is completed in this step     (sec. 5)
    value  : P(g <= t) = mass on goal states at t                              (sec. 7)

The question it answers is a DEADLINE, not a path: a failed attempt leaves its
mass in a non-goal state, which keeps moving and may still reach the goal.

Relaxations (all optimistic, so the value is meant as an upper bound; not yet
verified against the windows MDP):
* no deletes, no negative preconditions, no inExecution mutex;
* pre(a) = the start's positive preconditions only (end conditions dropped);
* start adds are applied as soon as pre(a) holds, union over start outcomes;
* zero-duration actions add the union of their outcomes at once (unlimited retries);
* a running action ignores pre(a): it completes at lo, lo + d, lo + 2d, ...
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, Hashable, Iterable, List, Optional, Sequence, Tuple

Fact = Hashable
_EPS = 1e-15
_GOAL = ("__goal__",)


@dataclass(frozen=True)
class RelaxedAction:
    name: str
    pre: FrozenSet
    duration: Fraction
    start_adds: FrozenSet
    end_outcomes: Tuple[Tuple[FrozenSet, float], ...]     # (adds, p), p sums to 1


# ---------------------------------------------------------------------------
# Relaxed actions from the two model adapters
# ---------------------------------------------------------------------------

def _is_exec_fact(fact) -> bool:
    return "inExecution" in str(fact)


def relaxed_actions_from_toy(model) -> List[RelaxedAction]:
    out = []
    for key, spec in model._spec.items():
        duration = Fraction(spec.get("duration", 1))
        start = spec.get("start", [((), (), 1.0)])
        end = spec.get("end", [((), (), 1.0)])
        if duration == 0:                       # instantaneous: its effects are the "start" branches
            end, start = start, [((), (), 1.0)]
        start_adds = frozenset().union(*(frozenset(a) for a, _d, _p in start))
        outcomes = _fold((frozenset(a), float(p)) for a, _d, p in end)
        out.append(RelaxedAction(key, frozenset(spec.get("pre", ())), duration, start_adds, outcomes))
    return out


def relaxed_actions_from_engine(model, probe_facts: Iterable[Fact]) -> List[RelaxedAction]:
    """Read adds and probabilities through the real transition function.

    Probabilities may depend on the state, so they are read at ``probe_facts``
    (the query state plus the action's preconditions): exact for the query,
    an approximation elsewhere.
    """
    probe = frozenset(probe_facts)
    # A fact-capped pattern counts preconditions outside the pattern as true.
    keep = getattr(model, "pattern_facts", None)
    out = []
    for key, op in model.ops().items():
        start = op.start_action
        pre = frozenset(f for f in start.pos_preconditions
                        if not _is_exec_fact(f) and (keep is None or f in keep))
        before = probe | pre
        start_results = model.outcomes(start, before)
        start_adds = frozenset().union(*(frozenset(f) - before for f, _p in start_results))
        if op.end_action is None:
            outcomes = _fold((frozenset(f) - before, float(p)) for f, p in start_results)
            out.append(RelaxedAction(key, pre, Fraction(0), frozenset(), outcomes))
            continue
        after_start = max(start_results, key=lambda fp: fp[1])[0] | frozenset(op.end_action.pos_preconditions)
        outcomes = _fold(
            (frozenset(f) - after_start - _exec_facts(after_start), float(p))
            for f, p in model.outcomes(op.end_action, after_start)
        )
        out.append(RelaxedAction(key, pre, Fraction(op.duration), start_adds, outcomes))
    return out


def _exec_facts(facts: FrozenSet) -> FrozenSet:
    return frozenset(f for f in facts if _is_exec_fact(f))


def _fold(pairs) -> Tuple[Tuple[FrozenSet, float], ...]:
    folded: Dict[FrozenSet, float] = defaultdict(float)
    for adds, p in pairs:
        if p > _EPS:
            folded[frozenset(a for a in adds if not _is_exec_fact(a))] += p
    return tuple(folded.items())


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

class SurvivorSweep:
    """``P(goal <= t)`` for every timestamp ``t <= horizon`` from one query.

    ``running`` is ``[(action name, lo)]``: the earliest end of each running
    action (the windows MDP's lower bound). Build once per query, then
    ``value(r)`` is a lookup.
    """

    def __init__(self, actions: Sequence[RelaxedAction], goals: Iterable[Fact]):
        self.actions = list(actions)
        self.goals = frozenset(goals)
        self.durative = [i for i, a in enumerate(self.actions) if a.duration > 0]
        self.instant = [a for a in self.actions if a.duration == 0]
        self.states_seen = 0

    # -- structure ---------------------------------------------------------
    def _closure(self, facts: FrozenSet) -> FrozenSet:
        facts = set(facts)
        changed = True
        while changed:
            changed = False
            for a in self.actions:
                if a.pre <= facts:
                    extra = a.start_adds - facts
                    if a.duration == 0:
                        extra |= frozenset().union(*(adds for adds, _p in a.end_outcomes)) - facts
                    if extra:
                        facts |= extra
                        changed = True
        return frozenset(facts)

    def _timestamps(self, facts: FrozenSet, horizon: Fraction, gates: Dict[int, Fraction],
                    anchors: Dict[int, Fraction]) -> List[Fraction]:
        """Section 1: T closed under u + d(a), a reachable by u. Seeded with the
        running actions' completions, which do not depend on the state."""
        earliest: Dict[Fact, Fraction] = {f: Fraction(0) for f in facts}
        T = {Fraction(0)}
        for i, lo in anchors.items():
            for adds, _p in self.actions[i].end_outcomes:
                for f in adds:
                    if lo <= horizon and earliest.get(f, horizon + 1) > lo:
                        earliest[f] = lo
            t = lo
            while t <= horizon:
                T.add(t)
                t += self.actions[i].duration
        for i, g in gates.items():
            if g <= horizon:
                T.add(g)
        changed = True
        while changed:
            changed = False
            for i in self.durative:
                a = self.actions[i]
                if not a.pre <= earliest.keys():
                    continue
                ready = max((earliest[f] for f in a.pre), default=Fraction(0))
                for f in a.start_adds:
                    if earliest.get(f, horizon + 1) > ready:
                        earliest[f] = ready
                        changed = True
                done = ready + a.duration
                for adds, _p in a.end_outcomes:
                    for f in adds:
                        if done <= horizon and earliest.get(f, horizon + 1) > done:
                            earliest[f] = done
                            changed = True
        frontier = sorted(T)
        while frontier:
            u = frontier.pop()
            for i in self.durative:
                a = self.actions[i]
                if all(earliest.get(f, horizon + 1) <= u for f in a.pre):
                    v = u + a.duration
                    if v <= horizon and v not in T:
                        T.add(v)
                        frontier.append(v)
        return sorted(T)

    def _step_outcomes(self, facts: FrozenSet, attempts: Dict[int, int]) -> Dict[FrozenSet, float]:
        """Section 4: every achiever with n >= 1 draws n times, independently."""
        dist: Dict[FrozenSet, float] = {frozenset(): 1.0}
        for i, n in attempts.items():
            a = self.actions[i]
            useful = [(adds - facts, p) for adds, p in a.end_outcomes]
            if all(not adds for adds, _p in useful):
                continue
            own: Dict[FrozenSet, float] = {frozenset(): 1.0}
            for _ in range(n):
                nxt: Dict[FrozenSet, float] = defaultdict(float)
                for got, q in own.items():
                    for adds, p in useful:
                        nxt[got | adds] += q * p
                own = nxt
            combined: Dict[FrozenSet, float] = defaultdict(float)
            for got, q in dist.items():
                for extra, p in own.items():
                    combined[got | extra] += q * p
            dist = combined
        return dist

    # -- sweep -------------------------------------------------------------
    def curve(self, facts: Iterable[Fact], horizon, running: Sequence[Tuple[str, object]] = ()
              ) -> List[Tuple[Fraction, float]]:
        horizon = Fraction(horizon)
        index = {a.name: i for i, a in enumerate(self.actions)}
        s0 = self._closure(frozenset(facts))

        # A running action: completions at lo, lo + d, ... regardless of pre.
        phases0: Dict[int, Fraction] = {}
        gates: Dict[int, Fraction] = {}
        anchors: Dict[int, Fraction] = {}
        at_zero: List[int] = []
        for name, lo in running:
            i = index.get(name)
            if i is None or self.actions[i].duration == 0:
                continue
            d, lo = self.actions[i].duration, Fraction(lo)
            anchors[i] = lo
            if lo == 0:
                at_zero.append(i)
                phases0[i] = Fraction(0)
            elif lo <= d:
                phases0[i] = d - lo
            else:
                gates[i] = lo - d
        for i in self.durative:
            if i not in phases0 and i not in gates and self.actions[i].pre <= s0:
                phases0[i] = Fraction(0)

        dist: Dict[Tuple, float] = defaultdict(float)
        for extra, p in self._step_outcomes(s0, {i: 1 for i in at_zero}).items():
            s = self._closure(s0 | extra)
            dist[self._key(s, phases0, s0, Fraction(0), gates)] += p

        T = self._timestamps(s0, horizon, gates, anchors)
        out = [(T[0], self._goal_mass(dist))]
        for prev, t in zip(T, T[1:]):
            delta = t - prev
            nxt: Dict[Tuple, float] = defaultdict(float)
            for key, mass in dist.items():
                if key == _GOAL:
                    nxt[_GOAL] += mass
                    continue
                facts_s, phases = key[0], dict(key[1])
                attempts = {i: int((c + delta) // self.actions[i].duration) for i, c in phases.items()}
                attempts = {i: n for i, n in attempts.items() if n >= 1}
                moved = {i: (c + delta) % self.actions[i].duration for i, c in phases.items()}
                for extra, p in self._step_outcomes(facts_s, attempts).items():
                    s2 = self._closure(facts_s | extra)
                    nxt[self._key(s2, moved, facts_s, t, gates)] += mass * p
            dist = nxt
            out.append((t, self._goal_mass(dist)))
        self.states_seen = max(self.states_seen, len(dist))
        return out

    def _key(self, facts: FrozenSet, phases: Dict[int, Fraction], before: FrozenSet,
             t: Fraction, gates: Dict[int, Fraction]):
        if self.goals <= facts:
            return _GOAL
        phases = dict(phases)
        for i in self.durative:
            if i in phases:
                continue
            gate = gates.get(i)
            if gate is not None:
                if t >= gate:
                    phases[i] = Fraction(0)        # running action: its next run starts at the gate
            elif self.actions[i].pre <= facts:
                phases[i] = Fraction(0)            # last precondition arrived in this step
        return (facts, tuple(sorted(phases.items())))

    def _goal_mass(self, dist) -> float:
        return float(dist.get(_GOAL, 0.0))

    # -- lookup ------------------------------------------------------------
    def value(self, facts: Iterable[Fact], r, running: Sequence[Tuple[str, object]] = ()) -> float:
        return lookup(self.curve(facts, r, running), r)


def lookup(curve: List[Tuple[Fraction, float]], r) -> float:
    """``P(goal <= r)``: the last timestamp at or before r."""
    times = [t for t, _p in curve]
    k = bisect.bisect_right(times, Fraction(r)) - 1
    return curve[k][1] if k >= 0 else 0.0
