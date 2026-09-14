"""Fact-capped patterns for the windows ILAO* PDB.

Growth (same backwards idea as ``survivor_pdb.build_pattern``, with the rule:
"add a fact; if it cannot hold as-is in the real problem, add its preconditions;
continue until reachable, then add more if there is room"):

    phi = goals
    repeat while |phi| < max_facts:
        candidates = preconditions of the achievers of facts in phi, not in phi
        pick, in this order:
            1. a precondition that is NOT true in the initial state   (needed to reach the goal)
            2. one that IS true initially but some action deletes      (a real threat)
            3. any other one
          ties: closest to the goal, then needed by more pattern achievers
    stop when there is no candidate

Projection (``FactPatternModel``):
    state   = facts in phi  +  the inExecution slots of the pattern's actions
    actions = every action that adds, deletes or may change a fact in phi
    a precondition on a fact outside the pattern counts as TRUE (so is a mutex
    with an action outside the pattern): only more is allowed, the value stays
    an upper bound.
    outcomes come from the real transition function on a completed state (the
    projected facts, the initial state outside the pattern, and the action's
    own preconditions), projected back onto the pattern.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, Hashable, Iterable, List, Sequence, Set, Tuple

Fact = Hashable


def _is_exec(fact) -> bool:
    return "inExecution" in str(fact)


def _events(op) -> list:
    return [op.start_action] + ([op.end_action] if op.end_action is not None else [])


def op_adds(op) -> Set[Fact]:
    out: Set[Fact] = set()
    for event in _events(op):
        out |= set(getattr(event, "add_effects", set()) or set())
        for pe in getattr(event, "probabilistic_effects", []) or []:
            out |= set(getattr(pe, "fluents", []) or [])
    return {f for f in out if not _is_exec(f)}


def op_touches(op) -> Set[Fact]:
    out = set(op_adds(op))
    for event in _events(op):
        out |= {f for f in (getattr(event, "del_effects", set()) or set()) if not _is_exec(f)}
    return out


def op_preconditions(op) -> Set[Fact]:
    out: Set[Fact] = set()
    for event in _events(op):
        out |= {f for f in event.pos_preconditions if not _is_exec(f)}
    return out


def op_slots(op) -> Set[Fact]:
    out: Set[Fact] = set()
    for event in _events(op):
        out |= {f for f in (getattr(event, "add_effects", set()) or set()) if _is_exec(f)}
    return out


def grow_pattern(ops: Dict[str, object], goals: Sequence[Fact], initial_facts: Iterable[Fact],
                 max_facts: int) -> List[Fact]:
    initial = set(initial_facts)
    achievers: Dict[Fact, List[str]] = defaultdict(list)
    deleted: Set[Fact] = set()
    for key, op in ops.items():
        for f in op_adds(op):
            achievers[f].append(key)
        for event in _events(op):
            deleted |= set(getattr(event, "del_effects", set()) or set())
    phi: List[Fact] = list(dict.fromkeys(goals))
    level: Dict[Fact, int] = {g: 0 for g in phi}
    while len(phi) < max_facts:
        chosen = set(phi)
        support: Dict[Fact, int] = defaultdict(int)
        depth: Dict[Fact, int] = {}
        for f in phi:
            for key in achievers.get(f, ()):
                for p in op_preconditions(ops[key]):
                    if p in chosen:
                        continue
                    support[p] += 1
                    depth[p] = min(depth.get(p, 10 ** 9), level[f] + 1)
        if not support:
            break

        def rank(p):
            tier = 0 if p not in initial else (1 if p in deleted else 2)
            return (tier, depth[p], -support[p], str(p))

        best = min(support, key=rank)
        phi.append(best)
        level[best] = depth[best]
    return phi


class FactPatternModel:
    """``EngineModel`` interface over a fact-capped pattern (see module doc)."""

    def __init__(self, base, pattern_facts: Sequence[Fact], goals: Iterable[Fact], initial_facts: Iterable[Fact]):
        self.base = base
        self.pattern_facts = frozenset(pattern_facts)
        self._goals = frozenset(goals)
        all_ops = base.ops()
        self._ops = {k: op for k, op in all_ops.items() if op_touches(op) & self.pattern_facts}
        slots: Set[Fact] = set()
        for op in self._ops.values():
            slots |= op_slots(op)
        self.keep = frozenset(self.pattern_facts | slots)
        self._outside_initial = frozenset(f for f in initial_facts if f not in self.keep)
        self._outcome_cache: Dict[Tuple[FrozenSet, int], tuple] = {}
        self.lost_probability_mass = 0.0

    # -- EngineModel interface -------------------------------------------------
    @property
    def goals(self) -> FrozenSet:
        return self._goals

    def deadline(self):
        return self.base.deadline()

    def ops(self):
        return self._ops

    def project(self, facts: Iterable[Fact]) -> FrozenSet:
        return frozenset(f for f in facts if f in self.keep)

    def _event_legal(self, event, facts: FrozenSet) -> bool:
        pos = {f for f in event.pos_preconditions if f in self.keep}
        neg = {f for f in event.neg_preconditions if f in self.keep}
        return pos <= facts and not (neg & facts)

    def legal_action_names(self, facts: FrozenSet) -> Tuple[str, ...]:
        return tuple(k for k, op in self._ops.items() if self._event_legal(op.start_action, facts))

    def end_legal_op(self, op, facts: FrozenSet) -> bool:
        return self._event_legal(op.end_action, facts)

    def outcomes(self, action, facts: FrozenSet):
        facts = self.project(facts)
        key = (facts, id(action))
        hit = self._outcome_cache.get(key)
        if hit is not None:
            return hit
        freed_pos = {f for f in action.pos_preconditions if f not in self.keep}
        freed_neg = {f for f in action.neg_preconditions if f not in self.keep}
        full = (set(self._outside_initial) | freed_pos) - freed_neg
        full |= facts
        folded: Dict[FrozenSet, float] = defaultdict(float)
        for next_facts, p in self.base.outcomes(action, frozenset(full)):
            folded[self.project(next_facts)] += p
        out = tuple(folded.items())
        self._outcome_cache[key] = out
        return out

    def start_is_inert(self, key: str) -> bool:
        return self.base.start_is_inert(key)
