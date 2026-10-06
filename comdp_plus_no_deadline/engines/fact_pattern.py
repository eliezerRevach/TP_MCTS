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
    a STATIC fact (below) is never a candidate: the projection already has its value.

Projection (``FactPatternModel``):
    state   = facts in phi  +  the inExecution slots of the pattern's actions
    actions = every action that adds, deletes or may change a fact in phi
    a precondition on a fact outside the pattern counts as TRUE (so is a mutex
    with an action outside the pattern): only more is allowed, the value stays
    an upper bound.
    except a STATIC fact -- read by some action, changed by none (good(h1),
    hand_of(h0, r0)): it keeps its initial value, so an action that needs
    good(h1) while good(h1) is false never runs in the pattern either. Exact, and
    free: a static never changes, so no state is added (impossible actions only
    remove states). TP_MCTS_WILAO_EXACT_STATICS = 0 restores the old relaxation.
    outcomes come from the real transition function on a completed state (the
    projected facts, the initial state outside the pattern, and the action's
    own non-static preconditions), projected back onto the pattern.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, FrozenSet, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

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


def op_reads(op) -> Set[Fact]:
    """Every precondition, positive AND negative (a negative one is still a fact
    the action depends on: sample needs "not full(store)")."""
    out: Set[Fact] = set()
    for event in _events(op):
        out |= {f for f in event.pos_preconditions if not _is_exec(f)}
        out |= {f for f in getattr(event, "neg_preconditions", set()) or set() if not _is_exec(f)}
    return out


def exact_statics_default() -> bool:
    """TP_MCTS_WILAO_EXACT_STATICS (1): static facts keep their initial value in every pattern."""
    return (os.environ.get("TP_MCTS_WILAO_EXACT_STATICS") or "1").strip().lower() not in ("0", "false")


def static_facts(ops: Dict[str, object]) -> Set[Fact]:
    """Facts some action reads (a precondition, positive or negative) and no action
    changes: they keep their initial value forever."""
    read: Set[Fact] = set()
    changed: Set[Fact] = set()
    for op in ops.values():
        read |= op_reads(op)
        changed |= op_touches(op)
    return read - changed


def independent_goal_groups(ops: Dict[str, object], goals: Sequence[Fact]) -> List[List[Fact]]:
    """Split the goals into groups that cannot influence each other.

    relevant(g): actions that change a fact g needs, grown backward to a fixpoint
                 (needed = g, then every precondition -- positive or negative -- of
                 a relevant action).
    footprint(g): every fact a relevant action reads or changes.
    Two goals are joined when they share a relevant action, or a relevant action of
    one changes a fact in the other's footprint. Different groups then share no
    action and no changed fact, so what happens for one group is independent of
    the other: P(all goals) <= product over groups of P(the group's goals)."""
    reads = {k: op_reads(op) for k, op in ops.items()}
    writes = {k: op_touches(op) for k, op in ops.items()}
    relevant: Dict[Fact, Set[str]] = {}
    footprint: Dict[Fact, Set[Fact]] = {}
    for g in goals:
        needed, rel = {g}, set()
        changed = True
        while changed:
            changed = False
            for k in ops:
                if k not in rel and writes[k] & needed:
                    rel.add(k)
                    needed |= reads[k]
                    changed = True
        relevant[g] = rel
        footprint[g] = set(needed).union(*(writes[k] for k in rel)) if rel else set(needed)
    parent = {g: g for g in goals}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for i, a in enumerate(goals):
        for b in goals[i + 1:]:
            wa = set().union(*(writes[k] for k in relevant[a])) if relevant[a] else set()
            wb = set().union(*(writes[k] for k in relevant[b])) if relevant[b] else set()
            if relevant[a] & relevant[b] or wa & footprint[b] or wb & footprint[a]:
                parent[find(a)] = find(b)
    groups: Dict[Fact, List[Fact]] = defaultdict(list)
    for g in goals:
        groups[find(g)].append(g)
    return list(groups.values())


def _objects(fact) -> Tuple[str, ...]:
    """The object arguments of a grounded fact: ``ready(h0, x1)`` -> ``("h0", "x1")``."""
    text = str(fact)
    if "(" not in text:
        return ()
    inside = text[text.index("(") + 1:text.rindex(")")]
    return tuple(a.strip() for a in inside.split(",") if a.strip())


def _goal_relevant_ops(ops: Dict[str, object], goal: Fact) -> Set[str]:
    """Actions that change something the goal needs, grown backward to a fixpoint
    (the same closure as independent_goal_groups)."""
    reads = {k: op_reads(op) for k, op in ops.items()}
    writes = {k: op_touches(op) for k, op in ops.items()}
    needed, rel = {goal}, set()
    changed = True
    while changed:
        changed = False
        for k in ops:
            if k not in rel and writes[k] & needed:
                rel.add(k)
                needed |= reads[k]
                changed = True
    return rel


def resource_patterns(ops: Dict[str, object], groups: Sequence[Sequence[Fact]], initial_facts: Iterable[Fact],
                      max_facts: int = 0) -> List[Tuple[List[Fact], List[Fact], str]]:
    """A pattern COLLECTION ("goal group x one shared resource"), ``[(goals, facts, resource)]``.

        base(G)     = the goals of G + every precondition of their achievers that is not true
                      initially, closed backward (have_rock, calibrated, ...), resource facts left out
        contested   = facts that relevant actions of TWO OR MORE goals of G read, and some action
                      changes (statics excepted): the things the goals fight over (free_h(h0), full(s0))
        resource(f) = every changeable fact that mentions an object of f other than the goals' own
                      objects (hand h0: free_h(h0), ready(h0, x0), ready(h0, x1)); same objects merge
        users(R)    = the goals of G whose OWN achievers (of the goal and of its base facts) read a
                      fact of R -- not the whole relevance closure, where everything touches everything
        patterns    = users(R) + their base facts + R, for every resource R of G;
                      goals that use no resource -> one pattern of those goals and their base

    Each pattern holds every goal that uses its resource, so taking the resource for one of
    them is part of that pattern's plan, not a loss for another pattern. Each is an upper
    bound on P(its goals) >= P(G): combine with min inside the group, product across
    independent groups. ``max_facts`` > 0 trims a
    resource (base facts are kept)."""
    initial = set(initial_facts)
    statics = static_facts(ops)
    changeable: Set[Fact] = set()
    for op in ops.values():
        changeable |= op_touches(op)
    achievers: Dict[Fact, List[str]] = defaultdict(list)
    for key, op in ops.items():
        for f in op_adds(op):
            achievers[f].append(key)
    out: List[Tuple[List[Fact], List[Fact], str]] = []
    for group in groups:
        goals = list(dict.fromkeys(group))
        goal_set = set(goals)
        rel = {g: _goal_relevant_ops(ops, g) for g in goals}
        readers: Dict[Fact, Set] = defaultdict(set)
        for g in goals:
            for k in rel[g]:
                for f in op_reads(ops[k]):
                    readers[f].add(g)
        contested = sorted((f for f, gs in readers.items()
                            if len(gs) >= 2 and f in changeable and f not in statics and f not in goal_set),
                           key=str)
        goal_objects = set().union(*(set(_objects(g)) for g in goals)) if goals else set()
        resources: Dict[Tuple[str, ...], List[Fact]] = {}
        for f in contested:
            objs = set(_objects(f)) - goal_objects          # the goals' own objects (x0) never define a resource
            if not objs:
                continue
            key = tuple(sorted(objs))
            if key in resources:
                continue
            members = sorted((x for x in changeable if x not in goal_set and objs & set(_objects(x))), key=str)
            if max_facts > 0:
                members = members[:max(0, max_facts - len(goals))]
            resources[key] = members
        in_resource = set().union(*resources.values()) if resources else set()
        def base_of(sub):
            base = list(sub)
            frontier = list(sub)
            while frontier:
                f = frontier.pop()
                for key in achievers.get(f, ()):
                    for p in op_preconditions(ops[key]):
                        if p in base or p in initial or p in statics or p in in_resource or p not in changeable:
                            continue
                        base.append(p)
                        frontier.append(p)
            return base

        own_reads = {}
        for g in goals:
            reads: Set[Fact] = set()
            for f in base_of([g]):
                for key in achievers.get(f, ()):
                    reads |= op_reads(ops[key])
            own_reads[g] = reads
        covered: Set[Fact] = set()
        for key, members in resources.items():
            users = [g for g in goals if own_reads[g] & set(members)]
            if not users:
                continue
            covered |= set(users)
            base = base_of(users)
            out.append((users, base + [m for m in members if m not in base], "+".join(key)))
        rest = [g for g in goals if g not in covered]
        if rest:
            out.append((rest, base_of(rest), "base"))
    return out


def grow_pattern(ops: Dict[str, object], goals: Sequence[Fact], initial_facts: Iterable[Fact],
                 max_facts: int, skip_statics: Optional[bool] = None) -> List[Fact]:
    initial = set(initial_facts)
    if skip_statics is None:
        skip_statics = exact_statics_default()
    statics = static_facts(ops) if skip_statics else set()
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
                    if p in chosen or p in statics:
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

    def __init__(self, base, pattern_facts: Sequence[Fact], goals: Iterable[Fact], initial_facts: Iterable[Fact],
                 exact_statics: Optional[bool] = None):
        self.base = base
        self.pattern_facts = frozenset(pattern_facts)
        self._goals = frozenset(goals)
        initial = frozenset(initial_facts)
        all_ops = base.ops()
        # Static facts outside the pattern keep their initial value (module doc).
        self.exact_statics = exact_statics_default() if exact_statics is None else bool(exact_statics)
        statics = (static_facts(all_ops) - self.pattern_facts) if self.exact_statics else set()
        self._statics = frozenset(statics)
        self._static_true = frozenset(f for f in statics if f in initial)
        self._static_false = self._statics - self._static_true
        self._static_ok: Dict[int, bool] = {}
        # An action whose start needs a static it can never have never runs: not a pattern action.
        self._ops = {k: op for k, op in all_ops.items()
                     if op_touches(op) & self.pattern_facts and self._statics_hold(op.start_action)}
        slots: Set[Fact] = set()
        for op in self._ops.values():
            slots |= op_slots(op)
        self.keep = frozenset(self.pattern_facts | slots)
        self._outside_initial = frozenset(f for f in initial if f not in self.keep)
        self._outcome_cache: Dict[Tuple[FrozenSet, int], tuple] = {}
        self.lost_probability_mass = 0.0

    def _statics_hold(self, event) -> bool:
        """The event's static preconditions, read in the initial state (they never change)."""
        ok = self._static_ok.get(id(event))
        if ok is None:
            ok = not (set(event.pos_preconditions) & self._static_false) and \
                not (set(getattr(event, "neg_preconditions", set()) or set()) & self._static_true)
            self._static_ok[id(event)] = ok
        return ok

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
        if not self._statics_hold(event):
            return False
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
        # outside conditions are assumed met -- statics excepted: they keep their initial value
        freed_pos = {f for f in action.pos_preconditions if f not in self.keep and f not in self._statics}
        freed_neg = {f for f in action.neg_preconditions if f not in self.keep and f not in self._statics}
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
