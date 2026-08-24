"""Exact pattern-MDP heuristic for CoMDP+ (``exact_pattern_mdp``).

This is NOT a member of the PTRPG family and does not import it. The PTRPG
strategies (including ``survivor_pdb``) are *delete-relaxed*: facts only ever
turn on, so no two facts can be mutex, every applicable achiever fires at every
layer, and a consumable is never consumed. Under that envelope a max over
action sets is provably vacuous -- "fire everything" is always optimal -- which
is why those strategies never need to choose.

Here we solve a genuine finite-horizon MDP instead. The ONLY relaxations are:

1. **Projection.** A pattern is a small fact set ``Phi``. Facts outside ``Phi``
   are not tracked, so preconditions outside ``Phi`` are dropped. Dropping a
   precondition is an AND-drop: the action becomes *more* applicable, the value
   goes UP, the bound stays admissible.

   The dual is NOT allowed: every action that touches ``Phi`` is kept. Dropping
   an achiever would be an OR-drop -- the fact becomes harder to reach, the
   value goes DOWN, and the bound falls below the truth.

2. **State-dependent effect probabilities.** A probability function that branches
   on the full state cannot be evaluated in an abstract state, so it is probed at
   two extremes (nothing true / everything true) and the branch most favourable
   to the pattern facts is used. Optimistic, hence admissible.

3. **At-end conditions are dropped** (``P_E`` is not modelled). This is again an
   AND-drop, so it is safe. It also buys the key temporal property below. None of
   the shipped domains uses ``EndPreconditionTiming``, so on those the relaxation
   is vacuous and the bound is exact w.r.t. this axis.

Everything else is exact *inside the pattern*: deletes apply, durations are
respected, effect outcomes stay jointly distributed (so exclusive branches are
not decorrelated), and the policy genuinely maximises.

The model
---------
State ``(m, rho, r, g)``:

* ``m``    -- bitmask of the pattern facts currently true
* ``rho``  -- in-flight operations as ``(op, remaining)``. A running operation
  keeps its slot even when its end effects miss ``Phi`` because the slot is what
  forbids an overlapping copy of the same ground action.
* ``r``    -- remaining time to the deadline
* ``g``    -- sticky "goal has held at some point" bit. This encodes ACHIEVEMENT
  ("goal reached before the deadline") rather than MAINTENANCE ("goal holds at
  the deadline"). With deletes inside ``Phi`` the two genuinely differ.

At a decision boundary, zero or more operations may be dispatched before time
advances. Constructing that same-time dispatch group is a *local* calculation;
its partial subsets are not pattern-database states. The persistent table only
contains canonical boundary states after time has advanced, so equivalent
groups meet at the same ``(m, rho, r)`` key.

Time advances to the earliest required boundary. If an idle applicable action
remains, the next duration-GCD boundary is retained so a policy may deliberately
delay its start. If no action can be started before the world changes, time jumps
directly to the next running completion. Thus the GCD bounds the step size when
a decision is possible, but does not force a one-tick sweep between events.

Why one backward pass suffices
------------------------------
Every persistent transition strictly decreases ``r``. Inside one boundary,
local dispatch recursion strictly grows ``rho`` and is discarded after the best
group has been evaluated. Both recursions are acyclic, so no value iteration,
policy iteration, discount factor, or convergence test is needed.

Admissibility, and the one thing that breaks it
-----------------------------------------------
``V^Phi >= V*`` because relaxations (1)-(3) all widen the set of available
policies, and a max over a superset of policies is an upper bound.

The restriction that would BREAK it is temporal: shrinking the set of start
times removes policies and pushes the value DOWN. Durations are integral in this
prototype, so the exact model retains deadline-aligned duration-GCD boundaries
whenever an idle action could still be started. Pure event jumps are used only
when no such decision is possible before the next running completion. At-end or
overall conditions remain outside this model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import gcd
from types import SimpleNamespace
from typing import Dict, FrozenSet, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Fact = Hashable

DEFAULT_MAX_FACTS = 4
DEFAULT_MAX_PATTERNS = 0          # 0 => one pattern per unachieved goal fact
PATTERN_HARD_CAP = 8
DEFAULT_MAX_STATES = 200_000
_EPS = 1e-12

# Drop the end-delete of facts an op adds at its own start (see _op_from_action).
# Admissible (fewer deletes = optimistic). Off by default; flip it, or set
# TP_MCTS_EXACT_PDB_HOLD_WINDOWS=1, to measure the state-space vs.
# informativeness trade-off.
HOLD_WINDOWS = os.environ.get("TP_MCTS_EXACT_PDB_HOLD_WINDOWS", "0") not in ("0", "", "false")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))


# ---------------------------------------------------------------------------
# Outcome extraction
# ---------------------------------------------------------------------------

Outcome = Tuple[FrozenSet, FrozenSet, float]      # (adds, deletes, probability)


def joint_outcomes(action, probe_predicates: Set[Fact]) -> List[Outcome]:
    """Joint distribution over the (adds, deletes) one execution produces.

    Joint rather than per-fact: an effect branching ``0.4 -> {goal}`` /
    ``0.6 -> {s1}`` must not be read as two independent coins, or the
    "failure enables the other path" structure is lost.

    Deterministic effects combine with each probabilistic effect by cartesian
    product. A SUB-stochastic probability function keeps its missing mass as an
    explicit "did not fire" outcome rather than being renormalised (which would
    silently promote a 0.3 chance to a certainty).
    """
    base_add = frozenset(getattr(action, "add_effects", set()) or set())
    base_del = frozenset(getattr(action, "del_effects", set()) or set())
    dist: Dict[Tuple[FrozenSet, FrozenSet], float] = {(base_add, base_del): 1.0}

    for effect in getattr(action, "probabilistic_effects", []) or []:
        try:
            outcomes = effect.probability_function(
                SimpleNamespace(predicates=set(probe_predicates)), None
            )
        except Exception:
            outcomes = {}
        if not outcomes:
            continue
        residual = 1.0 - sum(_clamp01(float(p)) for p in outcomes)
        expanded: Dict[Tuple[FrozenSet, FrozenSet], float] = {}
        if residual > _EPS:
            for key, mass in dist.items():
                expanded[key] = expanded.get(key, 0.0) + mass * residual
        for (prev_add, prev_del), mass in dist.items():
            for probability, assignments in outcomes.items():
                p = _clamp01(float(probability))
                if p <= 0.0:
                    continue
                adds, dels = set(prev_add), set(prev_del)
                for fact, value in assignments.items():
                    positive = (
                        bool(value.bool_constant_value())
                        if hasattr(value, "bool_constant_value") else bool(value)
                    )
                    if positive:
                        adds.add(fact)
                        dels.discard(fact)
                    else:
                        dels.add(fact)
                        adds.discard(fact)
                key = (frozenset(adds), frozenset(dels))
                expanded[key] = expanded.get(key, 0.0) + mass * p
        if expanded:
            dist = expanded

    total = sum(dist.values())
    if total <= 0.0:
        return [(frozenset(), frozenset(), 1.0)]
    if total > 1.0 + 1e-9:                       # malformed super-stochastic input
        dist = {k: v / total for k, v in dist.items()}
    return [(a, d, p) for (a, d), p in dist.items() if p > _EPS]


def best_joint_outcomes(action, probe_universe: Set[Fact], targets: Set[Fact]) -> List[Outcome]:
    """Probe both extremes and keep the branch most favourable to ``targets``.

    A state-dependent probability function that neither probe reaches is
    invisible; probing only the empty state is what once dropped machine_shop's
    sole ``free(machine)`` re-achiever. Taking the most favourable branch is the
    optimistic (hence admissible) reading.
    """
    best: Optional[List[Outcome]] = None
    best_score = -1.0
    for probe in (set(), set(probe_universe)):
        cand = joint_outcomes(action, probe)
        score = sum(p for a, _d, p in cand if a & targets)
        if score > best_score + 1e-15:
            best_score, best = score, cand
    return best if best is not None else [(frozenset(), frozenset(), 1.0)]


# ---------------------------------------------------------------------------
# Pattern structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternOp:
    """One durative operation seen through a pattern.

    ``pre`` holds only the start conditions inside ``Phi``; every other
    precondition is freed. ``start_outcomes`` land at the moment the operation
    starts, ``end_outcomes`` ``duration`` steps later. Masks are over ``Phi``.
    """

    name: str
    duration: int
    pre_mask: int
    start_outcomes: Tuple[Tuple[int, int, float], ...]     # (add_mask, del_mask, p)
    end_outcomes: Tuple[Tuple[int, int, float], ...]
    touches_end: bool                                       # end effects reach Phi
    footprint: int                                          # every fact it may touch


@dataclass(frozen=True)
class Pattern:
    facts: Tuple[Fact, ...]
    seed_goal: Fact
    ops: Tuple[PatternOp, ...]
    goal_mask: int
    conflicts: Tuple[int, ...]      # conflicts[i] = bitset of op indices mutex with i


def _mask_of(facts: Iterable[Fact], index: Mapping[Fact, int]) -> int:
    m = 0
    for f in facts:
        pos = index.get(f)
        if pos is not None:
            m |= 1 << pos
    return m


# ---------------------------------------------------------------------------
# The exact DP
# ---------------------------------------------------------------------------

class PatternSolver:
    """Memoised backward induction over boundary states of one pattern.

    ``memo`` IS the pattern database. Its keys ``(m, rho, r)`` are canonical
    states at which a new dispatch decision may be made. Partial same-time
    action groups live only in a local cache and never enter ``memo``. A solver
    instance must be kept ALIVE across heuristic calls -- build the table once,
    then every later query is a lookup plus whatever few boundary states its own
    start point reaches for the first time.
    Constructing a fresh solver per call throws the database away and pays the
    full DP again on every node, which is the difference between a pattern
    database and an exhaustive per-node search.
    """

    def __init__(self, pattern: Pattern, max_states: int = DEFAULT_MAX_STATES,
                 sequential: bool = False):
        self.p = pattern
        self.max_states = max_states
        self.memo: Dict[Tuple, float] = {}
        self.in_progress: Set[Tuple] = set()
        self.truncated = False
        self.dispatch_states_evaluated = 0
        self.quantum = 0
        for op in pattern.ops:
            self.quantum = gcd(self.quantum, max(1, int(op.duration)))
        self.quantum = max(1, self.quantum)
        self.symmetry_classes = self._build_symmetry_classes()
        self.add_masks = tuple(
            self._outcome_add_mask(op.start_outcomes + op.end_outcomes)
            for op in pattern.ops
        )
        # NOT admissible. Collapses each op to one atomic transition consuming
        # d(a) time, so actions can never overlap. That removes policies, which
        # pushes the value DOWN and can put it below the truth: two d=10 ops
        # achieving different goal facts with D=15 give 1.0 concurrently and 0.0
        # sequentially. Kept as a measurement/ablation mode -- the state space
        # collapses from (m, rho, r) to (m, r) -- never as the default.
        self.sequential = sequential

    def value_sequential(self, m: int, r: int, g: bool) -> float:
        if g:
            return 1.0
        if r <= 0:
            return 0.0
        key = (m, r)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        if len(self.memo) >= self.max_states:
            self.truncated = True
            return 1.0
        self.in_progress.add(key)

        best = 0.0
        for op in self.p.ops:
            if op.duration > r or m & op.pre_mask != op.pre_mask:
                continue
            total = 0.0
            for s_add, s_del, s_p in op.start_outcomes:
                m1 = self._apply(m, s_add, s_del)
                if (m1 & self.p.goal_mask) == self.p.goal_mask:
                    total += s_p                       # banked before it even ends
                    continue
                for e_add, e_del, e_p in op.end_outcomes:
                    m2 = self._apply(m1, e_add, e_del)
                    g2 = (m2 & self.p.goal_mask) == self.p.goal_mask
                    total += s_p * e_p * (
                        1.0 if g2 else self.value_sequential(m2, r - op.duration, False)
                    )
            if total > best:
                best = total

        self.in_progress.discard(key)
        self.memo[key] = best
        return best

    # -- helpers ---------------------------------------------------------
    def _apply(self, m: int, add: int, dele: int) -> int:
        return (m & ~dele) | add

    @staticmethod
    def _op_signature(op: PatternOp) -> Tuple:
        """Projected behaviour used to identify exchangeable ground actions."""
        return (
            op.duration,
            op.pre_mask,
            op.start_outcomes,
            op.end_outcomes,
            op.touches_end,
            op.footprint,
        )

    @staticmethod
    def _outcome_add_mask(outcomes: Sequence[Tuple[int, int, float]]) -> int:
        add_mask = 0
        for add, _dele, _prob in outcomes:
            add_mask |= add
        return add_mask

    def _build_symmetry_classes(self) -> Tuple[Tuple[int, ...], ...]:
        """Find projected action identities that can be safely permuted.

        Members must have identical projected transitions, must not conflict
        with one another, and must have the same conflict relation to every
        outside operation. A class of q actions remains able to run q copies;
        only their unobservable names are quotiented out.
        """
        unused = set(range(len(self.p.ops)))
        classes: List[Tuple[int, ...]] = []
        while unused:
            i = min(unused)
            unused.remove(i)
            block = [i]
            for j in sorted(unused):
                if self._op_signature(self.p.ops[i]) != self._op_signature(self.p.ops[j]):
                    continue
                if self.p.conflicts[i] >> j & 1:
                    continue
                equivalent = True
                for k in range(len(self.p.ops)):
                    if k == i or k == j:
                        continue
                    if bool(self.p.conflicts[i] >> k & 1) != bool(self.p.conflicts[j] >> k & 1):
                        equivalent = False
                        break
                if equivalent:
                    block.append(j)
            for j in block[1:]:
                unused.remove(j)
            if len(block) > 1:
                classes.append(tuple(block))
        return tuple(classes)

    def _canonical_rho(
        self, rho: Tuple[Tuple[int, int], ...],
    ) -> Tuple[Tuple[int, int], ...]:
        """Canonicalise remaining-time multisets inside symmetry classes."""
        if not self.symmetry_classes or not rho:
            return tuple(sorted(rho))
        remaining_by_op = dict(rho)
        class_members = {i for cls in self.symmetry_classes for i in cls}
        canonical = [(i, rem) for i, rem in rho if i not in class_members]
        for cls in self.symmetry_classes:
            remaining = sorted(remaining_by_op[i] for i in cls if i in remaining_by_op)
            canonical.extend((i, rem) for i, rem in zip(cls, remaining))
        return tuple(sorted(canonical))

    # An operation already in flight is retained because its slot forbids an
    # overlapping copy. That does not justify STARTING a projected no-op whose
    # end lies beyond the deadline; _late_projected_noop removes that dominated
    # dispatch before it can manufacture useless rho combinations.

    def _running_mask(self, rho: Tuple[Tuple[int, int], ...]) -> int:
        m = 0
        for i, _k in rho:
            m |= 1 << i
        return m

    def _late_projected_noop(self, op: PatternOp, r: int) -> bool:
        """Return true when starting ``op`` can only occupy its running slot."""
        return op.duration > r and all(
            add == 0 and dele == 0 for add, dele, _prob in op.start_outcomes
        )

    def _startable_indices(
        self, m: int, rho: Tuple[Tuple[int, int], ...], r: int,
    ) -> List[int]:
        running = self._running_mask(rho)
        startable: List[int] = []
        for i, op in enumerate(self.p.ops):
            if running >> i & 1:
                continue
            if m & op.pre_mask != op.pre_mask:
                continue
            if self.p.conflicts[i] & running:
                continue
            if self._late_projected_noop(op, r):
                continue
            # Preconditions and goals are positive-only, and the conflict table
            # forbids an overlapping operation from deleting anything this op
            # may add. If every possible add is already true, starting the op
            # can only preserve or delete facts and occupy a slot. Waiting until
            # a later boundary if one of those facts is deleted is never worse.
            if self.add_masks[i] & ~m == 0:
                continue
            startable.append(i)
        return startable

    def _grid_delta(self, r: int) -> int:
        """Next deadline-aligned duration-GCD boundary in remaining time."""
        phase = r % self.quantum
        return phase if phase else self.quantum

    # -- value -----------------------------------------------------------
    def value(self, m: int, rho: Tuple[Tuple[int, int], ...], r: int, g: bool) -> float:
        if g:
            return 1.0                              # goal banked; absorbing
        if r <= 0:
            return 0.0                              # deadline reached without it
        rho = self._canonical_rho(rho)
        key = (m, rho, r)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        if len(self.memo) >= self.max_states or key in self.in_progress:
            # Budget exhausted, or (impossible given the lexicographic measure)
            # a cycle. Return the safe UPPER fallback but do NOT store it --
            # writing it would poison a persistent table with a value that is
            # sound yet uninformative for every later query.
            self.truncated = True
            return 1.0
        self.in_progress.add(key)

        # Same-time group construction is deliberately local. Persisting its
        # partial subsets is what made a four-fact NASA pattern fill the table
        # with non-boundary queue configurations.
        dispatch_memo: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], float] = {}
        best = self._dispatch_value(m, rho, r, dispatch_memo)

        self.in_progress.discard(key)
        self.memo[key] = best
        return best

    def _dispatch_value(
        self,
        m: int,
        rho: Tuple[Tuple[int, int], ...],
        r: int,
        local_memo: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], float],
    ) -> float:
        """Optimise one same-time dispatch group without adding PDB rows."""
        if (m & self.p.goal_mask) == self.p.goal_mask:
            return 1.0
        rho = self._canonical_rho(rho)
        key = (m, rho)
        hit = local_memo.get(key)
        if hit is not None:
            return hit
        if len(local_memo) >= self.max_states:
            self.truncated = True
            return 1.0
        self.dispatch_states_evaluated += 1

        # Stop adding operations and advance to a canonical boundary.
        best = self._advance_value(m, rho, r)

        # Or add one more operation at this same time. Start outcomes are
        # observed immediately, so later choices in the group may adapt to them.
        for i in self._startable_indices(m, rho, r):
            op = self.p.ops[i]
            total = 0.0
            for add, dele, prob in op.start_outcomes:
                m2 = self._apply(m, add, dele)
                if (m2 & self.p.goal_mask) == self.p.goal_mask:
                    total += prob
                    continue
                rho2 = self._canonical_rho(rho + ((i, op.duration),))
                total += prob * self._dispatch_value(m2, rho2, r, local_memo)
            if total > best:
                best = total

        local_memo[key] = best
        return best

    def _advance_value(
        self, m: int, rho: Tuple[Tuple[int, int], ...], r: int,
    ) -> float:
        """Advance to the next completion or required decision-grid boundary."""
        if r <= 0:
            return 0.0
        next_completion = min((k for _i, k in rho), default=None)
        has_delayed_choice = bool(self._startable_indices(m, rho, r))

        if has_delayed_choice:
            delta = self._grid_delta(r)
            if next_completion is not None:
                delta = min(delta, next_completion)
        elif next_completion is not None:
            delta = next_completion
        else:
            return 0.0

        if delta <= 0 or delta > r:
            return 0.0
        finishing = [i for i, k in rho if k == delta]
        rest = self._canonical_rho(tuple((i, k - delta) for i, k in rho if k != delta))
        r2 = r - delta
        if finishing:
            return self._expand_completions(finishing, 0, m, rest, r2, 1.0)
        return self.value(m, rest, r2, False)

    def _expand_completions(
        self, finishing: List[int], idx: int, m: int,
        rest: Tuple[Tuple[int, int], ...], r: int, weight: float,
    ) -> float:
        """Fold the joint outcome distribution of simultaneously ending ops.

        Distinct operations draw independent coins; each op's own outcomes stay
        jointly distributed. Effects are applied in sequence, which is sound
        because ops with contradictory effects are forbidden from overlapping.
        """
        if weight <= _EPS:
            return 0.0
        if idx >= len(finishing):
            g = (m & self.p.goal_mask) == self.p.goal_mask
            return self.value(m, rest, r, g)
        op = self.p.ops[finishing[idx]]
        total = 0.0
        for add, dele, prob in op.end_outcomes:
            if prob <= _EPS:
                continue
            m2 = self._apply(m, add, dele)
            total += prob * self._expand_completions(
                finishing, idx + 1, m2, rest, r, weight * prob
            )
        return total


def solve_pattern(
    pattern: Pattern, initial_facts: Set[Fact], remaining: int,
    max_states: int = DEFAULT_MAX_STATES,
    solver: Optional[PatternSolver] = None,
) -> float:
    """Optimal probability of achieving the pattern goal within ``remaining``.

    Pass a long-lived ``solver`` to reuse the pattern database across calls; a
    fresh one is created only when the caller genuinely wants a cold table
    (tests, one-off analysis).
    """
    index = {f: i for i, f in enumerate(pattern.facts)}
    m0 = _mask_of(initial_facts, index)
    if (m0 & pattern.goal_mask) == pattern.goal_mask:
        return 1.0
    if solver is None:
        solver = PatternSolver(pattern, max_states=max_states)
    if solver.sequential:
        return _clamp01(solver.value_sequential(m0, max(0, int(remaining)), False))
    return _clamp01(solver.value(m0, (), max(0, int(remaining)), False))


# ---------------------------------------------------------------------------
# Pattern construction
# ---------------------------------------------------------------------------

def _op_from_action(
    action, index: Mapping[Fact, int], fact_set: Set[Fact],
    probe_universe: Set[Fact], duration: int, pre_facts: FrozenSet,
    end_action=None,
) -> Optional[PatternOp]:
    start_raw = best_joint_outcomes(action, probe_universe, fact_set)
    end_raw = (
        best_joint_outcomes(end_action, probe_universe, fact_set)
        if end_action is not None else [(frozenset(), frozenset(), 1.0)]
    )

    def project(raw: List[Outcome]) -> Tuple[Tuple[int, int, float], ...]:
        folded: Dict[Tuple[int, int], float] = {}
        for adds, dels, p in raw:
            key = (_mask_of(adds, index), _mask_of(dels, index))
            folded[key] = folded.get(key, 0.0) + p
        return tuple((a, d, p) for (a, d), p in folded.items() if p > _EPS)

    start_out = project(start_raw)
    end_out = project(end_raw)

    if HOLD_WINDOWS:
        # "I can hold it without conditions": a WINDOW fact is one this op adds
        # at its own start and removes at its own end (light_match: light on at
        # start, out at end). Dropping that end-delete is a delete relaxation --
        # optimistic, so admissible -- and it removes the only reason a consumer
        # had to OVERLAP the provider, which is what generates in-flight states.
        #
        # It is NOT the same as dropping a CONSUMABLE. has_pin is deleted at end
        # but never added at start, so it fails the test and its delete stays --
        # which is what makes the pin plateau instead of saturating.
        opened = 0
        for add, _d, _p in start_out:
            opened |= add
        if opened:
            folded: Dict[Tuple[int, int], float] = {}
            for a, d, p in end_out:
                key = (a, d & ~opened)
                folded[key] = folded.get(key, 0.0) + p
            end_out = tuple((a, d, p) for (a, d), p in folded.items())
    touches_start = any(a or d for a, d, _ in start_out)
    touches_end = any(a or d for a, d, _ in end_out)
    if not touches_start and not touches_end:
        return None                                  # no-op on Phi: safe to drop

    footprint = 0
    for a, d, _ in start_out + end_out:
        footprint |= a | d
    return PatternOp(
        name=getattr(action, "name", repr(action)),
        duration=max(1, int(duration)),
        pre_mask=_mask_of(pre_facts & fact_set, index),
        start_outcomes=start_out,
        end_outcomes=end_out,
        touches_end=touches_end,
        footprint=footprint,
    )


def _add_mask(op: PatternOp) -> int:
    m = 0
    for a, _d, _p in op.start_outcomes + op.end_outcomes:
        m |= a
    return m


def _relevant_op_indices(ops: Sequence[PatternOp], goal_mask: int) -> List[int]:
    adds = [_add_mask(op) for op in ops]
    relevant = goal_mask
    changed = True
    while changed:
        changed = False
        for i, op in enumerate(ops):
            if adds[i] & relevant and op.pre_mask & ~relevant:
                relevant |= op.pre_mask
                changed = True
    return [i for i, op in enumerate(ops) if adds[i] & relevant]


def relevant_ops(ops: Sequence[PatternOp], goal_mask: int) -> List[PatternOp]:
    """Backward regression from the goal -- keep only ops that can contribute.

        R <- {g}
        repeat:  R <- R + pre(op)  for every op adding some fact in R
        keep op  iff  adds(op) & R

    An op whose adds miss ``R`` can only act on the pattern by producing facts
    nothing needs, or by deleting. Preconditions here are positive-only, so more
    facts is always weakly better and neither route can raise the value -- a
    maximising policy would simply never start it. Dropping it is therefore
    EXACT, not merely optimistic: it removes an action that is never chosen.

    This is emphatically NOT the OR-drop that breaks admissibility. That one is
    dropping ONE ACHIEVER of a fact the pattern tracks while keeping the fact,
    which makes the fact look harder to reach and pushes the bound below the
    truth. Here every achiever of every relevant fact is kept; what goes away
    are ops achieving nothing any relevant op requires.

    Worth it because ``|rho|`` grows as ``prod (d_a + 1)`` over exactly this op
    set, so pruning ops collapses the state space multiplicatively rather than
    shaving it.
    """
    return [ops[i] for i in _relevant_op_indices(ops, goal_mask)]


def _conflict_table(ops: Sequence[PatternOp]) -> Tuple[int, ...]:
    """Ops whose potential Phi-effects contradict may not overlap.

    This is the paper's third mutex clause ("contradictory potential effects").
    Enforcing it removes only behaviour that is already illegal in the real
    model, so it tightens toward the truth and cannot break the upper bound.
    """
    adds = []
    dels = []
    for op in ops:
        a = d = 0
        for x, y, _ in op.start_outcomes + op.end_outcomes:
            a |= x
            d |= y
        adds.append(a)
        dels.append(d)
    table = []
    for i in range(len(ops)):
        bits = 0
        for j in range(len(ops)):
            if i != j and ((adds[i] & dels[j]) or (dels[i] & adds[j])):
                bits |= 1 << j
        table.append(bits)
    return tuple(table)


def _execution_conflict_table(start_actions: Sequence[object]) -> Tuple[int, ...]:
    """Static overlap mutexes encoded by converted ``inExecution`` facts.

    ``convert_problem`` adds an action's execution marker at start and places
    that marker in every action that may not overlap it as a negative start
    precondition. These are genuine concrete-domain mutexes, not assumptions
    about an untracked resource state, so retaining them in a projection is an
    exact tightening and preserves the optimistic upper bound.
    """
    own_markers: List[Set[Fact]] = []
    blocked_markers: List[Set[Fact]] = []
    for action in start_actions:
        markers = set()
        for fact in getattr(action, "add_effects", set()) or set():
            try:
                if fact.is_fluent_exp() and fact.fluent().name == "inExecution":
                    markers.add(fact)
            except Exception:
                continue
        own_markers.append(markers)
        blocked_markers.append(set(getattr(action, "neg_preconditions", set()) or set()))

    table = [0] * len(start_actions)
    for i in range(len(start_actions)):
        for j in range(i + 1, len(start_actions)):
            if ((own_markers[i] & blocked_markers[j])
                    or (own_markers[j] & blocked_markers[i])):
                table[i] |= 1 << j
                table[j] |= 1 << i
    return tuple(table)


def build_pattern(
    seed_goal: Fact,
    *,
    durative_ops: Sequence[Tuple[object, object, int, FrozenSet, FrozenSet]],
    initial_facts: Set[Fact],
    probe_universe: Set[Fact],
    max_facts: int,
) -> Optional[Pattern]:
    """Grow ``Phi`` backwards from ``seed_goal``, then freeze the operations.

    ``durative_ops`` entries are ``(start_action, end_action, duration,
    pre_facts, effect_facts)``.

    Growth is structural, not simulation-driven: the blockers are exactly the
    preconditions the projection dropped. Candidates are ranked by achiever
    count ascending -- a fact with one achiever genuinely bottlenecks, a fact
    with six is nearly free even once tracked -- and facts that can never fail
    (true initially and deleted by nobody) are skipped as wasted slots.
    """
    achievers: Dict[Fact, int] = {}
    deleted_anywhere: Set[Fact] = set()
    for _sa, _ea, _d, _pre, eff in durative_ops:
        for f in eff:
            achievers[f] = achievers.get(f, 0) + 1
    for _sa, _ea, _d, _pre, eff in durative_ops:
        deleted_anywhere.update(eff)

    phi: List[Fact] = [seed_goal]
    while len(phi) < max_facts:
        phi_set = set(phi)
        candidates: Set[Fact] = set()
        for _sa, _ea, _d, pre, eff in durative_ops:
            if not (eff & phi_set):
                continue                              # only ops touching Phi matter
            for c in pre:
                if c in phi_set:
                    continue
                if c in initial_facts and c not in deleted_anywhere:
                    continue                          # can never fail: wasted slot
                candidates.add(c)
        if not candidates:
            break
        phi.append(min(candidates, key=lambda c: (achievers.get(c, 0), repr(c))))

    facts = tuple(phi)
    fact_set = set(facts)
    index = {f: i for i, f in enumerate(facts)}

    ops: List[PatternOp] = []
    source_starts: List[object] = []
    for sa, ea, dur, pre, eff in durative_ops:
        if not (eff & fact_set):
            continue                                  # no-op on Phi
        op = _op_from_action(sa, index, fact_set, probe_universe, dur, pre, ea)
        if op is not None:
            ops.append(op)
            source_starts.append(sa)

    goal_mask = _mask_of({seed_goal}, index)
    relevant = _relevant_op_indices(ops, goal_mask)
    ops = [ops[i] for i in relevant]
    source_starts = [source_starts[i] for i in relevant]
    if not ops:
        return None
    effect_conflicts = _conflict_table(ops)
    execution_conflicts = _execution_conflict_table(source_starts)
    return Pattern(
        facts=facts,
        seed_goal=seed_goal,
        ops=tuple(ops),
        goal_mask=goal_mask,
        conflicts=tuple(a | b for a, b in zip(effect_conflicts, execution_conflicts)),
    )


# ---------------------------------------------------------------------------
# Heuristic entry point
# ---------------------------------------------------------------------------

def _extract_state_facts(state) -> Set[Fact]:
    preds = getattr(state, "predicates", None)
    if preds is None and isinstance(state, (set, frozenset)):
        return set(state)
    return set(preds or set())


class ExactPatternMDPHeuristic:
    """Admissible upper bound from exact small-pattern CoMDP+ MDPs.

    Usage mirrors the PTRPG heuristics so the MCTS driver can swap it in:
    ``from_problem(problem)`` then ``heuristic_score(state, goals, ...)``.
    """

    def __init__(self, actions, goal_facts: Set[Fact], initial_facts: Set[Fact]):
        self._actions = list(actions or [])
        self._goal_facts = set(goal_facts or set())
        self._initial_facts = set(initial_facts or set())
        self._max_facts = max(1, _env_int("TP_MCTS_EXACT_PDB_MAX_FACTS", DEFAULT_MAX_FACTS))
        self._max_patterns = max(0, _env_int("TP_MCTS_EXACT_PDB_MAX_PATTERNS", DEFAULT_MAX_PATTERNS))
        self._max_states = max(1000, _env_int("TP_MCTS_EXACT_PDB_MAX_STATES", DEFAULT_MAX_STATES))
        self._pattern_cache: Dict[Tuple, List[Pattern]] = {}
        self._solve_memo: Dict[Tuple, float] = {}
        # id(pattern) -> PatternSolver. Persistent: the solver's memo is the
        # pattern database and is amortised over the whole search.
        self._solvers: Dict[int, "PatternSolver"] = {}
        self._durative_ops = None
        self._probe_universe = None
        self.patterns_used = 0

    @classmethod
    def from_problem(cls, problem) -> "ExactPatternMDPHeuristic":
        initial = set()
        for fact, value in getattr(problem, "initial_values", {}).items():
            try:
                if value.bool_constant_value():
                    initial.add(fact)
            except Exception:
                if bool(value):
                    initial.add(fact)
        goals = set(getattr(problem, "goals", set()))
        return cls(getattr(problem, "actions", []), goal_facts=goals, initial_facts=initial)

    # -- model extraction -------------------------------------------------
    @staticmethod
    def _duration_of(action) -> int:
        for getter in (
            lambda a: int(a.duration_int()),
            lambda a: int(a.duration.lower.int_constant_value()),
        ):
            try:
                return max(1, getter(action))
            except Exception:
                continue
        return 1

    def _build_durative_ops(self):
        """Pair snap actions back into durative operations.

        ``convert_problem`` splits every durative action into ``start_X`` /
        ``end_X`` (the paper's offline preprocessing). Recombining them is what
        lets this model keep start effects and end effects at their real times
        instead of collapsing both into one layer.
        """
        if self._durative_ops is not None:
            return self._durative_ops
        ops = []
        seen_end = set()
        for action in self._actions:
            if hasattr(action, "actions") and not hasattr(action, "add_effects"):
                continue                                    # CombinationAction wrapper
            end_action = getattr(action, "end_action", None)
            start_action = getattr(action, "start_action", None)
            if end_action is not None:                      # this is a start action
                seen_end.add(id(end_action))
                pre = frozenset(getattr(action, "pos_preconditions", set()) or set())
                eff = self._effect_facts(action) | self._effect_facts(end_action)
                ops.append((action, end_action, self._duration_of(action), pre, eff))
            elif start_action is None:                      # plain instantaneous
                pre = frozenset(getattr(action, "pos_preconditions", set()) or set())
                eff = self._effect_facts(action)
                ops.append((action, None, 1, pre, eff))
        ops = [o for o in ops if id(o[0]) not in seen_end]
        self._durative_ops = ops
        return ops

    @staticmethod
    def _effect_facts(action) -> Set[Fact]:
        out = set(getattr(action, "add_effects", set()) or set())
        out |= set(getattr(action, "del_effects", set()) or set())
        for pe in getattr(action, "probabilistic_effects", []) or []:
            out.update(getattr(pe, "fluents", []) or [])
        return out

    def _probe_facts(self) -> Set[Fact]:
        if self._probe_universe is not None:
            return self._probe_universe
        universe = set(self._initial_facts) | set(self._goal_facts)
        for action in self._actions:
            universe |= set(getattr(action, "pos_preconditions", set()) or set())
            universe |= set(getattr(action, "neg_preconditions", set()) or set())
            universe |= self._effect_facts(action)
        self._probe_universe = universe
        return universe

    # -- patterns ---------------------------------------------------------
    def patterns_for(self, goal_facts: Sequence[Fact]) -> List[Pattern]:
        targets = [g for g in goal_facts]
        key = (frozenset(targets), self._max_facts)
        cached = self._pattern_cache.get(key)
        if cached is not None:
            return cached
        ops = self._build_durative_ops()
        probe = self._probe_facts()
        cap = self._max_patterns or min(len(targets), PATTERN_HARD_CAP)
        patterns: List[Pattern] = []
        for goal in targets[:max(1, cap)]:
            pat = build_pattern(
                goal,
                durative_ops=ops,
                initial_facts=self._initial_facts,
                probe_universe=probe,
                max_facts=self._max_facts,
            )
            if pat is not None:
                patterns.append(pat)
        self._pattern_cache[key] = patterns
        return patterns

    # -- scoring ----------------------------------------------------------
    def heuristic_score(
        self, state, goal_facts: Iterable[Fact], fixed_depth: int = 25,
        start_time: float = 0.0, **_ignored,
    ) -> float:
        """Admissible upper bound on P(goal before deadline) from ``state``.

        Goals are combined with MIN, not a product. For upper bounds ``U``,
        ``min(U(A), U(B)) >= min(P(A), P(B)) >= P(A and B)`` always holds, while
        ``U(A)*U(B)`` can fall BELOW ``P(A and B)`` when the goals are positively
        correlated. A goal no pattern covers contributes 1.0.
        """
        facts = _extract_state_facts(state)
        goals = [g for g in goal_facts]
        remaining = max(0, int(fixed_depth))
        unmet = [g for g in goals if g not in facts]
        if not unmet:
            return 1.0

        patterns = self.patterns_for(unmet)
        self.patterns_used = len(patterns)
        if not patterns:
            return 1.0

        covered: Dict[Fact, float] = {}
        for pat in patterns:
            # One long-lived solver per pattern: its memo IS the pattern
            # database, so it must survive across nodes. The outer cache below
            # only short-circuits the exact same query; the solver is what makes
            # every OTHER query cheap, because search nodes overwhelmingly share
            # abstract sub-states even when their concrete states differ.
            solver = self._solvers.get(id(pat))
            if solver is None:
                solver = PatternSolver(pat, max_states=self._max_states)
                self._solvers[id(pat)] = solver
            projection = frozenset(f for f in pat.facts if f in facts)
            key = (id(pat), projection, remaining)
            val = self._solve_memo.get(key)
            if val is None:
                val = solve_pattern(pat, facts, remaining, solver=solver)
                self._solve_memo[key] = val
            prev = covered.get(pat.seed_goal)
            covered[pat.seed_goal] = val if prev is None else min(prev, val)

        return _clamp01(min((covered.get(g, 1.0) for g in unmet), default=1.0))
