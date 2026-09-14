"""Temporal PDB with backward critical-path values -- first version, UPPER BOUND.

Spec: ``Temporal_PDB_Backward_Critical_Path5.docx``, sections 3-8.

    V_s = { (T, tails) : p }        V_s(r) = max{ p : max(T, max_a(r_a + tail_a)) <= r }

* Section 3, built FORWARD over ``s = (F, Q)``: choice node (start a / noop),
  interval node (one branch per closed gap of Q), outcome node (first end of Q
  fires, one branch per outcome). No deadline in the build.
* Sections 5-7, computed BACKWARD:
      regress E_x : tail_x = T
      regress S_a : T = max(T, d(a) + tail_a),  a leaves the tails
  ``tail_x = T`` is the doc's group rule: T only changes at a start, so ends
  regressed with no start between them get the same tail.
  Outcome node = pointwise sum (one entry per outcome, or none), choice and
  interval node = union, dominance on (T, tails, p). Cycles are iterated to a
  fixed point; entries with ``T > horizon`` are dropped.
* Section 8, lookup: the query's running actions ordered by remaining time.

Why the value is an UPPER bound
-------------------------------
1. The interval node's STN sees only what ``(F, Q)`` knows: each running end lies
   in ``[now, now + d]``. A gap reachable from SOME history is kept.
2. At an outcome node each branch's value is taken as it is, with the remaining
   time; the STN of the path that reached the split is never cross-checked
   against the branch's own paths.
Both only ADD schedules, so ``V >= V*``. Known loose case: retries of a short
action inside a running one are capped by the deadline, not by the running
action (see the test that documents it).

Not specified by the doc, handled explicitly
--------------------------------------------
* Zero-duration actions: T does not grow around their loops, so the iteration
  stops once no probability improves by more than ``p_tol`` (unlimited-retry
  limit). This is one of the open options, chosen only so the code terminates.
* Budget: states past ``max_states`` stay unexpanded and get the vacuous entry
  ``{0: 1.0}``, which keeps the upper bound; ``complete`` reports it.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from comdp_plus_no_deadline.engines.strict_dbm import StrictDBM

Fact = Hashable
NO_TAIL = -1                                 # this end is not needed before the goal
State = Tuple[FrozenSet, Tuple[str, ...]]    # (F, Q)
Branches = Tuple[Tuple[float, State], ...]

DEFAULT_MAX_STATES = 50_000
DEFAULT_P_TOL = 1e-9
DEFAULT_MAX_UPDATES = 2_000_000
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    """One pair of section 4 with its open tails, aligned with the state's Q.

    Times are integers in units of ``1/CriticalPathPDB.scale``. A tail is
    ``NO_TAIL`` (-1) when that end need not happen before the goal; real tails
    are >= 0, so ``<=`` and ``max`` on tails need no special case.
    """

    T: int
    tails: Tuple[int, ...]
    p: float


def dominates(a: Entry, b: Entry, p_tol: float = 0.0) -> bool:
    """Section 10: dominance compares the tails as well as (T, p)."""
    return a.T <= b.T and a.p >= b.p - p_tol and all(map(operator.le, a.tails, b.tails))


def prune(entries: Iterable[Entry]) -> List[Entry]:
    """Keep the non-dominated entries.

    Sorted by (T, -p, sum of tails), every dominator of an entry comes before
    it, so each entry is checked against the kept ones only -- and every kept
    one already has ``T <=``.
    """
    ordered = sorted(set(entries), key=lambda e: (e.T, -e.p, sum(e.tails)))
    kept: List[Entry] = []
    le = operator.le
    for e in ordered:
        p, tails = e.p, e.tails
        for k in kept:
            if k.p >= p and all(map(le, k.tails, tails)):
                break
        else:
            kept.append(e)
    return kept


def usable(entry: Entry, remaining: Sequence, r, scale: int = 1) -> bool:
    """Section 8's time filter: ``max(T, max_a(r_a + tail_a)) <= r``.

    Entry times are in units of ``1/scale``; ``remaining`` and ``r`` are not.
    """
    need = Fraction(entry.T)
    for r_a, tail in zip(remaining, entry.tails):
        if tail != NO_TAIL:
            need = max(need, Fraction(r_a) * scale + tail)
    return need <= Fraction(r) * scale


def gap_is_consistent(queue_durations: Sequence, duration, gap: int) -> bool:
    """Section 3's interval-node STN, restricted to what ``(F, Q)`` knows.

    ``now <= E_i <= now + d_i`` (started at or before now), ends in Q order,
    ``now <= S_a <= E_1``, ``E_a - S_a = d(a)``, ``E_(gap) <= E_a <= E_(gap+1)``.
    """
    z = StrictDBM("now")
    ends = [f"E{i}" for i in range(len(queue_durations))]
    ok = True
    for i, d in enumerate(queue_durations):
        ok = ok and z.add_le("now", ends[i], 0) and z.add_le(ends[i], "now", d)
        if i:
            ok = ok and z.add_le(ends[i - 1], ends[i], 0)
    ok = ok and z.add_le("now", "S", 0) and z.add_eq("Ea", "S", duration)
    if ends:
        ok = ok and z.add_le("S", ends[0], 0)
    if gap > 0:
        ok = ok and z.add_le(ends[gap - 1], "Ea", 0)
    if gap < len(ends):
        ok = ok and z.add_le("Ea", ends[gap], 0)
    return bool(ok) and z.consistent


# ---------------------------------------------------------------------------
# The PDB
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("goal", "expanded", "starts", "noop")

    def __init__(self, goal: bool):
        self.goal = goal
        self.expanded = goal
        # (op key, gap or None for an instantaneous op, outcome branches)
        self.starts: List[Tuple[str, Optional[int], Branches]] = []
        self.noop: Optional[Branches] = None


class CriticalPathPDB:
    """Table ``(F, Q) -> [Entry]`` for one model (a pattern) and one horizon.

    ``model`` is an ``EngineModel`` or ``ToyModel`` from ``temporal_stn_pdb``.
    The table grows across lookups; one build answers every ``r <= horizon``.
    """

    def __init__(
        self,
        model,
        horizon,
        *,
        max_states: int = DEFAULT_MAX_STATES,
        p_tol: float = DEFAULT_P_TOL,
        max_updates: int = DEFAULT_MAX_UPDATES,
    ):
        self.model = model
        self.horizon = Fraction(horizon)
        self.max_states = max(1, int(max_states))
        self.p_tol = float(p_tol)
        self.max_updates = max(1, int(max_updates))
        self.nodes: Dict[State, _Node] = {}
        self.preds: Dict[State, Set[State]] = {}
        self.entries: Dict[State, List[Entry]] = {}
        self.expanded_count = 0
        self.updates = 0
        self.converged = True
        self._frontier: deque = deque()
        self._dirty = False
        self._gap_cache: Dict[Tuple, bool] = {}
        # Entries keep T and tails as integers in units of 1/scale: comparing
        # Fractions was 97% of the solve time on prob_conc.
        denominators = [self.horizon.denominator]
        denominators += [Fraction(op.duration).denominator for op in model.ops().values()]
        self.scale = math.lcm(*denominators)
        self._horizon_units = int(self.horizon * self.scale)
        self._units = {k: int(Fraction(op.duration) * self.scale) for k, op in model.ops().items()}

    # -- lookup (section 8) --------------------------------------------------
    def value(self, facts: Iterable[Fact], r, running: Sequence[Tuple[str, object]] = ()) -> float:
        """``max p`` over usable entries. ``running`` is ``[(op key, remaining)]``."""
        if Fraction(r) > self.horizon:
            raise ValueError(f"r={r} is past the horizon {self.horizon}; entries with T > horizon were dropped")
        facts = frozenset(facts)
        if self.model.goals <= facts:
            return 1.0
        roots = self.root_states(facts, running)
        self.build(state for state, _ in roots)
        self.solve()
        return max((self.state_value(state, r, remaining) for state, remaining in roots), default=0.0)

    def state_value(self, state: State, r, remaining: Sequence = ()) -> float:
        best = 0.0
        for e in self.entries.get(state, ()):
            if e.p > best and usable(e, remaining, r, self.scale):
                best = e.p
        return best

    def root_states(self, facts: FrozenSet, running: Sequence[Tuple[str, object]] = ()):
        """Q ordered by remaining time. Tied ends may be processed in either order
        (closed gaps), so every tie order is a candidate root."""
        ops = self.model.ops()
        known = sorted((Fraction(rem), key) for key, rem in running if key in ops)
        groups = [list(g) for _, g in itertools.groupby(known, key=lambda x: x[0])]
        roots = {}
        for combo in itertools.product(*(itertools.permutations(g) for g in groups)):
            queue = tuple(key for group in combo for _, key in group)
            remaining = tuple(rem for group in combo for rem, _ in group)
            roots[(frozenset(facts), queue)] = remaining
        return list(roots.items())

    # -- build (section 3) ---------------------------------------------------
    def build(self, roots: Iterable[State]) -> None:
        for state in roots:
            if state not in self.nodes:
                self._add(state)
        while self._frontier:
            state = self._frontier.popleft()
            node = self.nodes[state]
            if node.expanded or self.expanded_count >= self.max_states:
                continue                      # unexpanded: vacuous upper-bound entry
            self._expand(state, node)

    def _add(self, state: State) -> None:
        node = _Node(goal=self.model.goals <= state[0])
        self.nodes[state] = node
        self.preds.setdefault(state, set())
        self._dirty = True
        if not node.goal:
            self._frontier.append(state)

    def _expand(self, state: State, node: _Node) -> None:
        facts, queue = state
        ops = self.model.ops()
        for name in self.model.legal_action_names(facts):
            op = ops.get(name)
            if op is None:
                continue
            outs = self._outcomes(op, "S", facts)
            if op.end_action is None:
                node.starts.append((name, None, self._link(state, outs, queue)))
                continue
            durations = tuple(ops[k].duration for k in queue)
            for gap in range(len(queue) + 1):
                if self._gap_ok(durations, op.duration, gap):
                    new_queue = queue[:gap] + (name,) + queue[gap:]
                    node.starts.append((name, gap, self._link(state, outs, new_queue)))
        if queue:
            op = ops.get(queue[0])
            if op is not None and op.end_action is not None and self._end_legal(op, facts):
                node.noop = self._link(state, self._outcomes(op, "E", facts), queue[1:])
        node.expanded = True
        self.expanded_count += 1
        self._dirty = True

    def _link(self, parent: State, outcomes, queue: Tuple[str, ...]) -> Branches:
        out = []
        for facts, p in outcomes:
            if p <= _EPS:
                continue
            child = (facts, queue)
            if child not in self.nodes:
                self._add(child)
            self.preds[child].add(parent)
            out.append((float(p), child))
        return tuple(out)

    def _gap_ok(self, durations: Tuple, duration, gap: int) -> bool:
        key = (durations, duration, gap)
        hit = self._gap_cache.get(key)
        if hit is None:
            hit = self._gap_cache[key] = gap_is_consistent(durations, duration, gap)
        return hit

    def _outcomes(self, op, which: str, facts: FrozenSet):
        if isinstance(op.start_action, str):          # ToyModel
            action = (op.key, which)
        else:
            action = op.start_action if which == "S" else op.end_action
        return self.model.outcomes(action, facts)

    def _end_legal(self, op, facts: FrozenSet) -> bool:
        if isinstance(op.start_action, str):          # ToyModel
            return self.model.end_legal(op.key, facts)
        action = op.end_action
        return action.pos_preconditions.issubset(facts) and action.neg_preconditions.isdisjoint(facts)

    # -- values (sections 5-7) -----------------------------------------------
    def solve(self) -> None:
        if not self._dirty:
            return
        self.entries = {}
        for state, node in self.nodes.items():
            k = len(state[1])
            vacuous = node.goal or not node.expanded
            self.entries[state] = [Entry(0, (NO_TAIL,) * k, 1.0)] if vacuous else []
        # Leaves first: reversed discovery order visits children before parents.
        work = deque(s for s, n in reversed(list(self.nodes.items())) if n.expanded and not n.goal)
        queued = set(work)
        self.updates = 0
        self.converged = True
        while work:
            if self.updates >= self.max_updates:
                self.converged = False
                break
            state = work.popleft()
            queued.discard(state)
            new = self._backup(state)
            old = self.entries[state]
            if all(any(dominates(o, e, self.p_tol) for o in old) for e in new):
                continue
            self.entries[state] = new
            self.updates += 1
            for parent in self.preds[state]:
                if parent not in queued:
                    work.append(parent)
                    queued.add(parent)
        self._dirty = False

    def _backup(self, state: State) -> List[Entry]:
        node = self.nodes[state]
        k = len(state[1])
        candidates: List[Entry] = []
        for key, gap, branches in node.starts:
            d = self._units[key]
            candidates.extend(self._outcome_sum(k, [
                (p, [self._regress_start(e, gap, d) for e in self.entries[child]])
                for p, child in branches
            ]))
        if node.noop is not None:
            candidates.extend(self._outcome_sum(k, [
                (p, [Entry(e.T, (e.T,) + e.tails, e.p) for e in self.entries[child]])
                for p, child in node.noop
            ]))
        return prune(candidates)

    @staticmethod
    def _regress_start(e: Entry, gap: Optional[int], duration) -> Entry:
        if gap is None:                               # instantaneous: no end, no time
            return e
        tail = e.tails[gap]
        T = e.T if tail == NO_TAIL else max(e.T, duration + tail)
        return Entry(T, e.tails[:gap] + e.tails[gap + 1:], e.p)

    def _outcome_sum(self, k: int, per_outcome) -> List[Entry]:
        """Section 6: pointwise sum. Each outcome contributes one of its entries
        or nothing (0); the combined requirement is the max of the chosen ones."""
        horizon = self._horizon_units
        if len(per_outcome) == 1:
            p, entries = per_outcome[0]
            return prune(Entry(e.T, e.tails, p * e.p) for e in entries if e.T <= horizon and p * e.p > _EPS)
        combos = [Entry(0, (NO_TAIL,) * k, 0.0)]
        for p, entries in per_outcome:
            nxt = list(combos)
            for c in combos:
                for e in entries:
                    T = max(c.T, e.T)
                    if T <= horizon:
                        nxt.append(Entry(T, tuple(map(max, c.tails, e.tails)), c.p + p * e.p))
            combos = prune(nxt)
        return [c for c in combos if c.p > _EPS]

    # -- reporting -----------------------------------------------------------
    @property
    def complete(self) -> bool:
        return all(n.expanded for n in self.nodes.values())

    def stats(self) -> Dict[str, object]:
        sizes = [len(v) for v in self.entries.values()]
        return {
            "states": len(self.nodes),
            "expanded": self.expanded_count,
            "unexpanded": sum(1 for n in self.nodes.values() if not n.expanded),
            "edges": sum(
                sum(len(b) for _, _, b in n.starts) + len(n.noop or ())
                for n in self.nodes.values()
            ),
            "entries": sum(sizes),
            "max_entries": max(sizes, default=0),
            "updates": self.updates,
            "converged": self.converged,
            "complete": self.complete,
        }
