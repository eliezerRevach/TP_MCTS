"""ILAO* over the time-left windows MDP (``Temporal_PDB_Time_Left_Windows``).

State ``(F, Q, W, r)``:
    F  facts;  Q  running ends in committed order;
    W  (lo, hi, e) per running action:
         [lo, hi]  time left in real time, measured from the last end (order checks)
         e         time left that is still guaranteed against the time CHARGED so far (deadline)
    r  time in hand. No clock.

Choice node (max) : start a with its end placed in a gap of Q  |  let the first end fire
Outcome node (sum): the event's outcomes, p(w) * V(child)

Window rules (doc section 4):
    start x in a gap : lo_x = max(d_x, lo before x);  hi_x = min(d_x + hi of first end, hi after x);  e_x = lo_x
                       ends before x: hi = min(hi, hi_x);  ends after x: lo = max(lo, lo_x)
                       keep Q order (lo, e never decrease and hi never increases along Q);
                       drop if some lo > hi
    first end x fires: r -= e_x
                       others hi = min(hi - lo_x, d), lo = max(0, lo - hi_x), e = max(lo, e - e_x)

Why e. Charging lo_x per end loses time: an end placed before b can shrink b's
lo by its hi while r is charged only its lo. e shrinks by exactly what was
charged, so charged + e_b >= d_b holds for every running b: overlapping actions
inside a charge max(d_b, d_c), sequential ones d_b + d_c.

Pruned, never generated or never expanded:
    * a gap whose windows are inconsistent (lo > hi)           -- STN reject
    * an end with r - e_x < 0                                   -- past the deadline
    * a state with h = 0                                        -- goal unreachable in r
    * an inert start (only occupies its slot) with lo_x > r     -- dominated by not starting

Values start at ``h`` (the survivor sweep, meant as an upper bound) and are only
ever backed up from children, so ``V(root)`` is an upper bound at any moment,
also when the expansion budget runs out. ILAO* (Hansen & Zilberstein 2001):
depth-first passes over the best partial policy, expanding its tips and
backing up in post-order, until no tip is left and the residual is < epsilon.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from fractions import Fraction
from typing import Dict, FrozenSet, Hashable, List, Optional, Sequence, Tuple

from comdp_plus_no_deadline.engines.survivor_sweep import (
    SurvivorSweep,
    lookup,
    relaxed_actions_from_engine,
    relaxed_actions_from_toy,
)

Fact = Hashable
Window = Tuple[Fraction, Fraction, Fraction]          # (lo, hi, e)


def _as_fraction(value) -> Fraction:
    """Remaining times from the search STN are floats; keep keys small."""
    if isinstance(value, float):
        return Fraction(value).limit_denominator(1000)
    return Fraction(value)
State = Tuple[FrozenSet, Tuple[str, ...], Tuple[Window, ...], Fraction]
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Window rules
# ---------------------------------------------------------------------------

def normalise(windows: Sequence[Window]) -> Optional[Tuple[Window, ...]]:
    """Q order: E_1 <= E_2 <= ..., so lo and e never decrease along Q and hi
    never increases backwards. e >= lo always (charged time never runs ahead
    of real time)."""
    lo = [w[0] for w in windows]
    hi = [w[1] for w in windows]
    e = [max(w[2], w[0]) for w in windows]
    for i in range(1, len(lo)):
        lo[i] = max(lo[i], lo[i - 1])
        e[i] = max(e[i], e[i - 1], lo[i])
    for i in range(len(hi) - 2, -1, -1):
        hi[i] = min(hi[i], hi[i + 1])
    if any(l > h for l, h in zip(lo, hi)):
        return None
    return tuple(zip(lo, hi, e))


def start_windows(windows: Sequence[Window], d: Fraction, gap: int) -> Optional[Tuple[Window, ...]]:
    if not windows:
        return ((d, d, d),)
    lo_x = max(d, windows[gap - 1][0] if gap > 0 else Fraction(0))
    hi_x = d + windows[0][1]
    if gap < len(windows):
        hi_x = min(hi_x, windows[gap][1])
    e_x = max(lo_x, windows[gap - 1][2] if gap > 0 else Fraction(0))
    new = [(lo, min(hi, hi_x), e) for lo, hi, e in windows[:gap]] + [(lo_x, hi_x, e_x)] + \
          [(max(lo, lo_x), hi, max(e, e_x)) for lo, hi, e in windows[gap:]]
    return normalise(new)


def end_windows(windows: Sequence[Window], durations: Sequence[Fraction]
                ) -> Tuple[Optional[Tuple[Window, ...]], Fraction]:
    """The first end fires; returns (windows of the rest, time charged).

    Everyone still running started before it, so hi <= d. The charge is e_x,
    and e of the others shrinks by exactly that charge."""
    lo_x, hi_x, e_x = windows[0]
    rest = []
    for (lo, hi, e), d in zip(windows[1:], durations):
        new_lo = max(Fraction(0), lo - hi_x)
        rest.append((new_lo, min(hi - lo_x, d), max(new_lo, e - e_x)))
    return normalise(rest), e_x


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class WindowsLAO:
    """``model`` is an ``EngineModel`` or ``ToyModel`` from ``temporal_stn_pdb``.

    ``heuristic``: ``"sweep"`` (survivor sweep) or ``"none"`` (h = 1), the
    control that shows what the heuristic saves.
    """

    def __init__(self, model, *, heuristic: str = "sweep", epsilon: float = 1e-6,
                 max_expansions: int = 2_000_000, prune_inert: bool = True,
                 time_budget: Optional[float] = None, sweep_horizon=None,
                 relaxed_actions: Optional[list] = None):
        self.model = model
        self.heuristic = heuristic
        self.epsilon = float(epsilon)
        self.max_expansions = int(max_expansions)
        self.prune_inert = prune_inert
        # Seconds per solve() call; None = no limit. A cut solve still returns an
        # upper bound: every value is h or backed up from values >= the truth.
        self.time_budget = time_budget
        self._deadline_at: Optional[float] = None
        self._expansion_cap = self.max_expansions
        self.ops = model.ops()
        self.durations = {k: Fraction(op.duration) for k, op in self.ops.items()}
        self.V: Dict[State, float] = {}
        self.best: Dict[State, Optional[int]] = {}
        self.options: Dict[State, List[Tuple[tuple, Tuple[Tuple[float, State], ...]]]] = {}
        self.fixed: set = set()                   # goal (1) or dead (0): never expanded
        self._curves: Dict[Tuple, list] = {}
        self._relaxed: Dict[FrozenSet, list] = {}
        # One relaxed action table for every state (probabilities read once, at
        # the probe state it was built from) instead of re-reading all actions
        # through the real transition function per state.
        self._shared_relaxed = relaxed_actions
        self.first_optimal: Optional[Dict[str, float]] = None
        self._toy = any(isinstance(op.start_action, str) for op in self.ops.values())
        # Horizon the sweep curves are built to. Fixing it (e.g. to the deadline)
        # lets one curve serve every later query with a smaller r.
        self.horizon = Fraction(sweep_horizon) if sweep_horizon is not None else Fraction(0)
        self.stats = {"expansions": 0, "backups": 0, "passes": 0, "h_builds": 0,
                      "pruned_h0": 0, "pruned_gap": 0, "pruned_deadline": 0, "pruned_inert": 0,
                      "cover_used": 0}
        self.complete = True
        # States whose value is final: every state of a converged best policy.
        # Indexed by (F, Q) for the covering lookup.
        self.solved: set = set()
        self._solved_index: Dict[Tuple[FrozenSet, Tuple[str, ...]], List[State]] = {}
        self._stop_at_solved = False
        # Branches the solved policies did not take: (depth, -branch value, n, state).
        self._candidates: List[Tuple[int, float, int, State]] = []
        self._candidate_count = 0

    # -- query ---------------------------------------------------------------
    def _roots(self, facts: FrozenSet, r: Fraction, running) -> List[State]:
        """Q ordered by remaining time, windows [rem, rem, rem].

        ``running`` is ``[(op key, remaining or None)]``. Tied ends may be
        processed in either order, so every tie order is a root. If any
        remaining time is unknown, every running action gets the window [0, d]
        (e = 0) and every end order is a root: an upper bound.
        """
        entries = [(key, rem) for key, rem in running if key in self.ops]
        roots = []
        if any(rem is None for _key, rem in entries):
            for order in itertools.permutations([key for key, _rem in entries]):
                windows = normalise(tuple((Fraction(0), self.durations[k], Fraction(0)) for k in order))
                if windows is not None:
                    roots.append((facts, order, windows, r))
        else:
            known = sorted((_as_fraction(rem), key) for key, rem in entries)
            groups = [list(g) for _, g in itertools.groupby(known, key=lambda x: x[0])]
            for combo in itertools.product(*(itertools.permutations(g) for g in groups)):
                queue = tuple(key for group in combo for _, key in group)
                windows = tuple((rem, rem, rem) for group in combo for rem, _ in group)
                roots.append((facts, queue, windows, r))
        return roots

    def solve(self, facts, r, running: Sequence[Tuple[str, object]] = (), *,
              time_budget: Optional[float] = None, max_expansions: Optional[int] = None,
              keep_exploring: bool = False, lazy: bool = False) -> float:
        """ILAO* from the query, until the best policy is solved (optimal).

        The table (values, expansions, sweep curves, solved marks) is kept
        between calls. ``time_budget`` and ``max_expansions`` limit THIS call;
        a cut call still returns an upper bound. A converged root marks its best
        policy as solved.

        ``lazy``: the MCTS miss path. Solved states are not re-explored -- their
        value is used as is -- and a new state covered by a solved one takes the
        solved value (see ``lookup``).
        """
        started = time.perf_counter()
        budget = self.time_budget if time_budget is None else time_budget
        self._deadline_at = None if budget is None else started + float(budget)
        cap = self.max_expansions if max_expansions is None else int(max_expansions)
        self._expansion_cap = self.stats["expansions"] + cap
        self.complete = True
        self._stop_at_solved = lazy
        r = Fraction(r)
        self.horizon = max(self.horizon, r)
        root_states = self._roots(frozenset(facts), r, running)
        best = 0.0
        try:
            for root in root_states:
                value, converged = self._ilao(root)
                best = max(best, value)
                if converged:
                    self._mark_solved(root)
            if self.complete and self.first_optimal is None:
                self.first_optimal = {"seconds": round(time.perf_counter() - started, 3),
                                      "expansions": self.stats["expansions"], "value": best}
            if keep_exploring and self._deadline_at is not None:
                self.extend(max(0.0, self._deadline_at - time.perf_counter()))
        finally:
            self._stop_at_solved = False
        self.stats["seconds"] = round(time.perf_counter() - started, 3)
        self.stats["states"] = len(self.V)
        return best

    # -- lookup ----------------------------------------------------------------
    def lookup(self, facts, r, running: Sequence[Tuple[str, object]] = ()) -> Tuple[Optional[float], str]:
        """``(value, "exact" | "cover")`` from solved states, or ``(None, "miss")``.

        cover: a solved state with the same F and Q whose windows are at least as
        loose (lo' <= lo, hi' >= hi, e' <= e) and with at least as much time
        (r' >= r) allows every schedule the query allows, so its value is an
        upper bound for the query; with several, the smallest is used.
        Several roots (tie orders) take the max, as in ``solve``.
        """
        facts, r = frozenset(facts), Fraction(r)
        if self.model.goals <= facts:
            return 1.0, "exact"
        best, kind = None, "exact"
        for root in self._roots(facts, r, running):
            if root in self.solved or (root in self.fixed and root in self.V):
                value = self.V[root]
            else:
                value = self._cover_value(root)
                if value is None:
                    return None, "miss"
                kind = "cover"
            best = value if best is None else max(best, value)
        return (0.0 if best is None else best), kind

    def _cover_value(self, s: State) -> Optional[float]:
        facts, queue, windows, r = s
        best = None
        for other in self._solved_index.get((facts, queue), ()):
            w2, r2 = other[2], other[3]
            if r2 < r:
                continue
            if all(lo2 <= lo and hi2 >= hi and e2 <= e
                   for (lo2, hi2, e2), (lo, hi, e) in zip(w2, windows)):
                value = self.V[other]
                best = value if best is None else min(best, value)
        return best

    def _mark_solved(self, root: State, level: int = 0) -> None:
        """Mark the converged best policy below ``root`` as solved, and queue the
        branches it did NOT take as candidates for ``extend``: (depth from the
        offline root, higher branch value first)."""
        stack, seen = [(root, level)], set()
        while stack:
            s, depth = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            if s not in self.solved:
                if s in self.fixed or s in self.options:
                    self.solved.add(s)
                    self._solved_index.setdefault((s[0], s[1]), []).append(s)
            if s in self.fixed or s not in self.options:
                continue
            b = self.best.get(s)
            for k, (_label, branches) in enumerate(self.options[s]):
                if k == b:
                    continue
                q = sum(p * self.V[c] for p, c in branches)
                for _p, child in branches:
                    if child not in self.solved:
                        self._candidate_count += 1
                        heapq.heappush(self._candidates, (depth + 1, -q, self._candidate_count, child))
            if b is not None:
                stack.extend((child, depth + 1) for _p, child in self.options[s][b][1])

    def extend(self, time_budget: float) -> bool:
        """Offline, after the optimal policy: solve the branches it did not take.

        Picks the unsolved child of a solved state closest to the root (higher
        branch value first), runs ILAO* from it -- stopping at solved states --
        until optimal, marks it solved, and repeats until ``time_budget`` runs
        out. Every state solved here is one fewer lazy solve during MCTS.
        Returns True while candidates remain.
        """
        started = time.perf_counter()
        self._deadline_at = started + float(time_budget)
        self._expansion_cap = self.stats["expansions"] + 10 ** 12
        self._stop_at_solved = True
        before = len(self.solved)
        try:
            while self._candidates and not self._out_of_budget():
                depth, neg_q, count, s = heapq.heappop(self._candidates)
                if s in self.solved:
                    continue
                value, converged = self._ilao(s)
                if converged:
                    self._mark_solved(s, depth)
                    self.stats["extend_solves"] = self.stats.get("extend_solves", 0) + 1
                else:
                    heapq.heappush(self._candidates, (depth, neg_q, count, s))   # retry next slice
                    break
        finally:
            self._stop_at_solved = False
        self.stats["extend_seconds"] = round(self.stats.get("extend_seconds", 0.0)
                                            + time.perf_counter() - started, 3)
        self.stats["extend_new_solved"] = self.stats.get("extend_new_solved", 0) + len(self.solved) - before
        return bool(self._candidates)

    def _out_of_budget(self) -> bool:
        if self.stats["expansions"] >= self._expansion_cap:
            return True
        return self._deadline_at is not None and time.perf_counter() >= self._deadline_at

    # -- ILAO* ---------------------------------------------------------------
    def _ilao(self, root: State) -> Tuple[float, bool]:
        """``(V(root), converged)``."""
        self._init(root)
        while True:
            if self._out_of_budget():
                self.complete = False
                return self.V[root], False
            expanded, residual = self._pass(root)
            self.stats["passes"] += 1
            if expanded == 0 and residual < self.epsilon:
                return self.V[root], True

    def _pass(self, root: State) -> Tuple[int, float]:
        """One depth-first pass over the best partial policy: expand its tips,
        back up every visited state in post-order."""
        expanded, residual = 0, 0.0
        visited = set()
        stack = [(root, False)]
        while stack:
            s, done = stack.pop()
            if done:
                residual = max(residual, self._backup(s))
                continue
            if s in visited or s in self.fixed:
                continue
            if self._stop_at_solved and s in self.solved and s != root:
                continue                                # value already final: no re-exploring
            visited.add(s)
            if s not in self.options:
                if self._out_of_budget():
                    continue
                self._expand(s)
                expanded += 1
                stack.append((s, True))             # new children are next pass's tips
                continue
            stack.append((s, True))
            b = self.best.get(s)
            if b is not None:
                for _p, child in self.options[s][b][1]:
                    if child not in visited:
                        stack.append((child, False))
        return expanded, residual

    def _backup(self, s: State) -> float:
        old = self.V[s]
        best_value, best_index = 0.0, None
        for k, (_label, branches) in enumerate(self.options.get(s, ())):
            q = sum(p * self.V[c] for p, c in branches)
            if q > best_value + _EPS:
                best_value, best_index = q, k
        self.V[s] = best_value
        self.best[s] = best_index
        self.stats["backups"] += 1
        return abs(old - best_value)

    def _init(self, s: State) -> None:
        if s in self.V:
            return
        facts, queue, windows, r = s
        if self.model.goals <= facts:
            self.V[s] = 1.0
            self.fixed.add(s)
            return
        if self._stop_at_solved:
            covered = self._cover_value(s)
            if covered is not None:
                self.V[s] = covered                     # a solved state covers it: take its value
                self.fixed.add(s)
                self.stats["cover_used"] += 1
                return
        h = self._h(s)
        self.V[s] = h
        if h <= _EPS:
            self.fixed.add(s)
            self.stats["pruned_h0"] += 1

    # -- expansion -----------------------------------------------------------
    def _expand(self, s: State) -> None:
        facts, queue, windows, r = s
        options = []
        for name in self.model.legal_action_names(facts):
            op = self.ops.get(name)
            if op is None:
                continue
            outs = self._outcomes(op, "S", facts)
            if op.end_action is None:
                options.append((("do", name), self._children(outs, queue, windows, r)))
                continue
            d = self.durations[name]
            for gap in range(len(queue) + 1):
                new = start_windows(windows, d, gap)
                if new is None:
                    self.stats["pruned_gap"] += 1
                    continue
                if self.prune_inert and new[gap][2] > r and self._inert(name):
                    self.stats["pruned_inert"] += 1
                    continue
                q2 = queue[:gap] + (name,) + queue[gap:]
                options.append((("start", name, gap), self._children(outs, q2, new, r)))
        if queue:
            x = queue[0]
            op = self.ops[x]
            charge = windows[0][2]
            if r - charge < 0:
                self.stats["pruned_deadline"] += 1
            elif self._end_legal(op, facts):
                rest, _ = end_windows(windows, [self.durations[k] for k in queue[1:]])
                if rest is None:
                    self.stats["pruned_gap"] += 1
                else:
                    outs = self._outcomes(op, "E", facts)
                    options.append((("end", x), self._children(outs, queue[1:], rest, r - charge)))
        self.options[s] = options
        self.stats["expansions"] += 1
        self._backup(s)

    def _children(self, outcomes, queue, windows, r):
        out = []
        for facts, p in outcomes:
            if p <= _EPS:
                continue
            child = (frozenset(facts), queue, windows, r)
            self._init(child)
            out.append((float(p), child))
        return tuple(out)

    # -- heuristic -----------------------------------------------------------
    def _h(self, s: State) -> float:
        if self.heuristic == "none":
            return 1.0
        facts, queue, windows, r = s
        running = tuple((k, w[2]) for k, w in zip(queue, windows))   # guaranteed end e, same clock as r
        # PDB lookup: with one shared relaxed table the sweep reads only the facts
        # its actions and goals mention, so states that agree on those facts share
        # one curve -- the projection is exact, not an approximation.
        if self._shared_relaxed is not None:
            if getattr(self, "_pattern_facts", None) is None:
                mentioned = set(self.model.goals)
                for a in self._shared_relaxed:
                    mentioned |= a.pre | a.start_adds
                    for adds, _p in a.end_outcomes:
                        mentioned |= adds
                self._pattern_facts = frozenset(mentioned)
            facts = facts & self._pattern_facts
        key = (facts, running)
        cached = self._curves.get(key)
        curve = cached[1] if cached is not None and cached[0] >= r else None
        if curve is None:
            if self._shared_relaxed is not None:
                actions = self._shared_relaxed
            elif self._toy:
                actions = self._relaxed.setdefault(None, relaxed_actions_from_toy(self.model))
            else:
                actions = self._relaxed.get(facts)
                if actions is None:
                    actions = self._relaxed[facts] = relaxed_actions_from_engine(self.model, facts)
            curve = SurvivorSweep(actions, self.model.goals).curve(facts, self.horizon, running)
            self._curves[key] = (self.horizon, curve)
            self.stats["h_builds"] += 1
        return lookup(curve, r)

    # -- model adapters --------------------------------------------------------
    def _outcomes(self, op, which, facts):
        if isinstance(op.start_action, str):
            return self.model.outcomes((op.key, which), facts)
        return self.model.outcomes(op.start_action if which == "S" else op.end_action, facts)

    def _end_legal(self, op, facts) -> bool:
        if isinstance(op.start_action, str):
            return self.model.end_legal(op.key, facts)
        a = op.end_action
        return a.pos_preconditions.issubset(facts) and a.neg_preconditions.isdisjoint(facts)

    def _inert(self, name: str) -> bool:
        getter = getattr(self.model, "start_is_inert", None)
        return bool(getter(name)) if getter is not None else False
