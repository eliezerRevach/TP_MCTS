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
    * an instant action whose every outcome is its own state    -- never leaves (``fold_zero_time_loop``)

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


def fold_zero_time_loop(branches, s):
    """An instant action whose outcome is its own state lands at the same state at
    the same r, so the best policy repeats it until it leaves. ``branches`` are
    ``(p, child)``; the part q that comes back to ``s`` is folded away:
        q = 1      -> None, the option never leaves (it would hold any value, V = V)
        0 < q < 1  -> the other outcomes, p / (1 - q)
    Only for zero time: a durative retry lands at r - c, a different value."""
    q = sum(p for p, child in branches if child == s)
    if q <= 0:
        return branches
    if q >= 1 - _EPS:
        return None
    return tuple((p / (1 - q), child) for p, child in branches if child != s)


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
                      "pruned_noop": 0, "cover_used": 0}
        self.complete = True
        # States whose value is final: every state of a converged best policy.
        # Indexed by (F, Q) for the covering lookup.
        self.solved: set = set()
        self._solved_index: Dict[Tuple[FrozenSet, Tuple[str, ...]], List[State]] = {}
        self._stop_at_solved = False
        # Branches the solved policies did not take: (depth, -branch value, n, state).
        self._candidates: List[Tuple[int, float, int, State]] = []
        self._candidate_count = 0
        self._policy_changed = False

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
            # A best choice that switched during the pass (e.g. between tied options)
            # points into a part this pass did not visit, whose values may be stale.
            if expanded == 0 and residual < self.epsilon and not self._policy_changed:
                return self.V[root], True

    def _pass(self, root: State) -> Tuple[int, float]:
        """One depth-first pass over the best partial policy: expand its tips,
        back up every visited state in post-order."""
        expanded, residual = 0, 0.0
        self._policy_changed = False
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
        if self.best.get(s, best_index) != best_index:
            self._policy_changed = True
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
    def _rest_options(self, facts, queue, windows):
        """The options of (F, Q, W), without r:
        ``(label, charge, inert_e, queue after, windows after, outcomes)``.

        charge  : what the deadline is charged -- 0 for a start or an instant
                  action, e_x for the first end.
        inert_e : e of an inert start (not taken when it exceeds r), else None.
        Shared by ILAO* (``_expand``) and the full table (``WindowsTable``).
        """
        for name in self.model.legal_action_names(facts):
            op = self.ops.get(name)
            if op is None:
                continue
            outs = self._outcomes(op, "S", facts)
            if op.end_action is None:
                yield ("do", name), Fraction(0), None, queue, windows, outs
                continue
            d = self.durations[name]
            inert = self.prune_inert and self._inert(name)
            for gap in range(len(queue) + 1):
                new = start_windows(windows, d, gap)
                if new is None:
                    self.stats["pruned_gap"] += 1
                    continue
                yield (("start", name, gap), Fraction(0), new[gap][2] if inert else None,
                       queue[:gap] + (name,) + queue[gap:], new, outs)
        if queue:
            x = queue[0]
            op = self.ops[x]
            if self._end_legal(op, facts):
                rest, charge = end_windows(windows, [self.durations[k] for k in queue[1:]])
                if rest is None:
                    self.stats["pruned_gap"] += 1
                else:
                    yield ("end", x), charge, None, queue[1:], rest, self._outcomes(op, "E", facts)

    def _expand(self, s: State) -> None:
        facts, queue, windows, r = s
        options = []
        for label, charge, inert_e, q2, w2, outs in self._rest_options(facts, queue, windows):
            if r - charge < 0:
                self.stats["pruned_deadline"] += 1
                continue
            if inert_e is not None and inert_e > r:
                self.stats["pruned_inert"] += 1
                continue
            branches = self._children(outs, q2, w2, r - charge)
            if label[0] == "do":
                branches = fold_zero_time_loop(branches, s)
                if branches is None:
                    self.stats["pruned_noop"] += 1
                    continue
            options.append((label, branches))
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
        pattern_check = getattr(self.model, "end_legal_op", None)
        if pattern_check is not None:                  # fact-capped pattern: outside facts are free
            return pattern_check(op, facts)
        a = op.end_action
        return a.pos_preconditions.issubset(facts) and a.neg_preconditions.isdisjoint(facts)

    def _inert(self, name: str) -> bool:
        getter = getattr(self.model, "start_is_inert", None)
        return bool(getter(name)) if getter is not None else False


# ---------------------------------------------------------------------------
# The whole pattern, every r
# ---------------------------------------------------------------------------

def _frac_gcd(a: Fraction, b: Fraction) -> Fraction:
    den = a.denominator * b.denominator // math.gcd(a.denominator, b.denominator)
    return Fraction(math.gcd(int(a * den), int(b * den)), den)


class WindowsTable:
    """Every state of the pattern at every r: a forward graph over (F, Q, W) and
    backward curves V(s, r). No heuristic, no ILAO*.

    forward : every state reachable from the root before the deadline, expanded once
              with ``WindowsLAO._rest_options`` -- the same rules as ILAO*, without r
              in the state. States are taken in order of tmin, the least time charged
              on any path to them; one with tmin > horizon is never expanded (it can
              only be reached after the deadline). tmin is kept beside the state, not
              in it. Each option keeps its charge (0 for a start or an instant action,
              e_x for an end) and, for an inert start, the e it needs. An instant
              action that comes back to its own state is folded
              (``fold_zero_time_loop``).
    backward: V(s, .) is a step function of r, kept on the grid g = gcd of every
              charge, inert threshold and the horizon. Goals are 1, everything else
              starts at 0, so a value is only ever built from real paths to the goal
              and a loop that never reaches it stays 0:
                  V(s, r) = max over options  sum p * V(s', r - c)
              (an option with r < c, or an inert start with r < e, is not taken).
              An end charges c > 0 and reads an earlier layer, so the layers
              r = 0, g, 2g, ... are solved in order. Inside a layer only charge-0
              edges are left: starts (Q grows) and ends with e = 0 (Q shrinks) cannot
              loop and instant self-loops are folded, so a layer is one sink-first
              pass (vectorised per level). If a zero-time cycle over several states is
              left (two instant actions undoing each other), a worklist from the goals
              is used instead; it terminates, but a cycle that leaks is summed as a
              series and stops within 1e-14 below the value.
              A state is only ever at r <= horizon - tmin; cells beyond that are set
              to 1 (never read by a reachable cell, and a sound answer if a lookup
              lands there).
    lookup  : the leaf's state itself, else the table states with the same F and Q that
              fit it some time d >= 0 after their event: a table state holds the windows
              at an event, a leaf sits d after it, so it needs
                  lo - d <= rem <= hi - d   and   e - d <= rem     for every running end
              and r + d <= horizon - tmin(s) (the event cannot come before tmin).
              EVERY fitting state is an upper bound on its own, so the smallest value
              over them is returned. Why V(s, r + d) >= V(leaf, r):
                1. the leaf dated back by d, windows rem + d at r + d, is worth at
                   least the leaf: after an end the two are the same state; after a
                   start the dated one is looser; every prune passes for it whenever
                   it passes for the leaf;
                2. the windows MDP is monotone in the order "same F and Q, lo <=,
                   hi >=, r >=, r - e >=": every start, end and instant maps a looser
                   state to a looser successor, and every prune is monotone. The
                   fitting condition says exactly s >= the dated leaf.
              The smallest fitting d gives the tightest value. None fits: a miss.
    """

    def __init__(self, lao: WindowsLAO, horizon):
        self.lao = lao
        self.horizon = Fraction(horizon)
        self.states: List[Tuple[FrozenSet, Tuple[str, ...], Tuple[Window, ...]]] = []
        self.index: Dict[tuple, int] = {}
        self.options: List[Optional[list]] = []
        self.V = None                                     # V[k, i] = V(state i, r = k * grid)
        self.grid: Optional[Fraction] = None
        self.complete = False
        self._by_fq: Dict[tuple, List[Tuple[tuple, int]]] = {}
        self.stats: Dict[str, object] = {}

    # -- forward -------------------------------------------------------------
    def build(self, root_facts, time_budget: Optional[float] = None) -> bool:
        """Forward graph, then backward curves. False if ``time_budget`` (seconds)
        ran out during the forward graph; the table is then not used."""
        started = time.perf_counter()
        stop_at = None if time_budget is None else started + float(time_budget)
        lao, goals = self.lao, self.lao.model.goals
        root = (frozenset(root_facts), (), ())
        pending = object()                                # created, not expanded (yet)
        index, states, options, tmin = {root: 0}, [root], [pending], [Fraction(0)]
        heap = [(Fraction(0), 0)]
        noops = folded = 0

        def node(facts, queue, windows):
            key = (frozenset(facts), queue, windows)
            j = index.get(key)
            if j is None:
                j = index[key] = len(states)
                states.append(key)
                options.append(pending)
                tmin.append(None)
            return j

        popped = 0
        while heap:
            if stop_at is not None and not popped & 255 and time.perf_counter() > stop_at:
                self.stats.update(states_when_cut=len(states),
                                  forward_seconds=round(time.perf_counter() - started, 2))
                return False
            t, i = heapq.heappop(heap)
            if options[i] is not pending or t > tmin[i]:
                continue
            popped += 1
            facts, queue, windows = states[i]
            if goals <= facts:
                options[i] = None
                continue
            row = []
            for label, charge, inert_e, q2, w2, outs in lao._rest_options(facts, queue, windows):
                branches = tuple((float(p), node(f, q2, w2)) for f, p in outs if p > _EPS)
                if label[0] == "do":
                    before = branches
                    branches = fold_zero_time_loop(branches, i)
                    if branches is None:
                        noops += 1
                        continue
                    folded += branches is not before
                row.append((charge, inert_e, branches, label))
                arrive = t + charge
                for _p, j in branches:
                    if (tmin[j] is None or arrive < tmin[j]) and options[j] is pending:
                        tmin[j] = arrive
                        if arrive <= self.horizon:
                            heapq.heappush(heap, (arrive, j))
            options[i] = row
        cut = 0
        for i, row in enumerate(options):
            if row is pending:                            # first reached after the deadline
                options[i] = []
                cut += 1
        self.states, self.index, self.options = states, index, options
        self.tmin = [self.horizon + 1 if x is None else x for x in tmin]
        forward_seconds = time.perf_counter() - started
        method = self._backward()
        for k, (facts, queue, windows) in enumerate(states):
            self._by_fq.setdefault((facts, queue), []).append((windows, k))
        self.complete = True
        self.stats.update(
            states=len(states), cut_after_deadline=cut,
            edges=sum(len(b) for row in options if row for _c, _e, b, _l in row),
            instant_noops_dropped=noops, instant_loops_folded=folded, backward=method, grid=str(self.grid),
            forward_seconds=round(forward_seconds, 2),
            backward_seconds=round(time.perf_counter() - started - forward_seconds, 2),
            seconds=round(time.perf_counter() - started, 2))
        return True

    # -- backward ------------------------------------------------------------
    def _backward(self) -> str:
        import numpy as np
        from scipy.sparse import csr_matrix

        n = len(self.states)
        g = self.horizon
        for row in self.options:
            for charge, inert_e, _b, _l in row or ():
                if charge:
                    g = _frac_gcd(g, Fraction(charge))
                if inert_e:
                    g = _frac_gcd(g, Fraction(inert_e))
        self.grid = g
        R = int(self.horizon / g) + 1
        owner, shift, need, rows, cols, probs = [], [], [], [], [], []
        for i, row in enumerate(self.options):
            for charge, inert_e, branches, _l in row or ():
                k = len(owner)
                owner.append(i)
                shift.append(int(Fraction(charge) / g))
                need.append(0 if inert_e is None else math.ceil(Fraction(inert_e) / g))
                for p, j in branches:
                    rows.append(k)
                    cols.append(j)
                    probs.append(p)
        owner = np.array(owner, dtype=int)
        shift = np.array(shift, dtype=int)
        need = np.array(need, dtype=int)
        m = len(owner)
        P = csr_matrix((probs, (rows, cols)), shape=(m, n))
        V = np.zeros((R, n))
        V[:, [i for i, row in enumerate(self.options) if row is None]] = 1.0

        # levels over the charge-0 edges, sinks first
        zero = shift == 0
        rows_a, cols_a = np.array(rows, dtype=int), np.array(cols, dtype=int)
        edge_zero = zero[rows_a] if len(rows_a) else np.zeros(0, dtype=bool)
        pairs = set(zip(owner[rows_a[edge_zero]].tolist(), cols_a[edge_zero].tolist()))
        pending = np.zeros(n, dtype=int)
        preds: List[List[int]] = [[] for _ in range(n)]
        for a, b in pairs:
            pending[a] += 1
            preds[b].append(a)
        level = np.zeros(n, dtype=int)
        frontier = [i for i in range(n) if pending[i] == 0]
        seen = 0
        while frontier:
            b = frontier.pop()
            seen += 1
            for a in preds[b]:
                level[a] = max(level[a], level[b] + 1)
                pending[a] -= 1
                if pending[a] == 0:
                    frontier.append(a)
        if seen < n:                                      # a zero-time cycle over several states
            self.V = self._worklist(V, P, owner, shift, need)
            self._mark_unreachable_cells()
            return "worklist"

        groups = {int(c): np.nonzero(shift == c)[0] for c in np.unique(shift) if c > 0}
        P_pos = {c: P[idx] for c, idx in groups.items()}
        opt_level = level[owner] if m else np.zeros(0, dtype=int)
        levels = []
        for L in range(int(level.max()) + 1 if n else 0):
            mine = np.nonzero(opt_level == L)[0]
            z = mine[zero[mine]]
            levels.append((z, P[z], mine))
        for k in range(R):
            q = np.zeros(m)
            for c, idx in groups.items():
                if c <= k:
                    q[idx] = P_pos[c] @ V[k - c]
            Vk = V[k]
            for z, Pz, mine in levels:
                if len(z):
                    q[z] = Pz @ Vk
                if len(mine):
                    vals = np.where(need[mine] > k, 0.0, q[mine])
                    np.maximum.at(Vk, owner[mine], vals)
        self.V = V
        self._mark_unreachable_cells()
        return "layers"

    def _mark_unreachable_cells(self) -> None:
        """A state is only ever at r <= horizon - tmin: later cells may read children
        that were never expanded, so they are set to 1 (sound, and never read by a
        reachable cell: a child's tmin is at most its parent's plus the charge)."""
        for i, t in enumerate(self.tmin):
            first = 0 if t > self.horizon else int((self.horizon - t) / self.grid) + 1
            self.V[first:, i] = 1.0

    def _worklist(self, V, P, owner, shift, need):
        """Fallback for zero-time cycles over several states: from the goals,
        recompute each predecessor's whole curve, requeue it when it went up."""
        import numpy as np
        from collections import deque

        R, n = V.shape
        by_state: List[List[int]] = [[] for _ in range(n)]
        for k, i in enumerate(owner.tolist()):
            by_state[i].append(k)
        preds: List[set] = [set() for _ in range(n)]
        coo = P.tocoo()
        for k, j in zip(coo.row.tolist(), coo.col.tolist()):
            preds[j].add(int(owner[k]))
        Vt = V.T.copy()                                   # Vt[i] = curve of state i
        goal = [row is None for row in self.options]
        work = deque(i for i in range(n) if goal[i])
        queued = np.array(goal, dtype=bool)
        grid = np.arange(R)
        while work:
            s = work.popleft()
            queued[s] = False
            for t in preds[s]:
                if goal[t]:
                    continue
                best = np.zeros(R)
                for k in by_state[t]:
                    lo, hi = P.indptr[k], P.indptr[k + 1]
                    acc = P.data[lo:hi] @ Vt[P.indices[lo:hi]]
                    q = np.zeros(R)
                    c = int(shift[k])
                    q[c:] = acc[:R - c]
                    q[grid < need[k]] = 0.0
                    np.maximum(best, q, out=best)
                if np.any(best > Vt[t] + 1e-14):
                    np.maximum(Vt[t], best, out=Vt[t])
                    if not queued[t]:
                        queued[t] = True
                        work.append(t)
        return Vt.T.copy()

    # -- lookup --------------------------------------------------------------
    def lookup(self, facts, r, running: Sequence[Tuple[str, object]] = ()) -> Tuple[Optional[float], str]:
        """``(value, "exact" | "cover")`` or ``(None, "miss")``; several tie orders
        of the running ends take the max, as in ``WindowsLAO``."""
        facts = frozenset(facts)
        if self.lao.model.goals <= facts:
            return 1.0, "exact"
        r = _as_fraction(r)
        if not self.complete or r < 0 or r > self.horizon:
            return None, "miss"
        best, kind = None, "exact"
        for f, q, w, _r in self.lao._roots(facts, r, running):
            i = self.index.get((f, q, w))
            if i is not None:
                value = self.value(i, r)
            else:
                value = self._shifted_cover(f, q, w, r)
                if value is None:
                    return None, "miss"
                kind = "cover"
            best = value if best is None else max(best, value)
        return (0.0 if best is None else best), kind

    def value(self, i: int, r) -> float:
        return float(self.V[int(min(Fraction(r), self.horizon) / self.grid), i])

    def policy(self, i: int, r):
        """The best option of state i at r, ``(charge, inert_e, branches, label)``,
        or None when nothing reaches the goal from there (value 0) or i is a goal."""
        row = self.options[i]
        if not row or r < 0 or r > self.horizon - self.tmin[i]:
            return None
        best, best_q = None, _EPS
        for option in row:
            charge, inert_e, branches, _label = option
            if charge > r or (inert_e is not None and inert_e > r):
                continue
            q = sum(p * self.value(j, r - charge) for p, j in branches)
            if q > best_q + 1e-12:
                best, best_q = option, q
        return best

    def fitting_values(self, facts, queue, windows, r) -> List[float]:
        """V(s, r + d) for every table state s with this F and Q that fits the leaf
        some d >= 0 after its event, d the smallest that fits (see the class doc)."""
        out = []
        for w2, j in self._by_fq.get((facts, queue), ()):
            d_lo, d_hi = Fraction(0), self.horizon - self.tmin[j] - r
            for (lo2, hi2, e2), (lo, hi, e) in zip(w2, windows):
                d_lo = max(d_lo, lo2 - lo, e2 - e)
                d_hi = min(d_hi, hi2 - hi)
                if d_lo > d_hi:
                    break
            else:
                if d_lo <= d_hi:
                    out.append(self.value(j, r + d_lo))
        return out

    def _shifted_cover(self, facts, queue, windows, r) -> Optional[float]:
        values = self.fitting_values(facts, queue, windows, r)
        return min(values) if values else None
