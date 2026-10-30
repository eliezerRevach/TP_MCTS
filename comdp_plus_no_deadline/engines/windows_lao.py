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

Blur (optional, ``blur`` / TP_MCTS_WILAO_BLUR = eps, 0 = off): every window a
start or an end produces is snapped OUTWARD to the grid eps,
    lo -> floor(lo / eps) * eps,   hi -> ceil(hi / eps) * eps,   e unchanged,
so states that differ only in nearby windows become one. Sound: the MDP is
monotone in "lo <=, hi >=, r >=, r - e >=", so a looser window is worth at
least as much -- the value stays an upper bound. e is NOT rounded: rounding it
down charges less than the real time (eps = 3 gave V = 1.0 everywhere on nasa).
Only the window variants shrink (facts and end orders are untouched).

Rules (``rules`` / TP_MCTS_WILAO_RULES): ``windows`` (default, the rules above) or
``fixed_stn`` -- the STN of the running actions never moves, only LB (the time
charged so far) goes up. Windows are kept relative to LB, so one situation reached at
different times is one state (no clock in the key) and e = lo always:
    start x in a gap : as above (e = lo)
    first end x fires: r -= lo_x
                       others lo = lo - lo_x,  hi = min(hi - lo_x, hi_x - lo_x + d)
                       (the same absolute window seen from the new LB; everyone left
                       started before x ended, so it ends by E_x + d)
Blur rounds only hi up there (lo is the charge).

Hidden reads: an outcome probability may read a fact outside a fact-capped pattern
(push reads rock_under_car). Each value of those facts gives its own option of the
same event, so the max over options picks the best one -- an upper bound
(``FactPatternModel.outcome_variants``). The survivor sweep ``h`` (ILAO*'s guide,
not used by the full table) still reads them at their initial value.

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

import collections
import heapq
import itertools
import math
import os
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


def end_windows_fixed(windows: Sequence[Window], durations: Sequence[Fraction]
                      ) -> Tuple[Optional[Tuple[Window, ...]], Fraction]:
    """fixed_stn: the first end fires, LB goes up by lo_x and nothing else moves
    (module doc, "Rules"); returns (windows of the rest relative to the new LB, lo_x)."""
    lo_x, hi_x, _e = windows[0]
    rest = []
    for (lo, hi, _e), d in zip(windows[1:], durations):
        new_lo = lo - lo_x
        rest.append((new_lo, min(hi - lo_x, hi_x - lo_x + d), new_lo))
    return normalise(rest), lo_x


def rules_default() -> str:
    """TP_MCTS_WILAO_RULES: windows (default) | fixed_stn."""
    text = (os.environ.get("TP_MCTS_WILAO_RULES") or "windows").strip().lower()
    if text not in ("windows", "fixed_stn"):
        raise ValueError(f"TP_MCTS_WILAO_RULES={text!r}: use windows | fixed_stn")
    return text


def fit_default() -> str:
    """TP_MCTS_WILAO_FIT: shift (default, the closed form) | stn (Floyd-Warshall)."""
    text = (os.environ.get("TP_MCTS_WILAO_FIT") or "shift").strip().lower()
    if text not in ("shift", "stn"):
        raise ValueError(f"TP_MCTS_WILAO_FIT={text!r}: use shift | stn")
    return text


def ub_default() -> bool:
    """TP_MCTS_WILAO_UB: 1 = the table computes T (UB) and answers r < T with 0; 0 = off (default)."""
    text = (os.environ.get("TP_MCTS_WILAO_UB") or "0").strip().lower()
    if text not in ("0", "1"):
        raise ValueError(f"TP_MCTS_WILAO_UB={text!r}: use 0 | 1")
    return text == "1"


def vi_default() -> str:
    """TP_MCTS_WILAO_VI: layers (default: one value per state and time step) | intervals (the PTA
    backward pass: a state's curve is computed and kept only where it changes; same values)."""
    text = (os.environ.get("TP_MCTS_WILAO_VI") or "layers").strip().lower()
    if text not in ("layers", "intervals"):
        raise ValueError(f"TP_MCTS_WILAO_VI={text!r}: use layers | intervals")
    return text


def blur_default() -> Fraction:
    """TP_MCTS_WILAO_BLUR: the window grid eps; 0 (default) = no blur."""
    text = (os.environ.get("TP_MCTS_WILAO_BLUR") or "0").strip()
    try:
        eps = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"TP_MCTS_WILAO_BLUR={text!r}: use a number >= 0 (0 = off)")
    if eps < 0:
        raise ValueError(f"TP_MCTS_WILAO_BLUR={text!r}: use a number >= 0 (0 = off)")
    return eps


def blur_windows(windows: Optional[Tuple[Window, ...]], eps: Fraction,
                 hi_only: bool = False) -> Optional[Tuple[Window, ...]]:
    """lo down and hi up to the grid eps, e exact (module doc); ``hi_only`` for fixed_stn."""
    if windows is None or not eps:
        return windows
    return normalise(tuple((lo if hi_only else Fraction(math.floor(lo / eps)) * eps,
                            Fraction(math.ceil(hi / eps)) * eps, e) for lo, hi, e in windows))


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
                 relaxed_actions: Optional[list] = None, blur=None, rules: Optional[str] = None):
        self.model = model
        # Window grid eps (module doc, "Blur"); None = TP_MCTS_WILAO_BLUR, 0 = off.
        self.blur = blur_default() if blur is None else Fraction(blur)
        # Window rules (module doc, "Rules"); None = TP_MCTS_WILAO_RULES.
        self.rules = rules_default() if rules is None else rules
        self._fixed = self.rules == "fixed_stn"
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

    def goal_out_of_reach(self, facts, r, running: Sequence[Tuple[str, object]] = (),
                          max_states: int = 5000) -> Optional[bool]:
        """The miss-path UB (TP_MCTS_WILAO_MISS_UB): Dijkstra forward from the leaf over
        (F, Q, W), every outcome a possible branch (the luckiest one), charges as time.
        True: no root reaches a goal within r -- even the relaxed problem cannot, so the
        value is 0. False: some root can. None: ``max_states`` ran out first, no answer."""
        facts, r = frozenset(facts), Fraction(r)
        if self.model.goals <= facts:
            return False
        for root in self._roots(facts, r, running):
            start = root[:3]
            best = {start: Fraction(0)}
            heap = [(Fraction(0), 0, start)]
            count, popped = 1, 0
            while heap:
                t, _n, s = heapq.heappop(heap)
                if t > best.get(s, t):
                    continue
                if self.model.goals <= s[0]:
                    return False
                popped += 1
                if popped > max_states:
                    return None
                for _label, charge, inert_e, q2, w2, outs in self._rest_options(*s):
                    arrive = t + charge
                    if arrive > r or (inert_e is not None and inert_e > r - t):
                        continue
                    for f2, p in outs:
                        if p <= _EPS:
                            continue
                        child = (frozenset(f2), q2, w2)
                        if arrive < best.get(child, r + 1):
                            best[child] = arrive
                            count += 1
                            heapq.heappush(heap, (arrive, count, child))
        return True

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
            variants = self._outcome_variants(op, "S", facts)
            if op.end_action is None:
                for outs in variants:
                    yield ("do", name), Fraction(0), None, queue, windows, outs
                continue
            d = self.durations[name]
            inert = self.prune_inert and self._inert(name)
            for gap in range(len(queue) + 1):
                new = blur_windows(start_windows(windows, d, gap), self.blur, self._fixed)
                if new is None:
                    self.stats["pruned_gap"] += 1
                    continue
                for outs in variants:
                    yield (("start", name, gap), Fraction(0), new[gap][2] if inert else None,
                           queue[:gap] + (name,) + queue[gap:], new, outs)
        if queue:
            x = queue[0]
            op = self.ops[x]
            if self._end_legal(op, facts):
                end_rule = end_windows_fixed if self._fixed else end_windows
                rest, charge = end_rule(windows, [self.durations[k] for k in queue[1:]])
                rest = blur_windows(rest, self.blur, self._fixed)
                if rest is None:
                    self.stats["pruned_gap"] += 1
                else:
                    for outs in self._outcome_variants(op, "E", facts):
                        yield ("end", x), charge, None, queue[1:], rest, outs

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

    def _outcome_variants(self, op, which, facts):
        """Outcome distributions of one event, one per value of the facts its probability reads
        outside a fact-capped pattern (``FactPatternModel.outcome_variants``); each is its own
        option, so the max over options picks the best value. One distribution otherwise."""
        getter = getattr(self.model, "outcome_variants", None)
        if getter is None or isinstance(op.start_action, str):
            return (self._outcomes(op, which, facts),)
        return getter(op.start_action if which == "S" else op.end_action, facts)

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


def _concat_ranges(starts, ends):
    """np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)]), without the loop."""
    import numpy as np
    lens = ends - starts
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return offsets + np.arange(total)


class IntervalValues:
    """V(state i, r = k * grid) of ``WindowsTable`` with ``vi = intervals``: only the layers where
    a curve changes are kept. State i's curve is ``vals[ptr[i]:ptr[i+1]]``, each value holding from
    its layer in ``ks`` up to the next one, 0 before the first. Layers k >= first[i] (r beyond
    horizon - tmin_i, never reached) read 1, as ``_mark_unreachable_cells`` sets them on the array.
    ``V[k, i]`` reads like the dense array."""

    def __init__(self, ptr, ks, vals, first, shape):
        self.ptr, self.ks, self.vals, self.first, self.shape = ptr, ks, vals, first, shape

    def __getitem__(self, key):
        import numpy as np
        k, i = key
        if k >= self.first[i]:
            return 1.0
        lo, hi = int(self.ptr[i]), int(self.ptr[i + 1])
        j = lo + int(np.searchsorted(self.ks[lo:hi], k, side="right")) - 1
        return float(self.vals[j]) if j >= lo else 0.0

    @property
    def nbytes(self) -> int:
        return self.ptr.nbytes + self.ks.nbytes + self.vals.nbytes + self.first.nbytes

    def dense(self):
        """The (layers, states) array the layered pass would hold (checks only)."""
        import numpy as np
        R, n = self.shape
        out = np.zeros((R, n))
        for i in range(n):
            lo, hi = int(self.ptr[i]), int(self.ptr[i + 1])
            for j in range(lo, hi):
                end = int(self.ks[j + 1]) if j + 1 < hi else R
                out[int(self.ks[j]):end, i] = self.vals[j]
            out[int(self.first[i]):, i] = 1.0
        return out


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
              ``vi`` / TP_MCTS_WILAO_VI = intervals: the same values, PTA-style -- a
              curve is a step function kept only where it changes (IntervalValues).
              Layer k recomputes only the states an input changed for: a successor
              that changed at layer k - c, an option whose charge or inert need is
              k, or a charge-0 successor that changed in this layer (levels, sinks
              first). Only the last max(c) + 1 layers are kept while solving.
    lookup  : the leaf's state itself, else SHIFT IT BACK: a table state holds the
              windows at an event, a leaf usually sits some d after one. Moving the
              leaf back by d gives every running end d more time left and r + d.
              It fits table state j (same F and Q) when, per running action,
                  lo_j <= lo + d,   e_j <= e + d,   hi + d <= hi_j
              and r + d <= horizon - tmin_j (j is never reached before tmin_j):
                  d in [max(0, lo_j - lo, e_j - e),  min(hi_j - hi, horizon - tmin_j - r)]
              Take the smallest d (least extra time = tightest) and read V(j, r + d).
              Every fitting j is an upper bound on its own, so the smallest value
              over them is returned; none fits: a miss. Why V(j, r + d) >= V(leaf, r):
                1. the leaf moved back by d is worth at least the leaf (d more time;
                   after an end the two are the same state, after a start the moved
                   one is looser; every prune passes for it whenever it does for the leaf);
                2. the windows MDP is monotone in "same F and Q, lo <=, hi >=, r >=,
                   r - e >=", and the fit says exactly j >= the moved leaf.
              ``fit`` / TP_MCTS_WILAO_FIT: shift (that closed form) or stn -- the same
              question asked of an STN and answered by Floyd-Warshall (``stn_shift``).
    UB      : (``ub`` / TP_MCTS_WILAO_UB = 1, off by default)
              T(s), the least time from s to a goal when every outcome is a possible
              branch (the luckiest one): one Dijkstra pass from the goals over the
              reversed graph, an option costing max(its inert need, charge + T(next)).
              No policy is faster, so V(s, r) = 0 for r < T(s) (UB(s) = D - T(s) in
              time-since-0 terms); lookups answer those with 0 at once. The curves are
              already 0 there, so only the placeholder cells beyond horizon - tmin can
              change (counted as ub_zero_curve_nonzero).
    """

    def __init__(self, lao: WindowsLAO, horizon, fit: Optional[str] = None, ub: Optional[bool] = None,
                 vi: Optional[str] = None):
        self.lao = lao
        self.horizon = Fraction(horizon)
        self.fit = fit_default() if fit is None else fit
        self.ub = ub_default() if ub is None else bool(ub)
        self.vi = vi_default() if vi is None else vi
        self.T: Optional[list] = None                     # T[i] = least time to a goal, None = never
        self.states: List[Tuple[FrozenSet, Tuple[str, ...], Tuple[Window, ...]]] = []
        self.index: Dict[tuple, int] = {}
        self.options: List[Optional[list]] = []
        self.V = None                                     # V[k, i] = V(state i, r = k * grid)
        self.grid: Optional[Fraction] = None
        self.complete = False
        self._by_fq: Dict[tuple, List[Tuple[tuple, int]]] = {}
        self.stats: Dict[str, object] = {}
        self.fit_stats: Dict[str, int] = collections.Counter()   # how lookups fit (diagnostics)
        self.fit_examples: List[dict] = []

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
        self._cut = []
        for i, row in enumerate(options):
            if row is pending:                            # first reached after the deadline
                options[i] = []
                cut += 1
                self._cut.append(i)
        self.states, self.index, self.options = states, index, options
        self.tmin = [self.horizon + 1 if x is None else x for x in tmin]
        forward_seconds = time.perf_counter() - started
        self.T = self._time_to_goal() if self.ub else None
        ub_seconds = time.perf_counter() - started - forward_seconds
        method = self._backward()
        for k, (facts, queue, windows) in enumerate(states):
            self._by_fq.setdefault((facts, queue), []).append((windows, k))
        self.complete = True
        self.stats.update(
            states=len(states), cut_after_deadline=cut,
            edges=sum(len(b) for row in options if row for _c, _e, b, _l in row),
            instant_noops_dropped=noops, instant_loops_folded=folded, backward=method, grid=str(self.grid),
            vi=self.vi, values_bytes=int(self.V.nbytes),
            forward_seconds=round(forward_seconds, 2), ub_seconds=round(ub_seconds, 2),
            backward_seconds=round(time.perf_counter() - started - forward_seconds - ub_seconds, 2),
            seconds=round(time.perf_counter() - started, 2), rules=lao.rules, ub=self.ub,
            never_reach_goal=None if self.T is None else sum(1 for t in self.T if t is None))
        return True

    # -- UB --------------------------------------------------------------------
    def _time_to_goal(self) -> List[Optional[Fraction]]:
        """T(s) by Dijkstra from the goals over the reversed graph (class doc, UB). The
        first child of an option to be settled is its luckiest one, so the option's cost
        max(need, charge + T(child)) is final then; it is >= T(child), so label-setting holds.
        A state never expanded (first reached after the deadline) is unknown, not dead:
        it starts at 0 like a goal, so T stays a lower bound everywhere."""
        n = len(self.states)
        into: List[List[Tuple[int, Fraction, Fraction]]] = [[] for _ in range(n)]
        for i, row in enumerate(self.options):
            for charge, inert_e, branches, _label in row or ():
                need = Fraction(0) if inert_e is None else Fraction(inert_e)
                for _p, j in branches:
                    into[j].append((i, Fraction(charge), need))
        T: List[Optional[Fraction]] = [None] * n
        heap = [(Fraction(0), i) for i, row in enumerate(self.options) if row is None]
        heap += [(Fraction(0), i) for i in self._cut]
        heapq.heapify(heap)
        while heap:
            t, j = heapq.heappop(heap)
            if T[j] is not None:
                continue
            T[j] = t
            for i, charge, need in into[j]:
                if T[i] is None:
                    heapq.heappush(heap, (max(need, charge + t), i))
        return T

    # -- backward ------------------------------------------------------------
    def _backward(self) -> str:
        """Backward curves on compact arrays (array / numpy, 4-8 bytes per entry; the same
        computation as keeping every edge in Python lists, at a fraction of the memory).
        Zero-charge levels: Kahn in rounds over the distinct (state, successor) pairs --
        round = longest charge-0 path to a sink. ``vi = intervals``: ``_backward_intervals``."""
        import numpy as np

        if self.vi == "intervals":
            return self._backward_intervals()
        n = len(self.states)
        R, owner, shift, need, P = self._option_arrays()
        m = len(owner)
        V = np.zeros((R, n))
        V[:, [i for i, row in enumerate(self.options) if row is None]] = 1.0
        zero = shift == 0
        level, seen, _pred_of, _pred_ptr = self._zero_levels(n, owner, zero, P)
        del _pred_of, _pred_ptr
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

    def _option_arrays(self):
        """The options as arrays: the layer count R (sets ``self.grid``, the gcd of every charge,
        inert need and the horizon) and per option its owner, charge and inert need in grid
        steps, with P[option, successor] = p."""
        import numpy as np
        from array import array
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
        owner, shift, need = array("i"), array("i"), array("i")
        rows, cols, probs = array("i"), array("i"), array("d")
        shift_of, need_of = {}, {}
        k = 0
        for i, row in enumerate(self.options):
            for charge, inert_e, branches, _l in row or ():
                owner.append(i)
                c = shift_of.get(charge)
                if c is None:
                    c = shift_of[charge] = int(Fraction(charge) / g)
                shift.append(c)
                e = need_of.get(inert_e)
                if e is None:
                    e = need_of[inert_e] = 0 if inert_e is None else math.ceil(Fraction(inert_e) / g)
                need.append(e)
                for p, j in branches:
                    rows.append(k)
                    cols.append(j)
                    probs.append(p)
                k += 1
        owner = np.frombuffer(owner, dtype=np.int32).astype(int)
        shift = np.frombuffer(shift, dtype=np.int32).astype(int)
        need = np.frombuffer(need, dtype=np.int32).astype(int)
        rows_a = np.frombuffer(rows, dtype=np.int32)
        cols_a = np.frombuffer(cols, dtype=np.int32)
        P = csr_matrix((np.frombuffer(probs, dtype=np.float64), (rows_a, cols_a)), shape=(len(owner), n))
        return R, owner, shift, need, P

    @staticmethod
    def _zero_levels(n, owner, zero, P):
        """Levels over the charge-0 edges, sinks first: (level, states levelled, pred_of, pred_ptr),
        pred_of[pred_ptr[j]:pred_ptr[j + 1]] = the states with a charge-0 option into j.
        Fewer states levelled than n means a zero-time cycle."""
        import numpy as np

        coo = P.tocoo()
        rows_a, cols_a = coo.row, coo.col
        edge_zero = zero[rows_a] if len(rows_a) else np.zeros(0, dtype=bool)
        key = np.unique(owner[rows_a[edge_zero]].astype(np.int64) * n + cols_a[edge_zero].astype(np.int64))
        pa, pb = key // n, key % n
        del key, edge_zero, coo
        pending = np.bincount(pa, minlength=n).astype(np.int64)
        pred_of = pa[np.argsort(pb, kind="stable")]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pb, minlength=n), out=indptr[1:])
        del pa, pb
        level = np.zeros(n, dtype=int)
        frontier = np.nonzero(pending == 0)[0]
        seen, rnd = 0, 0
        while len(frontier):
            seen += len(frontier)
            level[frontier] = rnd
            starts, counts = indptr[frontier], indptr[frontier + 1] - indptr[frontier]
            total = int(counts.sum())
            if total == 0:
                break
            idx = np.repeat(starts - np.cumsum(np.concatenate(([0], counts[:-1]))), counts) + np.arange(total)
            preds = pred_of[idx]
            np.subtract.at(pending, preds, 1)
            cand = np.unique(preds)
            frontier = cand[pending[cand] == 0]
            rnd += 1
        return level, seen, pred_of, indptr

    def _backward_intervals(self) -> str:
        """The PTA backward pass (class doc, ``vi = intervals``): the values of ``_backward``'s
        layers, computed only where a curve changes and kept as IntervalValues."""
        import numpy as np

        n = len(self.states)
        R, owner, shift, need, P = self._option_arrays()
        m = len(owner)
        zero = shift == 0
        level, seen, pred_of, pred_ptr = self._zero_levels(n, owner, zero, P)
        if seen < n:                                      # zero-time cycle: the layered pass's worklist
            self.stats["vi_fallback"] = "zero-time cycle: dense worklist"
            self.vi = "layers"
            method = self._backward()
            self.vi = "intervals"
            return method

        opt_ptr = np.zeros(n + 1, dtype=np.int64)         # options of state i: opt_ptr[i]:opt_ptr[i+1]
        np.cumsum(np.bincount(owner, minlength=n), out=opt_ptr[1:])
        has_options = opt_ptr[1:] > opt_ptr[:-1]
        goal = np.array([row is None for row in self.options], dtype=bool)
        coo = P.tocoo()                                   # who reads state j through a charge-c option
        into = {}
        for c in np.unique(shift[shift > 0]):
            sel = shift[coo.row] == c
            succ, opt = coo.col[sel].astype(np.int64), coo.row[sel].astype(np.int64)
            order = np.argsort(succ, kind="stable")
            ptr = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(np.bincount(succ, minlength=n), out=ptr[1:])
            into[int(c)] = (ptr, owner[opt[order]])
        del coo
        gate = {}                                         # the layer where an option starts to count
        for arr in (shift, need):
            for k in np.unique(arr[arr > 0]):
                gate.setdefault(int(k), []).append(np.unique(owner[arr == k]))
        W = int(shift.max()) + 1 if m else 1
        ring = np.zeros((W, n))                           # the last W layers only
        self.stats["ring_W"] = W
        changed_at = [np.zeros(0, dtype=np.int64)] * W
        rec_i, rec_k, rec_v = [], [], []
        max_level = int(level.max()) if n else 0
        entry_opt = np.repeat(np.arange(m), np.diff(P.indptr))   # the option of each entry of P
        opt_len = np.diff(P.indptr)

        def best_of(states, Vk, k):
            """For states with options: max over them of sum p * V(succ, k - c), 0 if none counts."""
            opts = _concat_ranges(opt_ptr[states], opt_ptr[states + 1])
            ents = _concat_ranges(P.indptr[opts], P.indptr[opts + 1])
            local = np.repeat(np.arange(len(opts)), opt_len[opts])
            c = shift[entry_opt[ents]]
            succ = P.indices[ents]
            vals = np.where(c == 0, Vk[succ], ring[(k - c) % W, succ])
            q = np.bincount(local, weights=P.data[ents] * vals, minlength=len(opts))
            q[(shift[opts] > k) | (need[opts] > k)] = 0.0
            per_state = opt_ptr[states + 1] - opt_ptr[states]
            starts = np.concatenate(([0], np.cumsum(per_state)[:-1]))
            return np.maximum(np.maximum.reduceat(q, starts), 0.0)

        for k in range(R):
            if k == 0:
                Vk = np.where(goal, 1.0, 0.0)
                flagged = [np.nonzero(has_options)[0]]
                changed = [np.nonzero(goal)[0]]
            else:
                Vk = ring[(k - 1) % W].copy()
                flagged, changed = [], []
                for c, (ptr, who) in into.items():
                    J = changed_at[(k - c) % W]
                    if c <= k and len(J):
                        flagged.append(who[_concat_ranges(ptr[J], ptr[J + 1])])
                flagged += gate.get(k, [])
            pending = [[] for _ in range(max_level + 1)]
            cand = np.unique(np.concatenate(flagged)) if flagged else np.zeros(0, dtype=np.int64)
            cand = cand[has_options[cand]]
            if len(cand):
                lv = level[cand]
                for L in np.unique(lv):
                    pending[int(L)].append(cand[lv == L])
            for L in range(max_level + 1):
                if not pending[L]:
                    continue
                S = np.unique(np.concatenate(pending[L]))
                new = best_of(S, Vk, k)
                moved = np.abs(new - Vk[S]) > 1e-15
                S, new = S[moved], new[moved]
                if not len(S):
                    continue
                Vk[S] = new
                changed.append(S)
                preds = pred_of[_concat_ranges(pred_ptr[S], pred_ptr[S + 1])]
                preds = np.unique(preds[has_options[preds]])
                for L2 in np.unique(level[preds]):
                    pending[int(L2)].append(preds[level[preds] == L2])
            ch = np.concatenate(changed) if changed else np.zeros(0, dtype=np.int64)
            ring[k % W] = Vk
            changed_at[k % W] = ch
            if len(ch):
                rec_i.append(ch.astype(np.int64))
                rec_k.append(np.full(len(ch), k, dtype=np.int32))
                rec_v.append(Vk[ch])
        del ring, changed_at

        first = np.array([0 if t > self.horizon else int((self.horizon - t) / self.grid) + 1
                          for t in self.tmin], dtype=np.int32)
        ri = np.concatenate(rec_i) if rec_i else np.zeros(0, dtype=np.int64)
        rk = np.concatenate(rec_k) if rec_k else np.zeros(0, dtype=np.int32)
        rv = np.concatenate(rec_v) if rec_v else np.zeros(0)
        keep = rk < first[ri]                             # layers past horizon - tmin read 1 anyway
        ri, rk, rv = ri[keep], rk[keep], rv[keep]
        order = np.lexsort((rk, ri))
        ri, rk, rv = ri[order], rk[order], rv[order]
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(ri, minlength=n), out=ptr[1:])
        self.V = IntervalValues(ptr, rk.astype(np.int32), rv, first, (R, n))
        return "intervals"

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
                value = self._read(i, r)
                self.fit_stats["exact"] += 1
            else:
                fits = self._fits(f, q, w, r)
                if not fits:
                    self.fit_stats["miss"] += 1
                    return None, "miss"
                value, d, w_j = min(((self._read(j, r + d), d, w_j) for d, j, w_j in fits),
                                    key=lambda x: (x[0], x[1]))
                self._record_fit(q, w, r, fits, value, d, w_j)
                kind = "cover"
            best = value if best is None else max(best, value)
        return (0.0 if best is None else best), kind

    def _record_fit(self, queue, windows, r, fits, value, d, w_j) -> None:
        """Diagnostics only (values unchanged): did the leaf fit a table state as it is (d = 0)
        or only after the shift back by d > 0; a few shifted examples."""
        st = self.fit_stats
        st["fit_d0" if any(dd == 0 for dd, _j, _w in fits) else "shift_only"] += 1
        st["d_used=" + str(d)] += 1
        if d and len(self.fit_examples) < 8:
            self.fit_examples.append(dict(
                queue=list(queue), query_time_left=[str(hi) for _lo, hi, _e in windows], r=str(r),
                table_windows_lo_hi_e=[[str(x) for x in win] for win in w_j],
                d=str(d), value=round(value, 6), fits=len(fits)))

    def value(self, i: int, r) -> float:
        return float(self.V[int(min(Fraction(r), self.horizon) / self.grid), i])

    def _read(self, i: int, r) -> float:
        """V(i, r); with UB on, 0 at once when r < T(i) (class doc, UB)."""
        if self.T is not None:
            t = self.T[i]
            if t is None or r < t:
                self.fit_stats["ub_zero"] += 1
                if self.value(i, r) > 0:
                    self.fit_stats["ub_zero_curve_nonzero"] += 1
                return 0.0
        return self.value(i, r)

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

    def shift(self, j: int, w_j, windows, r) -> Optional[Fraction]:
        """The smallest d >= 0 that moves the leaf back into table state j (windows w_j), or None
        (class doc: d in [max(0, lo_j - lo, e_j - e), min(hi_j - hi, horizon - tmin_j - r)])."""
        d_lo, d_hi = Fraction(0), self.horizon - self.tmin[j] - r
        for (lo_j, hi_j, e_j), (lo, hi, e) in zip(w_j, windows):
            d_lo = max(d_lo, lo_j - lo, e_j - e)
            d_hi = min(d_hi, hi_j - hi)
        return d_lo if d_lo <= d_hi else None

    def stn_shift(self, j: int, w_j, windows, r) -> Optional[Fraction]:
        """``shift`` asked of an STN (TP_MCTS_WILAO_FIT = stn). Nodes: O, table state j's event;
        N, the leaf's now (x = N - O is the shift); per running action the leaf's lo, hi and e
        end points, each placed from N and bounded from O by j's window:
            P - N = lo, P - O >= lo_j;   H - N = hi, H - O <= hi_j;   E - N = e, E - O >= e_j
            0 <= N - O <= horizon - tmin_j - r
        Every point of the leaf's window lands in j's (inclusion, not just overlap). Consistent
        (Floyd-Warshall, no negative cycle) iff the leaf fits; the smallest x is -dist[N][O]."""
        inf = None
        n = 2 + 3 * len(windows)
        dist = [[Fraction(0) if a == b else inf for b in range(n)] for a in range(n)]

        def edge(a, b, w):                              # b - a <= w
            if dist[a][b] is None or w < dist[a][b]:
                dist[a][b] = w

        edge(0, 1, self.horizon - self.tmin[j] - r)
        edge(1, 0, Fraction(0))
        for k, ((lo_j, hi_j, e_j), (lo, hi, e)) in enumerate(zip(w_j, windows)):
            p, h, q = 2 + 3 * k, 3 + 3 * k, 4 + 3 * k
            for node, at in ((p, lo), (h, hi), (q, e)):
                edge(1, node, at)
                edge(node, 1, -at)
            edge(p, 0, -lo_j)
            edge(0, h, hi_j)
            edge(q, 0, -e_j)
        for k in range(n):
            row_k = dist[k]
            for a in range(n):
                via = dist[a][k]
                if via is None:
                    continue
                row_a = dist[a]
                for b in range(n):
                    if row_k[b] is not None and (row_a[b] is None or via + row_k[b] < row_a[b]):
                        row_a[b] = via + row_k[b]
        if any(dist[a][a] < 0 for a in range(n)):
            return None
        return -dist[1][0]

    def _fits(self, facts, queue, windows, r) -> List[Tuple[Fraction, int, tuple]]:
        """(d, j, w_j) for every table state j with this F and Q that the leaf fits after shift d."""
        fit = self.stn_shift if self.fit == "stn" else self.shift
        out = []
        for w_j, j in self._by_fq.get((facts, queue), ()):
            d = fit(j, w_j, windows, r)
            if d is not None:
                out.append((d, j, w_j))
        return out

    def fitting_values(self, facts, queue, windows, r) -> List[float]:
        """V(j, r + d) for every fitting table state j."""
        return [self._read(j, r + d) for d, j, _w in self._fits(facts, queue, windows, r)]
