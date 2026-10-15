"""ILAO* on the MDP that TP-MCTS searches, solved offline from the initial state to optimal.

Same model as TP-MCTS, re-derived nowhere:
    node      = (state, STN) -- the real ``mdp`` state and the search STN
    actions   = ``mdp.legal_actions(state)`` (starts and ends), kept only if the STN stays
                consistent after ``update_stn(stn.clone(), action, previous node)``: exactly how
                ``C_ANode`` builds TP-MCTS's children. The deadline is a constraint inside the
                STN (``create_init_stn``), so an action that cannot fit is not consistent.
    "when"    = the STN: every event at its earliest consistent time; a later choice of end
                order may push an earlier start back. No time choice in the search.
    outcomes  = ``mdp.transition_function(state, action)``
    value     = 1 at a goal state (reward_mode terminal), 0 where no consistent action is left;
                V(n) = max over actions of sum p * V(child): the best success probability.

Merging (TP-MCTS does not merge): two prefixes are one node when they have the same facts and
the same constraints on the future. Every later event is tied only to the plan start, the plan
end, the last chosen event and the pending ends, so the node key is the facts plus the STN's
minimal network over exactly those points (all-pairs bounds t_u - t_v <= M[u, v]).

Heuristic: the windows PDB (``windows_ilao_pdb``), fed exactly as TP-MCTS feeds it: r = D - the
STN's current end time, running actions with the STN's earliest remaining times. It is an upper
bound, so ILAO* (Hansen and Zilberstein 2001) returns the optimum once nothing is left to expand.

Loops: time is part of the node, so every loop costs time and the graph is acyclic -- except
zero-time actions (instantaneous ones such as turn_on_hand, which fails 0.2 and stays put).
An option whose outcomes return to its own node is folded in closed form (p / (1 - q); dropped
if q = 1). Any other cycle is counted and reported (``stats["cycles"]``).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

_EPS = 1e-12


# ---------------------------------------------------------------------------
# The STN: TP-MCTS's own class, with consistency decided exactly (Floyd-Warshall)
# ---------------------------------------------------------------------------

def _num(x):
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x)
    return x


def _exact_stn_class():
    """TP-MCTS's DeltaSimpleTemporalNetwork with its incremental check switched off.

    That check compares events with ``==``, and STNPlanNode.__eq__ calls two runs of the same action
    equal, so after a retry it can report a consistent STN as inconsistent (depending on the order
    the edges were added). Here every edge is only recorded; consistency and the earliest times
    come from Floyd-Warshall on the whole graph: consistent iff no negative cycle, and the earliest
    time of v is -min(0, min_u M[u, v]) -- the library's own convention (a virtual source with a
    0-edge to every event). TP-MCTS itself is not touched."""
    from unified_planning.plans.stn.delta_stn import DeltaSimpleTemporalNetwork

    class ExactSTN(DeltaSimpleTemporalNetwork):
        def __init__(self, constraints=None, distances=None, is_sat=True, epsilon=0):
            super().__init__(constraints, distances, True, epsilon)
            self._solved = None                      # (consistent, index, M); None = edges changed

        def _inc_check(self, x, y, b):
            self._solved = None
            return True

        def remove_endPlan_constraint(self, x, end_plan):
            super().remove_endPlan_constraint(x, end_plan)
            self._solved = None

        def copy_stn(self):
            out = ExactSTN(self._constraints.copy(), self._distances.copy())
            out._solved = self._solved
            return out

        def solve(self):
            if self._solved is None:
                nodes = list(self._distances)
                index = {v: i for i, v in enumerate(nodes)}
                n = len(nodes)
                M = [[None] * n for _ in range(n)]
                for i in range(n):
                    M[i][i] = 0
                for x, nb in self._constraints.items():
                    i = index[x]
                    while nb is not None:            # x - dst <= bound
                        j, b = index[nb.dst], _num(nb.bound)
                        if M[i][j] is None or b < M[i][j]:
                            M[i][j] = b
                        nb = nb.next
                consistent = True
                for k in range(n):
                    Mk = M[k]
                    for i in range(n):
                        mik = M[i][k]
                        if mik is None:
                            continue
                        Mi = M[i]
                        for j in range(n):
                            mkj = Mk[j]
                            if mkj is not None and (Mi[j] is None or mik + mkj < Mi[j]):
                                Mi[j] = mik + mkj
                    if any(M[i][i] < 0 for i in range(n)):
                        consistent = False           # negative cycle
                        break
                if consistent:
                    for v, j in index.items():
                        col = [M[i][j] for i in range(n) if M[i][j] is not None]
                        self._distances[v] = min([0] + col)
                self._solved = (consistent, index, M)
            return self._solved

        def check_stn(self):
            return self.solve()[0]

        def get_stn_model(self, x):
            self.solve()
            return -1 * self._distances[x]

    return ExactSTN


_EXACT = None


def _exact_class():
    global _EXACT
    if _EXACT is None:
        _EXACT = _exact_stn_class()
    return _EXACT


def exact_stn(stn):
    """A copy of an STNPlan whose network is the exact (Floyd-Warshall) one."""
    from unified_planning.plans.stn.stn_plan import STNPlan

    net = stn._stn
    out = STNPlan([], _stn=_exact_class()(net._constraints.copy(), net._distances.copy()))
    out._potential_end_actions = dict(stn._potential_end_actions)
    return out


# ---------------------------------------------------------------------------
# Node key: facts + the STN's minimal network over the points the future can touch
# ---------------------------------------------------------------------------

def stn_signature(stn, prev) -> tuple:
    return stn_minimal_network(stn, prev)[0]


def stn_minimal_network(stn, prev):
    """All-pairs bounds t_u - t_v <= M[u, v] among plan start, plan end, the last chosen event and
    the pending ends (by action name), read off the Floyd-Warshall matrix of a consistent STN.
    Returns (signature, [(label, node)], {(u_node, v_node): bound})."""
    import unified_planning as up
    from unified_planning.model.timing import TimepointKind

    if not hasattr(stn._stn, "solve"):
        stn = exact_stn(stn)
    _consistent, index, M = stn._stn.solve()
    gs = up.plans.stn.STNPlanNode(TimepointKind.GLOBAL_START)
    ge = up.plans.stn.STNPlanNode(TimepointKind.GLOBAL_END)
    points: List[Tuple[str, object]] = [("GS", gs), ("GE", ge)]
    if prev is not None:
        points.append(("P", prev))
    points += sorted(((n.action_instance.action.name, n) for n in stn._potential_end_actions), key=lambda t: t[0])
    out = []
    bounds = {}
    for label, src in points:
        i = index.get(src)
        if i is None:
            out.append((label, None))
            continue
        row = []
        for lab2, dst in points:
            if dst is src:
                continue
            j = index.get(dst)
            b = None if j is None else M[i][j]
            if b is not None:
                bounds[(src, dst)] = b
            row.append((lab2, _num(b)))
        out.append((label, tuple(row)))
    return tuple(out), points, bounds


def compact_stn(stn, bounds):
    """A new (exact) STN holding only the kept points, with their minimal-network bounds: the same
    future (every later event is tied only to these points) at a constant size, instead of the
    whole history. The pending ends keep their start nodes as values (update_stn only compares them).

    update_stn later DELETES every direct edge "last event -> plan end" (remove_endPlan_constraint)
    and adds "last event <= new event <= plan end" in the same step, so in the full STN nothing
    implied is ever lost. Here every u -> GE bound is implied, so it goes through a helper point:
    deleting the direct edge must not delete it."""
    from unified_planning.plans.stn.stn_plan import STNPlan

    import unified_planning as up
    from unified_planning.model.timing import TimepointKind

    ge = up.plans.stn.STNPlanNode(TimepointKind.GLOBAL_END)
    gs = up.plans.stn.STNPlanNode(TimepointKind.GLOBAL_START)
    net = _exact_class()()
    for (u, v), b in bounds.items():
        if v == ge and u != gs:
            # u -> GE is implied (a path), but u may be the last event at some later step, and
            # update_stn then DELETES every direct u -> GE edge: route it through a helper point.
            aux = ("implied", object())
            net.add(u, aux, b)
            net.add(aux, v, 0)
        else:
            net.add(u, v, b)                             # t_u - t_v <= b
    out = STNPlan([], _stn=net)
    out._potential_end_actions = dict(stn._potential_end_actions)
    return out


class Node:
    __slots__ = ("key", "state", "stn", "prev", "value", "best", "options", "goal", "dead",
                 "current_time", "h0", "depth", "last")

    def __init__(self, key, state, stn, prev):
        self.key = key
        self.state = state
        self.stn = stn                 # kept until the node is expanded
        self.prev = prev
        self.value = 0.0
        self.best = None               # index into options
        self.options = None            # [(action name, ((p, child key), ...))]; None = not expanded
        self.goal = False
        self.dead = False
        self.current_time = 0.0
        self.h0 = 0.0                  # the heuristic's value when the node was created
        self.depth = 0                 # fewest steps from the root seen so far
        self.last = None               # canonical_starts: the last start, if it may be reordered


def _commute_table(mdp) -> Dict[str, frozenset]:
    """For every deterministic start: the deterministic starts it commutes with -- neither one's
    effects touch the other's preconditions or effects (so both orders are legal and reach the same
    facts). Ends, instantaneous and probabilistic actions are never reordered."""
    info = {}
    for a in mdp.problem.actions:
        if type(a).__name__ != "InstantaneousStartAction" or getattr(a, "probabilistic_effects", None):
            continue
        reads = frozenset(a.pos_preconditions) | frozenset(a.neg_preconditions)
        writes = frozenset(a.add_effects) | frozenset(a.del_effects)
        info[a.name] = (reads, writes)
    table = {}
    for x, (rx, wx) in info.items():
        table[x] = frozenset(y for y, (ry, wy) in info.items()
                             if y != x and not (wx & (ry | wy)) and not (wy & rx))
    return table


# ---------------------------------------------------------------------------
# Diagnostics (opt-in): what the search does, per depth
# ---------------------------------------------------------------------------

class LAODiagnostics:
    """Counters filled while ILAO* runs (one dict per depth, nothing per node beyond depth and h0),
    plus a lower bound on V(s0) every ``interval`` seconds. Read with ``TPMDPLAO.diagnostics()``.

    per expansion: legal actions, how many the STN rejects, options kept, outcomes per option,
    children that are new vs merged into an existing node, zero-time (instantaneous) options,
    folded self-loops, and the heuristic's own ranking of the options: Q_h(a) = sum p * h(child),
    the spread best - worst and how many options are within 1e-3 / 1e-2 of the best (ties: an
    upper bound can never cut those)."""

    TIES = (1e-3, 1e-2)

    def __init__(self, deadline: float, interval: float = 600.0):
        self.deadline = float(deadline)
        self.interval = float(interval)
        self.by_depth: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.progress: List[Dict[str, float]] = []
        self.started = time.perf_counter()
        self._last = self.started

    def expansion(self, node, legal: int, rejected: int, option_rows, q_values) -> None:
        d = self.by_depth[node.depth]
        d["expanded"] += 1
        d["legal"] += legal
        d["stn_rejected"] += rejected
        d["h0_sum"] += node.h0
        d["time_left_sum"] += self.deadline - node.current_time
        if not q_values:
            d["dead_ends"] += 1
            return
        for outcomes, new, merged, zero_time, folded in option_rows:
            d["options"] += 1
            d["outcomes"] += outcomes
            d["new_children"] += new
            d["merged_children"] += merged
            d["zero_time_options"] += zero_time
            d["folded_loops"] += folded
        best = max(q_values)
        spread = best - min(q_values)
        d["spread_sum"] += spread
        d["spread_max"] = max(d["spread_max"], spread)
        for eps in self.TIES:
            d[f"ties_{eps:g}"] += sum(1 for v in q_values if best - v <= eps) - 1
        if len(q_values) == 1:
            d["single_option"] += 1

    def due(self) -> bool:
        return self.interval > 0 and time.perf_counter() - self._last >= self.interval

    def record(self, solver: "TPMDPLAO") -> None:
        self._last = time.perf_counter()
        self.progress.append(dict(seconds=round(self._last - self.started, 1),
                                  generated=solver.stats["generated"], expanded=solver.stats["expanded"],
                                  lower_bound=round(solver.policy_lower_bound(), 6),
                                  upper_bound=round(solver.root.value, 6)))


# ---------------------------------------------------------------------------
# ILAO*
# ---------------------------------------------------------------------------

class TPMDPLAO:
    """ILAO* over TP-MCTS's (state, STN) MDP. ``heuristic(node) -> float`` must be an upper bound;
    ``heuristic=None`` is the blind control (h = 1)."""

    def __init__(self, mdp, heuristic: Optional[Callable] = None, epsilon: float = 1e-9,
                 max_nodes: int = 0, time_budget: float = 0.0, on_pass: Optional[Callable] = None,
                 compact: bool = True, diagnostics: bool = False, diagnostics_interval: float = 600.0,
                 canonical_starts: bool = False):
        from unified_planning.engines.utils import create_init_stn

        self.mdp = mdp
        self.heuristic = heuristic
        self.epsilon = float(epsilon)
        self.max_nodes = int(max_nodes)
        self.time_budget = float(time_budget)
        self.on_pass = on_pass          # called after every pass with (solver, seconds so far)
        self.compact = compact          # store each node's STN reduced to its kept points
        self.nodes: Dict[tuple, Node] = {}
        self.diag = LAODiagnostics(float(mdp.deadline()), diagnostics_interval) if diagnostics else None
        # canonical_starts: two independent starts in a row (no event between them) are only taken in
        # name order -- the other order reaches the same facts. Not exact in general: the STN keeps
        # "a before b" vs "b before a", and a later end order can need one of them.
        self.canonical = bool(canonical_starts)
        self._commute = _commute_table(mdp) if self.canonical else {}
        self._deadline_at = None        # perf_counter() time at which solve() stops (time_budget)
        self.stats = dict(generated=0, expanded=0, passes=0, backups=0, h_calls=0, h_seconds=0.0,
                          key_seconds=0.0, expand_seconds=0.0, inconsistent=0, folded=0, dropped_noop=0,
                          cycles=0, seconds=0.0, complete=False, canonical_skipped=0)
        stn = exact_stn(create_init_stn(mdp))           # exact consistency (Floyd-Warshall)
        self.root = self._node(mdp.initial_state(), stn, None)

    # -- nodes ---------------------------------------------------------------
    def node_key(self, state, signature, last=None) -> tuple:
        """Facts + the STN's bounds on the future; in canonical mode also the reorderable last start
        (which starts may follow depends on it)."""
        if self.canonical:
            return (frozenset(state.predicates), signature, last)
        return (frozenset(state.predicates), signature)

    def last_of(self, action):
        """The name a node remembers after `action` (canonical mode): a start that may be reordered."""
        return action.name if self.canonical and action.name in self._commute else None

    def _node(self, state, stn, prev, depth: int = 0, last=None) -> Node:
        t0 = time.perf_counter()
        signature, _points, bounds = stn_minimal_network(stn, prev)
        key = self.node_key(state, signature, last)
        self.stats["key_seconds"] += time.perf_counter() - t0
        node = self.nodes.get(key)
        if node is not None:
            if depth < node.depth:
                node.depth = depth
            return node
        if self.compact and prev is not None:
            stn = compact_stn(stn, bounds)
        node = Node(key, state, stn, prev)
        node.depth = depth
        node.last = last
        node.current_time = float(stn.get_current_end_time())
        self.nodes[key] = node
        self.stats["generated"] += 1
        if self.mdp.is_terminal(state):
            node.goal, node.value, node.options, node.stn = True, 1.0, [], None
        elif self.heuristic is None:
            node.value = 1.0
        else:
            t1 = time.perf_counter()
            node.value = float(self.heuristic(node))
            self.stats["h_seconds"] += time.perf_counter() - t1
            self.stats["h_calls"] += 1
        node.h0 = node.value
        return node

    def _expand(self, node: Node) -> None:
        from unified_planning.engines.utils import update_stn

        t0 = time.perf_counter()
        options = []
        legal = rejected = 0
        rows = []                                        # per option, for the diagnostics
        commute_with_last = self._commute.get(node.last, ()) if node.last else ()
        for action in self.mdp.legal_actions(node.state):
            legal += 1
            if commute_with_last and action.name in commute_with_last and action.name < node.last:
                self.stats["canonical_skipped"] += 1     # the same two starts in name order exist
                continue
            child_stn = node.stn.clone()
            new_prev = update_stn(child_stn, action, node.prev)
            if not child_stn.is_consistent():
                self.stats["inconsistent"] += 1
                rejected += 1
                continue
            folded: Dict[tuple, float] = {}
            children = {}
            new = merged = 0
            for next_state, p in self.mdp.transition_function(node.state, action):
                p = float(p)
                if p <= _EPS:
                    continue
                before = self.stats["generated"]
                child = self._node(next_state, child_stn, new_prev, node.depth + 1, self.last_of(action))
                if self.stats["generated"] > before:
                    new += 1
                else:
                    merged += 1
                folded[child.key] = folded.get(child.key, 0.0) + p
                children[child.key] = child
            q = folded.pop(node.key, 0.0)                # zero-time outcome back to this node
            if q > 0:
                if q >= 1 - 1e-12:
                    self.stats["dropped_noop"] += 1      # never leaves: no value of its own
                    continue
                self.stats["folded"] += 1
                folded = {k: p / (1 - q) for k, p in folded.items()}
            options.append((action.name, tuple((p, k) for k, p in folded.items())))
            rows.append((len(folded), new, merged, int(type(action).__name__ == "InstantaneousAction"),
                         int(q > 0)))
        node.options = options
        if self.diag is not None:
            q_values = [sum(p * self.nodes[k].value for p, k in branches) for _name, branches in options]
            self.diag.expansion(node, legal, rejected, rows, q_values)
        node.stn = None                                  # expanded: the children hold their own STNs
        if not options:
            node.dead, node.value = True, 0.0
        self.stats["expanded"] += 1
        self.stats["expand_seconds"] += time.perf_counter() - t0

    def _backup(self, node: Node) -> float:
        if node.goal or node.dead:
            return 0.0
        best_v, best_i = 0.0, None
        for i, (_name, branches) in enumerate(node.options):
            v = sum(p * self.nodes[k].value for p, k in branches)
            if best_i is None or v > best_v + _EPS:
                best_v, best_i = v, i
        residual = abs(node.value - best_v)
        if node.best != best_i:
            self._policy_changed = True
        node.value, node.best = best_v, best_i
        self.stats["backups"] += 1
        return residual

    # -- search --------------------------------------------------------------
    def solve(self) -> float:
        started = time.perf_counter()
        self._deadline_at = started + self.time_budget if self.time_budget else None
        while True:
            self.stats["passes"] += 1
            self._policy_changed = False
            expanded, residual = self._pass()
            if self.on_pass is not None:
                self.on_pass(self, time.perf_counter() - started)
            if self.max_nodes and len(self.nodes) >= self.max_nodes:
                break
            if self._deadline_at is not None and time.perf_counter() >= self._deadline_at:
                break
            if expanded == 0 and residual < self.epsilon and not self._policy_changed:
                self.stats["complete"] = True
                break
        self.stats["seconds"] = time.perf_counter() - started
        if self.diag is not None:
            self.diag.record(self)                       # the final lower / upper bound
        return self.root.value

    def _pass(self) -> Tuple[int, float]:
        """One depth-first pass over the best partial policy: expand its tips, back up in post-order."""
        expanded, residual = 0, 0.0
        on_stack, done = set(), set()
        stack = [(self.root, False)]
        while stack:
            node, returning = stack.pop()
            if returning:
                on_stack.discard(node.key)
                done.add(node.key)
                residual = max(residual, self._backup(node))
                continue
            if node.key in done:
                continue
            if node.key in on_stack:
                self.stats["cycles"] += 1               # a zero-time cycle over several nodes
                continue
            if node.goal or node.dead:
                done.add(node.key)
                continue
            if node.options is None:
                if self._deadline_at is not None and time.perf_counter() >= self._deadline_at:
                    break                                # out of time: values stay upper bounds
                self._expand(node)
                expanded += 1
                if self.diag is not None and self.diag.due():
                    self.diag.record(self)
                if node.dead:
                    done.add(node.key)
                    continue
                self._backup(node)                       # pick the best option from the children's h
            on_stack.add(node.key)
            stack.append((node, True))
            if node.best is not None:
                for _p, k in node.options[node.best][1]:
                    child = self.nodes[k]
                    if child.key not in done:
                        stack.append((child, False))
        return expanded, residual

    def expand_all(self) -> bool:
        """Every node reachable from the root, every action, every outcome -- no search, no pruning.
        The size of the whole MDP, to compare against what ILAO* touched. Returns True when done."""
        started = time.perf_counter()
        frontier = [self.root]
        while frontier:
            if self.max_nodes and len(self.nodes) >= self.max_nodes:
                return False
            if self.time_budget and time.perf_counter() - started > self.time_budget:
                return False
            node = frontier.pop()
            if node.goal or node.dead or node.options is not None:
                continue
            self._expand(node)
            for _name, branches in node.options:
                for _p, k in branches:
                    child = self.nodes[k]
                    if child.options is None and not child.goal:
                        frontier.append(child)
        self.stats["seconds"] = time.perf_counter() - started
        self.stats["complete"] = True
        return True

    # -- reporting -----------------------------------------------------------
    def policy_counts(self) -> Dict[str, int]:
        """Nodes of the optimal policy from the root (best action, every outcome): decision nodes
        (an action is chosen), goal leaves, dead ends (no consistent action) and unsolved tips."""
        seen, frontier = set(), [self.root]
        counts = dict(decision=0, goal=0, dead=0, tip=0)
        while frontier:
            n = frontier.pop()
            if n.key in seen:
                continue
            seen.add(n.key)
            if n.goal:
                counts["goal"] += 1
            elif n.dead:
                counts["dead"] += 1
            elif n.options is None or n.best is None:
                counts["tip"] += 1
            else:
                counts["decision"] += 1
                frontier += [self.nodes[k] for _p, k in n.options[n.best][1]]
        counts["total"] = len(seen)
        return counts

    def policy_lower_bound(self) -> float:
        """V of the current best partial policy with every unsolved tip counted as 0: a policy that
        can actually be followed, so a lower bound on V*(s0). Cycles count as 0 (conservative)."""
        value: Dict[tuple, float] = {}
        on_path = set()
        stack = [(self.root.key, False)]
        while stack:
            k, ready = stack.pop()
            if k in value:
                continue
            n = self.nodes[k]
            if n.goal:
                value[k] = 1.0
                continue
            if n.dead or n.options is None or n.best is None:
                value[k] = 0.0
                continue
            branches = n.options[n.best][1]
            if ready:
                on_path.discard(k)
                value[k] = sum(p * value.get(c, 0.0) for p, c in branches)
                continue
            on_path.add(k)
            stack.append((k, True))
            stack.extend((c, False) for _p, c in branches if c not in value and c not in on_path)
        return value[self.root.key]

    def diagnostics(self, walk: bool = True) -> Dict[str, object]:
        """The diagnostics report (needs diagnostics=True). walk=False skips the passes over the
        graph (h-pruning, tightness): only the counters and the lower-bound history."""
        if self.diag is None:
            return {}
        rows = []
        pruned_by_depth: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
        if walk:
            for n in self.nodes.values():
                if not n.options:
                    continue
                for i, (_name, branches) in enumerate(n.options):
                    explored = i == n.best or any(self.nodes[k].options is not None for _p, k in branches)
                    pruned_by_depth[n.depth][0] += 1
                    pruned_by_depth[n.depth][1] += 0 if explored else 1
        for depth in sorted(self.diag.by_depth):
            d = self.diag.by_depth[depth]
            e = max(1.0, d["expanded"])
            opts = max(1.0, d["options"])
            kids = max(1.0, d["new_children"] + d["merged_children"])
            row = dict(depth=depth, expanded=int(d["expanded"]),
                       legal_per_node=round(d["legal"] / e, 2),
                       stn_rejected_pct=round(100 * d["stn_rejected"] / max(1.0, d["legal"]), 1),
                       options_per_node=round(d["options"] / e, 2),
                       outcomes_per_option=round(d["outcomes"] / opts, 2),
                       merged_pct=round(100 * d["merged_children"] / kids, 1),
                       q_spread_mean=round(d["spread_sum"] / e, 5), q_spread_max=round(d["spread_max"], 5),
                       h0_mean=round(d["h0_sum"] / e, 5), time_left_mean=round(d["time_left_sum"] / e, 2),
                       zero_time_options=int(d["zero_time_options"]), folded_loops=int(d["folded_loops"]),
                       dead_ends=int(d["dead_ends"]))
            row["ties_within_0.001"] = round(d["ties_0.001"] / e, 2)    # per node, besides the best
            row["ties_within_0.01"] = round(d["ties_0.01"] / e, 2)
            if walk:
                total, pruned = pruned_by_depth.get(depth, [0, 0])
                row["h_pruned_pct"] = round(100 * pruned / max(1, total), 1)
            rows.append(row)
        if walk:
            groups: Dict[int, set] = defaultdict(set)
            counts: Dict[int, int] = defaultdict(int)
            for n in self.nodes.values():
                groups[n.depth].add((n.key[0], n.current_time))
                counts[n.depth] += 1
            for row in rows:
                g = len(groups.get(row["depth"], ())) or 1
                row["nodes_per_facts_time"] = round(counts.get(row["depth"], 0) / g, 2)
        out: Dict[str, object] = dict(by_depth=rows, progress=list(self.diag.progress),
                                      cycles=self.stats["cycles"], folded=self.stats["folded"],
                                      canonical_skipped=self.stats["canonical_skipped"],
                                      dropped_noops=self.stats["dropped_noop"])
        if walk:
            total = sum(v[0] for v in pruned_by_depth.values())
            pruned = sum(v[1] for v in pruned_by_depth.values())
            out["h_pruned_options_pct"] = round(100 * pruned / max(1, total), 2)
            out["tightness"] = self._tightness() if self.stats["complete"] else (
                "run not finished: V* unknown below the root -- see progress (lower / upper bound)")
        return out

    def _tightness(self) -> Dict[str, object]:
        """On a finished run every node of the optimal policy has V = V*: compare it with h0 there.
        h0 < V* would be an admissibility violation (h must be an upper bound)."""
        seen, frontier = set(), [self.root]
        by_depth: Dict[int, List[float]] = defaultdict(list)
        violations = []
        while frontier:
            n = frontier.pop()
            if n.key in seen:
                continue
            seen.add(n.key)
            if n.goal or n.dead or not n.options or n.best is None:
                continue
            gap = n.h0 - n.value
            by_depth[n.depth].append(gap)
            if gap < -1e-9:
                violations.append(dict(depth=n.depth, h0=n.h0, v_star=n.value))
            frontier.extend(self.nodes[k] for _p, k in n.options[n.best][1])
        rows = []
        for depth in sorted(by_depth):
            g = by_depth[depth]
            rows.append(dict(depth=depth, policy_nodes=len(g), gap_mean=round(sum(g) / len(g), 6),
                             gap_max=round(max(g), 6), exact_pct=round(100 * sum(1 for x in g if abs(x) < 1e-6) / len(g), 1)))
        allg = [x for g in by_depth.values() for x in g]
        return dict(policy_decision_nodes=len(allg), gap_mean=round(sum(allg) / max(1, len(allg)), 6),
                    gap_max=round(max(allg) if allg else 0.0, 6), admissibility_violations=violations[:20],
                    violations=len(violations), by_depth=rows)

    def policy_size(self) -> int:
        return self.policy_counts()["total"]

    def best_action(self, node: Optional[Node] = None) -> Optional[str]:
        node = node or self.root
        if not node.options or node.best is None:
            return None
        return node.options[node.best][0]


# ---------------------------------------------------------------------------
# Executing the policy in CoMDP+, the way TP-MCTS executes in the real world
# ---------------------------------------------------------------------------

class PolicyExecution:
    """Run the solved policy as TP-MCTS's ``plan`` loop runs an action in the real world:

        outcome  : ``mdp.step(state, action)`` (sampled) -- or every outcome of
                   ``mdp.transition_function`` (``exact``)
        time     : ``update_stn(stn, action, prev, type='SetTime')`` -- the action is fixed at its
                   earliest consistent time, and that time can never move again

    The search STN keeps every past time flexible: a later choice may push an earlier start back.
    Execution cannot. So every step checks what the policy took for granted:

        illegal       the policy's action is not legal in the real state
        unschedulable with the past times fixed, the action no longer fits (STN inconsistent,
                      deadline included) -- an optimistic timing assumption
        off_policy    the real outcome is not one of the outcomes the policy planned for
        goal_mismatch the policy and the real MDP disagree on whether the goal is reached

    A run that hits any of these counts as a failure (TP-MCTS would stop on its assert)."""

    KINDS = ("illegal", "unschedulable", "off_policy", "goal_mismatch")

    def __init__(self, lao: "TPMDPLAO"):
        from unified_planning.engines.utils import create_init_stn

        self.lao = lao
        self.mdp = lao.mdp
        self.actions = {a.name: a for a in self.mdp.problem.actions}
        self._init_stn = lambda: exact_stn(create_init_stn(self.mdp))

    def _plan_child(self, stn, prev, action, next_state):
        """The search's own successor: the same key ILAO* gave this outcome."""
        from unified_planning.engines.utils import update_stn

        child = stn.clone()
        new_prev = update_stn(child, action, prev)
        sig, _points, bounds = stn_minimal_network(child, new_prev)
        key = self.lao.node_key(next_state, sig, self.lao.last_of(action))
        if self.lao.compact and new_prev is not None:
            child = compact_stn(child, bounds)
        return key, child, new_prev

    def _execute(self, stn, prev, action):
        """Real-world time: fixed at the earliest consistent time. None if it does not fit."""
        from unified_planning.engines.utils import update_stn

        child = stn.clone()
        new_prev = update_stn(child, action, prev, type="SetTime")
        if not child.is_consistent():
            return None
        return child, new_prev, child.get_current_time(new_prev)

    def _decide(self, node, state):
        """The policy's action at this node, or the reason it cannot be applied."""
        if node.goal or node.dead or node.best is None:
            return None, None
        name = node.options[node.best][0]
        if name not in {a.name for a in self.mdp.legal_actions(state)}:
            return None, "illegal"
        return self.actions[name], None

    # -- sampled runs (mdp.step, as in TP-MCTS's plan loop) ------------------------------------------
    def run(self, runs: int = 1000, seed: int = 0, keep_examples: int = 3) -> Dict:
        import numpy as np

        np.random.seed(seed)
        lao, mdp, deadline = self.lao, self.mdp, float(self.mdp.deadline())
        out = dict(runs=runs, success=0, dead_end=0, end_times=[], examples=[],
                   **{k: 0 for k in self.KINDS})
        for _ in range(runs):
            node, state = lao.root, mdp.initial_state()
            p_stn, p_prev = self._init_stn(), None
            e_stn, e_prev = self._init_stn(), None
            trace, result = [], None
            while result is None:
                if node.goal:
                    ok = mdp.is_terminal(state) and e_stn.get_current_end_time() <= deadline
                    result = "success" if ok else "goal_mismatch"
                    break
                action, why = self._decide(node, state)
                if action is None:
                    result = why or "dead_end"
                    break
                executed = self._execute(e_stn, e_prev, action)
                if executed is None:
                    trace.append(f"{action.name} @ -- (does not fit)")
                    result = "unschedulable"
                    break
                e_stn, e_prev, t = executed
                trace.append(f"{action.name} @ {t}")
                terminal, next_state, _reward = mdp.step(state, action)
                key, p_stn, p_prev = self._plan_child(p_stn, p_prev, action, next_state)
                if key != node.key and key not in {k for _p, k in node.options[node.best][1]}:
                    result = "off_policy"
                    break
                state, node = next_state, lao.nodes[key]
                if terminal and not node.goal:
                    result = "goal_mismatch"
            out[result] += 1
            if result == "success":
                out["end_times"].append(float(e_stn.get_current_end_time()))
            elif result in self.KINDS and len(out["examples"]) < keep_examples:
                out["examples"].append(dict(kind=result, trace=trace))
        n = max(1, runs)
        rate = out["success"] / n
        out["success_rate"] = round(rate, 6)
        out["ci95"] = round(1.96 * math.sqrt(rate * (1 - rate) / n), 6)
        out["policy_value"] = round(lao.root.value, 6)
        out["violations"] = sum(out[k] for k in self.KINDS)
        times = out.pop("end_times")
        out["max_end_time"] = max(times) if times else None
        return out

    # -- every outcome (transition_function), exact probability ----------------------------------
    def exact(self, max_states: int = 200000) -> Dict:
        """Sum over every outcome of the policy, with real-world (fixed) times: the probability that
        executing the policy reaches the goal, and the probability mass of each violation. Merged on
        (search node, execution STN's minimal network), as in the search."""
        import sys

        sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
        lao, mdp, deadline = self.lao, self.mdp, float(self.mdp.deadline())
        memo: Dict[tuple, Tuple[float, Dict[str, float]]] = {}
        on_stack = set()
        info = dict(states=0, cycles=0, complete=True, examples=[])

        def key_of(node, e_stn, e_prev):
            return (node.key, stn_signature(e_stn, e_prev))

        def value(node, state, p_stn, p_prev, e_stn, e_prev, mkey, trace):
            if mkey in memo:
                return memo[mkey]
            info["states"] += 1
            if info["states"] > max_states:
                info["complete"] = False
                return 0.0, {}
            if node.goal:
                ok = mdp.is_terminal(state) and e_stn.get_current_end_time() <= deadline
                res = (1.0, {}) if ok else (0.0, {"goal_mismatch": 1.0})
                memo[mkey] = res
                return res
            action, why = self._decide(node, state)
            if action is None:
                res = (0.0, {why: 1.0} if why else {})
                memo[mkey] = res
                return res
            executed = self._execute(e_stn, e_prev, action)
            if executed is None:
                if len(info["examples"]) < 3:
                    info["examples"].append(trace + [f"{action.name} @ -- (does not fit)"])
                memo[mkey] = (0.0, {"unschedulable": 1.0})
                return memo[mkey]
            e2, ep2, t = executed
            planned = {k for _p, k in node.options[node.best][1]}
            on_stack.add(mkey)
            v, bad, loop_q = 0.0, defaultdict(float), 0.0
            for next_state, p in mdp.transition_function(state, action):
                p = float(p)
                if p <= _EPS:
                    continue
                key, p2, pp2 = self._plan_child(p_stn, p_prev, action, next_state)
                if key != node.key and key not in planned:
                    bad["off_policy"] += p
                    continue
                child = lao.nodes[key]
                ckey = key_of(child, e2, ep2)
                if ckey == mkey:
                    loop_q += p                           # zero-time retry back to this very state
                    continue
                if ckey in on_stack:
                    info["cycles"] += 1
                    continue
                cv, cbad = value(child, next_state, p2, pp2, e2, ep2, ckey, trace + [f"{action.name} @ {t}"])
                v += p * cv
                for k, m in cbad.items():
                    bad[k] += p * m
            on_stack.discard(mkey)
            if loop_q > 0:
                scale = 1.0 / (1.0 - loop_q) if loop_q < 1 - 1e-12 else 0.0
                v *= scale
                bad = defaultdict(float, {k: m * scale for k, m in bad.items()})
            memo[mkey] = (v, dict(bad))
            return memo[mkey]

        p0, e0 = self._init_stn(), self._init_stn()
        v, bad = value(lao.root, mdp.initial_state(), p0, None, e0, None, key_of(lao.root, e0, None), [])
        return dict(executed_value=round(v, 6), policy_value=round(lao.root.value, 6),
                    gap=round(lao.root.value - v, 6),
                    violation_mass={k: round(m, 6) for k, m in bad.items()},
                    states=info["states"], cycles=info["cycles"], complete=info["complete"],
                    examples=info["examples"])


def windows_pdb_heuristic(mdp):
    """h(node) = the windows PDB, given exactly what TP-MCTS gives it at a leaf."""
    from comdp_plus_no_deadline.engines.windows_ilao_pdb import WindowsILAOPDBHeuristic
    from unified_planning.engines.solvers.mcts import _effective_temporal_depth, _stn_running_remaining

    pdb = WindowsILAOPDBHeuristic(mdp)
    deadline = mdp.deadline()

    def h(node: Node) -> float:
        current = node.current_time
        return pdb.heuristic_score(
            node.state, mdp.problem.goals,
            fixed_depth=_effective_temporal_depth(int(float(deadline)), current, deadline),
            start_time=current,
            running_remaining=_stn_running_remaining(node.stn, current))

    h.pdb = pdb
    return h
