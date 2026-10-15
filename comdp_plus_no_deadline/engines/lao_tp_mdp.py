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
                 "current_time", "h0", "depth", "last", "parent", "via", "branch", "weak", "lower")

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
        self.parent = None             # diagnostics: the node it was first created from,
        self.via = None                # by this action,
        self.branch = None             # under this root option
        self.weak = None               # diagnostics: the PDB goal group with the lowest value at h0
        self.lower = 0.0               # algorithm="bounded": a value some policy is known to reach


_GAP_EDGES = (1e-4, 1e-3, 1e-2, 5e-2)


def _gap_bucket(gap: float) -> str:
    for edge in _GAP_EDGES:
        if gap < edge:
            return f"<{edge:g}"
    return f">={_GAP_EDGES[-1]:g}"


def _schema(action_name) -> str:
    """The lifted action name: a grounded name with its object arguments (letter + digits) dropped,
    e.g. start_sample_rock_good_r0_s0_x0_h0 -> start_sample_rock_good."""
    if not action_name:
        return "(root)"
    parts = action_name.split("_")
    while len(parts) > 1 and len(parts[-1]) >= 2 and parts[-1][0].isalpha() and parts[-1][1:].isdigit():
        parts.pop()
    return "_".join(parts)


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

    def __init__(self, deadline: float, interval: float = 600.0, tips: int = 0, goals=()):
        self.deadline = float(deadline)
        self.goals = frozenset(goals)
        self._is_exec: Dict[object, bool] = {}
        self._seen_facts: set = set()
        self.by_time_left: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.by_running: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.by_goals_done: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.by_schema: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.gap_hist: Dict[str, int] = defaultdict(int)
        self.interval = float(interval)
        self.tips = int(tips)           # per sample: the k tips carrying most of V(s0), expanded and shown
        self.by_depth: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.progress: List[Dict[str, float]] = []
        self._groups: Dict[int, set] = {}
        self.by_branch: Dict[object, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.started = time.perf_counter()
        self._last = self.started

    def _running(self, facts) -> int:
        is_exec = self._is_exec
        n = 0
        for f in facts:
            flag = is_exec.get(f)
            if flag is None:
                flag = is_exec[f] = "inExecution" in str(f)
            n += flag
        return n

    def expansion(self, node, legal: int, rejected: int, option_rows, q_values) -> None:
        self.by_branch[node.branch]["expanded"] += 1
        self.by_time_left[int(round(self.deadline - node.current_time))]["expanded"] += 1
        self.by_running[self._running(node.key[0])]["expanded"] += 1
        self.by_schema[_schema(node.via)]["expanded"] += 1
        if q_values:
            best = max(q_values)
            first = True
            for q in q_values:                     # how far each alternative sits below the best
                if first and q == best:
                    first = False
                    continue
                self.gap_hist[_gap_bucket(best - q)] += 1
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

    def created(self, node) -> None:
        """Every new node: count it per (creation depth) and per (facts, time) group, so the symmetry
        measure does not need a walk at the end; and per root option (which first move it is under)."""
        b = self.by_branch[node.branch]
        b["created"] += 1
        b["max_depth"] = max(b["max_depth"], node.depth)
        facts = node.key[0]
        self.by_time_left[int(round(self.deadline - node.current_time))]["created"] += 1
        self.by_running[self._running(facts)]["created"] += 1
        self.by_goals_done[len(self.goals & facts)]["created"] += 1
        sch = self.by_schema[_schema(node.via)]
        sch["created"] += 1
        fh = hash(facts)
        if fh not in self._seen_facts:             # a fact set nobody reached before
            self._seen_facts.add(fh)
            sch["new_fact_sets"] += 1
        d = self.by_depth[node.depth]
        d["created"] += 1
        groups = self._groups.setdefault(node.depth, set())
        before = len(groups)
        groups.add(hash((node.key[0], node.current_time)))
        if len(groups) > before:
            d["facts_time_groups"] += 1

    def due(self) -> bool:
        return self.interval > 0 and time.perf_counter() - self._last >= self.interval

    def record(self, solver: "TPMDPLAO") -> None:
        policy = solver.policy_stats()
        self._last = time.perf_counter()
        self.progress.append(dict(seconds=round(self._last - self.started, 1),
                                  generated=solver.stats["generated"], expanded=solver.stats["expanded"],
                                  lower_bound=round(max(policy["lower_bound"], getattr(solver.root, "lower", 0.0)), 6),
                                  upper_bound=round(solver.root.value, 6),
                                  policy_decision=policy["decision"], policy_tips=policy["tips"],
                                  policy_goals=policy["goals"], policy_dead=policy["dead"],
                                  policy_depth=policy["max_depth"]))
        if self.tips > 0:
            self.progress[-1]["top_tips"] = solver.top_tips(self.tips)


# ---------------------------------------------------------------------------
# ILAO*
# ---------------------------------------------------------------------------

class TPMDPLAO:
    """ILAO* over TP-MCTS's (state, STN) MDP. ``heuristic(node) -> float`` must be an upper bound;
    ``heuristic=None`` is the blind control (h = 1)."""

    def __init__(self, mdp, heuristic: Optional[Callable] = None, epsilon: float = 1e-9,
                 max_nodes: int = 0, time_budget: float = 0.0, on_pass: Optional[Callable] = None,
                 compact: bool = True, diagnostics: bool = False, diagnostics_interval: float = 600.0,
                 canonical_starts: bool = False, diagnostics_tips: int = 0, algorithm: str = "ilao",
                 gap_epsilon: float = 0.0):
        from unified_planning.engines.utils import create_init_stn

        self.mdp = mdp
        self.heuristic = heuristic
        self.epsilon = float(epsilon)
        self.max_nodes = int(max_nodes)
        self.time_budget = float(time_budget)
        self.on_pass = on_pass          # called after every pass with (solver, seconds so far)
        self.compact = compact          # store each node's STN reduced to its kept points
        self.nodes: Dict[tuple, Node] = {}
        self.diag = (LAODiagnostics(float(mdp.deadline()), diagnostics_interval, diagnostics_tips,
                                    goals=getattr(mdp.problem, "goals", ()))
                     if diagnostics else None)
        # canonical_starts: two independent starts in a row (no event between them) are only taken in
        # name order -- the other order reaches the same facts. Not exact in general: the STN keeps
        # "a before b" vs "b before a", and a later end order can need one of them.
        # algorithm: "ilao" = ILAO* (upper bounds only, depth-first passes over the best partial policy);
        # "bounded" = two bounds per node (upper = h backed up, lower = what a policy is known to reach),
        # FRTDP-style trials: follow the best upper-bound action, take the outcome with the largest
        # p * (upper - lower), back both bounds up; an option whose upper bound is <= the node's lower
        # bound can never win and is never followed. Stops when upper(s0) - lower(s0) <= gap_epsilon
        # (0 = exact; > 0 = the returned policy is provably within gap_epsilon of optimal).
        if algorithm not in ("ilao", "bounded"):
            raise ValueError(f"algorithm={algorithm!r}: use ilao | bounded")
        self.algorithm = algorithm
        self.gap_epsilon = float(gap_epsilon)
        self.canonical = bool(canonical_starts)
        self._commute = _commute_table(mdp) if self.canonical else {}
        self._deadline_at = None        # perf_counter() time at which solve() stops (time_budget)
        self.stats = dict(generated=0, expanded=0, passes=0, backups=0, h_calls=0, h_seconds=0.0,
                          key_seconds=0.0, expand_seconds=0.0, inconsistent=0, folded=0, dropped_noop=0,
                          cycles=0, seconds=0.0, complete=False, canonical_skipped=0, trials=0,
                          cut_options=0)
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

    def _node(self, state, stn, prev, depth: int = 0, last=None, parent: "Node" = None, via=None) -> Node:
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
        if self.diag is not None and parent is not None:
            node.parent, node.via = parent.key, via
            node.branch = via if parent is self.root else parent.branch
        node.current_time = float(stn.get_current_end_time())
        self.nodes[key] = node
        self.stats["generated"] += 1
        if self.mdp.is_terminal(state):
            node.goal, node.value, node.options, node.stn = True, 1.0, [], None
            node.lower = 1.0
        elif self.heuristic is None:
            node.value = 1.0
        else:
            t1 = time.perf_counter()
            node.value = float(self.heuristic(node))
            self.stats["h_seconds"] += time.perf_counter() - t1
            self.stats["h_calls"] += 1
        node.h0 = node.value
        if self.diag is not None:
            pdb = getattr(self.heuristic, "pdb", None)
            values = getattr(pdb, "last_values", None) if pdb is not None else None
            if values and getattr(pdb, "groups", None) and not node.goal:
                group_mins = [min(values[i] for i in m if i < len(values)) for m in pdb.groups
                              if any(i < len(values) for i in m)]
                if group_mins:
                    node.weak = min(range(len(group_mins)), key=group_mins.__getitem__)
            self.diag.created(node)
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
                child = self._node(next_state, child_stn, new_prev, node.depth + 1, self.last_of(action),
                                   parent=node, via=action.name)
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
        if self.algorithm == "bounded":
            return self._solve_bounded()
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

    # -- algorithm = "bounded": upper and lower bounds, FRTDP-style trials -------------------------
    def _backup_both(self, node: Node) -> None:
        """Both bounds from the children: upper = max_a sum p * upper(child), lower = max_a sum p *
        lower(child). best = the action with the highest upper bound (ties: the higher lower bound)."""
        if node.goal or node.dead or not node.options:
            return
        best_u = best_l = -1.0
        best_i = None
        lower = 0.0
        q_lower = []
        for i, (_name, branches) in enumerate(node.options):
            qu = sum(p * self.nodes[k].value for p, k in branches)
            ql = sum(p * self.nodes[k].lower for p, k in branches)
            q_lower.append((qu, ql))
            lower = max(lower, ql)
            if best_i is None or qu > best_u + _EPS or (abs(qu - best_u) <= _EPS and ql > best_l):
                best_u, best_l, best_i = qu, ql, i
        node.value, node.best = best_u, best_i
        node.lower = min(lower, best_u)                  # both are exact once they meet
        self.stats["backups"] += 1

    def _trial(self) -> bool:
        """One dive from the root: best upper-bound action, outcome with the largest p * gap, expanding
        tips on the way; then both bounds are backed up along the path. False = out of time."""
        path, on_path, n = [], set(), self.root
        tol = 1e-12
        while not (n.goal or n.dead):
            if n.key in on_path:
                self.stats["cycles"] += 1                # a zero-time cycle: end the dive here
                break
            on_path.add(n.key)
            path.append(n)
            if n.options is None:
                if self._deadline_at is not None and time.perf_counter() >= self._deadline_at:
                    for m in reversed(path):
                        self._backup_both(m)
                    return False
                self._expand(n)
                if self.diag is not None and self.diag.due():
                    self.diag.record(self)
                if n.dead:
                    break
                self._backup_both(n)
            if n.value - n.lower <= tol:
                break                                    # solved here: nothing to gain below
            branches = n.options[n.best][1]
            weight, child = max(((p * (self.nodes[k].value - self.nodes[k].lower), k) for p, k in branches),
                                key=lambda t: t[0])
            if weight <= tol:
                break
            n = self.nodes[child]
        for m in reversed(path):
            self._backup_both(m)
        return True

    def _solve_bounded(self) -> float:
        started = time.perf_counter()
        self._deadline_at = started + self.time_budget if self.time_budget else None
        target = max(self.gap_epsilon, 1e-12)
        while self.root.value - self.root.lower > target:
            self.stats["trials"] += 1
            self.stats["passes"] = self.stats["trials"]
            if not self._trial():
                break
            if self.on_pass is not None and self.stats["trials"] % 1000 == 0:
                self.on_pass(self, time.perf_counter() - started)
            if self.max_nodes and len(self.nodes) >= self.max_nodes:
                break
        gap = self.root.value - self.root.lower
        self.stats["complete"] = gap <= target
        self.stats["gap"] = gap
        # The policy to report and execute: the one the lower bound belongs to (its value is reached).
        cut = 0
        for n in self.nodes.values():
            if n.options and not n.goal and not n.dead:
                best_i, best_l = None, -1.0
                for i, (_name, branches) in enumerate(n.options):
                    ql = sum(p * self.nodes[k].lower for p, k in branches)
                    qu = sum(p * self.nodes[k].value for p, k in branches)
                    if qu <= n.lower + 1e-12 and ql < n.lower - 1e-12:
                        cut += 1                         # can never beat what is already reached
                    if ql > best_l + _EPS:
                        best_l, best_i = ql, i
                n.best = best_i
        self.stats["cut_options"] = cut
        self.stats["seconds"] = time.perf_counter() - started
        if self.diag is not None:
            self.diag.record(self)
        return self.root.lower

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
        return self.policy_stats()["lower_bound"]

    def policy_stats(self) -> Dict[str, float]:
        """One walk over the current best partial policy (best action, every outcome): its lower
        bound (tips = 0) and its size -- decision nodes, unexpanded tips, goal leaves, dead ends,
        depth. The size of the policy is the least work any exact method needs from here."""
        value: Dict[tuple, float] = {}
        size = dict(decision=0, tips=0, goals=0, dead=0, max_depth=0)
        on_path = set()
        stack = [(self.root.key, False)]
        while stack:
            k, ready = stack.pop()
            if k in value:
                continue
            n = self.nodes[k]
            if n.goal:
                value[k] = 1.0
                size["goals"] += 1
                continue
            if n.dead or n.options is None or n.best is None:
                value[k] = 0.0
                size["dead" if n.dead else "tips"] += 1
                continue
            branches = n.options[n.best][1]
            if ready:
                on_path.discard(k)
                value[k] = sum(p * value.get(c, 0.0) for p, c in branches)
                size["decision"] += 1
                size["max_depth"] = max(size["max_depth"], n.depth)
                continue
            on_path.add(k)
            stack.append((k, True))
            stack.extend((c, False) for _p, c in branches if c not in value and c not in on_path)
        return dict(lower_bound=value[self.root.key], **size)

    def top_tips(self, k: int = 5, expand: bool = True) -> List[Dict[str, object]]:
        """The k unexpanded tips of the current best policy that carry the most of V(s0):
        contribution = P(reach the tip under the policy) * its value (= h0, an upper bound). For each:
        time left, running actions, facts changed since s0, h per pattern (which pattern is the binding
        min), and -- with expand -- its value after one expansion (how far h drops one step later).
        Expanding them is what ILAO* would do next anyway."""
        order, seen, stack = [], set(), [(self.root.key, False)]
        while stack:                                     # post-order over the policy (a DAG here)
            key, ready = stack.pop()
            if ready:
                order.append(key)
                continue
            if key in seen:
                continue
            seen.add(key)
            n = self.nodes[key]
            stack.append((key, True))
            if n.options and n.best is not None and not n.goal:
                stack.extend((c, False) for _p, c in n.options[n.best][1] if c not in seen)
        reach: Dict[tuple, float] = defaultdict(float)
        reach[self.root.key] = 1.0
        tips = []
        for key in reversed(order):                      # parents before children
            n = self.nodes[key]
            if n.goal or n.dead:
                continue
            if n.options is None or n.best is None:
                tips.append(key)
                continue
            for p, c in n.options[n.best][1]:
                reach[c] += reach[key] * p
        upper = max(self.root.value, 1e-12)
        ranked = sorted(tips, key=lambda t: reach[t] * self.nodes[t].value, reverse=True)[:k]
        init = frozenset(self.mdp.initial_state().predicates)
        out = []
        for key in ranked:
            n = self.nodes[key]
            facts = n.key[0]
            row = dict(share_of_upper=round(reach[key] * n.value / upper, 4), reach=round(reach[key], 6),
                       value=round(n.value, 6), h0=round(n.h0, 6), depth=n.depth,
                       time_left=round(self.diag.deadline - n.current_time, 2) if self.diag else None,
                       running=sorted(str(f) for f in facts if "inExecution" in str(f)),
                       gained=sorted(str(f) for f in facts - init if "inExecution" not in str(f)),
                       lost=sorted(str(f) for f in init - facts if "inExecution" not in str(f)))
            pdb = getattr(self.heuristic, "pdb", None)
            if pdb is not None and n.stn is not None:
                self.heuristic(n)                        # recompute only to read the per-pattern values
                values = list(getattr(pdb, "last_values", []) or [])
                groups = getattr(pdb, "groups", None) or [list(range(len(values)))]
                pats = []
                for gi, members in enumerate(groups):
                    group_vals = [values[i] for i in members if i < len(values)]
                    low = min(group_vals) if group_vals else None
                    for i in members:
                        if i < len(values):
                            pat = pdb.patterns[i]
                            pats.append(dict(group=gi, goals=pat.get("goals"), facts=pat.get("facts"),
                                             value=round(values[i], 6), binding=values[i] == low))
                row["patterns"] = pats
            if expand and n.options is None and not n.goal:
                self._expand(n)
                self._backup(n)
                row["after_expand"] = round(n.value, 6)
                row["after_expand_dead"] = n.dead
            out.append(row)
        return out

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
        for row in rows:                                # symmetry, from the online counters
            d = self.diag.by_depth[row["depth"]]
            row["created"] = int(d["created"])
            row["nodes_per_facts_time"] = round(d["created"] / max(1.0, d["facts_time_groups"]), 2)
        out: Dict[str, object] = dict(by_depth=rows, progress=list(self.diag.progress),
                                      cycles=self.stats["cycles"], folded=self.stats["folded"],
                                      canonical_skipped=self.stats["canonical_skipped"],
                                      dropped_noops=self.stats["dropped_noop"])
        out["by_root_option"] = self._by_root_option()
        dg = self.diag

        def table(store, key_name, sort_key=None):
            rows = [dict({key_name: k}, **{n: int(v) for n, v in row.items()}) for k, row in store.items()]
            return sorted(rows, key=sort_key or (lambda r: r[key_name]))

        out["by_time_left"] = table(dg.by_time_left, "time_left")
        out["by_running_actions"] = table(dg.by_running, "running")
        out["by_goals_done"] = table(dg.by_goals_done, "goals_done")
        out["by_creating_action"] = table(dg.by_schema, "action", sort_key=lambda r: -r.get("created", 0))[:30]
        total_gaps = max(1, sum(dg.gap_hist.values()))
        out["alternative_gap_to_best"] = {k: dict(count=v, pct=round(100 * v / total_gaps, 2))
                                          for k, v in sorted(dg.gap_hist.items())}
        if walk:
            out["h_drop_by_depth"] = self._h_drop_by_depth()
            out["work_outside_best_policy"] = self._waste()
            out["dominance"] = self.dominance_summary()
            out["same_facts_groups"] = self._same_facts_groups()
            out["main_path"] = self._main_path()
        if walk:
            total = sum(v[0] for v in pruned_by_depth.values())
            pruned = sum(v[1] for v in pruned_by_depth.values())
            out["h_pruned_options_pct"] = round(100 * pruned / max(1, total), 2)
            out["tightness"] = self._tightness() if self.stats["complete"] else (
                "run not finished: V* unknown below the root -- see progress (lower / upper bound)")
        return out

    def _by_root_option(self) -> List[Dict[str, object]]:
        """Where the nodes are: per first move from the root, nodes created / expanded and the
        deepest one, with that option's current Q (upper bound) and whether it is the best now."""
        q_now = {}
        if self.root.options and all(k in self.nodes for _n, br in self.root.options for _p, k in br):
            for i, (name, branches) in enumerate(self.root.options):
                q_now[name] = (sum(p * self.nodes[k].value for p, k in branches), i == self.root.best)
        rows = []
        for branch, b in self.diag.by_branch.items():
            if branch is None:
                continue
            q, best = q_now.get(branch, (None, False))
            rows.append(dict(root_option=branch, created=int(b["created"]), expanded=int(b["expanded"]),
                             max_depth=int(b["max_depth"]), q_upper=None if q is None else round(q, 6),
                             best_now=best))
        rows.sort(key=lambda r: -r["created"])
        return rows

    def path_to(self, node) -> List[Dict[str, object]]:
        """The actions from the root to `node` along the parents it was first created from."""
        steps = []
        n = node
        while n is not None and n.parent is not None:
            steps.append(dict(action=n.via, time=round(n.current_time, 2), depth=n.depth))
            n = self.nodes.get(n.parent)
        return list(reversed(steps))

    def _h_drop_by_depth(self) -> List[Dict[str, object]]:
        """For expanded nodes: h0 - current value (how far the first estimate has come down so far; on a
        finished run it is h0 - V* wherever the node is solved). A lower bound on the overestimate."""
        rows: Dict[int, List[float]] = defaultdict(list)
        for n in self.nodes.values():
            if n.options is not None and not n.goal:
                rows[n.depth].append(n.h0 - n.value)
        out = []
        for depth in sorted(rows):
            g = rows[depth]
            out.append(dict(depth=depth, expanded=len(g), drop_mean=round(sum(g) / len(g), 6),
                            drop_max=round(max(g), 6),
                            dropped_gt_0_001_pct=round(100 * sum(1 for x in g if x > 1e-3) / len(g), 1),
                            dropped_gt_0_01_pct=round(100 * sum(1 for x in g if x > 1e-2) / len(g), 1)))
        return out

    def _waste(self) -> Dict[str, object]:
        """Nodes of the current best (partial) policy vs everything generated: the rest is the cost of
        refuting alternatives."""
        seen, stack = set(), [self.root.key]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            n = self.nodes[k]
            if n.options and n.best is not None and not n.goal:
                stack.extend(c for _p, c in n.options[n.best][1])
        total = len(self.nodes)
        return dict(best_policy_nodes=len(seen), generated=total,
                    outside_best_policy_pct=round(100 * (total - len(seen)) / max(1, total), 2))

    @staticmethod
    def _bound_map(signature) -> Dict[tuple, object]:
        out = {}
        for label, row in signature:
            if row is None:
                continue
            for lab2, b in row:
                out[(label, lab2)] = b
        return out

    @staticmethod
    def _looser(a: Dict[tuple, object], b: Dict[tuple, object]) -> bool:
        """Every bound of a is >= the same bound of b (None = no bound): every schedule b allows, a allows."""
        if a.keys() != b.keys():
            return False
        for k, vb in b.items():
            va = a[k]
            if va is None:
                continue
            if vb is None or va < vb:
                return False
        return True

    def dominance_pairs(self, max_group: int = 2000):
        """Yield (dominator, dominated) node pairs: same facts, the dominator's STN looser in every bound,
        so V(dominator) >= V(dominated). Groups larger than max_group are only scanned on their first
        max_group members (reported)."""
        groups: Dict[frozenset, List[Node]] = defaultdict(list)
        for n in self.nodes.values():
            groups[n.key[0]].append(n)
        for members in groups.values():
            if len(members) < 2:
                continue
            members = members[:max_group]
            maps = [self._bound_map(n.key[1]) for n in members]
            for i, bi in enumerate(maps):
                for j, bj in enumerate(maps):
                    if i != j and self._looser(bj, bi):
                        yield members[j], members[i]
                        break

    def dominance_summary(self, max_group: int = 1000, budget: float = 600.0) -> Dict[str, object]:
        """How much dominance would merge: nodes per equal-facts group that another node of the group
        dominates (looser STN, same facts), and what would be left (the undominated frontier)."""
        groups: Dict[frozenset, List[Node]] = defaultdict(list)
        for n in self.nodes.values():
            groups[n.key[0]].append(n)
        sizes = sorted((len(m) for m in groups.values()), reverse=True)
        dominated = 0
        clipped = 0
        largest = []
        started = time.perf_counter()
        scanned_nodes = scanned_groups = 0
        for facts, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(members) < 2:
                continue
            if time.perf_counter() - started > budget:
                break                                    # out of time: what was scanned is reported
            if len(members) > max_group:
                clipped += 1
            scan = members[:max_group]
            maps = [self._bound_map(n.key[1]) for n in scan]
            dom_here = sum(1 for i, bi in enumerate(maps)
                           if any(i != j and self._looser(bj, bi) for j, bj in enumerate(maps)))
            dominated += dom_here
            scanned_nodes += len(scan)
            scanned_groups += 1
            if len(largest) < 10:
                largest.append(dict(size=len(members), scanned=len(scan), dominated=dom_here,
                                    frontier=len(scan) - dom_here,
                                    times=sorted({round(n.current_time, 2) for n in scan})[:12]))
        total = len(self.nodes)
        return dict(nodes=total, distinct_facts=len(groups), largest_facts_group=sizes[0] if sizes else 0,
                    dominated=dominated, left_after_dominance=total - dominated,
                    dominated_pct=round(100 * dominated / max(1, total), 2), clipped_groups=clipped,
                    groups_with_2_or_more=sum(1 for m in groups.values() if len(m) >= 2),
                    scanned_groups=scanned_groups, scanned_nodes=scanned_nodes,
                    dominated_pct_of_scanned=round(100 * dominated / max(1, scanned_nodes), 2),
                    largest_groups=largest)

    def _same_facts_groups(self, groups: int = 3, members: int = 3) -> List[Dict[str, object]]:
        """The largest groups of nodes with the same facts at the same time (they differ only in the
        STN): for a few members, the path that made each one and its STN signature -- what differs."""
        by_group: Dict[tuple, List[Node]] = defaultdict(list)
        for n in self.nodes.values():
            if n.parent is not None:
                by_group[(n.key[0], n.current_time)].append(n)
        top = sorted(by_group.values(), key=len, reverse=True)[:groups]
        init = frozenset(self.mdp.initial_state().predicates)
        out = []
        for g in top:
            facts = g[0].key[0]
            entry = dict(size=len(g), time=g[0].current_time, depths=sorted({n.depth for n in g})[:10],
                         root_options=sorted({str(n.branch) for n in g})[:14],
                         running=sorted(str(f) for f in facts if "inExecution" in str(f)),
                         gained=sorted(str(f) for f in facts - init if "inExecution" not in str(f)),
                         members=[])
            for n in sorted(g, key=lambda n: n.depth)[:members]:
                entry["members"].append(dict(depth=n.depth, value=round(n.value, 6), expanded=n.options is not None,
                                             path=[f"{st['action']}@{st['time']}" for st in self.path_to(n)],
                                             signature=str(n.key[1])))
            out.append(entry)
        return out

    def _main_path(self, limit: int = 150) -> List[Dict[str, object]]:
        """From the root: the best action and its most likely outcome, step by step, until a tip or
        a leaf -- with h0, the current value, the options there and how many tie with the best."""
        steps, n, seen = [], self.root, set()
        while n is not None and n.key not in seen and len(steps) < limit:
            seen.add(n.key)
            row = dict(depth=n.depth, time_left=round(self.diag.deadline - n.current_time, 2),
                       h0=round(n.h0, 6), value=round(n.value, 6), goal=n.goal, dead=n.dead,
                       expanded=n.options is not None)
            if not n.options or n.best is None:
                steps.append(row)
                break
            qs = [sum(p * self.nodes[k].value for p, k in br) for _a, br in n.options]
            best_q = qs[n.best]
            name, branches = n.options[n.best]
            p_top, k_top = max(branches, key=lambda pk: pk[0])
            row.update(action=name, options=len(qs), ties_0_001=sum(1 for q in qs if best_q - q <= 1e-3) - 1,
                       outcomes=len(branches), p_followed=round(p_top, 4))
            steps.append(row)
            n = self.nodes[k_top]
        return steps

    def _tightness(self) -> Dict[str, object]:
        """On a finished run every node of the optimal policy has V = V*: compare it with h0 there.
        h0 < V* would be an admissibility violation (h must be an upper bound)."""
        seen, frontier = set(), [self.root]
        by_depth: Dict[int, List[float]] = defaultdict(list)
        violations = []
        weak_counts: Dict[object, int] = defaultdict(int)
        weak_gap: Dict[object, float] = defaultdict(float)
        while frontier:
            n = frontier.pop()
            if n.key in seen:
                continue
            seen.add(n.key)
            if n.goal or n.dead or not n.options or n.best is None:
                continue
            gap = n.h0 - n.value
            by_depth[n.depth].append(gap)
            if gap > 1e-2:
                weak_counts[n.weak] += 1
                weak_gap[n.weak] += gap
            if gap < -1e-9:
                violations.append(dict(depth=n.depth, h0=n.h0, v_star=n.value))
            frontier.extend(self.nodes[k] for _p, k in n.options[n.best][1])
        rows = []
        for depth in sorted(by_depth):
            g = by_depth[depth]
            rows.append(dict(depth=depth, policy_nodes=len(g), gap_mean=round(sum(g) / len(g), 6),
                             gap_max=round(max(g), 6), exact_pct=round(100 * sum(1 for x in g if abs(x) < 1e-6) / len(g), 1)))
        allg = [x for g in by_depth.values() for x in g]
        pdb = getattr(self.heuristic, "pdb", None)
        groups = getattr(pdb, "groups", None) or []

        def group_goals(gi):
            if gi is None or gi >= len(groups) or pdb is None:
                return str(gi)
            return str(sorted({g for i in groups[gi] for g in (pdb.patterns[i].get("goals") or [])}))
        loose = [dict(weakest_group=group_goals(gi), nodes=c, gap_mean=round(weak_gap[gi] / c, 6))
                 for gi, c in sorted(weak_counts.items(), key=lambda kv: -kv[1])]
        return dict(policy_decision_nodes=len(allg), gap_mean=round(sum(allg) / max(1, len(allg)), 6),
                    gap_max=round(max(allg) if allg else 0.0, 6), admissibility_violations=violations[:20],
                    violations=len(violations), by_depth=rows, loose_nodes_by_weakest_group=loose)

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
