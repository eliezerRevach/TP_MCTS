"""Backward plan PDB: regression from the goal, no temporal constraints.

A deliberately SIMPLE counterpart to ``temporal_stn_pdb``. There is no STN, no
deadline, no interval: a state maps to a list of PLANS, and a lookup picks the
best one that fits.

The three design points, as specified:

1. **Backward.** Start from the goal as a subgoal set and regress. Nothing is
   ever expanded forward, so an action that cannot contribute to the goal is
   never even considered -- which is what makes the 200 garbage actions of
   ``prob_conc`` cost exactly nothing.

2. **Multiple plans per state.** When two regression paths arrive at the same
   ``(subgoal, running)`` state -- typically one through an action's SUCCESS
   outcome and one through its FAILURE outcome and a retry -- both plans are
   kept. The state is not collapsed to a single best; the alternatives are what
   the lookup chooses between.

3. **One number per plan.** Its success probability. No upper/lower bound, no
   UNKNOWN, no certificate.

Running actions fall out of regression for free
-----------------------------------------------
Regressing an action's END means that action must have been RUNNING just before
it, so ``a`` enters the state's ``running`` set; regressing its START takes it
back out and replaces it with that start's preconditions. A state whose
``running`` set is non-empty is therefore a mid-execution state, and it matches
exactly the queries whose ``inExecution`` facts say the same thing.

Regression proposes, forward replay disposes
--------------------------------------------
Pure regression has to reason about negative preconditions, delete effects and
state-dependent probabilities on a PARTIAL state, none of which it can do
soundly. So regression is used only as a CANDIDATE GENERATOR: every candidate
event sequence is then replayed forward through the real
``mdp.transition_function`` from the actual query state, which both validates it
and gives its exact probability. Anything regression gets wrong is filtered out
by the replay rather than silently believed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

Fact = Hashable
Event = Tuple[str, str]              # (action key, "S" | "E")

DEFAULT_MAX_PLANS_PER_STATE = 8
DEFAULT_MAX_EVENTS = 12
_EPS = 1e-12


# ---------------------------------------------------------------------------
# States and plans
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressionState:
    """What must hold for the stored plans to reach the goal.

    ``subgoal`` -- facts that must be true. ``running`` -- action types that
    must already be executing. Both are REQUIREMENTS, not a full state: a query
    matches when it satisfies them, and may have anything else true besides.
    """

    subgoal: FrozenSet[Fact]
    running: FrozenSet[str]

    def matches(self, facts: FrozenSet[Fact], running: FrozenSet[str]) -> bool:
        return self.subgoal <= facts and self.running <= running

    def __repr__(self) -> str:
        run = f", running={sorted(self.running)}" if self.running else ""
        return f"State({sorted(str(f) for f in self.subgoal)}{run})"


@dataclass(frozen=True)
class Plan:
    """An ordered event sequence and the probability it reaches the goal."""

    events: Tuple[Event, ...]
    probability: float

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        body = " -> ".join(f"{k}:{w}" for k, w in self.events)
        return f"Plan(p={self.probability:.4f}, {len(self.events)} events: {body})"


# ---------------------------------------------------------------------------
# The PDB
# ---------------------------------------------------------------------------

class BackwardPlanPDB:
    """Regression table: ``RegressionState -> [Plan, ...]``.

    ``model`` is a :class:`~comdp_plus_no_deadline.engines.temporal_stn_pdb.EngineModel`
    (or the ``ToyModel``): anything exposing ``ops()``, ``goals``, ``outcomes``
    and ``legal_action_names``.
    """

    def __init__(
        self,
        model,
        goal_facts: Optional[Iterable[Fact]] = None,
        *,
        max_plans_per_state: int = DEFAULT_MAX_PLANS_PER_STATE,
        max_events: int = DEFAULT_MAX_EVENTS,
    ):
        self.model = model
        self.goals = frozenset(goal_facts if goal_facts is not None else model.goals)
        self.max_plans_per_state = max(1, int(max_plans_per_state))
        self.max_events = max(1, int(max_events))
        self.edges: Dict[RegressionState, List[Tuple[Event, RegressionState]]] = {}
        self.goal_state: Optional[RegressionState] = None
        self.merges = 0                  # times a path landed on a known state
        self.candidates = 0

    # -- structural helpers ------------------------------------------------
    def _adds(self, action) -> Set[Fact]:
        out = set(getattr(action, "add_effects", set()) or set())
        for pe in getattr(action, "probabilistic_effects", []) or []:
            out |= set(getattr(pe, "fluents", []) or [])
        return out

    def _pre(self, action) -> Set[Fact]:
        return set(getattr(action, "pos_preconditions", set()) or set())

    def _event_action(self, key: str, which: str):
        op = self.model.ops()[key]
        if isinstance(op.start_action, str):          # ToyModel
            return (key, which)
        return op.start_action if which == "S" else op.end_action

    def _event_adds(self, key: str, which: str) -> Set[Fact]:
        op = self.model.ops()[key]
        if isinstance(op.start_action, str):          # ToyModel: read the spec
            spec = self.model._spec[key]
            branches = spec.get("start" if which == "S" else "end", [])
            out = set()
            for adds, _dels, _p in branches:
                out |= set(adds)
            return out
        action = self._event_action(key, which)
        return self._adds(action) if action is not None else set()

    def _event_pre(self, key: str, which: str) -> Set[Fact]:
        op = self.model.ops()[key]
        if isinstance(op.start_action, str):          # ToyModel
            spec = self.model._spec[key]
            return set(spec.get("pre" if which == "S" else "end_pre", ()))
        action = self._event_action(key, which)
        return self._pre(action) if action is not None else set()

    # -- build -------------------------------------------------------------
    def build(self) -> "BackwardPlanPDB":
        """Regress from the goal. Each STATE is expanded once; plans are paths.

        The table is a DAG, not a list of sequences. Storing whole event
        sequences per state re-expands one state once per plan prefix, which
        re-creates the very factorial the backward direction exists to avoid --
        two ways to reach the same subgoal set are two PATHS through one node,
        not two nodes. So an edge ``state -(event)-> nearer-the-goal state`` is
        what is stored, and a plan is enumerated from it only at lookup.
        """
        ops = self.model.ops()
        self.goal_state = RegressionState(self.goals, frozenset())
        frontier: List[RegressionState] = [self.goal_state]
        self.edges = {self.goal_state: []}

        while frontier:
            state = frontier.pop()
            for pred, event in self._regress(state, ops):
                self.candidates += 1
                seen_before = pred in self.edges
                if not seen_before:
                    self.edges[pred] = []
                    frontier.append(pred)
                else:
                    # Two regression paths met at one state. Both continuations
                    # are kept as separate edges -- this is where alternative
                    # plans live.
                    self.merges += 1
                if len(self.edges[pred]) < self.max_plans_per_state:
                    self.edges[pred].append((event, state))
        return self

    def plans_from(self, state: "RegressionState", limit: int = 16) -> List[Tuple[Event, ...]]:
        """Enumerate event sequences from ``state`` to the goal state (DFS)."""
        out: List[Tuple[Event, ...]] = []

        def walk(node, acc, depth, on_path):
            if node == self.goal_state:
                out.append(tuple(acc))
                return
            if len(out) >= limit or depth >= self.max_events:
                return
            for event, nxt in self.edges.get(node, ()):
                if nxt in on_path:
                    continue                       # no cycles inside one plan
                walk(nxt, acc + [event], depth + 1, on_path | {nxt})

        walk(state, [], 0, frozenset({state}))
        return out

    def _regress(self, state: RegressionState, ops) -> List[Tuple[RegressionState, Event]]:
        """One backward step: un-do an END (action becomes running) or a START."""
        out: List[Tuple[RegressionState, Event]] = []

        # 1. Regress a START: the action is running, so its start came earlier.
        for key in state.running:
            adds = self._event_adds(key, "S")
            pre = self._event_pre(key, "S")
            subgoal = (state.subgoal - adds) | pre
            out.append((
                RegressionState(frozenset(subgoal), state.running - {key}),
                (key, "S"),
            ))

        # 2. Regress an END: it must ACHIEVE something still needed, and the
        #    action must not already be running (one instance at a time).
        for key, op in ops.items():
            if op.end_action is None or key in state.running:
                continue
            adds = self._event_adds(key, "E")
            if not (adds & state.subgoal):
                continue                              # contributes nothing: skip
            pre = self._event_pre(key, "E")
            subgoal = (state.subgoal - adds) | pre
            out.append((
                RegressionState(frozenset(subgoal), state.running | {key}),
                (key, "E"),
            ))
        return out

    # -- lookup ------------------------------------------------------------
    @staticmethod
    def _order_consistent(events: Sequence[Event], end_times: Dict[str, float]) -> bool:
        """Does this plan end the running actions in the order the clock says?

        The query knows each running instance's end time (start + duration), so
        the order in which they MUST complete is already determined -- it is not
        a choice left to the plan. A stored plan that ends them in a different
        order is a plan for a different execution, and is filtered out.

        This is the only place time enters the lookup. The PDB itself is
        untimed; the clock is used purely to decide WHICH stored plan the
        current execution is already committed to.
        """
        seen = [k for k, w in events if w == "E" and k in end_times]
        times = [end_times[k] for k in seen]
        return all(times[i] <= times[i + 1] + 1e-12 for i in range(len(times) - 1))

    def lookup(
        self,
        facts: Iterable[Fact],
        running=(),
    ) -> Tuple[Optional[Plan], List[Tuple[RegressionState, Plan]]]:
        """Best plan for this query, plus every candidate that matched.

        A stored state matches when its ``subgoal`` holds in ``facts`` and its
        ``running`` set is executing. Each matching plan is then REPLAYED
        forward through the real model to get its true probability -- so a plan
        that regression proposed but the domain forbids scores 0 and loses.
        """
        facts = frozenset(facts)
        # ``running`` may be a plain set (no clock) or {action_key: end_time}.
        end_times: Dict[str, float] = dict(running) if isinstance(running, dict) else {}
        running = frozenset(running)
        matched: List[Tuple[RegressionState, Plan]] = []
        for state in self.edges:
            # FILTER 1 -- facts: every fact the candidate needs must hold in the
            # query. The candidate may need FEWER facts than the query has
            # (that is the "don't care"); it may never need one the query lacks.
            if not state.matches(facts, running):
                continue
            for events in self.plans_from(state):
                # FILTER 2 -- plan: the running instances pin down which plan
                # this execution is already following.
                if end_times and not self._order_consistent(events, end_times):
                    continue
                if state.running and not all(
                    any(k == key and w == "E" for key, w in events) for k in state.running
                ):
                    continue                      # plan never completes what is running
                p = self.replay(events, facts, running)
                if p > _EPS:
                    matched.append((state, Plan(events, p)))
        if not matched:
            return None, []
        best = max(matched, key=lambda sp: (sp[1].probability, -len(sp[1].events)))
        return best[1], matched

    def replay(
        self,
        events: Sequence[Event],
        facts: FrozenSet[Fact],
        running: FrozenSet[str] = frozenset(),
    ) -> float:
        """Forward-execute ``events`` from ``facts``; return P(goal reached).

        This is the validator. It uses the model's OWN outcome kernel, so
        preconditions, deletes and state-dependent probabilities are handled
        exactly, and a regression candidate that is not actually executable
        simply returns 0.
        """
        frontier: List[Tuple[float, FrozenSet[Fact], FrozenSet[str]]] = [
            (1.0, frozenset(facts), frozenset(running))
        ]
        reached = 0.0
        for key, which in events:
            nxt: Dict[Tuple[FrozenSet[Fact], FrozenSet[str]], float] = {}
            for prob, cur, run in frontier:
                if self.goals <= cur:
                    reached += prob                    # already done; stop early
                    continue
                if which == "S":
                    if key in run or key not in self.model.legal_action_names(cur):
                        continue                       # not executable here
                    new_run = run | {key}
                else:
                    if key not in run:
                        continue
                    new_run = run - {key}
                action = self._event_action(key, which)
                if action is None:
                    continue
                if not self._event_pre(key, which) <= cur:
                    continue
                for out_facts, p in self.model.outcomes(action, cur):
                    if p <= _EPS:
                        continue
                    slot = (out_facts, new_run)
                    nxt[slot] = nxt.get(slot, 0.0) + prob * p
            frontier = [(p, f, r) for (f, r), p in nxt.items()]
            if not frontier:
                break
        for prob, cur, _run in frontier:
            if self.goals <= cur:
                reached += prob
        return reached

    # -- reporting ---------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {
            "states": len(self.edges),
            "edges": sum(len(v) for v in self.edges.values()),
            "multi_plan_states": sum(1 for v in self.edges.values() if len(v) > 1),
            "merges": self.merges,
            "candidates": self.candidates,
        }
