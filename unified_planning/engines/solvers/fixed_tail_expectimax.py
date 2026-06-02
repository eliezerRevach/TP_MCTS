"""
Expectimax prefix evaluation for fixed-tail MCTS.

V(s) = max_a Q(s,a) over STN-feasible controllable actions;
Q(s,a) = sum_o P(o|s,a) * V(s_o) using mdp.transition_function (no sampling).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import unified_planning as up
from unified_planning.engines.solvers.fixed_tail_ptrpg_rollout import (
    FixedTailSearchContext,
    _clamp01,
    _goal_reached,
    crossed_cutoff,
    elapsed_from_root,
    fixed_tail_dead_end_value,
    node_remaining,
    ptrpg_at_horizon,
)
from unified_planning.engines.utils import update_stn

OutcomeDetail = Tuple[float, float]


def _state_signature(state: "up.engines.State") -> object:
    preds = getattr(state, "predicates", None)
    sig: list = [frozenset(preds) if preds is not None else id(state)]
    active = getattr(state, "active_actions", None)
    if active is not None:
        sig.append(hash(active))
    ct = getattr(state, "current_time", None)
    if ct is not None:
        sig.append(float(ct))
    return tuple(sig)


def _stn_key(stn: "up.plans.stn.STNPlan") -> float:
    return float(stn.get_current_end_time())


def _fit_action_stn(
    mdp: "up.engines.MDP",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    action: "up.engines.Action",
) -> Optional[Tuple["up.plans.stn.STNPlan", "up.plans.stn.STNPlanNode"]]:
    candidate_stn = stn.clone()
    candidate_prev = previous_action_node
    try:
        candidate_prev = update_stn(
            candidate_stn,
            action,
            candidate_prev,
            type="SetTime",
        )
    except Exception:
        return None
    if not candidate_stn.is_consistent():
        return None
    if mdp.deadline() is not None and candidate_stn.get_current_end_time() > mdp.deadline():
        return None
    return candidate_stn, candidate_prev


@dataclass
class FixedTailExpectimaxGuards:
    max_nodes: int = 5000
    max_depth: int = 64
    max_time_sec: float = 0.0

    def __post_init__(self) -> None:
        self.max_nodes = max(1, int(self.max_nodes))
        self.max_depth = max(1, int(self.max_depth))


@dataclass
class FixedTailExpectimaxEvaluator:
    mdp: "up.engines.MDP"
    ctx: FixedTailSearchContext
    strategy: str
    guards: FixedTailExpectimaxGuards = field(default_factory=FixedTailExpectimaxGuards)

    _v_cache: Dict[Tuple, float] = field(default_factory=dict)
    _q_cache: Dict[Tuple, float] = field(default_factory=dict)
    _nodes_evaluated: int = 0
    _guard_tripped: bool = False
    _search_start: float = field(default_factory=time.perf_counter)

    def reset_search(self) -> None:
        self._v_cache.clear()
        self._q_cache.clear()
        self._nodes_evaluated = 0
        self._guard_tripped = False
        self._search_start = time.perf_counter()

    def _guard_ok(self, depth: int) -> bool:
        if self._guard_tripped:
            return False
        if self._nodes_evaluated >= self.guards.max_nodes:
            self._guard_tripped = True
            return False
        if depth >= self.guards.max_depth:
            self._guard_tripped = True
            return False
        if self.guards.max_time_sec > 0.0:
            if time.perf_counter() - self._search_start > self.guards.max_time_sec:
                self._guard_tripped = True
                return False
        return True

    def _v_cache_key(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
    ) -> Tuple:
        rem = node_remaining(self.mdp, state, stn)
        el = elapsed_from_root(self.ctx, rem)
        return (
            _state_signature(state),
            _stn_key(stn),
            rem,
            el,
            self.ctx.prefix_budget,
            self.strategy,
        )

    def _q_cache_key(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
        action: "up.engines.Action",
    ) -> Tuple:
        name = getattr(action, "name", None)
        return self._v_cache_key(state, stn) + (name if name is not None else id(action),)

    def _ptrpg_fallback(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
    ) -> float:
        rem = node_remaining(self.mdp, state, stn)
        return ptrpg_at_horizon(self.mdp, state, stn, rem, self.strategy)

    def value(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
        previous_action_node: "up.plans.stn.STNPlanNode",
        depth: int = 0,
    ) -> float:
        key = self._v_cache_key(state, stn)
        if key in self._v_cache:
            return self._v_cache[key]

        if not self._guard_ok(depth):
            val = _clamp01(self._ptrpg_fallback(state, stn))
            self._v_cache[key] = val
            return val

        self._nodes_evaluated += 1

        if _goal_reached(self.mdp, state):
            val = 1.0
        elif fixed_tail_dead_end_value(self.mdp, state, stn):
            val = 0.0
        else:
            rem = node_remaining(self.mdp, state, stn)
            el = elapsed_from_root(self.ctx, rem)
            if crossed_cutoff(self.ctx, el):
                val = ptrpg_at_horizon(self.mdp, state, stn, rem, self.strategy)
            else:
                legal = list(self.mdp.legal_actions(state))
                if not legal:
                    val = 0.0
                else:
                    best_q = -float("inf")
                    for action in legal:
                        fitted = _fit_action_stn(self.mdp, stn, previous_action_node, action)
                        if fitted is None:
                            continue
                        q_val = self.q_value(
                            state,
                            stn,
                            previous_action_node,
                            action,
                            depth=depth,
                        )
                        if q_val > best_q:
                            best_q = q_val
                    val = 0.0 if best_q == -float("inf") else best_q

        val = _clamp01(val)
        self._v_cache[key] = val
        return val

    def value_for_feasible_actions(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
        previous_action_node: "up.plans.stn.STNPlanNode",
        feasible_actions: List["up.engines.Action"],
        depth: int = 0,
    ) -> float:
        """V(s) using only STN-feasible actions (MCTS children), not all MDP-legal actions."""
        key = self._v_cache_key(state, stn) + ("feasible", tuple(getattr(a, "name", id(a)) for a in feasible_actions))
        if key in self._v_cache:
            return self._v_cache[key]

        if not self._guard_ok(depth):
            val = _clamp01(self._ptrpg_fallback(state, stn))
            self._v_cache[key] = val
            return val

        self._nodes_evaluated += 1

        if _goal_reached(self.mdp, state):
            val = 1.0
        elif fixed_tail_dead_end_value(self.mdp, state, stn):
            val = 0.0
        else:
            rem = node_remaining(self.mdp, state, stn)
            el = elapsed_from_root(self.ctx, rem)
            if crossed_cutoff(self.ctx, el):
                val = ptrpg_at_horizon(self.mdp, state, stn, rem, self.strategy)
            elif not feasible_actions:
                val = 0.0
            else:
                best_q = -float("inf")
                for action in feasible_actions:
                    q_val = self.q_value(
                        state,
                        stn,
                        previous_action_node,
                        action,
                        depth=depth,
                    )
                    if q_val > best_q:
                        best_q = q_val
                val = 0.0 if best_q == -float("inf") else best_q

        val = _clamp01(val)
        self._v_cache[key] = val
        return val

    def q_value(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
        previous_action_node: "up.plans.stn.STNPlanNode",
        action: "up.engines.Action",
        depth: int = 0,
        *,
        collect_outcomes: bool = False,
    ) -> float:
        q_key = self._q_cache_key(state, stn, action)
        if q_key in self._q_cache and not collect_outcomes:
            return self._q_cache[q_key]

        fitted = _fit_action_stn(self.mdp, stn, previous_action_node, action)
        if fitted is None:
            self._q_cache[q_key] = 0.0
            return 0.0

        stn_a, prev_a = fitted
        transitions = self.mdp.transition_function(state, action)
        if not transitions:
            self._q_cache[q_key] = 0.0
            return 0.0

        expected = 0.0
        for next_state, prob in transitions:
            p_o = float(prob)
            child_rem = node_remaining(self.mdp, next_state, stn_a)
            child_el = elapsed_from_root(self.ctx, child_rem)
            if crossed_cutoff(self.ctx, child_el):
                child_value = ptrpg_at_horizon(
                    self.mdp, next_state, stn_a, child_rem, self.strategy
                )
            else:
                child_value = self.value(
                    next_state,
                    stn_a,
                    prev_a,
                    depth=depth + 1,
                )
            expected += p_o * child_value
            if collect_outcomes:
                if not hasattr(self, "_last_outcome_details"):
                    self._last_outcome_details = []
                self._last_outcome_details.append((p_o, _clamp01(child_value)))

        q_val = _clamp01(expected)
        self._q_cache[q_key] = q_val
        return q_val

    def q_value_with_outcomes(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
        previous_action_node: "up.plans.stn.STNPlanNode",
        action: "up.engines.Action",
        depth: int = 0,
    ) -> Tuple[float, List[OutcomeDetail]]:
        self._last_outcome_details = []
        q_val = self.q_value(
            state,
            stn,
            previous_action_node,
            action,
            depth=depth,
            collect_outcomes=True,
        )
        details = list(getattr(self, "_last_outcome_details", []))
        return q_val, details


def uses_expectimax_prefix(config) -> bool:
    policy = getattr(config, "prefix_policy", "mcts_sampled")
    return str(policy).strip().lower() == "expectimax"


__all__ = [
    "FixedTailExpectimaxEvaluator",
    "FixedTailExpectimaxGuards",
    "OutcomeDetail",
    "uses_expectimax_prefix",
]
