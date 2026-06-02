"""
Fixed-tail PTRPG evaluation for MCTS leaf backup.

Prefix: random parallel sets of legal actions (no PTRPG) until the tail boundary.
If goal is reached during the prefix, return 1.0.
Otherwise one PTRPG evaluation at horizon FIXED_TAIL_H (e.g. atom_backtrack_exact_resolution).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import unified_planning as up
from unified_planning.engines.solvers.greedy_parallel import (
    GREEDY_MAX_PARALLEL_SET_SIZE,
    terminal_success_value,
)
from unified_planning.engines.solvers.ptrpg_guided_rollout import remaining_deadline
from unified_planning.engines.utils import update_stn

logger = logging.getLogger(__name__)


def _resolution_heuristic_kwargs_from_cli() -> dict:
    try:
        a = getattr(up, "args", None)
        if a is None:
            return {}
    except Exception:
        return {}
    ra = getattr(a, "resolution_alpha", 2.0)
    if ra is None:
        ra = 2.0
    else:
        ra = float(ra)
    return {
        "resolution_alpha": ra,
        "resolution_forced_minimum": bool(getattr(a, "resolution_forced_minimum", False)),
        "resolution_reference_t": getattr(a, "resolution_reference_t", None),
    }


def _aggregation_for_strategy(temporal_heuristic_strategy: str) -> str:
    strat = (temporal_heuristic_strategy or "").strip().lower()
    if strat == "baseline_survival_meanvar":
        return "meanvar"
    if strat == "baseline_time_to_goal":
        return "time_to_goal"
    return "product"


@dataclass
class FixedTailConfig:
    fixed_tail_h: int = 10
    tail_strategy: str = "atom_backtrack_exact_resolution"

    def __post_init__(self) -> None:
        self.fixed_tail_h = max(0, int(self.fixed_tail_h))


@dataclass
class _PrefixResult:
    state: "up.engines.State"
    stn: "up.plans.stn.STNPlan"
    previous_action_node: "up.plans.stn.STNPlanNode"
    waited: bool
    boundary_time: float
    goal_reached: bool = False
    dead_end: bool = False
    stn_infeasible: bool = False
    terminal_value: Optional[float] = None


def fixed_tail_config_from_args(args=None) -> FixedTailConfig:
    cli = args if args is not None else getattr(up, "args", None)
    fixed_tail_h = 10
    if cli is not None:
        fixed_tail_h = int(getattr(cli, "fixed_tail_h", fixed_tail_h))
    return FixedTailConfig(fixed_tail_h=fixed_tail_h)


def _goal_reached(mdp: "up.engines.MDP", state: "up.engines.State") -> bool:
    if not mdp.is_terminal(state):
        return False
    goals = set(mdp.problem.goals)
    preds = getattr(state, "predicates", None)
    return preds is not None and goals.issubset(set(preds))


def _failure_at(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
) -> tuple[bool, bool]:
    try:
        if not stn.is_consistent():
            return False, True
        if stn.get_current_end_time() > mdp.deadline():
            return False, True
    except Exception:
        return False, True
    if not mdp.legal_actions(state):
        return True, False
    return False, False


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ptrpg_at_horizon(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    current_time: float,
    horizon: int,
    strategy: str,
) -> float:
    from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
        TemporalProbabilisticRPGHeuristic,
    )

    heuristic = getattr(mdp, "_temporal_probabilistic_rpg_heuristic", None)
    if heuristic is None:
        heuristic = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)
        setattr(mdp, "_temporal_probabilistic_rpg_heuristic", heuristic)

    goals = set(mdp.problem.goals)
    eff = max(0, int(horizon))
    if mdp.deadline() is not None:
        eff = min(eff, max(0, int(math.floor(mdp.deadline() - current_time))))

    score = heuristic.heuristic_score(
        state,
        goals,
        aggregation=_aggregation_for_strategy(strategy),
        fixed_depth=eff,
        start_time=current_time,
        strategy=strategy,
        **_resolution_heuristic_kwargs_from_cli(),
    )
    return _clamp01(score)


def _stn_dispatch_fits(
    mdp: "up.engines.MDP",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    action: "up.engines.Action",
    gap: float,
) -> Optional[Tuple["up.plans.stn.STNPlan", "up.plans.stn.STNPlanNode", float]]:
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
    if candidate_stn.get_current_end_time() > mdp.deadline():
        return None
    dur = float(candidate_stn.get_current_end_time()) - float(stn.get_current_end_time())
    if dur <= 0.0 or dur > gap + 1e-9:
        return None
    return candidate_stn, candidate_prev, dur


def _run_prefix_rollout(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    *,
    fixed_tail_h: int,
    delta: int,
    max_parallel_set_size: int = GREEDY_MAX_PARALLEL_SET_SIZE,
) -> _PrefixResult:
    assert delta >= 0
    start_ct = float(stn.get_current_end_time())
    boundary_time = start_ct + float(delta)
    current_state = state
    current_stn = stn
    current_prev = previous_action_node
    waited = False

    while True:
        remaining = remaining_deadline(mdp, current_stn)
        if remaining <= fixed_tail_h:
            break

        elapsed = float(current_stn.get_current_end_time()) - start_ct
        if elapsed >= delta - 1e-9:
            break

        if terminal_success_value(mdp, current_state, current_stn) >= 1.0:
            return _PrefixResult(
                state=current_state,
                stn=current_stn,
                previous_action_node=current_prev,
                waited=waited,
                boundary_time=boundary_time,
                goal_reached=True,
                terminal_value=1.0,
            )

        dead_end, stn_bad = _failure_at(mdp, current_state, current_stn)
        if dead_end or stn_bad:
            return _PrefixResult(
                state=current_state,
                stn=current_stn,
                previous_action_node=current_prev,
                waited=waited,
                boundary_time=boundary_time,
                dead_end=dead_end,
                stn_infeasible=stn_bad,
                terminal_value=0.0,
            )

        decision_time = float(current_stn.get_current_end_time())
        gap = delta - (decision_time - start_ct)
        if gap <= 1e-9:
            break

        legal_actions = list(mdp.legal_actions(current_state))
        random.shuffle(legal_actions)
        chosen_in_set = 0

        for action in legal_actions:
            if chosen_in_set >= max_parallel_set_size:
                break

            elapsed = float(current_stn.get_current_end_time()) - start_ct
            gap = delta - elapsed
            if gap <= 1e-9:
                break

            fitted = _stn_dispatch_fits(
                mdp, current_stn, current_prev, action, gap
            )
            if fitted is None:
                continue

            candidate_stn, candidate_prev, _dur = fitted
            prev_elapsed = elapsed
            terminal, next_state, _reward = mdp.step(current_state, action)
            current_state = next_state
            current_stn = candidate_stn
            current_prev = candidate_prev
            chosen_in_set += 1

            new_elapsed = float(current_stn.get_current_end_time()) - start_ct
            if new_elapsed <= prev_elapsed + 1e-9:
                break

            if terminal:
                if _goal_reached(mdp, current_state):
                    return _PrefixResult(
                        state=current_state,
                        stn=current_stn,
                        previous_action_node=current_prev,
                        waited=waited,
                        boundary_time=boundary_time,
                        goal_reached=True,
                        terminal_value=1.0,
                    )
                return _PrefixResult(
                    state=current_state,
                    stn=current_stn,
                    previous_action_node=current_prev,
                    waited=waited,
                    boundary_time=boundary_time,
                    dead_end=True,
                    terminal_value=0.0,
                )

            dead_end, stn_bad = _failure_at(mdp, current_state, current_stn)
            if dead_end or stn_bad:
                return _PrefixResult(
                    state=current_state,
                    stn=current_stn,
                    previous_action_node=current_prev,
                    waited=waited,
                    boundary_time=boundary_time,
                    dead_end=dead_end,
                    stn_infeasible=stn_bad,
                    terminal_value=0.0,
                )

            if float(current_stn.get_current_end_time()) > decision_time + 1e-9:
                break

        if chosen_in_set == 0:
            waited = True
            break

    tail_time = (
        boundary_time
        if waited or float(current_stn.get_current_end_time()) - start_ct < delta - 1e-9
        else float(current_stn.get_current_end_time())
    )
    return _PrefixResult(
        state=current_state,
        stn=current_stn,
        previous_action_node=current_prev,
        waited=waited,
        boundary_time=tail_time,
    )


def fixed_tail_ptrpg_value(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    config: FixedTailConfig,
    tail_strategy: Optional[str] = None,
    **_kwargs,
) -> float:
    H = config.fixed_tail_h
    strategy = tail_strategy or config.tail_strategy

    if _goal_reached(mdp, state):
        return 1.0

    dead_end, stn_bad = _failure_at(mdp, state, stn)
    if dead_end or stn_bad:
        return 0.0

    R = remaining_deadline(mdp, stn)
    if R <= H:
        current_time = float(stn.get_current_end_time())
        return _ptrpg_at_horizon(mdp, state, current_time, R, strategy)

    delta = R - H
    prefix = _run_prefix_rollout(
        mdp,
        state,
        stn,
        previous_action_node,
        fixed_tail_h=H,
        delta=delta,
    )

    if prefix.terminal_value is not None:
        return _clamp01(prefix.terminal_value)

    remaining_after = remaining_deadline(mdp, prefix.stn)
    if remaining_after <= H:
        tail_horizon = remaining_after
        tail_time = float(prefix.stn.get_current_end_time())
    else:
        tail_horizon = H
        tail_time = prefix.boundary_time

    tail_val = _ptrpg_at_horizon(
        mdp,
        prefix.state,
        tail_time,
        tail_horizon,
        strategy,
    )
    return _clamp01(tail_val)


__all__ = [
    "FixedTailConfig",
    "fixed_tail_config_from_args",
    "fixed_tail_ptrpg_value",
]
