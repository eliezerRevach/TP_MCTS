"""
Fixed-tail PTRPG evaluation for MCTS leaf backup.

When remaining horizon R exceeds FIXED_TAIL_H, run a PTRPG-guided stochastic
prefix for delta = R - FIXED_TAIL_H time units (no overshoot past the tail
boundary), then evaluate PTRPG at horizon FIXED_TAIL_H only. Avoids mixing
PTRPG values computed with different remaining horizons in the same Q average.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import unified_planning as up
from unified_planning.engines.solvers.greedy_parallel import (
    pick_best_action,
    terminal_success_value,
)
from unified_planning.engines.solvers.ptrpg_guided_rollout import (
    remaining_deadline,
    resolve_rollout_policy,
)

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
    policy_strategy: str = "atom_backtrack_exact_resolution"
    tail_strategy: Optional[str] = None
    debug_eval_limit: int = 5
    debug_enabled: bool = False

    def __post_init__(self) -> None:
        self.fixed_tail_h = max(0, int(self.fixed_tail_h))
        if self.tail_strategy is None:
            self.tail_strategy = self.policy_strategy


@dataclass
class _EvalTrace:
    leaf_id: object = None
    original_remaining: int = 0
    fixed_tail_h: int = 0
    delta: int = 0
    prefix_actions: List[str] = field(default_factory=list)
    remaining_after_steps: List[int] = field(default_factory=list)
    waited_at_boundary: bool = False
    final_tail_horizon: int = 0
    tail_ptrpg_value: float = 0.0
    returned_value: float = 0.0
    goal_reached: bool = False
    dead_end: bool = False
    stn_infeasible: bool = False


def fixed_tail_config_from_args(args=None) -> FixedTailConfig:
    cli = args if args is not None else getattr(up, "args", None)
    fixed_tail_h = 10
    policy = "atomic_exact_resolution"
    debug = False
    if cli is not None:
        fixed_tail_h = int(getattr(cli, "fixed_tail_h", fixed_tail_h))
        policy = getattr(cli, "ptrpg_guided_rollout_policy", policy)
        debug = bool(getattr(cli, "fixed_tail_debug", False))
    return FixedTailConfig(
        fixed_tail_h=fixed_tail_h,
        policy_strategy=resolve_rollout_policy(policy),
        debug_enabled=debug,
    )


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
    """Return (dead_end, stn_infeasible)."""
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


def _log_eval_trace(trace: _EvalTrace) -> None:
    logger.info(
        "fixed_tail_ptrpg_rollout leaf_id=%s R=%d FIXED_TAIL_H=%d delta=%d "
        "prefix_actions=%s remaining_after=%s waited_at_boundary=%s "
        "final_tail_horizon=%d tail_ptrpg=%.6f returned=%.6f "
        "goal=%s dead_end=%s stn_infeasible=%s",
        trace.leaf_id,
        trace.original_remaining,
        trace.fixed_tail_h,
        trace.delta,
        trace.prefix_actions,
        trace.remaining_after_steps,
        trace.waited_at_boundary,
        trace.final_tail_horizon,
        trace.tail_ptrpg_value,
        trace.returned_value,
        trace.goal_reached,
        trace.dead_end,
        trace.stn_infeasible,
    )


def fixed_tail_ptrpg_value(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    config: FixedTailConfig,
    heuristic_name: str = "temporal_probabilistic_rpg",
    temporal_heuristic_depth: int = 25,
    leaf_id: object = None,
    debug_emit: bool = False,
) -> float:
    trace = _EvalTrace(leaf_id=leaf_id)
    H = config.fixed_tail_h
    trace.fixed_tail_h = H
    tail_strategy = config.tail_strategy or config.policy_strategy

    R = remaining_deadline(mdp, stn)
    trace.original_remaining = R

    if _goal_reached(mdp, state):
        trace.goal_reached = True
        trace.returned_value = 1.0
        if debug_emit:
            _log_eval_trace(trace)
        return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

    dead_end, stn_bad = _failure_at(mdp, state, stn)
    if dead_end or stn_bad:
        trace.dead_end = dead_end
        trace.stn_infeasible = stn_bad
        trace.returned_value = 0.0
        if debug_emit:
            _log_eval_trace(trace)
        return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

    if R <= H:
        current_time = float(stn.get_current_end_time())
        trace.final_tail_horizon = R
        tail_val = _ptrpg_at_horizon(
            mdp, state, current_time, R, tail_strategy
        )
        trace.tail_ptrpg_value = tail_val
        trace.returned_value = tail_val
        if debug_emit:
            _log_eval_trace(trace)
        return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

    delta = R - H
    trace.delta = delta
    start_ct = float(stn.get_current_end_time())
    current_state = state
    current_stn = stn
    current_prev = previous_action_node
    elapsed = 0.0
    waited = False

    while elapsed < delta:
        if terminal_success_value(mdp, current_state, current_stn) >= 1.0:
            trace.goal_reached = True
            trace.returned_value = 1.0
            if debug_emit:
                _log_eval_trace(trace)
            return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

        dead_end, stn_bad = _failure_at(mdp, current_state, current_stn)
        if dead_end or stn_bad:
            trace.dead_end = dead_end
            trace.stn_infeasible = stn_bad
            trace.returned_value = 0.0
            if debug_emit:
                _log_eval_trace(trace)
            return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

        gap = delta - elapsed
        picked = pick_best_action(
            mdp=mdp,
            state=current_state,
            stn=current_stn,
            previous_action_node=current_prev,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=config.policy_strategy,
            tie_break="legacy",
        )
        if picked is None:
            trace.dead_end = True
            trace.returned_value = 0.0
            if debug_emit:
                _log_eval_trace(trace)
            return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

        _score, action, candidate_stn, candidate_prev, _any_term, _cache = picked
        dur = float(candidate_stn.get_current_end_time()) - float(
            current_stn.get_current_end_time()
        )
        if dur > gap:
            waited = True
            break

        terminal, next_state, _reward = mdp.step(current_state, action)
        trace.prefix_actions.append(getattr(action, "name", str(action)))
        current_state = next_state
        current_stn = candidate_stn
        current_prev = candidate_prev
        elapsed = float(current_stn.get_current_end_time()) - start_ct
        trace.remaining_after_steps.append(remaining_deadline(mdp, current_stn))

        if terminal:
            if _goal_reached(mdp, current_state):
                trace.goal_reached = True
                trace.returned_value = 1.0
            else:
                trace.dead_end = True
                trace.returned_value = 0.0
            if debug_emit:
                trace.waited_at_boundary = waited
                _log_eval_trace(trace)
            return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

        dead_end, stn_bad = _failure_at(mdp, current_state, current_stn)
        if dead_end or stn_bad:
            trace.dead_end = dead_end
            trace.stn_infeasible = stn_bad
            trace.returned_value = 0.0
            if debug_emit:
                trace.waited_at_boundary = waited
                _log_eval_trace(trace)
            return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)

    trace.waited_at_boundary = waited
    trace.final_tail_horizon = H
    current_time = float(current_stn.get_current_end_time())
    tail_val = _ptrpg_at_horizon(
        mdp, current_state, current_time, H, tail_strategy
    )
    trace.tail_ptrpg_value = tail_val
    trace.returned_value = tail_val
    if debug_emit:
        _log_eval_trace(trace)
    return _assert_return(trace.returned_value, R, H, trace.final_tail_horizon)


def _assert_return(value: float, original_r: int, fixed_tail_h: int, tail_horizon: int) -> float:
    assert 0.0 <= value <= 1.0, f"fixed_tail value out of range: {value}"
    if original_r > fixed_tail_h and tail_horizon > 0:
        assert tail_horizon == fixed_tail_h, (
            f"expected tail horizon {fixed_tail_h}, got {tail_horizon}"
        )
    return value


__all__ = [
    "FixedTailConfig",
    "fixed_tail_config_from_args",
    "fixed_tail_ptrpg_value",
]
