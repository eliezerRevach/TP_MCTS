"""
Fixed-tail PTRPG evaluation for MCTS leaf backup.

When remaining horizon R exceeds FIXED_TAIL_H, run a prefix rollout for
delta = R - FIXED_TAIL_H: parallel sets of legal actions per time slice (as in
greedy_parallel), no PTRPG in the prefix. Return 1.0 if goal is reached early;
otherwise one PTRPG evaluation at horizon FIXED_TAIL_H from the tail boundary.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import unified_planning as up
from unified_planning.engines.solvers.greedy_parallel import (
    GREEDY_MAX_PARALLEL_SET_SIZE,
    pick_best_action,
    pick_first_legal_fitting_action,
    terminal_success_value,
)
from unified_planning.engines.solvers.ptrpg_guided_rollout import (
    remaining_deadline,
    resolve_rollout_policy,
)

logger = logging.getLogger(__name__)

MAX_PREFIX_STEPS = 20
MAX_POLICY_HEURISTIC_CALLS_PER_EVAL = 200
MAX_SECONDS_PER_FIXED_TAIL_EVAL = 2.0

PREFIX_POLICY_FIRST_LEGAL = "first_legal_fitting"
PREFIX_POLICY_PTRPG_GREEDY = "ptrpg_greedy"


class FixedTailSafetyError(RuntimeError):
    """Raised when fixed-tail evaluation exceeds safety budgets."""

    def __init__(
        self,
        message: str,
        *,
        remaining: int,
        fixed_tail_h: int,
        delta: int,
        prefix_steps: int,
        selected_action: object,
        selected_duration: float,
        legal_action_count: int,
        heuristic_calls: int,
    ):
        super().__init__(message)
        self.remaining = remaining
        self.fixed_tail_h = fixed_tail_h
        self.delta = delta
        self.prefix_steps = prefix_steps
        self.selected_action = selected_action
        self.selected_duration = selected_duration
        self.legal_action_count = legal_action_count
        self.heuristic_calls = heuristic_calls


@dataclass
class FixedTailProfiler:
    fixed_tail_evaluations: int = 0
    total_eval_seconds: float = 0.0
    total_prefix_steps: int = 0
    total_policy_heuristic_calls: int = 0
    total_tail_ptrpg_calls: int = 0
    total_mdp_step_calls: int = 0
    wait_boundary_hits: int = 0
    max_delta_seen: int = 0
    max_prefix_steps_seen: int = 0
    _first_trace_printed: bool = False

    def record_eval(
        self,
        elapsed_s: float,
        prefix_steps: int,
        policy_calls: int,
        tail_calls: int,
        mdp_steps: int,
        waited: bool,
        delta: int,
    ) -> None:
        self.fixed_tail_evaluations += 1
        self.total_eval_seconds += elapsed_s
        self.total_prefix_steps += prefix_steps
        self.total_policy_heuristic_calls += policy_calls
        self.total_tail_ptrpg_calls += tail_calls
        self.total_mdp_step_calls += mdp_steps
        if waited:
            self.wait_boundary_hits += 1
        if delta > self.max_delta_seen:
            self.max_delta_seen = delta
        if prefix_steps > self.max_prefix_steps_seen:
            self.max_prefix_steps_seen = prefix_steps
        if self.fixed_tail_evaluations % 10 == 0:
            self._print_summary()

    def _print_summary(self) -> None:
        n = self.fixed_tail_evaluations
        avg_s = self.total_eval_seconds / n if n else 0.0
        avg_prefix = self.total_prefix_steps / n if n else 0.0
        avg_policy = self.total_policy_heuristic_calls / n if n else 0.0
        print(
            f"[fixed_tail profiler] evals={n} avg_seconds_per_eval={avg_s:.4f} "
            f"total_prefix_steps={self.total_prefix_steps} "
            f"avg_prefix_steps_per_eval={avg_prefix:.2f} "
            f"total_policy_heuristic_calls={self.total_policy_heuristic_calls} "
            f"avg_policy_heuristic_calls_per_eval={avg_policy:.2f} "
            f"total_tail_ptrpg_calls={self.total_tail_ptrpg_calls} "
            f"total_mdp_step_calls={self.total_mdp_step_calls} "
            f"wait_boundary_hits={self.wait_boundary_hits} "
            f"max_delta_seen={self.max_delta_seen} "
            f"max_prefix_steps_seen={self.max_prefix_steps_seen}",
            flush=True,
        )


_PROFILER = FixedTailProfiler()


def get_fixed_tail_profiler() -> FixedTailProfiler:
    return _PROFILER


def reset_fixed_tail_profiler() -> None:
    global _PROFILER
    _PROFILER = FixedTailProfiler()


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


def _resolve_prefix_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized in (PREFIX_POLICY_FIRST_LEGAL, PREFIX_POLICY_PTRPG_GREEDY):
        return normalized
    raise ValueError(
        f"Unknown fixed_tail prefix policy {policy!r}; "
        f"expected {PREFIX_POLICY_FIRST_LEGAL!r} or {PREFIX_POLICY_PTRPG_GREEDY!r}"
    )


@dataclass
class FixedTailConfig:
    fixed_tail_h: int = 10
    policy_strategy: str = "atom_backtrack_exact_resolution"
    tail_strategy: Optional[str] = None
    prefix_policy: str = PREFIX_POLICY_FIRST_LEGAL
    debug_eval_limit: int = 5
    debug_enabled: bool = False

    def __post_init__(self) -> None:
        self.fixed_tail_h = max(0, int(self.fixed_tail_h))
        self.prefix_policy = _resolve_prefix_policy(self.prefix_policy)
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
    tail_start_time: float = 0.0
    tail_ptrpg_value: float = 0.0
    returned_value: float = 0.0
    goal_reached: bool = False
    dead_end: bool = False
    stn_infeasible: bool = False
    prefix_steps: int = 0
    policy_heuristic_calls: int = 0
    tail_ptrpg_calls: int = 0
    mdp_step_calls: int = 0


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
    trace: _EvalTrace = field(default_factory=_EvalTrace)


def fixed_tail_config_from_args(args=None) -> FixedTailConfig:
    cli = args if args is not None else getattr(up, "args", None)
    fixed_tail_h = 10
    policy = "atomic_exact_resolution"
    prefix_policy = PREFIX_POLICY_FIRST_LEGAL
    debug = False
    if cli is not None:
        fixed_tail_h = int(getattr(cli, "fixed_tail_h", fixed_tail_h))
        policy = getattr(cli, "ptrpg_guided_rollout_policy", policy)
        prefix_policy = getattr(cli, "fixed_tail_prefix_policy", prefix_policy)
        debug = bool(getattr(cli, "fixed_tail_debug", False))
    return FixedTailConfig(
        fixed_tail_h=fixed_tail_h,
        policy_strategy=resolve_rollout_policy(policy),
        prefix_policy=_resolve_prefix_policy(prefix_policy),
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


def _check_safety(
    *,
    t0: float,
    remaining: int,
    fixed_tail_h: int,
    delta: int,
    prefix_steps: int,
    action: object,
    duration: float,
    legal_count: int,
    heuristic_calls: int,
) -> None:
    elapsed_s = time.time() - t0
    if elapsed_s > MAX_SECONDS_PER_FIXED_TAIL_EVAL:
        raise FixedTailSafetyError(
            f"fixed_tail eval exceeded {MAX_SECONDS_PER_FIXED_TAIL_EVAL}s "
            f"({elapsed_s:.3f}s elapsed)",
            remaining=remaining,
            fixed_tail_h=fixed_tail_h,
            delta=delta,
            prefix_steps=prefix_steps,
            selected_action=action,
            selected_duration=duration,
            legal_action_count=legal_count,
            heuristic_calls=heuristic_calls,
        )
    if prefix_steps > MAX_PREFIX_STEPS:
        raise FixedTailSafetyError(
            f"fixed_tail prefix exceeded {MAX_PREFIX_STEPS} steps",
            remaining=remaining,
            fixed_tail_h=fixed_tail_h,
            delta=delta,
            prefix_steps=prefix_steps,
            selected_action=action,
            selected_duration=duration,
            legal_action_count=legal_count,
            heuristic_calls=heuristic_calls,
        )
    if heuristic_calls > MAX_POLICY_HEURISTIC_CALLS_PER_EVAL:
        raise FixedTailSafetyError(
            f"fixed_tail exceeded {MAX_POLICY_HEURISTIC_CALLS_PER_EVAL} "
            f"policy heuristic calls",
            remaining=remaining,
            fixed_tail_h=fixed_tail_h,
            delta=delta,
            prefix_steps=prefix_steps,
            selected_action=action,
            selected_duration=duration,
            legal_action_count=legal_count,
            heuristic_calls=heuristic_calls,
        )


def _estimate_ptrpg_greedy_calls(mdp: "up.engines.MDP", state: "up.engines.State") -> int:
    legal = mdp.legal_actions(state)
    total = len(legal)
    for action in legal:
        transitions = mdp.transition_function(state, action)
        total += max(0, len(transitions) - 1)
    return total


def _pick_prefix_action(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    gap: float,
    config: FixedTailConfig,
    heuristic_name: str,
    temporal_heuristic_depth: int,
    policy_calls: List[int],
) -> Optional[Tuple]:
    legal_count = len(mdp.legal_actions(state))
    if config.prefix_policy == PREFIX_POLICY_FIRST_LEGAL:
        return pick_first_legal_fitting_action(
            mdp=mdp,
            state=state,
            stn=stn,
            previous_action_node=previous_action_node,
            max_duration=gap,
        )

    remaining = remaining_deadline(mdp, stn)
    score_depth = min(
        int(temporal_heuristic_depth),
        max(0, int(math.floor(gap))),
        remaining,
    )
    policy_calls[0] += _estimate_ptrpg_greedy_calls(mdp, state)
    return pick_best_action(
        mdp=mdp,
        state=state,
        stn=stn,
        previous_action_node=previous_action_node,
        heuristic_name=heuristic_name,
        temporal_heuristic_depth=score_depth,
        temporal_heuristic_strategy=config.policy_strategy,
        tie_break="legacy",
    )


def _run_prefix_rollout(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    *,
    fixed_tail_h: int,
    delta: int,
    config: FixedTailConfig,
    heuristic_name: str,
    temporal_heuristic_depth: int,
    trace: _EvalTrace,
    t0: float,
    max_parallel_set_size: int = GREEDY_MAX_PARALLEL_SET_SIZE,
) -> _PrefixResult:
    """
    Prefix rollout until tail boundary: dispatch parallel sets of legal actions
    per time slice (same structure as greedy_parallel), then caller runs PTRPG once.
    """
    assert delta >= 0, f"delta must be non-negative, got {delta}"
    start_ct = float(stn.get_current_end_time())
    boundary_time = start_ct + float(delta)
    current_state = state
    current_stn = stn
    current_prev = previous_action_node
    waited = False
    policy_calls = [0]
    prefix_steps = 0
    selected_action = None
    selected_duration = 0.0

    while True:
        remaining = remaining_deadline(mdp, current_stn)
        if remaining <= fixed_tail_h:
            break

        elapsed = float(current_stn.get_current_end_time()) - start_ct
        if elapsed >= delta - 1e-9:
            break

        _check_safety(
            t0=t0,
            remaining=remaining,
            fixed_tail_h=fixed_tail_h,
            delta=delta,
            prefix_steps=prefix_steps,
            action=selected_action,
            duration=selected_duration,
            legal_count=len(mdp.legal_actions(current_state)),
            heuristic_calls=policy_calls[0],
        )

        if terminal_success_value(mdp, current_state, current_stn) >= 1.0:
            return _PrefixResult(
                state=current_state,
                stn=current_stn,
                previous_action_node=current_prev,
                waited=waited,
                boundary_time=boundary_time,
                goal_reached=True,
                terminal_value=1.0,
                trace=trace,
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
                trace=trace,
            )

        decision_time = float(current_stn.get_current_end_time())
        gap = delta - (decision_time - start_ct)
        if gap <= 1e-9:
            break

        chosen_in_set = 0
        while chosen_in_set < max_parallel_set_size:
            elapsed = float(current_stn.get_current_end_time()) - start_ct
            gap = delta - elapsed
            if gap <= 1e-9:
                break

            picked = _pick_prefix_action(
                mdp,
                current_state,
                current_stn,
                current_prev,
                gap,
                config,
                heuristic_name,
                temporal_heuristic_depth,
                policy_calls,
            )
            if picked is None:
                break

            _score, action, candidate_stn, candidate_prev, _any_term, _cache = picked
            selected_action = action
            dur = float(candidate_stn.get_current_end_time()) - float(
                current_stn.get_current_end_time()
            )
            selected_duration = dur

            if dur <= 0.0 or dur > gap + 1e-9:
                break

            prev_elapsed = elapsed
            terminal, next_state, _reward = mdp.step(current_state, action)
            trace.mdp_step_calls += 1
            prefix_steps += 1
            trace.prefix_actions.append(getattr(action, "name", str(action)))
            current_state = next_state
            current_stn = candidate_stn
            current_prev = candidate_prev
            chosen_in_set += 1
            trace.remaining_after_steps.append(remaining_deadline(mdp, current_stn))

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
                        trace=trace,
                    )
                return _PrefixResult(
                    state=current_state,
                    stn=current_stn,
                    previous_action_node=current_prev,
                    waited=waited,
                    boundary_time=boundary_time,
                    dead_end=True,
                    terminal_value=0.0,
                    trace=trace,
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
                    trace=trace,
                )

            if float(current_stn.get_current_end_time()) > decision_time + 1e-9:
                break

        if chosen_in_set == 0:
            waited = True
            break

    trace.prefix_steps = prefix_steps
    trace.policy_heuristic_calls = policy_calls[0]
    trace.waited_at_boundary = waited
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
        trace=trace,
    )


def _log_eval_trace(trace: _EvalTrace) -> None:
    logger.info(
        "fixed_tail_ptrpg_rollout leaf_id=%s R=%d FIXED_TAIL_H=%d delta=%d "
        "prefix_actions=%s remaining_after=%s waited_at_boundary=%s "
        "final_tail_horizon=%d tail_start_time=%.3f tail_ptrpg=%.6f returned=%.6f "
        "goal=%s dead_end=%s stn_infeasible=%s prefix_steps=%d policy_calls=%d",
        trace.leaf_id,
        trace.original_remaining,
        trace.fixed_tail_h,
        trace.delta,
        trace.prefix_actions,
        trace.remaining_after_steps,
        trace.waited_at_boundary,
        trace.final_tail_horizon,
        trace.tail_start_time,
        trace.tail_ptrpg_value,
        trace.returned_value,
        trace.goal_reached,
        trace.dead_end,
        trace.stn_infeasible,
        trace.prefix_steps,
        trace.policy_heuristic_calls,
    )


def _print_first_eval_trace(trace: _EvalTrace) -> None:
    print(
        f"[fixed_tail first eval] leaf_id={trace.leaf_id} R={trace.original_remaining} "
        f"FIXED_TAIL_H={trace.fixed_tail_h} delta={trace.delta} "
        f"prefix_actions={trace.prefix_actions} remaining_after={trace.remaining_after_steps} "
        f"waited_at_boundary={trace.waited_at_boundary} "
        f"final_tail_horizon={trace.final_tail_horizon} tail_start_time={trace.tail_start_time:.3f} "
        f"tail_ptrpg={trace.tail_ptrpg_value:.6f} returned={trace.returned_value:.6f} "
        f"prefix_steps={trace.prefix_steps} policy_heuristic_calls={trace.policy_heuristic_calls} "
        f"tail_ptrpg_calls={trace.tail_ptrpg_calls} mdp_step_calls={trace.mdp_step_calls}",
        flush=True,
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
    t0 = time.time()
    trace = _EvalTrace(leaf_id=leaf_id)
    H = config.fixed_tail_h
    trace.fixed_tail_h = H
    tail_strategy = config.tail_strategy or config.policy_strategy

    R = remaining_deadline(mdp, stn)
    trace.original_remaining = R

    def _finish(value: float, tail_horizon: int) -> float:
        trace.returned_value = value
        elapsed_s = time.time() - t0
        _PROFILER.record_eval(
            elapsed_s=elapsed_s,
            prefix_steps=trace.prefix_steps,
            policy_calls=trace.policy_heuristic_calls,
            tail_calls=trace.tail_ptrpg_calls,
            mdp_steps=trace.mdp_step_calls,
            waited=trace.waited_at_boundary,
            delta=trace.delta,
        )
        trace_first = (
            debug_emit
            or bool(os.environ.get("FIXED_TAIL_TRACE_FIRST"))
            or not _PROFILER._first_trace_printed
        )
        if trace_first and _PROFILER.fixed_tail_evaluations == 1:
            _PROFILER._first_trace_printed = True
            _print_first_eval_trace(trace)
        if debug_emit:
            _log_eval_trace(trace)
        return _assert_return(value, R, H, tail_horizon)

    if _goal_reached(mdp, state):
        trace.goal_reached = True
        return _finish(1.0, trace.final_tail_horizon)

    dead_end, stn_bad = _failure_at(mdp, state, stn)
    if dead_end or stn_bad:
        trace.dead_end = dead_end
        trace.stn_infeasible = stn_bad
        return _finish(0.0, trace.final_tail_horizon)

    if R <= H:
        _check_safety(
            t0=t0,
            remaining=R,
            fixed_tail_h=H,
            delta=0,
            prefix_steps=0,
            action=None,
            duration=0.0,
            legal_count=len(mdp.legal_actions(state)),
            heuristic_calls=0,
        )
        current_time = float(stn.get_current_end_time())
        trace.final_tail_horizon = R
        trace.tail_start_time = current_time
        trace.tail_ptrpg_calls += 1
        tail_val = _ptrpg_at_horizon(mdp, state, current_time, R, tail_strategy)
        _check_safety(
            t0=t0,
            remaining=R,
            fixed_tail_h=H,
            delta=0,
            prefix_steps=0,
            action=None,
            duration=0.0,
            legal_count=len(mdp.legal_actions(state)),
            heuristic_calls=0,
        )
        trace.tail_ptrpg_value = tail_val
        return _finish(tail_val, trace.final_tail_horizon)

    delta = R - H
    trace.delta = delta
    assert delta >= 0

    prefix = _run_prefix_rollout(
        mdp,
        state,
        stn,
        previous_action_node,
        fixed_tail_h=H,
        delta=delta,
        config=config,
        heuristic_name=heuristic_name,
        temporal_heuristic_depth=temporal_heuristic_depth,
        trace=trace,
        t0=t0,
    )
    trace.prefix_steps = prefix.trace.prefix_steps
    trace.policy_heuristic_calls = prefix.trace.policy_heuristic_calls
    trace.waited_at_boundary = prefix.waited
    trace.prefix_actions = prefix.trace.prefix_actions
    trace.remaining_after_steps = prefix.trace.remaining_after_steps
    trace.mdp_step_calls = prefix.trace.mdp_step_calls

    if prefix.terminal_value is not None:
        if prefix.goal_reached:
            trace.goal_reached = True
        elif prefix.dead_end:
            trace.dead_end = True
        elif prefix.stn_infeasible:
            trace.stn_infeasible = True
        return _finish(prefix.terminal_value, trace.final_tail_horizon)

    remaining_after = remaining_deadline(mdp, prefix.stn)
    if remaining_after <= H:
        tail_horizon = remaining_after
        tail_time = float(prefix.stn.get_current_end_time())
    else:
        tail_horizon = H
        tail_time = prefix.boundary_time
    _check_safety(
        t0=t0,
        remaining=remaining_after,
        fixed_tail_h=H,
        delta=delta,
        prefix_steps=trace.prefix_steps,
        action=None,
        duration=0.0,
        legal_count=len(mdp.legal_actions(prefix.state)),
        heuristic_calls=trace.policy_heuristic_calls,
    )
    trace.final_tail_horizon = tail_horizon
    trace.tail_start_time = tail_time
    trace.tail_ptrpg_calls += 1
    tail_val = _ptrpg_at_horizon(
        mdp,
        prefix.state,
        tail_time,
        tail_horizon,
        tail_strategy,
    )
    _check_safety(
        t0=t0,
        remaining=remaining_after,
        fixed_tail_h=H,
        delta=delta,
        prefix_steps=trace.prefix_steps,
        action=None,
        duration=0.0,
        legal_count=len(mdp.legal_actions(prefix.state)),
        heuristic_calls=trace.policy_heuristic_calls,
    )
    trace.tail_ptrpg_value = tail_val
    if R > H and remaining_after > H:
        assert tail_horizon == H
    return _finish(tail_val, trace.final_tail_horizon)


def _assert_return(value: float, original_r: int, fixed_tail_h: int, tail_horizon: int) -> float:
    assert 0.0 <= value <= 1.0, f"fixed_tail value out of range: {value}"
    if original_r > fixed_tail_h and tail_horizon > 0:
        assert tail_horizon <= fixed_tail_h, (
            f"tail horizon {tail_horizon} exceeds fixed_tail_h {fixed_tail_h}"
        )
    return value


__all__ = [
    "FixedTailConfig",
    "FixedTailProfiler",
    "FixedTailSafetyError",
    "PREFIX_POLICY_FIRST_LEGAL",
    "PREFIX_POLICY_PTRPG_GREEDY",
    "fixed_tail_config_from_args",
    "fixed_tail_ptrpg_value",
    "get_fixed_tail_profiler",
    "reset_fixed_tail_profiler",
]
