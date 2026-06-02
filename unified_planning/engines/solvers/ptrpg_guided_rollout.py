"""
PTRPG-guided terminal rollout for MCTS leaf evaluation.

Uses greedy PTRPG action scoring only as a rollout policy; backs up terminal 0/1.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import unified_planning as up
from unified_planning.engines.solvers.greedy_parallel import (
    GREEDY_HEURISTIC_WEIGHT,
    pick_best_action,
)
logger = logging.getLogger(__name__)

PTRPG_GUIDED_POLICY_ALIASES = {
    "atomic_exact_resolution": "atom_backtrack_exact_resolution",
    "atom_backtrack_exact_resolution": "atom_backtrack_exact_resolution",
    "baseline_survival_resolution": "baseline_survival_resolution",
}


def resolve_rollout_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized in PTRPG_GUIDED_POLICY_ALIASES:
        return PTRPG_GUIDED_POLICY_ALIASES[normalized]
    raise ValueError(
        f"Unknown ptrpg guided rollout policy {policy!r}; "
        f"expected one of: {', '.join(sorted(set(PTRPG_GUIDED_POLICY_ALIASES)))}"
    )


@dataclass
class RolloutConfig:
    policy_strategy: str
    max_steps: int
    epsilon: float = 0.0
    loop_repeat_limit: int = 3
    debug_first_rollout: bool = False


def rollout_config_from_args(mdp: "up.engines.MDP", args=None) -> RolloutConfig:
    cli = args if args is not None else getattr(up, "args", None)
    policy = "baseline_survival_resolution"
    max_steps = None
    epsilon = 0.0
    debug = False
    if cli is not None:
        policy = getattr(cli, "ptrpg_guided_rollout_policy", policy)
        max_steps = getattr(cli, "ptrpg_guided_rollout_max_steps", None)
        epsilon = float(getattr(cli, "ptrpg_guided_rollout_epsilon", 0.0))
        debug = bool(getattr(cli, "ptrpg_guided_rollout_debug", False))
    if max_steps is None:
        deadline = mdp.deadline()
        max_steps = int(deadline) if deadline is not None else 500
    return RolloutConfig(
        policy_strategy=resolve_rollout_policy(policy),
        max_steps=max(1, int(max_steps)),
        epsilon=epsilon,
        debug_first_rollout=debug,
    )


def remaining_deadline(mdp: "up.engines.MDP", stn: "up.plans.stn.STNPlan") -> int:
    deadline = mdp.deadline()
    if deadline is None:
        return 0
    return max(0, int(math.floor(deadline - stn.get_current_end_time())))


def state_time_key(state: "up.engines.State", remaining: int) -> Tuple[object, int]:
    preds = getattr(state, "predicates", None)
    if preds is not None:
        return (frozenset(preds), remaining)
    return (id(state), remaining)


def terminal_success_value(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
) -> float:
    if not mdp.is_terminal(state):
        return 0.0
    if stn.get_current_end_time() > mdp.deadline():
        return 0.0
    return 1.0


def pick_greedy_rollout_action(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    policy_strategy: str,
    heuristic_name: str,
    temporal_heuristic_depth: int,
    heuristic_weight: float = GREEDY_HEURISTIC_WEIGHT,
):
    return pick_best_action(
        mdp=mdp,
        state=state,
        stn=stn,
        previous_action_node=previous_action_node,
        heuristic_name=heuristic_name,
        temporal_heuristic_depth=temporal_heuristic_depth,
        temporal_heuristic_strategy=policy_strategy,
        heuristic_weight=heuristic_weight,
        tie_break="stable",
    )


def _log_rollout_step(
    step_idx: int,
    remaining: int,
    legal_count: int,
    action_name: str,
    action_score: float,
    top_scores: list,
    terminal: bool,
    stn_infeasible: bool,
    dead_end: bool,
):
    top_str = ", ".join(f"{name}={score:.4f}" for name, score in top_scores)
    logger.info(
        "ptrpg_guided_rollout step=%d remaining=%d legal=%d selected=%s score=%.4f "
        "top5=[%s] terminal=%s stn_infeasible=%s dead_end=%s",
        step_idx,
        remaining,
        legal_count,
        action_name,
        action_score,
        top_str,
        terminal,
        stn_infeasible,
        dead_end,
    )


def ptrpg_guided_terminal_rollout(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    config: RolloutConfig,
    heuristic_name: str = "temporal_probabilistic_rpg",
    temporal_heuristic_depth: int = 25,
    heuristic_weight: float = GREEDY_HEURISTIC_WEIGHT,
    debug_emit: bool = False,
) -> float:
    del config.epsilon  # reserved for future epsilon-greedy ablations

    current_state = state
    current_stn = stn
    current_prev = previous_action_node
    visited: Dict[Tuple[object, int], int] = {}
    policy_strategy = config.policy_strategy

    for step_idx in range(config.max_steps):
        if mdp.is_terminal(current_state):
            final_v = terminal_success_value(mdp, current_state, current_stn)
            if debug_emit:
                logger.info("ptrpg_guided_rollout final_value=%.1f goal_reached=yes", final_v)
            return final_v

        remaining = remaining_deadline(mdp, current_stn)
        if remaining <= 0:
            final_v = terminal_success_value(mdp, current_state, current_stn)
            if debug_emit:
                logger.info(
                    "ptrpg_guided_rollout final_value=%.1f goal_reached=%s remaining=0",
                    final_v,
                    final_v >= 1.0,
                )
            return final_v

        loop_key = state_time_key(current_state, remaining)
        visited[loop_key] = visited.get(loop_key, 0) + 1
        if visited[loop_key] >= config.loop_repeat_limit:
            if debug_emit:
                logger.info("ptrpg_guided_rollout final_value=0.0 loop_guard=yes")
            return 0.0

        legal_actions = mdp.legal_actions(current_state)
        if not legal_actions:
            if debug_emit:
                logger.info("ptrpg_guided_rollout final_value=0.0 dead_end=yes legal=0")
            return 0.0

        from unified_planning.engines.solvers.greedy_parallel import rank_actions_by_score

        ranked = rank_actions_by_score(
            mdp=mdp,
            state=current_state,
            stn=current_stn,
            previous_action_node=current_prev,
            legal_actions=legal_actions,
            heuristic_name=heuristic_name,
            temporal_heuristic_depth=temporal_heuristic_depth,
            temporal_heuristic_strategy=policy_strategy,
            heuristic_weight=heuristic_weight,
            tie_break="stable",
        )
        if not ranked:
            if debug_emit:
                logger.info("ptrpg_guided_rollout final_value=0.0 no_feasible_action=yes")
            return 0.0

        best_score, best_action, next_stn, next_prev, _, _ = ranked[0]
        if debug_emit:
            top_scores = [
                (getattr(a, "name", str(a)), s) for s, a, *_ in ranked[:5]
            ]
            _log_rollout_step(
                step_idx=step_idx,
                remaining=remaining,
                legal_count=len(legal_actions),
                action_name=getattr(best_action, "name", str(best_action)),
                action_score=best_score,
                top_scores=top_scores,
                terminal=False,
                stn_infeasible=False,
                dead_end=False,
            )

        terminal, next_state, _reward = mdp.step(current_state, best_action)
        if not next_stn.is_consistent() or next_stn.get_current_end_time() > mdp.deadline():
            if debug_emit:
                logger.info(
                    "ptrpg_guided_rollout final_value=0.0 stn_infeasible=yes "
                    "stochastic_outcome terminal=%s",
                    terminal,
                )
            return 0.0

        if debug_emit:
            logger.info(
                "ptrpg_guided_rollout stochastic_outcome terminal=%s goal=%s",
                terminal,
                mdp.is_terminal(next_state),
            )

        current_state = next_state
        current_stn = next_stn
        current_prev = next_prev

        if terminal:
            final_v = terminal_success_value(mdp, current_state, current_stn)
            if debug_emit:
                logger.info("ptrpg_guided_rollout final_value=%.1f goal_reached=%s", final_v, final_v >= 1.0)
            return final_v

        if not mdp.legal_actions(current_state) and not mdp.is_terminal(current_state):
            if debug_emit:
                logger.info("ptrpg_guided_rollout final_value=0.0 dead_end=yes")
            return 0.0

    final_v = terminal_success_value(mdp, current_state, current_stn)
    if debug_emit:
        logger.info(
            "ptrpg_guided_rollout final_value=%.1f max_steps_reached=yes goal_reached=%s",
            final_v,
            final_v >= 1.0,
        )
    return final_v
