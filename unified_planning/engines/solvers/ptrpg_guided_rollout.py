"""
PTRPG-guided terminal rollout for MCTS leaf evaluation.

Uses the same greedy MDP dispatcher as ``greedy_parallel.plan()``; backs up
terminal 0/1 only (goal before deadline = 1.0, else 0.0).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import unified_planning as up
from unified_planning.engines.solvers.greedy_parallel import (
    GREEDY_DEFAULT_TIME_SLICES,
    GREEDY_HEURISTIC_WEIGHT,
    GREEDY_MAX_PARALLEL_SET_SIZE,
    pick_best_action,
    simulate_greedy_mdp_until_terminal,
    terminal_success_value,
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
    max_time_slices: int = GREEDY_DEFAULT_TIME_SLICES
    max_parallel_set_size: int = GREEDY_MAX_PARALLEL_SET_SIZE
    max_mdp_steps: Optional[int] = None
    epsilon: float = 0.0
    debug_first_rollout: bool = False


def rollout_config_from_args(mdp: "up.engines.MDP", args=None) -> RolloutConfig:
    cli = args if args is not None else getattr(up, "args", None)
    policy = "baseline_survival_resolution"
    max_mdp_steps = None
    max_time_slices = GREEDY_DEFAULT_TIME_SLICES
    epsilon = 0.0
    debug = False
    if cli is not None:
        policy = getattr(cli, "ptrpg_guided_rollout_policy", policy)
        max_mdp_steps = getattr(cli, "ptrpg_guided_rollout_max_steps", None)
        max_time_slices = getattr(
            cli, "ptrpg_guided_rollout_max_time_slices", max_time_slices
        )
        epsilon = float(getattr(cli, "ptrpg_guided_rollout_epsilon", 0.0))
        debug = bool(getattr(cli, "ptrpg_guided_rollout_debug", False))
    if max_mdp_steps is None:
        max_mdp_steps = GREEDY_MAX_PARALLEL_SET_SIZE * max(1, int(max_time_slices))
    return RolloutConfig(
        policy_strategy=resolve_rollout_policy(policy),
        max_time_slices=max(1, int(max_time_slices)),
        max_mdp_steps=max(1, int(max_mdp_steps)),
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
        tie_break="legacy",
    )


def _log_rollout_step(
    step_idx: int,
    remaining: int,
    action_name: str,
    terminal: bool,
):
    logger.info(
        "ptrpg_guided_rollout step=%d remaining=%d selected=%s terminal=%s",
        step_idx,
        remaining,
        action_name,
        terminal,
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
    _ = getattr(config, "epsilon", 0.0)
    policy_strategy = config.policy_strategy
    step_log: list = []

    def on_mdp_step(mdp_step, _state, current_stn, action, terminal):
        if debug_emit:
            step_log.append((mdp_step, action, terminal, current_stn))

    value, final_stn = simulate_greedy_mdp_until_terminal(
        mdp=mdp,
        state=state,
        stn=stn,
        previous_action_node=previous_action_node,
        heuristic_name=heuristic_name,
        temporal_heuristic_depth=temporal_heuristic_depth,
        temporal_heuristic_strategy=policy_strategy,
        max_time_slices=config.max_time_slices,
        max_parallel_set_size=config.max_parallel_set_size,
        max_mdp_steps=config.max_mdp_steps,
        tie_break="legacy",
        heuristic_weight=heuristic_weight,
        on_mdp_step=on_mdp_step if debug_emit else None,
    )

    if debug_emit:
        for mdp_step, action, terminal, current_stn in step_log:
            _log_rollout_step(
                step_idx=mdp_step,
                remaining=remaining_deadline(mdp, current_stn),
                action_name=getattr(action, "name", str(action)),
                terminal=terminal,
            )
        logger.info(
            "ptrpg_guided_rollout final_value=%.1f goal_reached=%s",
            value,
            value >= 1.0,
        )

    return value


__all__ = [
    "RolloutConfig",
    "PTRPG_GUIDED_POLICY_ALIASES",
    "pick_greedy_rollout_action",
    "ptrpg_guided_terminal_rollout",
    "remaining_deadline",
    "resolve_rollout_policy",
    "rollout_config_from_args",
    "state_time_key",
    "terminal_success_value",
]
