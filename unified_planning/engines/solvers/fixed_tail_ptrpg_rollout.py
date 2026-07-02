"""
Fixed-tail PTRPG evaluation for MCTS leaf backup.

Per search: prefix_budget = floor(prefix_frac * root_remaining), tail_horizon =
root_remaining - prefix_budget. Ephemeral prefix rollouts run while remaining > tail_horizon,
then PTRPG(tail_horizon). Nodes already in the tail zone (remaining <= tail_horizon) skip
rollout and use PTRPG(actual remaining). MCTS does not expand when remaining <= tail_horizon.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import unified_planning as up
from unified_planning.engines.solvers.greedy_parallel import terminal_success_value

logger = logging.getLogger(__name__)

_FIXED_TAIL_DEBUG_MAX = 5


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
    import os

    env_agg = (os.environ.get("TP_MCTS_HEURISTIC_AGGREGATION") or "").strip().lower()
    if env_agg:
        return env_agg
    strat = (temporal_heuristic_strategy or "").strip().lower()
    if strat == "baseline_survival_meanvar":
        return "meanvar"
    if strat == "baseline_time_to_goal":
        return "time_to_goal"
    if strat == "baseline_admissible_paths_table":
        return "kernelized"
    return "product"


@dataclass
class FixedTailConfig:
    prefix_frac: float = 0.10
    tail_strategy: str = "atom_backtrack_exact_resolution"
    prefix_policy: str = "mcts_sampled"
    max_expectimax_nodes: int = 5000
    max_expectimax_depth: int = 64
    max_expectimax_time_sec: float = 0.0
    rollout_samples: int = 1
    rollout_policy: str = "random_legal_fitting"

    def __post_init__(self) -> None:
        frac = float(self.prefix_frac)
        if frac <= 0.0:
            self.prefix_frac = 0.0
        elif frac > 1.0:
            self.prefix_frac = 1.0
        else:
            self.prefix_frac = frac
        self.prefix_policy = str(self.prefix_policy).strip().lower()
        self.max_expectimax_nodes = max(1, int(self.max_expectimax_nodes))
        self.max_expectimax_depth = max(1, int(self.max_expectimax_depth))
        self.max_expectimax_time_sec = max(0.0, float(self.max_expectimax_time_sec))
        self.rollout_samples = max(1, int(self.rollout_samples))
        self.rollout_policy = str(self.rollout_policy).strip().lower()


@dataclass(frozen=True)
class FixedTailSearchContext:
    """Constant for one MCTS search; recomputed when online root changes."""

    root_remaining: int
    prefix_budget: int
    prefix_frac: float

    @property
    def tail_horizon(self) -> int:
        """Fixed PTRPG horizon for this search: root_remaining - prefix_budget."""
        return max(0, int(self.root_remaining) - int(self.prefix_budget))


def fixed_tail_config_from_args(args=None) -> FixedTailConfig:
    cli = args if args is not None else getattr(up, "args", None)
    prefix_frac = 0.10
    prefix_policy = "mcts_sampled"
    max_nodes = 5000
    max_depth = 64
    max_time = 0.0
    if cli is not None:
        prefix_frac = float(getattr(cli, "fixed_tail_prefix_frac", prefix_frac))
        prefix_policy = str(getattr(cli, "fixed_tail_prefix_policy", prefix_policy))
        max_nodes = int(getattr(cli, "fixed_tail_expectimax_max_nodes", max_nodes))
        max_depth = int(getattr(cli, "fixed_tail_expectimax_max_depth", max_depth))
        max_time = float(getattr(cli, "fixed_tail_expectimax_max_time_sec", max_time))
        rollout_samples = int(getattr(cli, "fixed_tail_rollout_samples", 1))
        rollout_policy = str(getattr(cli, "fixed_tail_rollout_policy", "random_legal_fitting"))
    else:
        rollout_samples = 1
        rollout_policy = "random_legal_fitting"
    return FixedTailConfig(
        prefix_frac=prefix_frac,
        prefix_policy=prefix_policy,
        max_expectimax_nodes=max_nodes,
        max_expectimax_depth=max_depth,
        max_expectimax_time_sec=max_time,
        rollout_samples=rollout_samples,
        rollout_policy=rollout_policy,
    )


def build_fixed_tail_search_context(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    config: FixedTailConfig,
) -> FixedTailSearchContext:
    root_remaining = node_remaining(mdp, state, stn)
    frac = config.prefix_frac
    if frac <= 0.0:
        prefix_budget = 0
    else:
        prefix_budget = max(0, int(math.floor(frac * root_remaining)))
    return FixedTailSearchContext(
        root_remaining=root_remaining,
        prefix_budget=prefix_budget,
        prefix_frac=frac,
    )


def _goal_reached(mdp: "up.engines.MDP", state: "up.engines.State") -> bool:
    if not mdp.is_terminal(state):
        return False
    goals = set(mdp.problem.goals)
    preds = getattr(state, "predicates", None)
    return preds is not None and goals.issubset(set(preds))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clock_time(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
) -> float:
    """Prefer combination-state simulated time; fall back to STN end time."""
    ct = getattr(state, "current_time", None)
    if ct is not None:
        return float(ct)
    return float(stn.get_current_end_time())


def node_remaining(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
) -> int:
    deadline = mdp.deadline()
    if deadline is None:
        return 0
    return max(0, int(math.floor(deadline - _clock_time(mdp, state, stn))))


def elapsed_from_root(ctx: FixedTailSearchContext, node_rem: int) -> int:
    return max(0, ctx.root_remaining - node_rem)


def crossed_cutoff(ctx: FixedTailSearchContext, elapsed: int) -> bool:
    return elapsed >= ctx.prefix_budget


def at_or_past_tail_horizon(ctx: FixedTailSearchContext, node_rem: int) -> bool:
    """True when the node is in the tail zone (no further MCTS expansion)."""
    return int(node_rem) <= ctx.tail_horizon


def fixed_tail_ptrpg_horizon(ctx: FixedTailSearchContext, node_rem: int) -> int:
    """Horizon passed to PTRPG at tail evaluation (fixed for the search)."""
    return ctx.tail_horizon


def fixed_tail_dead_end_value(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
) -> bool:
    """True when the node should backup 0.0 (infeasible / past deadline / no legal actions)."""
    if not stn.is_consistent():
        return True
    end = _clock_time(mdp, state, stn)
    if mdp.deadline() is not None and end > mdp.deadline() + 1e-9:
        return True
    if len(mdp.legal_actions(state)) == 0 and not _goal_reached(mdp, state):
        return True
    return False


def ptrpg_at_horizon(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
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
    current_time = _clock_time(mdp, state, stn)
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


def fixed_tail_bootstrap_value(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    strategy: str,
    *,
    ctx: Optional[FixedTailSearchContext] = None,
    debug_emit: bool = False,
    debug_label: str = "",
    action_duration: Optional[float] = None,
    child_remaining: Optional[int] = None,
    child_elapsed: Optional[int] = None,
    crossed: Optional[bool] = None,
) -> float:
    """
    Bootstrap leaf: goal 1.0, dead-end 0.0, else PTRPG with horizon = node remaining.
    """
    if _goal_reached(mdp, state):
        value = 1.0
        horizon_used = 0
    elif terminal_success_value(mdp, state, stn) >= 1.0:
        value = 1.0
        horizon_used = 0
    elif fixed_tail_dead_end_value(mdp, state, stn):
        value = 0.0
        horizon_used = 0
    else:
        rem = node_remaining(mdp, state, stn)
        if ctx is not None:
            if rem <= ctx.tail_horizon:
                horizon_used = rem
            else:
                horizon_used = ctx.tail_horizon
            value = ptrpg_at_horizon(mdp, state, stn, horizon_used, strategy)
        else:
            horizon_used = rem
            value = ptrpg_at_horizon(mdp, state, stn, rem, strategy)

    if debug_emit and ctx is not None:
        node_rem = node_remaining(mdp, state, stn)
        el = elapsed_from_root(ctx, node_rem)
        _emit_fixed_tail_debug(
            label=debug_label,
            ctx=ctx,
            node_remaining=node_rem,
            elapsed_from_root=el,
            action_duration=action_duration,
            child_remaining=child_remaining,
            child_elapsed=child_elapsed,
            crossed_cutoff=crossed if crossed is not None else crossed_cutoff(ctx, el),
            ptrpg_horizon=horizon_used,
            returned_value=value,
        )

    return value


def _emit_fixed_tail_debug(
    *,
    label: str,
    ctx: FixedTailSearchContext,
    node_remaining: int,
    elapsed_from_root: int,
    action_duration: Optional[float],
    child_remaining: Optional[int],
    child_elapsed: Optional[int],
    crossed_cutoff: bool,
    ptrpg_horizon: int,
    returned_value: float,
) -> None:
    parts = [
        f"[fixed_tail] {label}",
        f"root_remaining={ctx.root_remaining}",
        f"prefix_frac={ctx.prefix_frac}",
        f"prefix_budget={ctx.prefix_budget}",
        f"tail_horizon={ctx.tail_horizon}",
        f"node_remaining={node_remaining}",
        f"elapsed_from_root={elapsed_from_root}",
    ]
    if action_duration is not None:
        parts.append(f"action_duration={action_duration}")
    if child_remaining is not None:
        parts.append(f"child_remaining={child_remaining}")
    if child_elapsed is not None:
        parts.append(f"child_elapsed={child_elapsed}")
    parts.append(f"crossed_cutoff={'yes' if crossed_cutoff else 'no'}")
    parts.append(f"ptrpg_horizon={ptrpg_horizon}")
    parts.append(f"returned_value={returned_value:.6f}")
    print(" ".join(parts), flush=True)


__all__ = [
    "FixedTailConfig",
    "FixedTailSearchContext",
    "build_fixed_tail_search_context",
    "at_or_past_tail_horizon",
    "crossed_cutoff",
    "elapsed_from_root",
    "fixed_tail_ptrpg_horizon",
    "fixed_tail_bootstrap_value",
    "fixed_tail_config_from_args",
    "fixed_tail_dead_end_value",
    "node_remaining",
    "ptrpg_at_horizon",
]
