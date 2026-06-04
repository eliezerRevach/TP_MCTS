"""
Ephemeral fixed-tail prefix rollouts for MCTS leaf evaluation (Option A).

From a selected leaf: K independent rollout samples on copied state/STN until the
time prefix budget; at most one PTRPG per sample. No MCTS tree mutation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import unified_planning as up
from unified_planning.engines.solvers.fixed_tail_expectimax import (
    _feasible_actions,
    _fit_action_stn,
)
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

_ROLLOUT_POLICIES = frozenset({"random_legal_fitting", "first_legal_fitting"})
_FIXED_TAIL_RANDOM_DEBUG_MAX = 5


def _action_name(action: "up.engines.Action") -> str:
    return str(getattr(action, "name", action))


def _action_positive_duration(action: "up.engines.Action") -> bool:
    if hasattr(action, "duration_int"):
        try:
            return int(action.duration_int()) > 0
        except Exception:
            pass
    if hasattr(action, "actions"):
        for sub in action.actions:
            if _action_positive_duration(sub):
                return True
        return False
    return False


def rollout_legal_fitting_actions(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
) -> List["up.engines.Action"]:
    feasible = _feasible_actions(mdp, state, stn, previous_action_node)
    return [a for a in feasible if _action_positive_duration(a)]


def copy_state_for_rollout(state: "up.engines.State") -> "up.engines.State":
    preds = set(getattr(state, "predicates", set()) or set())
    if isinstance(state, up.engines.CombinationState):
        active = state.active_actions.clone()
        return up.engines.CombinationState(preds, active, state.current_time)
    return up.engines.State(preds)


def pick_rollout_action(
    legal: List["up.engines.Action"],
    policy: str,
    rng: random.Random,
) -> Optional["up.engines.Action"]:
    if not legal:
        return None
    if policy == "first_legal_fitting":
        return sorted(legal, key=_action_name)[0]
    if policy == "random_legal_fitting":
        return rng.choice(legal)
    raise ValueError(
        f"Unknown rollout policy {policy!r}; expected one of: {sorted(_ROLLOUT_POLICIES)}"
    )


@dataclass
class FixedTailRandomRolloutConfig:
    num_samples: int = 1
    rollout_policy: str = "random_legal_fitting"

    def __post_init__(self) -> None:
        self.num_samples = max(1, int(self.num_samples))
        policy = str(self.rollout_policy).strip().lower()
        if policy not in _ROLLOUT_POLICIES:
            raise ValueError(
                f"rollout_policy must be one of {sorted(_ROLLOUT_POLICIES)}, got {policy!r}"
            )
        self.rollout_policy = policy


def random_rollout_config_from_args(args=None) -> FixedTailRandomRolloutConfig:
    cli = args if args is not None else getattr(up, "args", None)
    num_samples = 1
    rollout_policy = "random_legal_fitting"
    if cli is not None:
        num_samples = int(getattr(cli, "fixed_tail_rollout_samples", num_samples))
        rollout_policy = str(
            getattr(cli, "fixed_tail_rollout_policy", rollout_policy)
        )
    return FixedTailRandomRolloutConfig(
        num_samples=num_samples,
        rollout_policy=rollout_policy,
    )


@dataclass
class _SampleTrace:
    actions: List[str] = field(default_factory=list)
    sample_value: float = 0.0
    ptrpg_horizon: int = 0
    final_remaining: int = 0
    final_elapsed: int = 0
    terminated_early: bool = False


@dataclass
class FixedTailRandomRolloutEvaluator:
    mdp: "up.engines.MDP"
    ctx: FixedTailSearchContext
    config: FixedTailRandomRolloutConfig
    strategy: str
    rng: random.Random = field(default_factory=random.Random)

    _leaf_eval_count: int = 0
    _next_leaf_id: int = 0

    def reset_search(self) -> None:
        self._leaf_eval_count = 0
        self._next_leaf_id = 0

    def _should_debug(self) -> bool:
        cli = getattr(up, "args", None)
        if cli is None or not getattr(cli, "fixed_tail_debug", False):
            return False
        return self._leaf_eval_count < _FIXED_TAIL_RANDOM_DEBUG_MAX

    def _run_one_sample(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
        previous_action_node: "up.plans.stn.STNPlanNode",
    ) -> Tuple[float, _SampleTrace]:
        trace = _SampleTrace()
        current_state = copy_state_for_rollout(state)
        current_stn = stn.clone()
        current_prev = previous_action_node

        rem = node_remaining(self.mdp, current_state, current_stn)
        el = elapsed_from_root(self.ctx, rem)
        max_steps = max(1, self.ctx.prefix_budget + 10)
        steps = 0

        while el < self.ctx.prefix_budget and steps < max_steps:
            steps += 1
            if _goal_reached(self.mdp, current_state):
                trace.sample_value = 1.0
                trace.terminated_early = True
                trace.final_remaining = rem
                trace.final_elapsed = el
                return 1.0, trace

            if fixed_tail_dead_end_value(self.mdp, current_state, current_stn):
                trace.sample_value = 0.0
                trace.terminated_early = True
                trace.final_remaining = rem
                trace.final_elapsed = el
                return 0.0, trace

            legal = rollout_legal_fitting_actions(
                self.mdp, current_state, current_stn, current_prev
            )
            if not legal:
                trace.sample_value = 0.0
                trace.terminated_early = True
                trace.final_remaining = rem
                trace.final_elapsed = el
                return 0.0, trace

            action = pick_rollout_action(legal, self.config.rollout_policy, self.rng)
            if action is None:
                trace.sample_value = 0.0
                trace.terminated_early = True
                trace.final_remaining = rem
                trace.final_elapsed = el
                return 0.0, trace

            trace.actions.append(_action_name(action))
            terminal, next_state, _reward = self.mdp.step(current_state, action)
            if terminal:
                val = 1.0 if _goal_reached(self.mdp, next_state) else 0.0
                trace.sample_value = val
                trace.terminated_early = True
                rem = node_remaining(self.mdp, next_state, current_stn)
                trace.final_remaining = rem
                trace.final_elapsed = elapsed_from_root(self.ctx, rem)
                return val, trace

            fitted = _fit_action_stn(
                self.mdp, current_stn, current_prev, action
            )
            if fitted is None:
                trace.sample_value = 0.0
                trace.terminated_early = True
                trace.final_remaining = rem
                trace.final_elapsed = el
                return 0.0, trace

            current_stn, current_prev = fitted
            current_state = copy_state_for_rollout(next_state)
            rem = node_remaining(self.mdp, current_state, current_stn)
            el = elapsed_from_root(self.ctx, rem)

            if crossed_cutoff(self.ctx, el):
                break

        rem = node_remaining(self.mdp, current_state, current_stn)
        el = elapsed_from_root(self.ctx, rem)
        trace.final_remaining = rem
        trace.final_elapsed = el
        trace.ptrpg_horizon = rem
        trace.sample_value = ptrpg_at_horizon(
            self.mdp, current_state, current_stn, rem, self.strategy
        )
        return trace.sample_value, trace

    def evaluate_leaf(
        self,
        state: "up.engines.State",
        stn: "up.plans.stn.STNPlan",
        previous_action_node: "up.plans.stn.STNPlanNode",
        *,
        leaf_id: Optional[int] = None,
    ) -> float:
        if leaf_id is None:
            leaf_id = self._next_leaf_id
            self._next_leaf_id += 1

        leaf_rem = node_remaining(self.mdp, state, stn)
        leaf_elapsed = elapsed_from_root(self.ctx, leaf_rem)

        sample_values: List[float] = []
        sample_traces: List[_SampleTrace] = []
        for _ in range(self.config.num_samples):
            val, tr = self._run_one_sample(state, stn, previous_action_node)
            sample_values.append(val)
            sample_traces.append(tr)

        averaged = _clamp01(sum(sample_values) / len(sample_values))

        if self._should_debug():
            self._emit_debug(
                leaf_id=leaf_id,
                leaf_remaining=leaf_rem,
                leaf_elapsed=leaf_elapsed,
                sample_traces=sample_traces,
                averaged=averaged,
            )
            self._leaf_eval_count += 1

        return averaged

    def _emit_debug(
        self,
        *,
        leaf_id: int,
        leaf_remaining: int,
        leaf_elapsed: int,
        sample_traces: List[_SampleTrace],
        averaged: float,
    ) -> None:
        print(
            f"[fixed_tail_random_rollout_eval] selected_leaf_id={leaf_id} "
            f"K={self.config.num_samples} "
            f"root_remaining={self.ctx.root_remaining} "
            f"prefix_budget={self.ctx.prefix_budget} "
            f"leaf_remaining={leaf_remaining} "
            f"leaf_elapsed_from_root={leaf_elapsed} "
            f"rollout_policy={self.config.rollout_policy} "
            f"averaged_return_value={averaged:.6f} "
            f"rollout_path_inserted_into_tree=false",
            flush=True,
        )
        for idx, tr in enumerate(sample_traces):
            print(
                f"  sample={idx} actions=[{', '.join(tr.actions)}] "
                f"final_remaining={tr.final_remaining} "
                f"final_elapsed_from_root={tr.final_elapsed} "
                f"PTRPG_horizon={tr.ptrpg_horizon} "
                f"sample_value={tr.sample_value:.6f} "
                f"terminated_early={tr.terminated_early}",
                flush=True,
            )


__all__ = [
    "FixedTailRandomRolloutConfig",
    "FixedTailRandomRolloutEvaluator",
    "copy_state_for_rollout",
    "pick_rollout_action",
    "random_rollout_config_from_args",
    "rollout_legal_fitting_actions",
]
