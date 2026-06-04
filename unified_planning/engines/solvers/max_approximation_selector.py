"""
Approximate max action-set selection for parallel greedy dispatch.

Samples valid concurrent action sets stochastically (score^alpha), evaluates each
with a PTRPG-like heuristic on the successor state, and returns the best set.
Inadmissible search bias — not a lower bound on true value.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import unified_planning as up
from unified_planning.engines.utils import update_stn
from unified_planning.engines.solvers.fixed_tail_expectimax import _fit_action_stn
from unified_planning.engines.solvers.mcts import (
    _effective_temporal_depth,
    _resolution_heuristic_kwargs_from_cli,
    _uses_tprpg_family,
)


@dataclass
class MaxApproximationConfig:
    alpha: float = 1.5
    num_samples: int = 32
    seed: Optional[int] = None
    debug: bool = False


@dataclass
class MaxApproximationDebug:
    heuristic_variant: str = ""
    action_scores: Dict[str, float] = field(default_factory=dict)
    sampled_sets: List[List[str]] = field(default_factory=list)
    rejections: List[Dict[str, str]] = field(default_factory=list)
    best_set: List[str] = field(default_factory=list)
    best_value: float = 0.0
    best_next_time: float = 0.0


class HeuristicAdapter(Protocol):
    variant_label: str

    def action_scores(
        self,
        state: "up.engines.State",
        legal_actions: Sequence["up.engines.Action"],
        remaining_deadline: float,
        current_time: float,
    ) -> Dict[str, float]:
        ...

    def evaluate(
        self,
        state: "up.engines.State",
        current_time: float,
        remaining_deadline: float,
    ) -> float:
        ...

    def actions_are_mutex(self, name_a: str, name_b: str) -> bool:
        ...


def _action_name(action: "up.engines.Action") -> str:
    name = getattr(action, "name", None)
    return name if name is not None else str(action)


def _sorted_actions(actions: Sequence["up.engines.Action"]) -> List["up.engines.Action"]:
    return sorted(actions, key=_action_name)


@dataclass
class _TprpgHeuristicAdapter:
    mdp: "up.engines.MDP"
    heuristic_name: str
    temporal_heuristic_strategy: str
    temporal_heuristic_depth: int
    resolution_kwargs: Dict[str, object]
    variant_label: str = ""

    def __post_init__(self) -> None:
        if not self.variant_label:
            self.variant_label = (
                f"{self.heuristic_name}/{self.temporal_heuristic_strategy}"
            )

    def _heuristic(self):
        from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
            TemporalProbabilisticRPGHeuristic,
        )

        h = getattr(self.mdp, "_temporal_probabilistic_rpg_heuristic", None)
        if h is None:
            h = TemporalProbabilisticRPGHeuristic.from_problem(self.mdp.problem)
            setattr(self.mdp, "_temporal_probabilistic_rpg_heuristic", h)
        return h

    def _effective_depth(self, current_time: float, remaining_deadline: float) -> int:
        depth = _effective_temporal_depth(
            self.temporal_heuristic_depth,
            current_time,
            self.mdp.deadline(),
        )
        if remaining_deadline is not None and math.isfinite(remaining_deadline):
            depth = min(depth, max(0, int(math.floor(remaining_deadline))))
        return depth

    def action_scores(
        self,
        state: "up.engines.State",
        legal_actions: Sequence["up.engines.Action"],
        remaining_deadline: float,
        current_time: float,
    ) -> Dict[str, float]:
        heuristic = self._heuristic()
        depth = self._effective_depth(current_time, remaining_deadline)
        names = [_action_name(a) for a in legal_actions]
        return heuristic.action_contribution_scores(
            state,
            self.mdp.problem.goals,
            fixed_depth=depth,
            start_time=current_time,
            strategy=self.temporal_heuristic_strategy,
            legal_action_names=names,
            **self.resolution_kwargs,
        )

    def evaluate(
        self,
        state: "up.engines.State",
        current_time: float,
        remaining_deadline: float,
    ) -> float:
        from unified_planning.engines.solvers.mcts import _temporal_heuristic

        depth = self._effective_depth(current_time, remaining_deadline)
        return float(
            _temporal_heuristic(
                self.mdp,
                state,
                current_time,
                depth,
                self.temporal_heuristic_strategy,
                leaf_heuristic_name=self.heuristic_name,
            )
        )

    def actions_are_mutex(self, name_a: str, name_b: str) -> bool:
        return self._heuristic().actions_are_mutex(name_a, name_b)


@dataclass
class _TrpgFallbackAdapter:
    mdp: "up.engines.MDP"
    variant_label: str = "trpg_fallback"

    def action_scores(
        self,
        state: "up.engines.State",
        legal_actions: Sequence["up.engines.Action"],
        remaining_deadline: float,
        current_time: float,
    ) -> Dict[str, float]:
        del remaining_deadline
        base = up.engines.heuristics.TRPG(self.mdp, state, int(current_time)).get_heuristic()
        scores: Dict[str, float] = {}
        for action in legal_actions:
            preds = set(state.predicates)
            preds |= action.add_effects
            preds -= action.del_effects
            next_state = up.engines.State(preds)
            h_next = up.engines.heuristics.TRPG(
                self.mdp, next_state, int(current_time)
            ).get_heuristic()
            scores[_action_name(action)] = max(0.0, float(base) - float(h_next))
        if not any(v > 0 for v in scores.values()):
            return {_action_name(a): 1.0 for a in legal_actions}
        return scores

    def evaluate(
        self,
        state: "up.engines.State",
        current_time: float,
        remaining_deadline: float,
    ) -> float:
        del remaining_deadline
        return float(
            up.engines.heuristics.TRPG(self.mdp, state, int(current_time)).get_heuristic()
        )

    def actions_are_mutex(self, name_a: str, name_b: str) -> bool:
        del name_a, name_b
        return False


def build_heuristic_adapter(
    mdp: "up.engines.MDP",
    heuristic_name: str,
    temporal_heuristic_strategy: str,
    temporal_heuristic_depth: int,
    *,
    resolution_kwargs: Optional[Dict[str, object]] = None,
) -> HeuristicAdapter:
    if _uses_tprpg_family(heuristic_name):
        return _TprpgHeuristicAdapter(
            mdp=mdp,
            heuristic_name=heuristic_name,
            temporal_heuristic_strategy=temporal_heuristic_strategy,
            temporal_heuristic_depth=temporal_heuristic_depth,
            resolution_kwargs=resolution_kwargs or _resolution_heuristic_kwargs_from_cli(),
        )
    return _TrpgFallbackAdapter(mdp=mdp)


def _stn_feasible_with_set(
    mdp: "up.engines.MDP",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    action_set: Sequence["up.engines.Action"],
    candidate: "up.engines.Action",
) -> bool:
    trial_stn = stn.clone()
    trial_prev = previous_action_node
    ordered = _sorted_actions(list(action_set) + [candidate])
    for action in ordered:
        fitted = _fit_action_stn(mdp, trial_stn, trial_prev, action)
        if fitted is None:
            return False
        trial_stn, trial_prev = fitted
    return True


def _sample_action_set(
    legal_actions: Sequence["up.engines.Action"],
    action_scores: Dict[str, float],
    mdp: "up.engines.MDP",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    adapter: HeuristicAdapter,
    alpha: float,
    rng: random.Random,
    rejections: List[Dict[str, str]],
) -> List["up.engines.Action"]:
    chosen: List["up.engines.Action"] = []

    while True:
        candidates: List["up.engines.Action"] = []
        for action in legal_actions:
            name = _action_name(action)
            if action in chosen:
                continue
            score = action_scores.get(name, 0.0)
            if score <= 0.0:
                if name not in {r.get("action") for r in rejections}:
                    rejections.append({"action": name, "reason": "zero_score"})
                continue
            blocked = False
            for other in chosen:
                if adapter.actions_are_mutex(name, _action_name(other)):
                    rejections.append(
                        {"action": name, "reason": f"mutex_with:{_action_name(other)}"}
                    )
                    blocked = True
                    break
            if blocked:
                continue
            if not _stn_feasible_with_set(mdp, stn, previous_action_node, chosen, action):
                rejections.append({"action": name, "reason": "stn"})
                continue
            candidates.append(action)

        if not candidates:
            break

        weights = [
            max(0.0, action_scores.get(_action_name(a), 0.0)) ** alpha for a in candidates
        ]
        total = sum(weights)
        if total <= 0.0:
            break
        picked = rng.choices(candidates, weights=weights, k=1)[0]
        chosen.append(picked)

    return _sorted_actions(chosen)


def _apply_action_set_sampled(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    action_set: Sequence["up.engines.Action"],
    rng: random.Random,
) -> Tuple["up.engines.State", "up.plans.stn.STNPlan", "up.plans.stn.STNPlanNode"]:
    current_state = state
    current_stn = stn
    current_prev = previous_action_node
    for action in _sorted_actions(action_set):
        fitted = _fit_action_stn(mdp, current_stn, current_prev, action)
        if fitted is None:
            break
        current_stn, current_prev = fitted
        # One stochastic draw per action (evaluation-only trajectory).
        _terminal, next_state, _reward = mdp.step(current_state, action)
        current_state = next_state
    return current_state, current_stn, current_prev


def select_max_approximation_action_set(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    legal_actions: Sequence["up.engines.Action"],
    adapter: HeuristicAdapter,
    remaining_deadline: float,
    config: Optional[MaxApproximationConfig] = None,
) -> Tuple[List["up.engines.Action"], MaxApproximationDebug]:
    """
    Stochastic greedy approximation of argmax_A h(next_state(s, A), H').

    Returns (best_action_set, debug_info).
    """
    cfg = config or MaxApproximationConfig()
    rng = random.Random(cfg.seed)
    current_time = float(stn.get_current_end_time())
    debug_info = MaxApproximationDebug(heuristic_variant=adapter.variant_label)

    if not legal_actions:
        return [], debug_info

    action_scores = adapter.action_scores(
        state, legal_actions, remaining_deadline, current_time
    )
    debug_info.action_scores = dict(action_scores)

    best_set: List["up.engines.Action"] = []
    best_value = -math.inf
    best_next_time = current_time

    num_samples = max(1, int(cfg.num_samples))
    for _ in range(num_samples):
        sample_rejections: List[Dict[str, str]] = []
        sampled = _sample_action_set(
            legal_actions=legal_actions,
            action_scores=action_scores,
            mdp=mdp,
            stn=stn,
            previous_action_node=previous_action_node,
            adapter=adapter,
            alpha=float(cfg.alpha),
            rng=rng,
            rejections=sample_rejections,
        )
        debug_info.sampled_sets.append([_action_name(a) for a in sampled])
        if cfg.debug:
            debug_info.rejections.extend(sample_rejections)

        if not sampled:
            value = adapter.evaluate(state, current_time, remaining_deadline)
            next_time = current_time
        else:
            next_state, next_stn, _ = _apply_action_set_sampled(
                mdp, state, stn, previous_action_node, sampled, rng
            )
            next_time = float(next_stn.get_current_end_time())
            rem = (
                remaining_deadline
                if remaining_deadline is not None and math.isfinite(remaining_deadline)
                else max(0.0, float(mdp.deadline()) - next_time)
            )
            value = adapter.evaluate(next_state, next_time, rem)

        if value > best_value or (
            math.isclose(value, best_value)
            and [_action_name(a) for a in sampled] < [_action_name(a) for a in best_set]
        ):
            best_value = value
            best_set = list(sampled)
            best_next_time = next_time

    debug_info.best_set = [_action_name(a) for a in best_set]
    debug_info.best_value = best_value if math.isfinite(best_value) else 0.0
    debug_info.best_next_time = best_next_time

    if cfg.debug:
        _print_max_approximation_debug(debug_info)

    return best_set, debug_info


def _print_max_approximation_debug(debug: MaxApproximationDebug) -> None:
    print(f"[max_approximation] heuristic={debug.heuristic_variant}")
    print(f"[max_approximation] action_scores={debug.action_scores}")
    for idx, sampled in enumerate(debug.sampled_sets):
        print(f"[max_approximation] sampled_set[{idx}]={sampled}")
    for rejection in debug.rejections[:50]:
        print(f"[max_approximation] rejection {rejection}")
    if len(debug.rejections) > 50:
        print(f"[max_approximation] ... {len(debug.rejections) - 50} more rejections")
    print(
        f"[max_approximation] best_set={debug.best_set} "
        f"value={debug.best_value:.6f} next_time={debug.best_next_time}"
    )
