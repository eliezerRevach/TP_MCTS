"""
Approximate max action-set selection via goal-backtrack groups.

Builds a valid concurrent action set by greedily/stochastically committing the
action whose effects most raise the relaxed goal probability of the *current
table*, re-scoring after each commit (so overlap between groups is priced in),
then evaluates the resulting successor with the same PTRPG-like heuristic and
returns the best set over a few samples.

This is the "pick the highest-contribution group, re-score the new table, take
the next" loop. Scores come from the goal-backtrack DP itself (the marginal lift
``h(working ∪ add(a)) − h(working)``), so actions that contribute *indirectly*
(achieve a precondition of a goal-achiever) get positive credit — unlike a
forward "directly adds a goal fact" score, which collapses to zero on multi-step
problems. Inadmissible search bias — not a lower bound on true value.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Protocol, Sequence, Set, Tuple

import unified_planning as up
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
    # Minimal goal-probability lift for an action to be considered a positive
    # contributor during group construction.
    marginal_eps: float = 1e-9


@dataclass
class MaxApproximationDebug:
    heuristic_variant: str = ""
    action_scores: Dict[str, float] = field(default_factory=dict)
    sampled_sets: List[List[str]] = field(default_factory=list)
    rejections: List[Dict[str, str]] = field(default_factory=list)
    best_set: List[str] = field(default_factory=list)
    best_value: float = 0.0
    best_next_time: float = 0.0
    forced_fallback: bool = False


class HeuristicAdapter(Protocol):
    variant_label: str

    def eval_facts(
        self,
        fact_set: Set["object"],
        current_time: float,
        remaining_deadline: float,
    ) -> float:
        """Relaxed goal value of a raw fact set (higher = closer to goal)."""
        ...

    def action_add_facts(self, action: "up.engines.Action") -> FrozenSet["object"]:
        """Facts the action contributes to the relaxed table."""
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


def _state_facts(state) -> Set["object"]:
    preds = getattr(state, "predicates", None)
    if preds is not None:
        return set(preds)
    return set(state)


@dataclass
class _TprpgHeuristicAdapter:
    mdp: "up.engines.MDP"
    heuristic_name: str
    temporal_heuristic_strategy: str
    temporal_heuristic_depth: int
    resolution_kwargs: Dict[str, object]
    variant_label: str = ""
    # Scoring uses a TIME-GRADED measure (area = mean P_t(goal) over layers) on
    # the forward baseline table. The configured strategy (e.g. resolution) only
    # fills layers {0, depth} and its product value saturates to 1.0 at generous
    # horizons, giving every action a zero marginal. The area integral rewards
    # achieving the goal EARLIER, so it keeps a usable gradient for ranking; the
    # final set is still scored with the configured strategy in ``evaluate``.
    score_strategy: str = "baseline"
    score_aggregation: str = "area"

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

    def eval_facts(
        self,
        fact_set: Set["object"],
        current_time: float,
        remaining_deadline: float,
    ) -> float:
        heuristic = self._heuristic()
        depth = self._effective_depth(current_time, remaining_deadline)
        return float(
            heuristic.heuristic_score(
                set(fact_set),
                list(self.mdp.problem.goals),
                aggregation=self.score_aggregation,
                fixed_depth=depth,
                start_time=current_time,
                strategy=self.score_strategy,
            )
        )

    def action_add_facts(self, action: "up.engines.Action") -> FrozenSet["object"]:
        return self._heuristic().action_add_facts(_action_name(action))

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

    def eval_facts(
        self,
        fact_set: Set["object"],
        current_time: float,
        remaining_deadline: float,
    ) -> float:
        del remaining_deadline
        # TRPG returns a cost-to-go (lower = better); negate so the selector can
        # maximize uniformly.
        cost = up.engines.heuristics.TRPG(
            self.mdp, up.engines.State(set(fact_set)), int(current_time)
        ).get_heuristic()
        return -float(cost)

    def action_add_facts(self, action: "up.engines.Action") -> FrozenSet["object"]:
        adds = set(getattr(action, "add_effects", set()))
        return frozenset(adds)

    def evaluate(
        self,
        state: "up.engines.State",
        current_time: float,
        remaining_deadline: float,
    ) -> float:
        del remaining_deadline
        return -float(
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


def _build_action_set(
    legal_actions: Sequence["up.engines.Action"],
    mdp: "up.engines.MDP",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    adapter: HeuristicAdapter,
    base_facts: Set["object"],
    current_time: float,
    remaining_deadline: float,
    alpha: float,
    rng: random.Random,
    marginal_eps: float,
    stochastic: bool,
    eval_cache: Dict[FrozenSet["object"], float],
    rejections: List[Dict[str, str]],
    first_scores: Optional[Dict[str, float]] = None,
) -> List["up.engines.Action"]:
    """
    Greedy goal-backtrack group builder.

    Repeatedly commits the legal, non-mutex, STN-feasible action whose effects
    most raise the relaxed goal value of the running fact set (``working``),
    re-scoring against the *updated* table after each commit. Stops when no
    remaining action gives a positive marginal lift.
    """

    def cached_eval(facts: FrozenSet["object"]) -> float:
        hit = eval_cache.get(facts)
        if hit is None:
            hit = adapter.eval_facts(set(facts), current_time, remaining_deadline)
            eval_cache[facts] = hit
        return hit

    committed: List["up.engines.Action"] = []
    working: Set["object"] = set(base_facts)
    base_h = cached_eval(frozenset(working))

    while True:
        scored: List[Tuple["up.engines.Action", float, FrozenSet["object"]]] = []
        for action in legal_actions:
            if action in committed:
                continue
            name = _action_name(action)
            adds = adapter.action_add_facts(action)
            if not adds:
                continue
            if working.issuperset(adds):
                # Action adds nothing new to the current table; cannot lift the
                # goal probability from here.
                continue
            if any(
                adapter.actions_are_mutex(name, _action_name(other))
                for other in committed
            ):
                rejections.append({"action": name, "reason": "mutex"})
                continue
            if not _stn_feasible_with_set(
                mdp, stn, previous_action_node, committed, action
            ):
                rejections.append({"action": name, "reason": "stn"})
                continue
            new_facts = frozenset(working | set(adds))
            marginal = cached_eval(new_facts) - base_h
            if first_scores is not None and not committed:
                first_scores[name] = marginal
            if marginal > marginal_eps:
                scored.append((action, marginal, new_facts))

        if not scored:
            break

        if stochastic and len(scored) > 1:
            weights = [max(0.0, m) ** alpha for (_, m, _) in scored]
            total = sum(weights)
            if total <= 0.0:
                break
            picked = rng.choices(scored, weights=weights, k=1)[0]
        else:
            # Deterministic: highest marginal, name as a stable tie-break.
            picked = min(scored, key=lambda item: (-item[1], _action_name(item[0])))

        action, _marg, new_facts = picked
        committed.append(action)
        working = set(new_facts)
        base_h = eval_cache[new_facts]

    return _sorted_actions(committed)


def _single_best_fallback(
    legal_actions: Sequence["up.engines.Action"],
    mdp: "up.engines.MDP",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    adapter: HeuristicAdapter,
    base_facts: Set["object"],
    current_time: float,
    remaining_deadline: float,
    eval_cache: Dict[FrozenSet["object"], float],
) -> List["up.engines.Action"]:
    """
    When no action gives a positive lift (e.g. the heuristic is saturated),
    pick the single STN-feasible action with the highest absolute table value so
    the dispatcher still makes progress instead of stalling.
    """
    best_action = None
    best_value = -math.inf
    for action in legal_actions:
        adds = adapter.action_add_facts(action)
        if not adds:
            continue
        if not _stn_feasible_with_set(mdp, stn, previous_action_node, [], action):
            continue
        facts = frozenset(set(base_facts) | set(adds))
        value = eval_cache.get(facts)
        if value is None:
            value = adapter.eval_facts(set(facts), current_time, remaining_deadline)
            eval_cache[facts] = value
        if value > best_value or (
            math.isclose(value, best_value)
            and best_action is not None
            and _action_name(action) < _action_name(best_action)
        ):
            best_value = value
            best_action = action
    return [best_action] if best_action is not None else []


def _apply_action_set_sampled(
    mdp: "up.engines.MDP",
    state: "up.engines.State",
    stn: "up.plans.stn.STNPlan",
    previous_action_node: "up.plans.stn.STNPlanNode",
    action_set: Sequence["up.engines.Action"],
    rng: random.Random,
) -> Tuple["up.engines.State", "up.plans.stn.STNPlan", "up.plans.stn.STNPlanNode"]:
    del rng  # mdp.step samples internally; kept for signature stability.
    current_state = state
    current_stn = stn
    current_prev = previous_action_node
    for action in _sorted_actions(action_set):
        fitted = _fit_action_stn(mdp, current_stn, current_prev, action)
        if fitted is None:
            break
        current_stn, current_prev = fitted
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
    Goal-backtrack group approximation of ``argmax_A h(next_state(s, A), H')``.

    Builds candidate concurrent sets by greedily committing the highest
    marginal-lift action (re-scoring the table after each commit), evaluates each
    sampled set's real successor, and returns the best set.

    Returns ``(best_action_set, debug_info)``.
    """
    cfg = config or MaxApproximationConfig()
    rng = random.Random(cfg.seed)
    current_time = float(stn.get_current_end_time())
    debug_info = MaxApproximationDebug(heuristic_variant=adapter.variant_label)

    if not legal_actions:
        return [], debug_info

    base_facts = _state_facts(state)
    eval_cache: Dict[FrozenSet["object"], float] = {}

    best_set: List["up.engines.Action"] = []
    best_value = -math.inf
    best_next_time = current_time

    num_samples = max(1, int(cfg.num_samples))
    for sample_idx in range(num_samples):
        sample_rejections: List[Dict[str, str]] = []
        # First sample is the deterministic greedy build; the rest explore.
        stochastic = sample_idx > 0
        first_scores = debug_info.action_scores if sample_idx == 0 else None
        sampled = _build_action_set(
            legal_actions=legal_actions,
            mdp=mdp,
            stn=stn,
            previous_action_node=previous_action_node,
            adapter=adapter,
            base_facts=base_facts,
            current_time=current_time,
            remaining_deadline=remaining_deadline,
            alpha=float(cfg.alpha),
            rng=rng,
            marginal_eps=float(cfg.marginal_eps),
            stochastic=stochastic,
            eval_cache=eval_cache,
            rejections=sample_rejections,
            first_scores=first_scores,
        )
        debug_info.sampled_sets.append([_action_name(a) for a in sampled])
        if cfg.debug:
            debug_info.rejections.extend(sample_rejections)

        if not sampled:
            continue

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

    # Saturation fallback: nothing gave a positive lift across all samples.
    if not best_set:
        forced = _single_best_fallback(
            legal_actions,
            mdp,
            stn,
            previous_action_node,
            adapter,
            base_facts,
            current_time,
            remaining_deadline,
            eval_cache,
        )
        if forced:
            debug_info.forced_fallback = True
            next_state, next_stn, _ = _apply_action_set_sampled(
                mdp, state, stn, previous_action_node, forced, rng
            )
            best_set = forced
            best_next_time = float(next_stn.get_current_end_time())
            best_value = adapter.evaluate(
                next_state, best_next_time, remaining_deadline
            )

    debug_info.best_set = [_action_name(a) for a in best_set]
    debug_info.best_value = best_value if math.isfinite(best_value) else 0.0
    debug_info.best_next_time = best_next_time

    if cfg.debug:
        _print_max_approximation_debug(debug_info)

    return best_set, debug_info


def _print_max_approximation_debug(debug: MaxApproximationDebug) -> None:
    print(f"[max_approximation] heuristic={debug.heuristic_variant}")
    print(f"[max_approximation] first_marginal_scores={debug.action_scores}")
    for idx, sampled in enumerate(debug.sampled_sets):
        print(f"[max_approximation] sampled_set[{idx}]={sampled}")
    for rejection in debug.rejections[:50]:
        print(f"[max_approximation] rejection {rejection}")
    if len(debug.rejections) > 50:
        print(f"[max_approximation] ... {len(debug.rejections) - 50} more rejections")
    print(
        f"[max_approximation] best_set={debug.best_set} "
        f"value={debug.best_value:.6f} next_time={debug.best_next_time} "
        f"forced_fallback={debug.forced_fallback}"
    )
