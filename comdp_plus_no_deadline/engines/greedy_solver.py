import math
from dataclasses import dataclass

from comdp_plus_no_deadline.engines.probabilistic_rpg import (
    ProbabilisticOptimisticRPGHeuristic,
)
from comdp_plus_no_deadline.engines.temporal_probabilistic_rpg import (
    TemporalProbabilisticRPGHeuristic,
)


@dataclass
class PlanResult:
    success: int
    plan_length: int
    makespan: float
    cumulative_reward: float


def _unsatisfied_goals(problem, predicates: set) -> int:
    return len(problem.goals.difference(predicates))


def _get_probabilistic_rpg_heuristic(mdp):
    heuristic = getattr(mdp, "_probabilistic_rpg_heuristic", None)
    if heuristic is None:
        heuristic = ProbabilisticOptimisticRPGHeuristic.from_problem(mdp.problem)
        setattr(mdp, "_probabilistic_rpg_heuristic", heuristic)
    return heuristic


def _get_temporal_probabilistic_rpg_heuristic(mdp):
    heuristic = getattr(mdp, "_temporal_probabilistic_rpg_heuristic", None)
    if heuristic is None:
        heuristic = TemporalProbabilisticRPGHeuristic.from_problem(mdp.problem)
        setattr(mdp, "_temporal_probabilistic_rpg_heuristic", heuristic)
    return heuristic


def _effective_temporal_depth(
    temporal_heuristic_depth: int,
    temporal_current_time: float,
    deadline: float | None,
) -> int:
    configured_depth = max(0, int(temporal_heuristic_depth))
    if deadline is None:
        return configured_depth
    remaining = max(0, int(math.floor(deadline - temporal_current_time)))
    return min(configured_depth, remaining)


def _state_heuristic_score(
    mdp,
    state,
    heuristic_name,
    heuristic_aggregation,
    heuristic_layers,
    heuristic_epsilon,
    goal_threshold,
    temporal_heuristic_depth,
    temporal_current_time,
    deadline,
):
    if heuristic_name == "goal_count":
        return -_unsatisfied_goals(mdp.problem, state.predicates)
    if heuristic_name == "probabilistic_rpg":
        heuristic = _get_probabilistic_rpg_heuristic(mdp)
        return heuristic.heuristic_score(
            state,
            mdp.problem.goals,
            aggregation=heuristic_aggregation,
            max_layers=heuristic_layers,
            epsilon=heuristic_epsilon,
            goal_threshold=goal_threshold,
        )
    if heuristic_name == "temporal_probabilistic_rpg":
        heuristic = _get_temporal_probabilistic_rpg_heuristic(mdp)
        effective_depth = _effective_temporal_depth(
            temporal_heuristic_depth,
            temporal_current_time,
            deadline,
        )
        return heuristic.heuristic_score(
            state,
            mdp.problem.goals,
            aggregation=heuristic_aggregation,
            fixed_depth=effective_depth,
            start_time=temporal_current_time,
        )
    raise ValueError(f"Unknown heuristic_name: {heuristic_name}")


def _score_regular_action(
    mdp,
    state,
    action,
    stn,
    previous_action_node,
    heuristic_weight,
    heuristic_name,
    heuristic_aggregation,
    heuristic_layers,
    heuristic_epsilon,
    goal_threshold,
    temporal_heuristic_depth,
    deadline,
):
    from unified_planning.engines.utils import update_stn

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
        return -math.inf, None, None, None, False, 0.0

    if not candidate_stn.is_consistent():
        return -math.inf, None, None, None, False, 0.0
    if deadline is not None and candidate_stn.get_current_end_time() > deadline:
        return -math.inf, None, None, None, False, 0.0

    terminal, next_state, reward = mdp.step(state, action)
    h = _state_heuristic_score(
        mdp,
        next_state,
        heuristic_name,
        heuristic_aggregation,
        heuristic_layers,
        heuristic_epsilon,
        goal_threshold,
        temporal_heuristic_depth,
        candidate_stn.get_current_end_time(),
        deadline,
    )
    score = reward + heuristic_weight * h - 0.001 * candidate_stn.get_current_end_time()
    return score, next_state, candidate_stn, candidate_prev, terminal, reward


def regular_greedy_plan(
    mdp,
    max_steps: int = 200,
    heuristic_weight: float = 0.2,
    heuristic_name: str = "goal_count",
    heuristic_aggregation: str = "product",
    heuristic_layers: int = 25,
    heuristic_epsilon: float = 1e-6,
    goal_threshold: float = 0.99,
    temporal_heuristic_depth: int = 25,
    deadline: float | None = None,
) -> PlanResult:
    from unified_planning.engines.utils import create_init_stn

    stn = create_init_stn(mdp)
    root_state = mdp.initial_state()
    previous_action_node = None
    cumulative_reward = 0.0

    for step in range(max_steps):
        legal_actions = mdp.legal_actions(root_state)
        if not legal_actions:
            return PlanResult(0, step, stn.get_current_end_time(), cumulative_reward)

        best = None
        best_score = -math.inf
        for action in legal_actions:
            cand = _score_regular_action(
                mdp,
                root_state,
                action,
                stn,
                previous_action_node,
                heuristic_weight,
                heuristic_name,
                heuristic_aggregation,
                heuristic_layers,
                heuristic_epsilon,
                goal_threshold,
                temporal_heuristic_depth,
                deadline,
            )
            if cand[0] > best_score:
                best_score = cand[0]
                best = cand

        if best is None or not math.isfinite(best_score):
            return PlanResult(0, step, stn.get_current_end_time(), cumulative_reward)

        _, root_state, stn, previous_action_node, terminal, reward = best
        if deadline is not None and stn.get_current_end_time() > deadline:
            return PlanResult(0, step + 1, stn.get_current_end_time(), cumulative_reward)
        cumulative_reward += reward
        if terminal:
            return PlanResult(1, step + 1, stn.get_current_end_time(), cumulative_reward)

    return PlanResult(0, max_steps, stn.get_current_end_time(), cumulative_reward)


def _score_combination_action(
    mdp,
    state,
    action,
    heuristic_weight,
    noop_penalty,
    heuristic_name,
    heuristic_aggregation,
    heuristic_layers,
    heuristic_epsilon,
    goal_threshold,
    temporal_heuristic_depth,
    deadline,
):
    terminal, next_state, reward = mdp.step(state, action)
    if deadline is not None and getattr(next_state, "current_time", 0.0) > deadline:
        return -math.inf, None, False, 0.0
    h = _state_heuristic_score(
        mdp,
        next_state,
        heuristic_name,
        heuristic_aggregation,
        heuristic_layers,
        heuristic_epsilon,
        goal_threshold,
        temporal_heuristic_depth,
        getattr(next_state, "current_time", 0.0),
        deadline,
    )
    score = reward + heuristic_weight * h - 0.001 * getattr(next_state, "current_time", 0)
    if action.name == "noop":
        score -= noop_penalty
    return score, next_state, terminal, reward


def combination_greedy_plan(
    mdp,
    max_steps: int = 200,
    heuristic_weight: float = 0.2,
    noop_penalty: float = 0.02,
    heuristic_name: str = "goal_count",
    heuristic_aggregation: str = "product",
    heuristic_layers: int = 25,
    heuristic_epsilon: float = 1e-6,
    goal_threshold: float = 0.99,
    temporal_heuristic_depth: int = 25,
    deadline: float | None = None,
) -> PlanResult:
    root_state = mdp.initial_state()
    cumulative_reward = 0.0

    for step in range(max_steps):
        legal_actions = mdp.legal_actions(root_state)
        if not legal_actions:
            return PlanResult(0, step, root_state.current_time, cumulative_reward)

        best = None
        best_score = -math.inf
        for action in legal_actions:
            cand = _score_combination_action(
                mdp,
                root_state,
                action,
                heuristic_weight,
                noop_penalty,
                heuristic_name,
                heuristic_aggregation,
                heuristic_layers,
                heuristic_epsilon,
                goal_threshold,
                temporal_heuristic_depth,
                deadline,
            )
            if cand[0] > best_score:
                best_score = cand[0]
                best = cand

        if best is None or not math.isfinite(best_score):
            return PlanResult(0, step, root_state.current_time, cumulative_reward)

        _, root_state, terminal, reward = best
        if deadline is not None and root_state.current_time > deadline:
            return PlanResult(0, step + 1, root_state.current_time, cumulative_reward)
        cumulative_reward += reward
        if terminal:
            return PlanResult(1, step + 1, root_state.current_time, cumulative_reward)

    return PlanResult(0, max_steps, root_state.current_time, cumulative_reward)

