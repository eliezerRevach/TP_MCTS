import math
from dataclasses import dataclass


@dataclass
class PlanResult:
    success: int
    plan_length: int
    makespan: float
    cumulative_reward: float


def _unsatisfied_goals(problem, predicates: set) -> int:
    return len(problem.goals.difference(predicates))


def _score_regular_action(mdp, state, action, stn, previous_action_node, heuristic_weight):
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

    terminal, next_state, reward = mdp.step(state, action)
    h = -_unsatisfied_goals(mdp.problem, next_state.predicates)
    score = reward + heuristic_weight * h - 0.001 * candidate_stn.get_current_end_time()
    return score, next_state, candidate_stn, candidate_prev, terminal, reward


def regular_greedy_plan(
    mdp,
    max_steps: int = 200,
    heuristic_weight: float = 0.2,
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
                mdp, root_state, action, stn, previous_action_node, heuristic_weight
            )
            if cand[0] > best_score:
                best_score = cand[0]
                best = cand

        if best is None or not math.isfinite(best_score):
            return PlanResult(0, step, stn.get_current_end_time(), cumulative_reward)

        _, root_state, stn, previous_action_node, terminal, reward = best
        cumulative_reward += reward
        if terminal:
            return PlanResult(1, step + 1, stn.get_current_end_time(), cumulative_reward)

    return PlanResult(0, max_steps, stn.get_current_end_time(), cumulative_reward)


def _score_combination_action(mdp, state, action, heuristic_weight, noop_penalty):
    terminal, next_state, reward = mdp.step(state, action)
    h = -_unsatisfied_goals(mdp.problem, next_state.predicates)
    score = reward + heuristic_weight * h - 0.001 * getattr(next_state, "current_time", 0)
    if action.name == "noop":
        score -= noop_penalty
    return score, next_state, terminal, reward


def combination_greedy_plan(
    mdp,
    max_steps: int = 200,
    heuristic_weight: float = 0.2,
    noop_penalty: float = 0.02,
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
                mdp, root_state, action, heuristic_weight, noop_penalty
            )
            if cand[0] > best_score:
                best_score = cand[0]
                best = cand

        if best is None or not math.isfinite(best_score):
            return PlanResult(0, step, root_state.current_time, cumulative_reward)

        _, root_state, terminal, reward = best
        cumulative_reward += reward
        if terminal:
            return PlanResult(1, step + 1, root_state.current_time, cumulative_reward)

    return PlanResult(0, max_steps, root_state.current_time, cumulative_reward)

